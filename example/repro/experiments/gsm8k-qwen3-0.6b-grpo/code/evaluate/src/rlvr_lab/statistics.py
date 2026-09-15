"""Paired, fail-closed held-out evaluation acceptance (no SciPy required)."""

from __future__ import annotations

import argparse
import io
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_manifest_rows(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                            predictions_sha256: str | None = None) -> None:
    unsigned = {key: value for key, value in metadata.items() if key != "manifest_sha256"}
    if metadata.get("manifest_sha256") != _digest(unsigned):
        raise ValueError("Corrupt or missing manifest SHA256")
    if metadata.get("evaluation_config_sha256") != _digest(metadata.get("evaluation_config")):
        raise ValueError("Corrupt evaluation config SHA256")
    row_order = [[r["id"], r["question_hash"], r["project"], r.get("source_split")] for r in rows]
    if metadata.get("row_order_sha256") != _digest(row_order):
        raise ValueError("Manifest row order SHA256 does not match predictions")
    if predictions_sha256 is None:
        # evaluate_rows emits precisely this JSONL encoding. For in-memory calls,
        # bind the supplied dictionaries to that original artifact representation.
        raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode()
        predictions_sha256 = hashlib.sha256(raw).hexdigest()
    if metadata.get("predictions_sha256") != predictions_sha256:
        raise ValueError("Manifest prediction SHA256 does not match supplied rows")
    if not metadata.get("predictions_file"):
        raise ValueError("Manifest is missing predictions_file")
    for key in ("reward_contract_sha256", "dataset_content_sha256", "rendered_prompts_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid manifest {key}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar_pvalue(losses: int, wins: int) -> float:
    """Two-sided exact binomial McNemar test; stable for large discordant counts."""
    if losses < 0 or wins < 0:
        raise ValueError("Discordant counts cannot be negative")
    n = losses + wins
    if n == 0:
        return 1.0
    k = min(losses, wins)
    log_peak = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
    # Sum relative to the largest term in the lower tail to avoid underflow.
    relative_sum = term = 1.0
    for j in range(k, 0, -1):
        term *= j / (n - j + 1)
        relative_sum += term
    return min(1.0, math.exp(log_peak + math.log(relative_sum) + math.log(2)))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    if not pvalues or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Holm correction requires finite p-values in [0, 1]")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    result = [1.0] * len(pvalues)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        result[i] = previous
    return result


def _validate_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    if not rows:
        raise ValueError(f"{label}: empty predictions")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for key in ("id", "project", "question_hash", "correctness"):
            if key not in row:
                raise ValueError(f"{label}: missing {key}")
        item_id = row["id"]
        if not isinstance(item_id, str) or not item_id.strip() or item_id in indexed:
            raise ValueError(f"{label}: corrupt or duplicate ID {item_id!r}")
        if not isinstance(row["project"], str) or not row["project"]:
            raise ValueError(f"{label}: invalid project for {item_id}")
        question_hash = row["question_hash"]
        if not isinstance(question_hash, str) or len(question_hash) != 64 or any(c not in "0123456789abcdef" for c in question_hash):
            raise ValueError(f"{label}: invalid question_hash for {item_id}")
        correctness = row["correctness"]
        if isinstance(correctness, bool) or not isinstance(correctness, (int, float)) or not math.isfinite(correctness) or correctness not in (0, 1):
            raise ValueError(f"{label}: correctness must be finite and binary for {item_id}")
        indexed[item_id] = row
    return indexed


def _verify_pair_metadata(before: Mapping[str, Any], after: Mapping[str, Any], count: int) -> None:
    for label, meta in (("before", before), ("after", after)):
        if meta.get("status") != "complete" or meta.get("counts", {}).get("evaluated") != count:
            raise ValueError(f"{label}: incomplete evaluation manifest")
        if meta.get("counts", {}).get("expected") != count:
            raise ValueError(f"{label}: expected/evaluated row count mismatch")
    # Model and adapter are intentionally excluded: their change is the treatment.
    for key in ("evaluation_config_sha256", "row_order_sha256", "reward_contract_sha256", "dataset_content_sha256", "rendered_prompts_sha256"):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError(f"Incompatible evaluation manifests: {key}")
    if before.get("evaluation_config") != after.get("evaluation_config"):
        raise ValueError("Incompatible generation/evaluation settings")


def compare_predictions(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260910,
    min_absolute_gain: float = 0.05,
    family_size: int = 2,
    before_metadata: Mapping[str, Any] | None = None,
    after_metadata: Mapping[str, Any] | None = None,
    require_manifests: bool = False,
) -> dict[str, Any]:
    """Compare one domain; standalone decisions reserve multiplicity for two domains.

    The CLI requires verified manifests. In-memory callers may omit manifests for
    statistical exploration; their result explicitly reports that limitation.
    """
    if bootstrap_samples < 1000 or family_size < 1 or not math.isfinite(min_absolute_gain) or not 0 < min_absolute_gain <= 1:
        raise ValueError("Invalid bootstrap, family size, or gain threshold")
    b = _validate_rows(before, "before")
    a = _validate_rows(after, "after")
    if set(b) != set(a):
        raise ValueError("Before/after ID sets differ; missing rows cannot be dropped")
    if list(b) != list(a):
        raise ValueError("Before/after row order differs; batch seed pairing is invalid")
    projects = {row["project"] for row in before}
    if len(projects) != 1:
        raise ValueError("One comparison must contain exactly one project")
    manifests_verified = before_metadata is not None and after_metadata is not None
    if require_manifests and not manifests_verified:
        raise ValueError("Complete evaluation manifests are required for acceptance")
    if (before_metadata is None) != (after_metadata is None):
        raise ValueError("Both manifests must be supplied together")
    for item_id, left in b.items():
        right = a[item_id]
        for key in ("project", "question_hash", "source_split", "evaluation_config_sha256", "reward_contract_sha256"):
            if left.get(key) != right.get(key):
                raise ValueError(f"Mismatched {key} for ID {item_id}")
        if require_manifests:
            if not left.get("source_split"):
                raise ValueError(f"Missing source split: {item_id}")
            for key in ("evaluation_config_sha256", "reward_contract_sha256"):
                if not left.get(key) or left.get(key) != (before_metadata or {}).get(key):
                    raise ValueError(f"Missing or inconsistent row contract: {key}")
    if manifests_verified:
        _verify_pair_metadata(before_metadata, after_metadata, len(b))
        _validate_manifest_rows(before_metadata, before)
        _validate_manifest_rows(after_metadata, after)
    n = len(b)
    before_correct = sum(int(row["correctness"]) for row in before)
    after_correct = sum(int(row["correctness"]) for row in after)
    wins = sum(b[item_id]["correctness"] == 0 and a[item_id]["correctness"] == 1 for item_id in b)
    losses = sum(b[item_id]["correctness"] == 1 and a[item_id]["correctness"] == 0 for item_id in b)
    ties = n - wins - losses
    delta = (wins - losses) / n
    # For binary accuracy, paired nonparametric resampling is exactly this
    # multinomial over {-1, 0, +1}; avoid allocating bootstrap_samples x n.
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [losses / n, ties / n, wins / n], size=bootstrap_samples)
    bootstrap_deltas = (draws[:, 2] - draws[:, 0]) / n
    interval = np.quantile(bootstrap_deltas, [0.025, 0.975]).tolist()
    pvalue = exact_mcnemar_pvalue(losses, wins)
    adjusted = min(1.0, pvalue * family_size)
    checks = {
        "absolute_gain_at_least_threshold": delta >= min_absolute_gain - 1e-12,
        "paired_bootstrap_lower_above_zero": interval[0] > 0,
        "multiplicity_adjusted_mcnemar_below_0_05": adjusted < 0.05,
    }
    result = {
        "project": next(iter(projects)), "n": n,
        "before": {"accuracy": before_correct / n, "correct": before_correct, "wilson_95_ci": wilson_interval(before_correct, n)},
        "after": {"accuracy": after_correct / n, "correct": after_correct, "wilson_95_ci": wilson_interval(after_correct, n)},
        "absolute_gain": delta, "absolute_gain_percentage_points": 100 * delta,
        "paired_flips": {"wrong_to_right": wins, "right_to_wrong": losses, "unchanged": ties},
        "paired_bootstrap_95_ci": interval, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "mcnemar_exact_two_sided_p": pvalue, "adjusted_p": adjusted,
        "multiplicity_method": "Bonferroni bound on Holm for standalone comparison",
        "family_size": family_size, "minimum_absolute_gain": min_absolute_gain,
        "checks": checks, "statistical_pass": all(checks.values()),
        "manifests_verified": manifests_verified,
        "accepted": all(checks.values()) and manifests_verified,
        "limitations": [
            "Inference concerns the locked held-out questions, not all tasks in the domain.",
            "One sampled response per model/question; batch seeds do not couple identical random draws after generation paths diverge.",
            "Seed and batch order matching do not remove generation variance or establish multi-seed training robustness.",
            "A single-domain result does not establish completion of the two-domain objective.",
        ],
    }
    json.dumps(result, allow_nan=False)
    return result


def load_predictions(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(io.StringIO(raw.decode("utf-8")), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank prediction row")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(row)
    summary_path = path.parent / "evaluation_summary.json"
    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    if metadata.get("predictions_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"{path}: prediction SHA256 does not match completed manifest")
    if metadata.get("predictions_file") != path.name:
        raise ValueError(f"{path}: manifest names a different predictions file")
    _validate_rows(rows, str(path))
    _validate_manifest_rows(metadata, rows, hashlib.sha256(raw).hexdigest())
    return rows, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--comparison", nargs=2, action="append", metavar=("BEFORE", "AFTER"), help="Repeat for final Holm correction across domain comparisons")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--min-absolute-gain", type=float, default=0.05)
    parser.add_argument("--family-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.comparison and (args.before or args.after):
        parser.error("Use either --comparison or --before/--after")
    if not args.comparison and not (args.before and args.after):
        parser.error("Provide --before and --after, or repeated --comparison pairs")
    pairs = args.comparison or [(args.before, args.after)]
    results = []
    try:
        for before_path, after_path in pairs:
            before, before_meta = load_predictions(before_path)
            after, after_meta = load_predictions(after_path)
            result = compare_predictions(before, after, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                min_absolute_gain=args.min_absolute_gain, family_size=max(args.family_size, len(pairs)),
                before_metadata=before_meta, after_metadata=after_meta, require_manifests=True)
            result["evidence"] = {"before": str(Path(before_path).resolve()), "after": str(Path(after_path).resolve()),
                "before_manifest_sha256": before_meta["manifest_sha256"], "after_manifest_sha256": after_meta["manifest_sha256"]}
            results.append(result)
        if len(results) > 1:
            if len({r["project"] for r in results}) != len(results):
                raise ValueError("Final domain comparisons must have distinct project names")
            # Include unreached family members as p=1 so a larger declared family
            # never silently receives a weaker multiplicity correction.
            pvalues = [r["mcnemar_exact_two_sided_p"] for r in results]
            pvalues += [1.0] * max(0, args.family_size - len(results))
            adjusted = holm_adjust(pvalues)
            for result, pvalue in zip(results, adjusted):
                result["adjusted_p"] = pvalue
                result["multiplicity_method"] = "Holm step-down exact McNemar"
                result["checks"]["multiplicity_adjusted_mcnemar_below_0_05"] = pvalue < 0.05
                result["statistical_pass"] = all(result["checks"].values())
                result["accepted"] = result["statistical_pass"] and result["manifests_verified"]
        report = results[0] if len(results) == 1 else {
            "comparisons": results, "accepted": all(r["accepted"] for r in results),
            "two_domain_statistical_acceptance": len(results) >= 2 and all(r["accepted"] for r in results),
            "note": "Statistical acceptance must be combined with training-integrity and provenance evidence.",
        }
    except (ValueError, OSError, KeyError, TypeError) as error:
        report = {"accepted": False, "status": "invalid_evidence", "error": str(error)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report.get("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
