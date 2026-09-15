"""Canonical public-dataset preparation with split and provenance audits.

Importing this module has no network effects. The CLI imports ``datasets`` only
when downloading/loading a source dataset, so verifier tests need only Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .rewards import parse_numeric

COD_INSTRUCTION = (
    "Think step by step, but only keep a minimum draft for each thinking step, "
    "with 5 words at most. Return the answer at the end of the response."
)
SOURCES = {
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "url": "https://huggingface.co/datasets/openai/gsm8k", "upstream": "https://github.com/openai/grade-school-math"},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def question_hash(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if not normalized:
        raise ValueError("question cannot be empty")
    return sha256_text(normalized)


def build_prompt(
    question: str,
    choices: Sequence[Mapping[str, str]] | Mapping[str, str] | None = None,
    mode: str = "cot",
) -> list[dict[str, str]]:
    """Make chat messages; ``choices=None`` selects the numeric answer contract."""
    if mode not in {"cot", "cod"}:
        raise ValueError(f"unsupported prompt mode: {mode}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty")
    system = "Solve the problem carefully. "
    system += COD_INSTRUCTION if mode == "cod" else "Reason step by step before giving the final answer."
    if choices is None:
        system += "\nPut the final answer on the last line in exactly this format:\nAnswer: <number>"
        user = question.strip()
    else:
        raise ValueError("Only numeric GSM8K prompts are included")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def canonicalize_row(project: str, raw: Mapping[str, Any], source_split: str, index: int, seed: int, mode: str) -> dict[str, Any]:
    question = str(raw["question"]).strip()
    qhash = question_hash(question)
    source_id = str(raw.get("id", index))
    row_id = f"{project}:{source_split}:{source_id}"
    metadata: dict[str, Any] = {"source_id": source_id, "source_index": index, "prompt_mode": mode}
    choices = None
    if project == "gsm8k":
        original_answer = str(raw["answer"])
        if "####" not in original_answer:
            raise ValueError(f"missing GSM8K answer delimiter: {row_id}")
        number = parse_numeric(original_answer.rsplit("####", 1)[1].strip())
        answer = str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"
        # Keep solution provenance as a hash; never insert solutions into prompts.
        metadata["source_answer_sha256"] = sha256_text(original_answer)
    else:
        raise ValueError(f"unsupported project: {project}")
    row = {
        "id": row_id,
        "project": project,
        "prompt": build_prompt(question, choices, mode),
        "answer": answer,
        "question_hash": qhash,
        "source_split": source_split,
        "metadata": metadata,
        "source_record_sha256": sha256_text(canonical_json(dict(raw))),
    }
    row["record_sha256"] = sha256_text(canonical_json(row))
    return row


class SplitOverlapError(ValueError):
    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        super().__init__(f"split audit failed: {audit['overlap_question_count']} overlapping question hashes, {audit.get('conflicting_answer_question_count', 0)} conflicting-answer hashes")


def _answer_identity(row: Mapping[str, Any]) -> str | None:
    if "answer" not in row:
        return None
    answer = str(row["answer"])
    choices = row.get("metadata", {}).get("choices", [])
    if choices:
        # Comparing letters alone would misclassify reordered equivalent choices.
        answer = next(str(choice["text"]) for choice in choices if choice["label"] == answer)
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())


def _conflict_identity(row: Mapping[str, Any]) -> str:
    """A repeated MCQ stem is not the same full question if options differ."""
    choices = row.get("metadata", {}).get("choices", [])
    if not choices:
        return str(row["question_hash"])
    contents = sorted({" ".join(unicodedata.normalize("NFKC", str(choice["text"])).casefold().split()) for choice in choices})
    return sha256_text(canonical_json({"question_hash": row["question_hash"], "option_contents": contents}))


def audit_split_overlap(splits: Mapping[str, Sequence[Mapping[str, Any]]], *, fail: bool = True) -> dict[str, Any]:
    """Check full normalized-question hashes, not only the selected experiment subset."""
    locations: dict[str, dict[str, list[str]]] = {}
    answers: dict[str, dict[str, list[dict[str, str]]]] = {}
    conflict_stems: dict[str, str] = {}
    for split, rows in splits.items():
        ids: set[str] = set()
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"duplicate row id in {split}: {row['id']}")
            ids.add(row["id"])
            locations.setdefault(str(row["question_hash"]), {}).setdefault(split, []).append(str(row["id"]))
            identity = _answer_identity(row)
            if identity is not None:
                full_question = _conflict_identity(row)
                conflict_stems[full_question] = str(row["question_hash"])
                answers.setdefault(full_question, {}).setdefault(identity, []).append({"split": split, "id": str(row["id"])})
    overlap = [{"question_hash": digest, "splits": located} for digest, located in sorted(locations.items()) if len(located) > 1]
    conflicts = [{"question_hash": conflict_stems[digest], "full_question_hash": digest, "answers": values} for digest, values in sorted(answers.items()) if len(values) > 1]
    audit = {
        "status": "FAIL" if overlap or conflicts else "PASS",
        "normalization": "Unicode NFKC, casefold, whitespace collapse; question stem only",
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "overlap_question_count": len(overlap),
        "overlaps": overlap,
        "conflicting_answer_question_count": len(conflicts),
        "conflicting_answers": conflicts,
        "answer_conflict_identity": "Math: normalized question stem. MCQ: normalized stem plus order-insensitive normalized option-content set; compare correct option content, not letter.",
        "within_split_duplicate_question_count": {
            name: sum(1 for located in locations.values() if len(located.get(name, [])) > 1) for name in splits
        },
        "scope": "Exact normalized question overlap across supplied splits only; does not establish absence of pretraining contamination or semantic duplicates.",
    }
    if audit["status"] == "FAIL" and fail:
        raise SplitOverlapError(audit)
    return audit


def _shuffle(rows: Iterable[dict[str, Any]], seed: int, split: str) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(sha256_text(f"{seed}:{split}")).shuffle(result)
    return result


def prepare_splits(project: str, dataset: Mapping[str, Any], *, seed: int = 20260910, mode: str = "cot", dev_size: int = 512, overlap_policy: str = "error") -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Canonicalize official splits and reserve GSM8K development rows from train."""
    if project not in SOURCES:
        raise ValueError(f"unsupported project: {project}")
    if overlap_policy not in {"error", "exclude-lower-priority"}:
        raise ValueError(f"unsupported overlap policy: {overlap_policy}")
    required = ("train", "test")
    missing = set(required) - set(dataset)
    if missing:
        raise ValueError(f"missing official splits: {sorted(missing)}")
    canonical = {split: [canonicalize_row(project, row, split, i, seed, mode) for i, row in enumerate(dataset[split])] for split in required}
    if any(not rows for rows in canonical.values()):
        raise ValueError("official source splits must be nonempty")
    if project == "gsm8k":
        if not 0 < dev_size < len(canonical["train"]):
            raise ValueError("GSM8K --dev-size must leave nonempty train and dev splits")
        shuffled = _shuffle(canonical["train"], seed, "gsm8k:reserve_dev")
        splits = {"train": shuffled[dev_size:], "dev": shuffled[:dev_size], "audit": canonical["test"]}
    else:
        raise ValueError("Only GSM8K is included")
    source_audit = audit_split_overlap(splits, fail=False)
    if overlap_policy == "error" and source_audit["status"] == "FAIL":
        raise SplitOverlapError(source_audit)
    excluded = []
    if overlap_policy == "exclude-lower-priority":
        higher: dict[str, tuple[str, list[str]]] = {}
        derived = {}
        for split in ("audit", "dev", "train"):
            kept = []
            for row in splits[split]:
                previous = higher.get(row["question_hash"])
                if previous is None:
                    kept.append(row)
                else:
                    excluded.append({"row": row, "excluded_split": split, "kept_split": previous[0], "kept_ids": previous[1], "reason": "question_hash_present_in_higher_priority_split"})
            # Same-split duplicates remain present and are reported by the audit.
            for row in kept:
                if row["question_hash"] not in higher:
                    higher[row["question_hash"]] = (split, [])
                higher[row["question_hash"]][1].append(row["id"])
            derived[split] = kept
        splits = {name: derived[name] for name in ("train", "dev", "audit")}
    derived_audit = audit_split_overlap(splits, fail=False)
    evidence = {**derived_audit, "source_overlap_audit": source_audit, "derived_overlap_audit": derived_audit, "excluded_rows": excluded}
    if derived_audit["status"] == "FAIL" or any(not rows for rows in splits.values()):
        # In particular, never silently remove conflicting labels inside audit.
        if any(not rows for rows in splits.values()):
            evidence["status"] = "FAIL"
            evidence["empty_derived_splits"] = [name for name, rows in splits.items() if not rows]
        raise SplitOverlapError(evidence)
    return {name: _shuffle(rows, seed, f"{project}:{name}") for name, rows in splits.items()}, evidence


def _source_metadata(dataset: Mapping[str, Any]) -> dict[str, Any]:
    details = {}
    for name, split in dataset.items():
        info = getattr(split, "info", None)
        details[name] = {
            "rows": len(split),
            "fingerprint": getattr(split, "_fingerprint", None),
            "dataset_version": str(getattr(info, "version", "")) or None,
            "homepage": getattr(info, "homepage", None),
            "license": getattr(info, "license", None),
            "citation": getattr(info, "citation", None),
        }
    return details


def write_dataset(project: str, dataset: Mapping[str, Any], output: str | Path, *, seed: int = 20260910, revision: str | None = None, requested_revision: str | None = None, transport_endpoint: str | None = None, cache_dir: str | None = None, mode: str = "cot", dev_size: int = 512, limits: Mapping[str, int | None] | None = None, overlap_policy: str = "error") -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous canonical dataset silently.
    targets = [output / f"{name}.jsonl" for name in ("train", "dev", "audit")] + [output / "manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"output already contains canonical dataset artifacts: {output}")
    try:
        splits, audit = prepare_splits(project, dataset, seed=seed, mode=mode, dev_size=dev_size, overlap_policy=overlap_policy)
    except SplitOverlapError as exc:
        failure = output / "overlap_failure.json"
        if not failure.exists():
            failure.write_text(json.dumps(exc.audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        source_audit = exc.audit.get("source_overlap_audit", exc.audit)
        source_path = output / "source_overlap_audit.json"
        if not source_path.exists():
            source_path.write_text(json.dumps(source_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    limits = dict(limits or {})
    for name, value in limits.items():
        if name not in splits or (value is not None and value <= 0):
            raise ValueError("limits must name train/dev/audit and be positive integers")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "seed": seed,
        "prompt_mode": mode,
        "source": {
            **SOURCES[project],
            "requested_revision": requested_revision if requested_revision is not None else revision,
            "resolved_revision": revision if revision and re.fullmatch(r"[a-fA-F0-9]{40}", revision) else None,
            "transport_endpoint": transport_endpoint,
            "cache_dir": cache_dir,
            "provenance_class": "public official dataset splits from publisher repository",
            "split_metadata": _source_metadata(dataset),
            "revision_note": "The production CLI resolves a dataset commit through HfApi at the recorded transport endpoint and passes that immutable SHA to load_dataset; raw/canonical record and file hashes additionally pin materialized data.",
        },
        "split_policy": {
            "train": "official train excluding reserved dev" if project == "gsm8k" else "official train",
            "dev": f"{dev_size} seeded rows from official train" if project == "gsm8k" else "official validation",
            "audit": "official test; held out from training and development decisions",
            "shuffle": "Python random.Random seeded by SHA256(seed:project:split); limits applied after full-split overlap audit and fixed shuffle",
            "requested_limits": limits,
            "overlap_policy": overlap_policy,
            "overlap_priority": ["audit", "dev", "train"],
            "audit_preservation": "No official test rows removed by overlap policy; any audit limit is a separate explicitly recorded subset choice",
        },
        "overlap_audit": audit["derived_overlap_audit"],
        "source_overlap_audit": audit["source_overlap_audit"],
        "derived_overlap_audit": audit["derived_overlap_audit"],
        "files": {},
    }
    for name in ("source_overlap_audit", "derived_overlap_audit"):
        path = output / f"{name}.json"
        content = json.dumps(audit[name], indent=2, ensure_ascii=False) + "\n"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != audit[name]:
                raise FileExistsError(f"refusing to overwrite different audit evidence: {path}")
        else:
            path.write_text(content, encoding="utf-8")
    excluded_content = "".join(canonical_json(row) + "\n" for row in audit["excluded_rows"])
    excluded_path = output / "excluded_overlaps.jsonl"
    if excluded_path.exists():
        raise FileExistsError(f"refusing to overwrite exclusion evidence: {excluded_path}")
    excluded_path.write_text(excluded_content, encoding="utf-8")
    manifest["excluded_overlaps"] = {"path": excluded_path.name, "rows": len(audit["excluded_rows"]), "sha256": sha256_text(excluded_content)}
    for name, rows in splits.items():
        selected = rows[:limits.get(name)] if limits.get(name) is not None else rows
        path = output / f"{name}.jsonl"
        content = "".join(canonical_json(row) + "\n" for row in selected)
        path.write_text(content, encoding="utf-8")
        manifest["files"][name] = {"path": path.name, "rows": len(selected), "sha256": sha256_text(content), "available_rows": len(rows)}
    manifest["manifest_payload_sha256"] = sha256_text(canonical_json(manifest))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def resolve_source_revision(project: str, revision: str | None = None, *, endpoint: str | None = None) -> str:
    """Resolve and validate the immutable source SHA before any production load."""
    from huggingface_hub import HfApi

    endpoint = endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    info = HfApi(endpoint=endpoint).dataset_info(SOURCES[project]["path"], revision=revision or "main")
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", resolved):
        raise ValueError("dataset endpoint did not return an immutable 40-character source commit SHA")
    if revision and re.fullmatch(r"[a-fA-F0-9]{40}", revision) and resolved.lower() != revision.lower():
        raise ValueError("dataset endpoint resolved a different commit than explicitly requested")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--revision", help="Prefer an immutable dataset repository commit SHA")
    parser.add_argument("--cache-dir")
    parser.add_argument("--prompt-mode", choices=("cot", "cod"), default="cot")
    parser.add_argument("--dev-size", type=int, default=512, help="GSM8K only: reserved official train rows")
    parser.add_argument("--overlap-policy", choices=("error", "exclude-lower-priority"), default="error", help="Explicitly opt into audit > dev > train duplicate exclusion; raw sources are retained")
    for name in ("train", "dev", "audit"):
        parser.add_argument(f"--{name}-limit", type=int)
    args = parser.parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the experiment runtime's datasets package before preparing data.") from exc
    source = SOURCES[args.project]
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    resolved_revision = resolve_source_revision(args.project, args.revision, endpoint=endpoint)
    kwargs = {"revision": resolved_revision, "cache_dir": args.cache_dir}
    if source["name"] is not None:
        kwargs["name"] = source["name"]
    dataset = load_dataset(source["path"], **kwargs)
    manifest = write_dataset(args.project, dataset, args.output, seed=args.seed, revision=resolved_revision, requested_revision=args.revision or "main", transport_endpoint=endpoint, cache_dir=args.cache_dir, mode=args.prompt_mode, dev_size=args.dev_size, limits={name: getattr(args, f"{name}_limit") for name in ("train", "dev", "audit")}, overlap_policy=args.overlap_policy)
    print(json.dumps({"output": str(Path(args.output).resolve()), "files": manifest["files"], "overlap_audit": manifest["overlap_audit"]["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
