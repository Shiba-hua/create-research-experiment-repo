#!/usr/bin/env python3
"""Plot completion-gated accuracy against retrospective token thresholds.

This CPU-only postprocessor replays one or more already saved evaluation
trajectories.  It does not generate text, truncate responses, or re-score
prefixes.  ``--arm label=predictions.jsonl`` may be repeated to put all arms
of one experiment on the same two-panel figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty predictions file: {path}")
    return rows


def _summary_for(path: Path) -> dict[str, Any] | None:
    summary = path.parent / "evaluation_summary.json"
    if not summary.is_file():
        return None
    value = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation summary is not an object: {summary}")
    return value


def replay(records: Sequence[Mapping[str, Any]], budget: int, *, label: str = "arm",
           summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compute every integer gate, retaining only first-rise points for plots."""

    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not records:
        raise ValueError(f"{label}: empty predictions")
    ids = [row.get("id") for row in records]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{label}: missing or duplicate question IDs")
    lengths = [0] * (budget + 1)
    correct_lengths = [0] * (budget + 1)
    for row in records:
        score, length = row.get("correctness"), row.get("response_tokens")
        if type(score) not in (bool, int, float) or score not in (0, 1):
            raise ValueError(f"{label}: correctness must be binary")
        if type(length) is not int or not 0 <= length <= budget:
            raise ValueError(f"{label}: invalid response_tokens for {row.get('id')}")
        if type(row.get("eos")) is not bool or type(row.get("truncated")) is not bool:
            raise ValueError(f"{label}: missing EOS/truncation status")
        if score and (row["eos"] is not True or row["truncated"] is True or length == 0):
            raise ValueError(f"{label}: a correct response must be nonempty, EOS-terminated, and untruncated")
        lengths[length] += 1
        correct_lengths[length] += int(score)

    n = len(records)
    total_tokens = sum(int(row["response_tokens"]) for row in records)
    accuracy = sum(correct_lengths) / n
    mean_tokens = total_tokens / n
    if summary is not None:
        counts = summary.get("counts", {})
        if counts.get("evaluated") != n or summary.get("accuracy") != accuracy:
            raise ValueError(f"{label}: endpoint disagrees with evaluation summary accuracy/count")
        mean = summary.get("response_tokens", {}).get("mean")
        if mean is not None and not math.isclose(float(mean), mean_tokens, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{label}: endpoint disagrees with evaluation summary token mean")

    accuracy_by_threshold = [0.0] * (budget + 1)
    charged_by_threshold = [0] * (budget + 1)
    completed_correct_counts = [0] * (budget + 1)
    for threshold in range(1, budget + 1):
        completed_correct_counts[threshold] = (
            completed_correct_counts[threshold - 1] + correct_lengths[threshold]
        )
        accuracy_by_threshold[threshold] = completed_correct_counts[threshold] / n
        charged_by_threshold[threshold] = (
            charged_by_threshold[threshold - 1] + n - sum(lengths[:threshold])
        )
    # At threshold zero no response token is charged and no positive-length
    # response has completed, so the origin is explicit even if malformed data
    # contains zero-length incorrect responses.
    jump_thresholds = [threshold for threshold in range(1, budget + 1)
                       if correct_lengths[threshold] > 0]

    def point(threshold: int) -> dict[str, Any]:
        return {
            "budget_threshold": threshold,
            "completed_correct_count": completed_correct_counts[threshold],
            "budget_gated_val_acc": accuracy_by_threshold[threshold],
            "charged_decode_tokens_total": charged_by_threshold[threshold],
            "mean_charged_decode_tokens": charged_by_threshold[threshold] / n,
            "is_origin": threshold == 0,
            "is_jump": threshold > 0 and correct_lengths[threshold] > 0,
            "is_full_horizon": threshold == budget,
        }

    endpoint = point(budget)
    checks = {
        "all_questions_retained": n == len(records),
        "correct_count_matches": endpoint["completed_correct_count"] == sum(int(row["correctness"]) for row in records),
        "accuracy_matches": endpoint["budget_gated_val_acc"] == accuracy,
        "decode_token_total_matches": endpoint["charged_decode_tokens_total"] == total_tokens,
        "mean_decode_tokens_matches": endpoint["mean_charged_decode_tokens"] == mean_tokens,
        "actual_recorded_cap": endpoint["budget_threshold"] == budget,
    }
    if not all(checks.values()):
        raise ValueError(f"{label}: full-horizon replay endpoint failed validation: {checks}")
    return {
        "label": label,
        "n": n,
        "measured_output_budget": budget,
        "score_field": "correctness",
        "recorded_point": {"correct": sum(int(row["correctness"]) for row in records),
                           "val_acc": accuracy, "decode_tokens_total": total_tokens,
                           "avg_decode_tokens": mean_tokens},
        "budget_thresholds": list(range(budget + 1)),
        "completed_correct_counts": completed_correct_counts,
        "budget_gated_val_acc": accuracy_by_threshold,
        "charged_decode_tokens_totals": charged_by_threshold,
        "mean_charged_decode_tokens": [value / n for value in charged_by_threshold],
        "jump_points": [point(threshold) for threshold in jump_thresholds],
        "plot_points": [point(threshold) for threshold in sorted({0, *jump_thresholds, budget})],
        "full_horizon": endpoint,
        "endpoint_validation": {"status": "PASS", "checks": checks},
    }


def _relative_source(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base.resolve())


def collect(arms: Sequence[tuple[str, Path]], budget: int, *, output: Path,
            title: str) -> dict[str, Any]:
    if len(arms) < 2:
        raise ValueError("At least two arms are required")
    labels = [label for label, _ in arms]
    if len(set(labels)) != len(labels):
        raise ValueError("Arm labels must be unique")
    order: list[tuple[str, str]] | None = None
    result_arms: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    repo = Path.cwd().resolve()
    for label, path in arms:
        path = path.resolve()
        records = _read_jsonl(path)
        current_order = [(str(row["id"]), str(row["question_hash"])) for row in records]
        if order is None:
            order = current_order
        elif current_order != order:
            raise ValueError(f"{label}: question identity/order differs from other arms")
        summary = _summary_for(path)
        curve = replay(records, budget, label=label, summary=summary)
        curve["predictions"] = _relative_source(path, repo)
        result_arms.append(curve)
        sources.append({"path": _relative_source(path, repo), **{k: v for k, v in _file_identity(path).items() if k != "path"}})
        if summary is not None:
            sources.append({"path": _relative_source(path.parent / "evaluation_summary.json", repo),
                            **{k: v for k, v in _file_identity(path.parent / "evaluation_summary.json").items() if k != "path"}})
    return {
        "schema": "rlvr.budget-gated-val-acc/v1",
        "title": title,
        "arms": result_arms,
        "sources": sources,
        "method": {
            "name": "Recorded-trajectory replay with completion gating",
            "accuracy_formula": "Y(t) = sum_i(c_i * I[L_i <= t]) / N",
            "cost_formula": "C(t) = sum_i(min(L_i, t)) / N",
            "length": "L_i = recorded response_tokens, including returned EOS",
            "correctness": "c_i = original recorded strict binary correctness; never reassigned",
            "denominator": "N = every recorded question, including wrong, format-failed, and truncated responses",
            "domain": "Every integer threshold t from 0 through the recorded output cap",
            "plot_selection": "Origin, every threshold where Y first rises, and the original cap endpoint",
            "display_interpolation": "Connect retained jump points in ascending threshold/cost order with ordinary line segments; segments are visual interpolation, not additional measurements",
            "interpretation": "Retrospective completion-gated replay; no new inference or prefix regrading",
        },
        "axis_contract": {
            "left_x": "Per-question decode budget threshold t",
            "right_x": "Mean charged decode tokens C(t)",
            "y": "Recorded completed-correct fraction / budget-gated val_acc Y(t)",
            "decode_includes": ["reasoning", "final answer", "returned EOS"],
            "decode_excludes": ["prompt", "padding", "new inference"],
        },
        "measured_output_budget": budget,
        "independent_budgets_per_arm": 1,
        "independent_multi_budget_evaluation": False,
        "new_model_run": False,
        "new_inference": False,
        "prefix_regrading": False,
        "extrapolation_beyond_recorded_cap": False,
        "producer": {"path": "repro/budget_gated_curve.py",
                     "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }


def _coordinates(arm: Mapping[str, Any], x_key: str) -> tuple[list[float], list[float]]:
    points = []
    for point in arm["plot_points"]:
        xy = (point[x_key], point["budget_gated_val_acc"])
        if not points or xy != points[-1]:
            points.append(xy)
    return [float(x) for x, _ in points], [float(y) for _, y in points]


def draw(data: Mapping[str, Any]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    colors = ("#296a9d", "#d37528", "#2a9d61", "#8d5aa7", "#b34d4d")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), dpi=180, sharey=True)
    handles = []
    for index, (axis, x_key) in enumerate(zip(axes, ("budget_threshold", "mean_charged_decode_tokens"))):
        for arm, color in zip(data["arms"], colors):
            xs, ys = _coordinates(arm, x_key)
            line, = axis.plot(xs, ys, color=color, linewidth=1.8, label=arm["label"])
            if index == 0:
                handles.append(line)
            endpoint = arm["full_horizon"]
            axis.scatter([endpoint[x_key]], [endpoint["budget_gated_val_acc"]],
                         marker="D", s=33, color=color, zorder=5, clip_on=False)
        axis.scatter([0], [0], s=13, color="#444444", zorder=6, clip_on=False)
        xmax = max(float(arm["full_horizon"][x_key]) for arm in data["arms"])
        axis.set_xlim(0, max(1, xmax * 1.04))
        axis.set_ylim(0, 1)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=index == 0))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.2)
    axes[0].set_title("Completion fraction versus budget threshold", fontsize=11, pad=12)
    axes[1].set_title("Completion fraction versus charged decode cost", fontsize=11, pad=12)
    axes[0].set_xlabel("Per-question decode budget threshold (tokens)")
    axes[1].set_xlabel("Mean charged decode tokens per question\n(reasoning + final + returned EOS; all questions)")
    axes[0].set_ylabel("Recorded completed-correct fraction\n(budget-gated val_acc)", labelpad=12)
    handles.append(Line2D([], [], marker="D", color="#555555", linestyle="none", markersize=5,
                          label="Full-horizon endpoint"))
    fig.suptitle(data["title"], fontsize=14, y=.975)
    fig.text(.56, .927, "Recorded-trajectory replay | no new inference | no prefix regrading",
             ha="center", fontsize=10, color="#454545")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .90),
               ncol=min(4, len(handles)), frameon=False)
    endpoint_text = " | ".join(
        f"{arm['label']}: B={arm['measured_output_budget']}, C(B)={arm['full_horizon']['mean_charged_decode_tokens']:.2f}, Y(B)={arm['full_horizon']['budget_gated_val_acc']:.2%}"
        for arm in data["arms"])
    fig.text(.025, .025,
             "Full horizon (diamond): " + endpoint_text + "\n"
             f"Each arm: N={data['arms'][0]['n']}; one saved inference cap. Thresholds are not independent budget evaluations.\n"
             "Wrong, format-failed, and truncated responses remain in the cost and accuracy denominator; prompt tokens are excluded.\n"
             "Paths retain the origin, every first-rise threshold, and the original cap endpoint; no extrapolation beyond that cap.",
             fontsize=8.5, va="bottom", color="#454545", linespacing=1.5)
    fig.subplots_adjust(left=.14, right=.98, bottom=.29, top=.78, wspace=.18)
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="Arm specification label=predictions.jsonl; repeat for each arm")
    parser.add_argument("--budget", required=True, type=int,
                        help="Recorded max output token budget (not an extrapolation target)")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, help="New figure directory")
    args = parser.parse_args(argv)
    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        if "=" not in spec:
            parser.error("Each --arm must be label=predictions.jsonl")
        label, path = spec.split("=", 1)
        if not label or not path:
            parser.error("Each --arm must have a nonempty label and path")
        arms.append((label, Path(path)))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure directory: {output}")
    data = collect(arms, args.budget, output=output, title=args.title)
    output.mkdir(parents=True)
    curves = output / "curves.json"
    curves.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fig = draw(data)
    files = []
    try:
        for suffix in ("png", "svg"):
            path = output / f"budget_gated_val_acc.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            files.append({"path": path.name, **{k: v for k, v in _file_identity(path).items() if k != "path"}})
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    manifest = {
        "schema": "rlvr.budget-gated-plot/v1",
        "scope": "CPU replay of saved trajectories; not an independently evaluated budget curve",
        "curves": {"path": "curves.json", **{k: v for k, v in _file_identity(curves).items() if k != "path"}},
        "files": files,
        "producer": data["producer"],
        "new_inference": False,
        "prefix_regrading": False,
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "arms": [arm["label"] for arm in data["arms"]],
                      "endpoints": {arm["label"]: arm["full_horizon"] for arm in data["arms"]}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
