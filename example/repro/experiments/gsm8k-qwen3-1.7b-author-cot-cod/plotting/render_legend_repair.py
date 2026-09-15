"""Redraw verified plot data with one timing-legend layout repair; CPU only."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/gsm8k-author-figures-001"
OUTPUT = ROOT / "results/gsm8k-author-figures-002"
PLOT_DATA_SHA256 = "4294d7697a1199820a9290f5a394d0f0d772f0a7f584822e4d88f7dbc35992e6"
RENDERER_SHA256 = "993d6b3939ed0236b0dbec27b9eaf37b6c7aaeb5e7b585eb11f7961cfd0b14cb"
COMPARISON_SHA256 = "bb9d6e5798f7cc16cec5a118202c0b235c59173b9621515e899f208a92295712"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main():
    require(not OUTPUT.exists(), "Use a fresh figure output directory")
    require(not (EVIDENCE / "repair_verification.json").exists(), "Repair verification already exists")
    source_manifest = json.loads((SOURCE / "plot_manifest.json").read_text())
    data_path = SOURCE / "plot_data.json"
    require(identity(data_path)["sha256"] == PLOT_DATA_SHA256 == source_manifest["plot_data"]["sha256"], "Original plot data differs")
    data = json.loads(data_path.read_text())
    source_snapshot = {identity(SOURCE / "plot_manifest.json")["path"]: identity(SOURCE / "plot_manifest.json"),
                       identity(data_path)["path"]: identity(data_path)}
    for chart in source_manifest["charts"]:
        for item in chart["files"]:
            path = SOURCE / item["path"]
            record = identity(path)
            require(all(record[key] == item[key] for key in ("sha256", "bytes")), "Original figure differs: " + str(path))
            source_snapshot[record["path"]] = record
    renderer = ROOT / "scripts/plot_cod_author.py"
    comparison = ROOT / "results/gsm8k-author-comparison-001.json"
    require(identity(renderer)["sha256"] == RENDERER_SHA256, "Original renderer differs")
    require(identity(comparison)["sha256"] == COMPARISON_SHA256, "Verified comparison differs")
    require(source_manifest["full_comparison_recomputed"] is True, "Source figures were not fully reverified")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MPLCONFIGDIR"] = str(EVIDENCE / ".matplotlib")
    spec = importlib.util.spec_from_file_location("original_cod_author_plot", renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module._digest({k: v for k, v in source_manifest.items() if k != "manifest_sha256"})
            == source_manifest["manifest_sha256"], "Original manifest digest differs")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    original_legend = Axes.legend
    patched_axes = []

    def repaired_legend(ax, *args, **kwargs):
        tick_text = [tick.get_text() for tick in ax.get_xticklabels()]
        if ax.get_ylabel() == "Seconds" and tick_text == ["Evaluation wall", "Generation batches"]:
            require(not args and not kwargs, "Unexpected timing legend arguments")
            patched_axes.append(ax)
            return original_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                                   ncol=2, borderaxespad=0.0)
        return original_legend(ax, *args, **kwargs)

    source_records = source_manifest["sources"] + [identity(SOURCE / "plot_manifest.json"),
                                                  identity(data_path), identity(Path(__file__))]
    Axes.legend = repaired_legend
    try:
        manifest = module.render(data, OUTPUT, source_records)
    finally:
        Axes.legend = original_legend
    require(len(patched_axes) == 1, "Legend repair must match exactly one timing panel")
    new_data = json.loads((OUTPUT / "plot_data.json").read_text())
    require(new_data == data, "Redraw changed plot data")
    require(identity(OUTPUT / "plot_data.json")["sha256"] == PLOT_DATA_SHA256, "Plot data bytes differ")
    for relative, expected in source_snapshot.items():
        require(identity(ROOT / relative) == expected, "Original plot artifact changed")
    require(identity(renderer)["sha256"] == RENDERER_SHA256, "Original renderer was changed")
    # The original render function labels a fresh collect+render as recomputed.
    # This derived render inherits prior verification but does not repeat collect.
    manifest.update(full_comparison_recomputed=False, upstream_full_comparison_recomputed=True,
                    render_only_from_verified_plot_data=True,
                    repair={"kind": "timing_legend_layout", "matched_axes": 1,
                            "legend_location": "above timing left panel, centered in two columns",
                            "source_plot_manifest": identity(SOURCE / "plot_manifest.json"),
                            "source_plot_data": identity(data_path),
                            "original_renderer": identity(renderer),
                            "repair_script": identity(Path(__file__)),
                            "plot_data_unchanged": True})
    manifest["manifest_sha256"] = module._digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    module.author.teacher_generate._write_json(OUTPUT / "plot_manifest.json", manifest)
    report = {"kind": "render_only_timing_legend_repair", "status": "RENDERED_PENDING_VISUAL_AUDIT",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "source_plot_artifacts_unchanged": list(source_snapshot.values()),
              "verified_comparison": identity(comparison), "original_renderer": identity(renderer),
              "repair_script": identity(Path(__file__)), "new_plot_data": identity(OUTPUT / "plot_data.json"),
              "new_plot_manifest": identity(OUTPUT / "plot_manifest.json"),
              "plot_data_dictionary_equal": new_data == data, "plot_data_bytes_equal": True,
              "matched_timing_axes": len(patched_axes), "charts": len(manifest["charts"]),
              "figure_files": [identity(OUTPUT / item["path"]) for chart in manifest["charts"] for item in chart["files"]],
              "source_code_modified": False, "remote_operations": False, "gpu_operations": False,
              "comparison_recomputed_this_render": False, "upstream_full_comparison_recomputed": True,
              "python_executable": sys.executable, "matplotlib_version": matplotlib.__version__,
              "numpy_version": module.np.__version__}
    write_json(EVIDENCE / "repair_verification.json", report)
    print(json.dumps({"status": report["status"], "charts": report["charts"],
                      "output": str(OUTPUT), "plot_data_sha256": PLOT_DATA_SHA256,
                      "repair_script_sha256": report["repair_script"]["sha256"]}))


if __name__ == "__main__":
    main()
