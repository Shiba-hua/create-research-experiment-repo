"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def local_artifact_identity(model_path: str | Path, adapter_path: str | Path | None = None) -> dict[str, Any]:
    """Hash local files before loading; verify the fetch lock when one exists."""
    base = Path(model_path).resolve()
    if not base.is_dir():
        raise ValueError("Artifact-bound evaluation requires a local model directory")
    manifest_path = base / "snapshot_manifest.json"
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    suffixes = {".safetensors", ".json", ".model", ".txt", ".jinja"}
    paths = {str(path.relative_to(base)): path for path in sorted(base.rglob("*"))
        if path.is_file() and not {".cache", ".git"}.intersection(path.relative_to(base).parts)
        and path.suffix in suffixes and path.name != "snapshot_manifest.json"}
    weights = {name for name in paths if name.endswith(".safetensors")}
    if not weights:
        raise ValueError("Local model has no safetensors weights to bind")
    if snapshot is not None:
        if not isinstance(snapshot.get("files"), dict) or not snapshot.get("model_id") or not snapshot.get("revision"):
            raise ValueError("Invalid model snapshot manifest")
        expected_weights = {name for name in snapshot["files"] if name.endswith(".safetensors")}
        if weights != expected_weights:
            raise ValueError("Actual safetensors files do not match the fetched model snapshot")
        # Also bind each fetched tokenizer/config/source file, including README
        # and LICENSE. Reject missing or escaping paths before hashing anything.
        for name in snapshot["files"]:
            path = base / name
            if not path.resolve().is_relative_to(base) or not path.is_file():
                raise ValueError(f"Missing or invalid fetched snapshot file: {name}")
            paths[name] = path
    files = {name: _file_identity(path) for name, path in sorted(paths.items())}
    if snapshot is not None:
        for name, actual in files.items():
            expected = snapshot["files"].get(name)
            if expected is None or expected.get("sha256") != actual["sha256"] or expected.get("bytes") != actual["bytes"]:
                raise ValueError(f"Model file differs from fetched snapshot: {name}")
    identity = {"binding": "actual local files SHA256 before model load", "base": {
        "path": str(base), "model_id": snapshot.get("model_id") if snapshot else None,
        "revision": snapshot.get("revision") if snapshot else None,
        "snapshot_manifest_sha256": _file_identity(manifest_path)["sha256"] if snapshot else None,
        "fetch_manifest_verified": snapshot is not None, "files": files, "files_sha256": _digest(files)}, "adapter": None}
    if adapter_path is not None:
        adapter = Path(adapter_path).resolve()
        if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
            raise ValueError("Adapter identity requires a local directory with adapter_config.json")
        adapter_paths = sorted(adapter.glob("*.safetensors"))
        if not adapter_paths:
            raise ValueError("Local adapter has no safetensors weights to bind")
        adapter_files = {path.name: _file_identity(path) for path in [adapter / "adapter_config.json", *adapter_paths]}
        identity["adapter"] = {"path": str(adapter), "files": adapter_files, "files_sha256": _digest(adapter_files)}
    return identity


def _quantiles(values: Sequence[int | float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty/nonfinite measurements")
    result = {key: float(value) for key, value in zip(("min", "p25", "median", "p75", "p90", "p95", "max"),
        np.quantile(array, [0, .25, .5, .75, .9, .95, 1]))}
    result["mean"] = float(array.mean())
    return result


def _validate_data(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    seen = set()
    for row in rows:
        for field in ("id", "project", "prompt", "answer", "question_hash", "source_split"):
            if field not in row:
                raise ValueError(f"Missing canonical field: {field}")
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError(f"Invalid/duplicate evaluation ID: {row['id']!r}")
        seen.add(row["id"])
        question_hash = row["question_hash"]
        if not isinstance(question_hash, str) or len(question_hash) != 64 or any(c not in "0123456789abcdef" for c in question_hash):
            raise ValueError(f"Invalid question hash: {row['id']}")
        if not isinstance(row["project"], str) or not row["project"] or not row["source_split"]:
            raise ValueError(f"Invalid project/split: {row['id']}")
        if not isinstance(row["prompt"], list) or not row["prompt"]:
            raise ValueError(f"Prompt must be nonempty chat messages: {row['id']}")
        for message in row["prompt"]:
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str):
                raise ValueError(f"Invalid chat message: {row['id']}")
    if len({row["project"] for row in rows}) != 1:
        raise ValueError("Evaluate one project at a time")


def _thinking_setting(model: Any, tokenizer: Any, requested: Any) -> tuple[bool | None, dict[str, Any]]:
    if isinstance(requested, bool):
        requested = str(requested).lower()
    if requested not in ("auto", "true", "false"):
        raise ValueError("thinking must be auto, true, or false")
    model_type = str(getattr(model.config, "model_type", ""))
    template = tokenizer.get_chat_template()
    supported = model_type.startswith("qwen3") and "enable_thinking" in template
    if requested == "true" and not supported:
        raise ValueError("Explicit thinking=true requires a supported native-thinking Qwen3 chat template")
    effective = (requested != "false") if supported else None
    kwargs = {"enable_thinking": effective} if supported else {}
    return effective, kwargs


def _thinking_lengths(text: str, tokenizer: Any, native_thinking: bool | None) -> tuple[int, int, str]:
    if "</think>" in text:
        thought, final = text.split("</think>", 1)
        thought = thought.removeprefix("<think>")
    elif native_thinking or text.lstrip().startswith("<think>"):
        thought, final = text.removeprefix("<think>"), ""
    else:
        thought, final = "", text
    return (len(tokenizer.encode(thought, add_special_tokens=False)) if thought else 0,
        len(tokenizer.encode(final, add_special_tokens=False)) if final else 0,
        "decoded segments re-tokenized independently; delimiters excluded; totals may not sum")


