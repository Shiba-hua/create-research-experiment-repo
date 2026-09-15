"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping
from .data import COD_INSTRUCTION
from .evaluate import _digest, _validate_data
def _hash_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(io.StringIO(path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: blank row")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(row)
    return rows


def _require_sha(value: Any, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise ValueError(f"Missing or invalid {label}")
    return value


def _signed_manifest(meta: Mapping[str, Any], key: str, label: str) -> None:
    if meta.get(key) != _digest({k: v for k, v in meta.items() if k != key}):
        raise ValueError(f"Corrupt {label} SHA256")


def _canonical_train(data: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest_path = data.parent / "manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    _signed_manifest(meta, "manifest_payload_sha256", "canonical manifest")
    source = meta.get("source", {})
    _require_sha(source.get("resolved_revision"), "dataset source revision", 40)
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError("Missing dataset source identity")
    if meta.get("prompt_mode") != "cot":
        raise ValueError("Canonical data must use the common neutral cot prompt, without a teacher CoD instruction")
    audit = meta.get("derived_overlap_audit", meta.get("overlap_audit", {}))
    if audit.get("status") != "PASS" or audit.get("overlap_question_count") != 0:
        raise ValueError("A passing full-split overlap audit is required")
    if meta.get("split_policy", {}).get("requested_limits", {}).get("train") is not None:
        raise ValueError("Cold-start generation must cover full train, not a train limit")
    inputs = {"canonical_manifest": _hash_file(manifest_path)}
    splits = {}
    for split in ("train", "dev", "audit"):
        entry = meta.get("files", {}).get(split, {})
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Invalid canonical {split} file path")
        path = data.parent / name
        if split == "train" and path.resolve() != data.resolve():
            raise ValueError("--data must be the canonical train file, never dev or audit")
        identity = _hash_file(path)
        rows = _read_rows(path)
        _validate_data(rows)
        if identity["sha256"] != entry.get("sha256") or len(rows) != entry.get("rows"):
            raise ValueError(f"Canonical {split} file does not match its manifest")
        if split == "train" and entry.get("available_rows") != len(rows):
            raise ValueError("Canonical train must contain every available training point")
        for row in rows:
            if row["project"] != meta.get("project"):
                raise ValueError(f"Canonical project mismatch: {row['id']}")
            _require_sha(row.get("source_record_sha256"), "canonical source record SHA256")
            if row.get("record_sha256") != _digest({k: v for k, v in row.items() if k != "record_sha256"}):
                raise ValueError(f"Corrupt canonical record SHA256: {row['id']}")
        splits[split] = rows
        inputs[f"canonical_{split}"] = identity
    train = splits["train"]
    train_ids = {row["id"] for row in train}
    train_hashes = {row["question_hash"] for row in train}
    for split in ("dev", "audit"):
        if train_ids.intersection(row["id"] for row in splits[split]) or train_hashes.intersection(row["question_hash"] for row in splits[split]):
            raise ValueError(f"Canonical train overlaps held-out {split} rows")
    for row in train:
        if row["source_split"] != "train":
            raise ValueError(f"Non-train source row cannot enter cold-start data: {row['id']}")
        if row.get("metadata", {}).get("prompt_mode") != "cot":
            raise ValueError(f"Canonical row has a non-neutral prompt mode: {row['id']}")
        if any(COD_INSTRUCTION in message["content"] for message in row["prompt"]):
            raise ValueError(f"Teacher CoD prompt instruction must not enter student prompts: {row['id']}")
    return train, meta, inputs


def _artifact_identity(meta: Mapping[str, Any]) -> dict[str, Any]:
    artifact = meta.get("evaluated_artifact")
    if meta.get("artifact_identity_recorded") is not True or not isinstance(artifact, dict):
        raise ValueError("Teacher model artifact identity is required")
    base = artifact.get("base", {})
    if base.get("fetch_manifest_verified") is not True or not isinstance(base.get("model_id"), str) or not base["model_id"]:
        raise ValueError("Teacher base must have a verified source model identity")
    _require_sha(base.get("revision"), "teacher model source revision", 40)
    _require_sha(base.get("snapshot_manifest_sha256"), "teacher snapshot manifest SHA256")
    for label, part in (("base", base), ("adapter", artifact.get("adapter"))):
        if part is None:
            continue
        files = part.get("files", {})
        if not isinstance(files, dict) or not files or not any(name.endswith(".safetensors") for name in files):
            raise ValueError(f"Teacher {label} has no bound model weights")
        if part.get("files_sha256") != _digest(files):
            raise ValueError(f"Corrupt teacher {label} file identity SHA256")
        for name, identity in files.items():
            if not isinstance(identity, dict):
                raise ValueError(f"Invalid teacher artifact file: {name}")
            _require_sha(identity.get("sha256"), f"teacher artifact file SHA256: {name}")
            if type(identity.get("bytes")) is not int or identity["bytes"] <= 0:
                raise ValueError(f"Invalid teacher artifact file size: {name}")
    return copy.deepcopy(artifact)


