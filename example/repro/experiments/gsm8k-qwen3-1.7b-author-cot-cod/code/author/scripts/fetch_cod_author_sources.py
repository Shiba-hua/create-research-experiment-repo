"""Fetch the pinned Chain-of-Draft sources to an external input directory.

The author tree has no license file. This script stores no upstream source in
the checkout, never imports upstream Python, and never calls a model API.
Download and verification use only the Python standard library. The optional
static audit checks all eight author examples against complete canonical splits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import unicodedata
import urllib.request


REVISION = "a7dbf5dea808b1aa1e12f7a90ea573321581df78"
REPOSITORY = "https://github.com/sileix/chain-of-draft"
SOURCE_PATHS = (
    "configs/gsm8k_cot.yaml", "configs/gsm8k_cod.yaml", "utils.py",
    "tasks/gsm8k.py", "llm_client.py",
)
REFERENCE_PATH = "reference/gsm8k_test.jsonl"
DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "configs/cod_author_upstream.json"


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if lock.get("schema_version") != 1 or lock.get("repository") != REPOSITORY or lock.get("revision") != REVISION:
        raise ValueError("Source lock must identify the expected immutable author revision")
    files = lock.get("files")
    if not isinstance(files, dict) or set(files) != set(SOURCE_PATHS):
        raise ValueError("Source lock must contain exactly the five expected author paths")
    for name, item in files.items():
        expected_url = f"https://raw.githubusercontent.com/sileix/chain-of-draft/{REVISION}/{name}"
        if not isinstance(item, dict) or item.get("url") != expected_url:
            raise ValueError(f"Source URL does not match the pinned revision: {name}")
        for key, length in (("sha256", 64), ("git_blob_sha", 40)):
            if not isinstance(item.get(key), str) or re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", item[key]) is None:
                raise ValueError(f"Invalid {key} in source lock: {name}")
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError(f"Invalid source byte count: {name}")
    reference = lock.get("reference_dataset", {})
    if (reference.get("revision") != "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
            or reference.get("url") != "https://raw.githubusercontent.com/openai/grade-school-math/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/test.jsonl"
            or reference.get("sha256") != "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
            or reference.get("bytes") != 749738 or reference.get("rows") != 1319):
        raise ValueError("Source lock must pin the official GSM8K test reference")
    return dict(lock)


def load_lock(path: str | Path | None = None) -> dict[str, Any]:
    """Load URLs and hashes; reject moving branches and unexpected paths."""
    return _validated_lock(json.loads(Path(path or DEFAULT_LOCK).read_text(encoding="utf-8")))


def _verify_bytes(raw: bytes, name: str, expected: Mapping[str, Any]) -> None:
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Source SHA256 or size mismatch: {name}")
    if "git_blob_sha" in expected:
        blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if blob_sha != expected["git_blob_sha"]:
            raise ValueError(f"Source Git blob SHA mismatch: {name}")


def verify_downloaded(source_dir: str | Path, lock: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Read and verify all five sources without importing or executing them."""
    lock = _validated_lock(lock) if lock is not None else load_lock()
    source_dir = Path(source_dir).resolve()
    identities = {}
    for name, expected in lock["files"].items():
        path = source_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(source_dir):
            raise ValueError(f"Source path escapes the input directory or is a symlink: {name}")
        _verify_bytes(path.read_bytes(), name, expected)
        identities[name] = {"path": str(path.resolve()), **expected}
    return identities


def _external_destination(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    # Check every containing repository, including the outer research checkout.
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() and output.is_relative_to(parent):
            raise ValueError("Author sources must be downloaded outside the Git checkout")
    return output


def fetch_sources(output_dir: str | Path, lock: Mapping[str, Any] | None = None, *, timeout: float = 30) -> dict[str, Any]:
    """Download missing files, verify before publication, and reject corrupt cache."""
    lock = _validated_lock(lock) if lock is not None else load_lock()
    output = _external_destination(output_dir)
    if timeout <= 0:
        raise ValueError("Download timeout must be positive")
    download_files = {**lock["files"], REFERENCE_PATH: lock["reference_dataset"]}
    # Verify every existing source first; corruption never triggers a silent repair.
    for name, expected in download_files.items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError(f"Unsafe existing source path: {name}")
        if path.exists():
            _verify_bytes(path.read_bytes(), name, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cod-source-staging-", dir=output.parent) as temporary:
        staged = Path(temporary)
        for name, expected in download_files.items():
            if (output / name).exists():
                continue
            request = urllib.request.Request(expected["url"], headers={"User-Agent": "rlvr-lab-source-lock/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(expected["bytes"] + 1)
            _verify_bytes(raw, name, expected)
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        # No files become inputs until every missing file has passed validation.
        for name in download_files:
            path = staged / name
            if path.exists():
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
    files = verify_downloaded(output, lock)
    receipt = {
        "status": "PASS", "kind": "cod_author_source_verification", "revision": REVISION,
        "repository": REPOSITORY, "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_lock_sha256": _digest(lock), "files": files,
        "reference_dataset": {**lock["reference_dataset"], "local_path": str(output / REFERENCE_PATH)},
        "upstream_code_executed": False, "model_api_calls": 0,
    }
    (output / "source_verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def normalize_question(question: str) -> str:
    """Match rlvr_lab.data.question_hash: NFKC, casefold, whitespace collapse."""
    if not isinstance(question, str):
        raise ValueError("Question must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if not normalized:
        raise ValueError("Question cannot be empty")
    return normalized


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def _question_examples(source_dir: Path) -> list[str]:
    """Read only the question blocks in the exact hash-verified YAML subset.

    Scalar whitespace style is irrelevant after canonical question normalization.
    This intentionally is not a general YAML parser or prompt compositor.
    """
    all_questions = []
    for mode in ("cot", "cod"):
        content = (source_dir / f"configs/gsm8k_{mode}.yaml").read_text(encoding="utf-8")
        questions = re.findall(r"^  - question:[ \t]*(?:\|[ \t]*)?\n(.*?)^    answer:", content, re.MULTILINE | re.DOTALL)
        questions = [normalize_question(question) for question in questions]
        if len(questions) != 8 or len(set(questions)) != 8:
            raise ValueError(f"Expected eight distinct author examples: {mode}")
        all_questions.append(questions)
    if all_questions[0] != all_questions[1]:
        raise ValueError("Author CoT and CoD examples must have matching questions and order")
    return all_questions[0]


def _phrase_tokens(question: str) -> list[str]:
    return re.findall(r"\w+", normalize_question(question))


def audit_fewshot_overlap(
    source_dir: str | Path, canonical_dir: str | Path,
    output_path: str | Path | None = None,
    lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit all eight static shots against signed, complete train/dev/audit files.

    Phrase hits are review candidates and cause FAIL even without exact overlap.
    Reports contain hashes/IDs only, not the author's full few-shot questions.
    """
    lock = _validated_lock(lock) if lock is not None else load_lock()
    source_files = verify_downloaded(source_dir, lock)
    examples = _question_examples(Path(source_dir))
    example_hashes = [question_hash(question) for question in examples]
    # A normalized seven-word prefix catches punctuation or suffix variations.
    anchors = [_phrase_tokens(question)[:7] for question in examples]
    canonical_dir = Path(canonical_dir).resolve()
    manifest_path = canonical_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_payload_sha256") != _digest({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}):
        raise ValueError("Canonical manifest SHA256 mismatch")
    if manifest.get("project") != "gsm8k":
        raise ValueError("Author GSM8K examples require the canonical GSM8K dataset")
    split_counts, canonical_files, exact_hits, phrase_hits = {}, {}, [], []
    for split in ("train", "dev", "audit"):
        entry = manifest.get("files", {}).get(split, {})
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Invalid canonical filename: {split}")
        path = canonical_dir / name
        if not path.resolve().is_relative_to(canonical_dir):
            raise ValueError(f"Canonical file escapes input directory: {split}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"Canonical file SHA256 mismatch: {split}")
        rows = []
        # StringIO iterates physical LF records without splitting Unicode U+2028.
        for number, line in enumerate(io.StringIO(raw.decode("utf-8")), 1):
            if not line.strip():
                raise ValueError(f"Blank canonical JSONL record: {split}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid canonical JSON object: {split}:{number}")
            rows.append(row)
        if not rows or len(rows) != entry.get("rows") or len(rows) != entry.get("available_rows"):
            raise ValueError(f"Static source-shot audit requires the entire canonical {split} split")
        split_counts[split] = len(rows)
        canonical_files[split] = {"path": str(path), "sha256": digest, "bytes": len(raw), "rows": len(rows)}
        ids = set()
        for row in rows:
            if row.get("project") != "gsm8k" or not isinstance(row.get("id"), str) or row["id"] in ids:
                raise ValueError(f"Invalid or duplicate canonical GSM8K row: {split}")
            ids.add(row["id"])
            if row.get("record_sha256") != _digest({k: v for k, v in row.items() if k != "record_sha256"}):
                raise ValueError(f"Canonical record SHA256 mismatch: {row['id']}")
            messages = row.get("prompt")
            if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
                raise ValueError(f"Invalid canonical messages: {row['id']}")
            questions = [message.get("content") for message in messages if message.get("role") == "user"]
            if len(questions) != 1:
                raise ValueError(f"Expected one canonical user question: {row['id']}")
            question = normalize_question(questions[0])
            digest = question_hash(question)
            if digest != row.get("question_hash"):
                raise ValueError(f"Canonical question hash mismatch: {row['id']}")
            tokens = _phrase_tokens(question)
            for index, (example_hash, anchor) in enumerate(zip(example_hashes, anchors)):
                identity = {"split": split, "row_id": row["id"], "source_split": row.get("source_split"),
                            "author_example_index": index, "question_hash": digest,
                            "author_question_hash": example_hash}
                if digest == example_hash:
                    exact_hits.append(identity)
                if any(tokens[start:start + len(anchor)] == anchor for start in range(len(tokens) - len(anchor) + 1)):
                    phrase_hits.append(identity)
    report = {
        "schema_version": 1, "kind": "cod_author_static_eight_shot_overlap_audit",
        "status": "FAIL" if exact_hits or phrase_hits else "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(), "author_revision": REVISION,
        "source_lock_sha256": _digest(lock), "source_files": source_files,
        "canonical_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                               "manifest_payload_sha256": manifest["manifest_payload_sha256"]},
        "canonical_files": canonical_files, "split_counts": split_counts,
        "author_example_count_per_mode": {"cot": 8, "cod": 8},
        "author_question_hashes": example_hashes, "same_ordered_questions_in_both_modes": True,
        "normalization": "Unicode NFKC, casefold, whitespace collapse; matches rlvr_lab.data.question_hash",
        "phrase_rule": "First seven Unicode word tokens of each author question, matched as a contiguous token sequence anywhere in each canonical question; casefolded NFKC with punctuation ignored",
        "phrase_anchor_sha256": [hashlib.sha256(" ".join(anchor).encode("utf-8")).hexdigest() for anchor in anchors],
        "exact_overlap_count": len(exact_hits), "exact_overlaps": exact_hits,
        "identifying_phrase_hit_count": len(phrase_hits), "identifying_phrase_hits": phrase_hits,
        "scope": "Every one of the eight ordered static author questions in both verified configs against every row in canonical train, dev, and audit. Uses actual canonical user text, recomputed question_hash, signed row/manifest hashes, and complete file counts. No dataset row is generated, modified, filtered, or used for tuning.",
        "limitations": "PASS excludes these exact and identifying-prefix matches only; it does not establish absence of semantic duplicates, pretraining contamination, or author exemplar provenance outside the supplied canonical dataset.",
        "model_api_calls": 0, "upstream_code_executed": False,
    }
    report["report_payload_sha256"] = _digest(report)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def audit_expected_answer_equivalence(
    canonical_dir: str | Path, reference_file: str | Path,
    output_path: str | Path | None = None,
    lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove original author reference strings equal every canonical audit answer.

    Exact strings matter: float/fraction normalization can change the upstream
    scorer's behavior. The official raw test file remains external to Git.
    """
    lock = _validated_lock(lock) if lock is not None else load_lock()
    source = lock.get("reference_dataset", {})
    if (source.get("revision") != "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
            or source.get("sha256") != "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"):
        raise ValueError("Expected the pinned official GSM8K test reference")
    reference_file = Path(reference_file).resolve()
    raw = reference_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"] or len(raw) != source.get("bytes"):
        raise ValueError("Official reference dataset SHA256 or byte count mismatch")
    original = [json.loads(line) for line in io.StringIO(raw.decode("utf-8"))]
    if len(original) != source.get("rows"):
        raise ValueError("Official reference dataset row count mismatch")
    canonical_dir = Path(canonical_dir).resolve()
    manifest_file = canonical_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("manifest_payload_sha256") != _digest({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}):
        raise ValueError("Canonical manifest SHA256 mismatch")
    entry = manifest.get("files", {}).get("audit", {})
    name = entry.get("path")
    if not isinstance(name, str) or Path(name).name != name or manifest.get("project") != "gsm8k":
        raise ValueError("Expected canonical GSM8K audit file")
    audit_file = canonical_dir / name
    audit_raw = audit_file.read_bytes()
    if hashlib.sha256(audit_raw).hexdigest() != entry.get("sha256"):
        raise ValueError("Canonical audit file SHA256 mismatch")
    rows = [json.loads(line) for line in io.StringIO(audit_raw.decode("utf-8"))]
    if len(rows) != entry.get("rows") or len(rows) != entry.get("available_rows") or len(rows) != len(original):
        raise ValueError("Reference answer audit requires the full official audit set")
    mismatches: dict[str, list[Any]] = {name: [] for name in (
        "source_record_hash", "source_answer_hash", "question", "expected_answer_string")}
    seen_indices = set()
    for row in rows:
        if row.get("record_sha256") != _digest({k: v for k, v in row.items() if k != "record_sha256"}):
            raise ValueError("Canonical audit record SHA256 mismatch")
        index = row.get("metadata", {}).get("source_index")
        if type(index) is not int or not 0 <= index < len(original) or index in seen_indices:
            raise ValueError("Invalid or repeated canonical source index")
        if row.get("source_split") != "test" or row.get("project") != "gsm8k":
            raise ValueError("Reference audit requires official GSM8K test rows")
        seen_indices.add(index)
        record = original[index]
        if _digest(record) != row.get("source_record_sha256"):
            mismatches["source_record_hash"].append(row["id"])
        if hashlib.sha256(record["answer"].encode("utf-8")).hexdigest() != row["metadata"].get("source_answer_sha256"):
            mismatches["source_answer_hash"].append(row["id"])
        questions = [message.get("content") for message in row["prompt"] if message.get("role") == "user"]
        if questions != [record["question"].strip()] or row.get("question_hash") != question_hash(record["question"]):
            mismatches["question"].append(row["id"])
        # Independently implement the verified tasks/gsm8k.py extraction order.
        answer = record["answer"]
        if "####" in answer:
            answer = answer.split("####")[1]
        answer = answer.strip().replace(",", "").replace("$", "").replace("%", "")
        if answer != row.get("answer"):
            mismatches["expected_answer_string"].append({
                "id": row["id"], "author_expected_answer": answer, "canonical_answer": row.get("answer"),
                "source_record_sha256": row["source_record_sha256"],
                "source_answer_sha256": row["metadata"]["source_answer_sha256"],
            })
    report = {
        "schema_version": 1, "kind": "cod_author_expected_answer_equivalence",
        "status": "FAIL" if any(mismatches.values()) else "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(), "author_revision": REVISION,
        "source_lock_sha256": _digest(lock), "source": {**source, "local_path": str(reference_file)},
        "canonical_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "canonical_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "canonical_audit": {"path": str(audit_file), "sha256": entry["sha256"], "rows": len(rows)},
        "all_original_test_rows_covered_once": len(seen_indices) == len(original),
        "mismatch_counts": {name: len(values) for name, values in mismatches.items()}, "mismatches": mismatches,
        "scope": "All canonical audit rows aligned by metadata.source_index with the hash-locked original official test file. Verifies full source_record_sha256, original source_answer_sha256, question text, and exact author extract_answer(raw answer) equality to canonical answer. This proof is bound to the reported canonical audit SHA256; it does not justify numeric reconstruction for other datasets.",
        "model_api_calls": 0, "upstream_code_executed": False,
    }
    report["report_payload_sha256"] = _digest(report)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def verify_reference_equivalence(
    source_dir: str | Path, canonical_dir: str | Path,
    output_path: str | Path | None = None,
    lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify downloaded author inputs and the colocated official test reference."""
    lock = _validated_lock(lock) if lock is not None else load_lock()
    verify_downloaded(source_dir, lock)
    source_dir = Path(source_dir).resolve()
    reference = source_dir / REFERENCE_PATH
    if reference.is_symlink() or not reference.resolve().is_relative_to(source_dir):
        raise ValueError("Official reference path escapes the external source directory")
    return audit_expected_answer_equivalence(canonical_dir, reference, output_path, lock)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Explicit external directory for upstream input files")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--verify-only", action="store_true", help="Verify existing inputs with no network access")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--audit-canonical-dir", type=Path)
    parser.add_argument("--audit-output", type=Path, help="New JSON report path, required with --audit-canonical-dir")
    parser.add_argument("--reference-test-file", type=Path, help="External pinned official test.jsonl for exact answer-string audit")
    parser.add_argument("--reference-audit-output", type=Path, help="New JSON answer-equivalence report path")
    args = parser.parse_args(argv)
    if bool(args.audit_canonical_dir) != bool(args.audit_output):
        parser.error("--audit-canonical-dir and --audit-output must be supplied together")
    if bool(args.reference_test_file) != bool(args.reference_audit_output) or (args.reference_test_file and not args.audit_canonical_dir):
        parser.error("Reference audit requires --reference-test-file, --reference-audit-output, and --audit-canonical-dir")
    lock = load_lock(args.lock)
    if args.verify_only:
        summary: dict[str, Any] = {"status": "PASS", "files": verify_downloaded(args.output_dir, lock)}
    else:
        summary = fetch_sources(args.output_dir, lock, timeout=args.timeout)
    if args.audit_canonical_dir:
        audit = audit_fewshot_overlap(args.output_dir, args.audit_canonical_dir, args.audit_output, lock)
        summary["static_eight_shot_audit"] = {key: audit[key] for key in (
            "status", "split_counts", "exact_overlap_count", "identifying_phrase_hit_count", "report_payload_sha256")}
        summary["static_eight_shot_audit"]["output"] = str(args.audit_output.resolve())
        summary["status"] = audit["status"]
    if args.reference_test_file:
        reference = audit_expected_answer_equivalence(args.audit_canonical_dir, args.reference_test_file, args.reference_audit_output, lock)
        summary["expected_answer_equivalence"] = {key: reference[key] for key in (
            "status", "mismatch_counts", "report_payload_sha256")}
        summary["expected_answer_equivalence"]["output"] = str(args.reference_audit_output.resolve())
        if reference["status"] != "PASS":
            summary["status"] = "FAIL"
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
