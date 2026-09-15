"""Offline BF16 Transformers evaluation with auditable per-question predictions."""

from __future__ import annotations

import argparse
import io
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import rewards
from .data import COD_INSTRUCTION


DEFAULTS = {"max_new_tokens": 1024, "batch_size": 16, "seed": 20260910,
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "greedy": False,
    "mode": "cot", "thinking": "auto"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def align_special_tokens(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Apply Trainer's BOS/PAD alignment before the first baseline evaluation."""
    changes = {}
    for owner_name in ('config', 'generation_config'):
        owner = getattr(model, owner_name)
        for key in ('bos_token_id', 'pad_token_id'):
            target = getattr(tokenizer, key)
            if key == 'pad_token_id' and target is None:
                target = tokenizer.eos_token_id
            current = getattr(owner, key, None)
            if current != target:
                changes[owner_name + '.' + key] = {'from': current, 'to': target}
                setattr(owner, key, target)
    return {'rule': 'align BOS and PAD with tokenizer before baseline, matching Transformers Trainer', 'changes': changes}


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


def _prompts(rows: Sequence[Mapping[str, Any]], mode: str) -> list[list[dict[str, str]]]:
    result = []
    for row in rows:
        messages = [dict(message) for message in row["prompt"]]
        contains_cod = any(COD_INSTRUCTION in message["content"] for message in messages)
        if mode == "cot" and contains_cod:
            raise ValueError(f"CoT evaluation requires canonical CoT data; CoD instruction found in {row['id']}")
        if mode == "cod" and not contains_cod:
            systems = [message for message in messages if message["role"] == "system"]
            if systems:
                systems[0]["content"] += "\n" + COD_INSTRUCTION
            else:
                messages.insert(0, {"role": "system", "content": COD_INSTRUCTION})
        result.append(messages)
    return result


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


@contextmanager
def _preserve_rng(torch: Any):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


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


def evaluate_rows(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]],
                  output_dir: str | Path, config_dict: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate existing weights; preserve the caller's training mode and RNG state.

    Configuration keys are DEFAULTS plus optional model/adapter labels and
    predictions_file (default predictions.jsonl). No global gradient setting changes.
    """
    import torch

    rows = list(rows)
    _validate_data(rows)
    config = {**DEFAULTS, **dict(config_dict or {})}
    if config["mode"] not in ("cot", "cod"):
        raise ValueError("mode must be cot or cod")
    for key in ("batch_size", "max_new_tokens"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(config["top_k"], int) or config["top_k"] < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if not math.isfinite(config["temperature"]) or config["temperature"] <= 0 or not 0 < config["top_p"] <= 1:
        raise ValueError("Invalid generation temperature/top_p")
    effective_thinking, chat_kwargs = _thinking_setting(model, tokenizer, config["thinking"])
    if config["greedy"] and effective_thinking:
        raise ValueError("Greedy evaluation is only supported for nonthinking mode")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    old_padding_side = tokenizer.padding_side
    old_pad_token = tokenizer.pad_token
    prompts = _prompts(rows, config["mode"])
    rendered = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs) for messages in prompts]
    evaluation_config = {key: config[key] for key in DEFAULTS}
    evaluation_config.update({"thinking": str(config["thinking"]).lower(), "effective_thinking": effective_thinking,
        "do_sample": not config["greedy"], "padding_side": "left", "dtype": str(next(model.parameters()).dtype),
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "cod_instruction": COD_INSTRUCTION if config["mode"] == "cod" else None,
        "seed_scheme": "seed plus zero-based batch index modulo 2**32",
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_vocab_sha256": _digest(tokenizer.get_vocab()),
        "inherited_generation_config": model.generation_config.to_dict(),
        "torch_version": torch.__version__})
    for package in ("transformers", "peft", "numpy"):
        try:
            evaluation_config[package + "_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            evaluation_config[package + "_version"] = "unavailable"
    config_hash = _digest(evaluation_config)
    reward_hash = hashlib.sha256(Path(rewards.__file__).read_bytes()).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = config.get("predictions_file", "predictions.jsonl")
    if Path(predictions_file).name != predictions_file:
        raise ValueError("predictions_file must be a basename")
    destination = output_dir / predictions_file
    summary_path = output_dir / "evaluation_summary.json"
    if destination.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation evidence in {output_dir}")
    pending = destination.with_suffix(destination.suffix + ".partial")
    if pending.exists():
        raise FileExistsError(f"Partial evaluation already exists: {pending}")
    was_training = model.training
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    predictions = []
    generation_config = {"max_new_tokens": config["max_new_tokens"], "do_sample": not config["greedy"],
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, "use_cache": True,
        "return_dict_in_generate": False, "output_scores": False}
    if not config["greedy"]:
        generation_config.update({key: config[key] for key in ("temperature", "top_p", "top_k")})
    eos_ids = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    device = next(model.parameters()).device
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        with _preserve_rng(torch), torch.inference_mode(), pending.open("x", encoding="utf-8") as output:
            for batch_index, offset in enumerate(range(0, len(rows), config["batch_size"])):
                batch = rows[offset:offset + config["batch_size"]]
                batch_seed = (config["seed"] + batch_index) % 2**32
                random.seed(batch_seed)
                np.random.seed(batch_seed)
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(batch_seed)
                inputs = tokenizer(rendered[offset:offset + len(batch)], padding=True, truncation=False, return_tensors="pt").to(device)
                width = inputs["input_ids"].shape[1]
                max_positions = getattr(model.config, "max_position_embeddings", None)
                if max_positions and width + config["max_new_tokens"] > max_positions:
                    raise ValueError("Prompt plus generation exceeds model context; evaluation never silently truncates prompts")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                batch_started = time.perf_counter()
                generated = model.generate(**inputs, **generation_config)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - batch_started
                if generated.shape[0] != len(batch) or not math.isfinite(elapsed):
                    raise ValueError("Generation returned an incomplete batch or nonfinite timing")
                generated_ids = generated[:, width:].tolist()
                for index, (row, token_ids) in enumerate(zip(batch, generated_ids)):
                    eos_index = next((i for i, token_id in enumerate(token_ids) if token_id in eos_ids), None)
                    eos = eos_index is not None
                    if eos:
                        token_ids = token_ids[:eos_index + 1]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    verification_text = text
                    # Native Qwen3 starts <think> in the prompt, outside generation.
                    # Make that opening explicit so unfinished thoughts cannot score.
                    if effective_thinking and not text.lstrip().startswith("<think>"):
                        verification_text = "<think>\n" + text
                    verdict = rewards.verify_completion(verification_text, row["answer"], row["project"])
                    score = float(verdict["correctness"])
                    if not math.isfinite(score) or score not in (0.0, 1.0):
                        raise ValueError(f"Verifier returned invalid correctness: {row['id']}")
                    thought_tokens, final_tokens, length_method = _thinking_lengths(text, tokenizer, effective_thinking)
                    prediction = {"id": row["id"], "project": row["project"], "question_hash": row["question_hash"],
                        "source_split": row["source_split"], "response": text, "correctness": score,
                        "format_ok": bool(verdict["format_ok"]), "parsed": verdict["parsed"], "reason": verdict["reason"],
                        "response_tokens": len(token_ids), "thinking_response_tokens": thought_tokens,
                        "final_response_tokens": final_tokens, "segment_token_count_method": length_method,
                        "prompt_tokens": int(inputs["attention_mask"][index].sum().item()),
                        "eos": eos, "truncated": not eos and len(token_ids) >= config["max_new_tokens"],
                        "elapsed": elapsed / len(batch), "batch_elapsed": elapsed,
                        "elapsed_method": "batch generation wall time divided by batch size; not individual latency",
                        "batch_index": batch_index, "batch_seed": batch_seed,
                        "evaluation_config_sha256": config_hash, "reward_contract_sha256": reward_hash}
                    output.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                    predictions.append(prediction)
                output.flush()
                print(json.dumps({"event": "eval_batch", "evaluated": len(predictions), "expected": len(rows),
                    "accuracy_so_far": sum(p["correctness"] for p in predictions) / len(predictions), "batch_elapsed": elapsed}), flush=True)
                del generated, inputs
        if len(predictions) != len(rows) or [p["id"] for p in predictions] != [r["id"] for r in rows]:
            raise ValueError("Incomplete evaluation; predictions do not match canonical row order")
        elapsed = time.perf_counter() - started
        accuracy = sum(p["correctness"] for p in predictions) / len(predictions)
        summary = {"status": "complete", "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": config.get("model", getattr(model.config, "_name_or_path", "in_process")), "adapter": config.get("adapter"),
            "special_token_alignment": config.get("special_token_alignment"),
            "evaluated_artifact": config.get("artifact_identity"),
            "artifact_identity_recorded": config.get("artifact_identity") is not None,
            "data_file": str(Path(config["data"]).resolve()) if config.get("data") else None,
            "data_file_sha256": hashlib.sha256(Path(config["data"]).read_bytes()).hexdigest() if config.get("data") else None,
            "source_splits": sorted({r["source_split"] for r in rows}),
            "project": rows[0]["project"], "accuracy": accuracy, "acc": accuracy, "val_acc": accuracy,
            "counts": {"expected": len(rows), "evaluated": len(predictions), "correct": int(sum(p["correctness"] for p in predictions)),
                "format_ok": sum(p["format_ok"] for p in predictions), "eos": sum(p["eos"] for p in predictions),
                "truncated": sum(p["truncated"] for p in predictions)},
            "response_tokens": _quantiles([p["response_tokens"] for p in predictions]),
            "thinking_response_tokens": _quantiles([p["thinking_response_tokens"] for p in predictions]),
            "final_response_tokens": _quantiles([p["final_response_tokens"] for p in predictions]),
            "prompt_tokens": _quantiles([p["prompt_tokens"] for p in predictions]),
            "elapsed_seconds": elapsed, "generated_tokens_per_second": sum(p["response_tokens"] for p in predictions) / elapsed,
            "evaluation_config": evaluation_config, "evaluation_config_sha256": config_hash,
            "row_order_sha256": _digest([[r["id"], r["question_hash"], r["project"], r["source_split"]] for r in rows]),
            "dataset_content_sha256": _digest(rows), "rendered_prompts_sha256": _digest(rendered),
            "reward_contract_sha256": reward_hash, "predictions_file": predictions_file,
            "predictions_sha256": hashlib.sha256(pending.read_bytes()).hexdigest(),
            "limitations": ["One sampled response per question; same seeds and row order do not guarantee identical random-number consumption across models.",
                "No generation full-vocabulary entropy is retained; training records entropy separately.",
                "Thinking/final segments are re-tokenized independently; only response_tokens is an exact generated-token count, including EOS."]}
        summary["manifest_sha256"] = _digest(summary)
        json.dumps(summary, allow_nan=False)
        pending.replace(destination)
        summary_pending = summary_path.with_suffix(".json.partial")
        summary_pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary_pending.replace(summary_path)
        return summary
    finally:
        model.train(was_training)
        tokenizer.padding_side = old_padding_side
        tokenizer.pad_token = old_pad_token


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="Output directory, or a .jsonl predictions path")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--temperature", type=float, default=.6)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=("cot", "cod"), default="cot")
    parser.add_argument("--thinking", choices=("true", "false", "auto"), default="auto")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This evaluation CLI requires a CUDA GPU with native BF16 support")
    rows = []
    for line_number, line in enumerate(io.StringIO(Path(args.data).read_text(encoding="utf-8")), 1):
        if not line.strip():
            raise ValueError(f"Blank dataset row at line {line_number}")
        rows.append(json.loads(line))
    _validate_data(rows)
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(rows):
            raise ValueError("limit must be positive and cannot exceed available rows")
        rows = rows[:args.limit]
    artifact_identity = local_artifact_identity(args.model, args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0}, local_files_only=True, trust_remote_code=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False, local_files_only=True)
    output = Path(args.output)
    config = vars(args).copy()
    config["special_token_alignment"] = align_special_tokens(model, tokenizer)
    config["artifact_identity"] = artifact_identity
    if output.suffix == ".jsonl":
        config["predictions_file"] = output.name
        output = output.parent
    summary = evaluate_rows(model, tokenizer, rows, output, config)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
