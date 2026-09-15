"""Optional fixed-dev-prompt attention diagnostic; never generates model answers.

Run before/after with the same --data/--row-index and pass the first tokens.json
as --reference-input in the second run. --max-tokens crops the display only.
This qualitative diagnostic does not grant experimental acceptance.
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize_layer(attention):
    """Full-prompt statistics: per-head causal distributions, then mean over H,Q."""
    import numpy as np
    array = np.asarray(attention, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] != array.shape[2] or min(array.shape) < 1:
        raise ValueError("Expected nonempty attention with shape [heads, query, key]")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Nonfinite or negative attention probabilities")
    n = array.shape[-1]
    causal = np.tril(np.ones((n, n), dtype=bool))
    future_mass = float((array * ~causal).sum(-1).max())
    row_mass = (array * causal).sum(-1, keepdims=True)
    if future_mass > 1e-5 or np.any(np.abs(row_mass - 1) > .05):
        raise ValueError("Attention is not a normalized causal distribution within BF16 tolerance")
    # Re-normalize only for statistics, avoiding BF16 row-sum drift. Raw head
    # means are saved separately and are not replaced with normalized values.
    probability = array * causal / row_mass
    entropy = -(probability * np.log(np.maximum(probability, np.finfo(np.float32).tiny))).sum(-1)
    distance = np.arange(n)[:, None] - np.arange(n)[None, :]
    stats = {"heads": array.shape[0], "query_tokens": n,
        "attention_entropy_nats_mean_head_query": float(entropy.mean()),
        "diagonal_self_mass_mean_head_query": float(np.diagonal(probability, axis1=-2, axis2=-1).mean()),
        "backward_distance_tokens_mean_head_query": float((probability * distance).sum(-1).mean()),
        "raw_causal_row_mass_min": float(row_mass.min()), "raw_causal_row_mass_max": float(row_mass.max()),
        "raw_future_mass_max": future_mass}
    return array.mean(axis=0), stats


def run(args) -> dict:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .evaluate import _thinking_setting, _validate_data, local_artifact_identity

    output, data_path = Path(args.output).resolve(), Path(args.data).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite attention evidence; use a new output directory")
    if args.row_index < 0 or not 1 <= args.max_tokens <= 256 or args.max_input_tokens < 1:
        raise ValueError("Require row-index >= 0, 1 <= max-tokens <= 256, max-input-tokens > 0")
    raw = data_path.read_bytes()
    rows = [json.loads(line) for line in io.StringIO(raw.decode("utf-8"))]
    _validate_data(rows)
    if args.row_index >= len(rows):
        raise ValueError("row-index lies outside the fixed dataset")
    data_hash, dev_verified = hashlib.sha256(raw).hexdigest(), False
    dataset_manifest = data_path.parent / "manifest.json"
    if dataset_manifest.is_file():
        meta = json.loads(dataset_manifest.read_text())
        expected = meta.get("files", {}).get("dev", {})
        if expected.get("sha256") != data_hash or expected.get("rows") != len(rows):
            raise ValueError("Dataset does not match the prepared dev split manifest")
        dev_verified = True
    row = rows[args.row_index]
    identity = local_artifact_identity(args.model, args.adapter)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no automatic device substitution")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager").to(device)
    native_qwen3 = str(model.config.model_type).startswith("qwen3")
    thinking, chat_kwargs = _thinking_setting(model, tokenizer, "true" if native_qwen3 else "auto")
    rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
        add_generation_prompt=True, **chat_kwargs)
    inputs = tokenizer(rendered, return_tensors="pt", truncation=False).to(device)
    ids = inputs["input_ids"][0].tolist()
    n = len(ids)
    context_limit = getattr(model.config, "max_position_embeddings", args.max_input_tokens)
    if n > min(args.max_input_tokens, context_limit):
        raise ValueError("Full prompt exceeds diagnostic/context budget; input is never silently cropped")
    tokens = {"data_sha256": data_hash, "row_index": args.row_index, "id": row["id"],
        "question_hash": row["question_hash"], "prompt_messages": row["prompt"], "rendered_prompt": rendered,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "input_ids": ids, "input_ids_sha256": _digest(ids), "tokens": tokenizer.convert_ids_to_tokens(ids),
        "decoded_tokens": [tokenizer.decode([token_id]) for token_id in ids],
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "enable_thinking": thinking, "tokenizer_source": str(Path(args.model).resolve())}
    reference_hash = None
    if args.reference_input:
        reference_raw = Path(args.reference_input).read_bytes()
        reference_hash = hashlib.sha256(reference_raw).hexdigest()
        reference = json.loads(reference_raw)
        for key in ("data_sha256", "row_index", "id", "question_hash", "input_ids", "rendered_prompt",
                    "chat_template_sha256", "enable_thinking"):
            if reference.get(key) != tokens[key]:
                raise ValueError(f"Before/after fixed input mismatch: {key}")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    # Call the transformer backbone: LoRA layers remain installed, while the
    # vocabulary projection/logits are never requested or retained.
    causal_lm = model.get_base_model() if args.adapter else model
    with torch.inference_mode():
        result = causal_lm.base_model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    if not result.attentions or any(layer is None for layer in result.attentions):
        raise ValueError("Eager forward did not return attention matrices")
    matrices, statistics = {}, []
    for index, layer in enumerate(result.attentions):
        if layer.shape[0] != 1 or layer.shape[-2:] != (n, n):
            raise ValueError("Attention shape does not match the complete fixed prompt")
        matrix, stats = summarize_layer(layer[0].float().cpu().numpy())
        matrices[f"layer_{index:03d}"] = matrix
        statistics.append({"layer": index, **stats})
    del result, layer, model, causal_lm, inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "tokens.json", tokens)
    _write(output / "layer_statistics.json", {"layers": statistics,
        "entropy_definition": "Mean over heads and full-prompt query rows of -sum_key p log p, nats",
        "normalization": "Mask future keys, normalize each head/query row for BF16 drift; no padding exists",
        "distance_definition": "Mean over heads/queries of sum_key p(key|query)*(query_index-key_index)"})
    np.savez_compressed(output / "attention_head_mean.npz", **matrices)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limit = min(n, args.max_tokens)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    selected = (0, len(matrices) - 1)
    vmax = max(float(matrices[f"layer_{index:03d}"][:limit, :limit].max()) for index in selected)
    ticks = np.unique(np.linspace(0, limit - 1, min(7, limit), dtype=int))
    for ax, index in zip(axes, selected):
        shown = matrices[f"layer_{index:03d}"][:limit, :limit]
        graphic = ax.imshow(shown, origin="upper", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set(title=f"Layer {index} | mean over attention heads", xlabel="Key token position / token ID",
            ylabel="Query token position / token ID", xticks=ticks, yticks=ticks)
        labels = [f"{i} / {ids[i]}" for i in ticks]
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(graphic, ax=axes, label="Attention probability (linear, shared scale)", shrink=.75)
    fig.suptitle(f"Fixed dev prompt | row {args.row_index} | display [0:{limit}] of {n} tokens\n"
        "Full prompt forwarded; rows = queries, columns = keys; no generated answer", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"attention_first_last.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    files = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file()}
    manifest = {"status": "complete_diagnostic_not_acceptance", "utc": datetime.now(timezone.utc).isoformat(),
        "model_artifact_identity": identity, "data": str(data_path), "data_sha256": data_hash,
        "prepared_dev_manifest_verified": dev_verified,
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest() if dev_verified else None,
        "project": row["project"], "row_index": args.row_index,
        "id": row["id"], "question_hash": row["question_hash"], "input_ids_sha256": tokens["input_ids_sha256"],
        "reference_input_sha256": reference_hash, "paired_input_verified": reference_hash is not None,
        "input_tokens": n, "display_tokens": limit,
        "display_crop": f"Leading query/key positions [0:{limit}]; model receives all {n} prompt tokens",
        "dtype": "bfloat16", "device": str(device), "attention_implementation": "eager", "enable_thinking": thinking,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "files": files,
        "limitations": ["Fixed dataset row; do not choose a new example separately for each model based on its answer.",
            "Attention entropy measures attention over input positions, not predictive entropy over vocabulary tokens.",
            "Means hide individual heads; full-prompt position count and special tokens affect the statistics.",
            "Qualitative prompt behavior is not proof of reasoning, learning, correctness, or causal explanation.",
            "No generation, chosen answer, output likelihood, or model weights are saved."]}
    _write(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local base model directory")
    parser.add_argument("--adapter", help="Optional local adapter directory")
    parser.add_argument("--data", required=True, help="Canonical fixed dev JSONL")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256, help="Display crop only; at most 256; no input truncation")
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Reject longer full prompts; never crop inference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Explicit execution device; run separately from training")
    parser.add_argument("--reference-input", help="Before-run tokens.json; require identical rendered tokens for comparison")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
