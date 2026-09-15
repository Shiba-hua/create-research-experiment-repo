"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
from typing import Any
import numpy as np
def _length_summary(tokens: np.ndarray) -> dict[str, Any] | None:
    if not len(tokens):
        return None
    return {"n": int(len(tokens)), "mean": float(tokens.mean()), "median": float(np.median(tokens)),
        "p90": float(np.quantile(tokens, .9)), "p95": float(np.quantile(tokens, .95)), "total": int(tokens.sum())}


def _paired_bootstrap(cot_correct: np.ndarray, cod_correct: np.ndarray, cot_tokens: np.ndarray,
                      cod_tokens: np.ndarray, samples: int, seed: int) -> dict[str, list[float]]:
    n = len(cot_tokens)
    rng = np.random.default_rng(seed)
    distributions = {key: np.empty(samples) for key in ("accuracy_delta", "mean_token_reduction", "relative_token_reduction")}
    batch = max(1, min(256, 1_000_000 // n))
    accuracy_delta = cod_correct - cot_correct
    for offset in range(0, samples, batch):
        count = min(batch, samples - offset)
        indices = rng.integers(0, n, size=(count, n))
        before_mean = cot_tokens[indices].mean(axis=1)
        after_mean = cod_tokens[indices].mean(axis=1)
        distributions["accuracy_delta"][offset:offset + count] = accuracy_delta[indices].mean(axis=1)
        distributions["mean_token_reduction"][offset:offset + count] = before_mean - after_mean
        distributions["relative_token_reduction"][offset:offset + count] = 1 - after_mean / before_mean
    return {key: np.quantile(values, [.025, .975]).tolist() for key, values in distributions.items()}


