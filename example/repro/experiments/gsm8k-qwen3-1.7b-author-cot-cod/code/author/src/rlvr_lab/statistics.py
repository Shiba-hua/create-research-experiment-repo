"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import math
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


