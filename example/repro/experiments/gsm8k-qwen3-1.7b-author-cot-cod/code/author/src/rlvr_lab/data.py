"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import hashlib
import unicodedata
COD_INSTRUCTION = (
    "Think step by step, but only keep a minimum draft for each thinking step, "
    "with 5 words at most. Return the answer at the end of the response."
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def question_hash(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if not normalized:
        raise ValueError("question cannot be empty")
    return sha256_text(normalized)


