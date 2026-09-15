"""Small, deterministic final-answer verifiers. Never executes model output."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

MAX_COMPLETION_CHARS = 100_000
MAX_NUMBER_CHARS = 256
MAX_DECIMAL_EXPONENT = 1_000
MATH_PROJECTS = frozenset({"gsm8k"})
_NUMBER = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?$", re.ASCII
)
_ANSWER_LINE = re.compile(r"^\s*Answer\s*:\s*(.*?)\s*$", re.IGNORECASE)
_BOX_START = re.compile(r"\\boxed\s*\{")


def _decimal_fraction(value: str) -> Fraction:
    if len(value) > MAX_NUMBER_CHARS or not _NUMBER.fullmatch(value):
        raise ValueError("not a bounded numeric literal")
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    if not number.is_finite() or abs(number.as_tuple().exponent) > MAX_DECIMAL_EXPONENT:
        raise ValueError("numeric exponent outside bounds")
    return Fraction(number)


def parse_numeric(value: str) -> Fraction:
    """Parse one bounded integer/decimal/rational literal, never an expression.

    Thousands separators must be correctly grouped. Fractions may use ``a/b``
    or a simple LaTeX ``\\frac{a}{b}``; arbitrary arithmetic is rejected.
    """
    if not isinstance(value, str) or len(value) > MAX_NUMBER_CHARS:
        raise ValueError("numeric literal outside bounds")
    value = unicodedata.normalize("NFKC", value).strip().replace("−", "-")
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    latex = re.fullmatch(r"([+-]?)\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", value)
    if latex:
        sign, numerator, denominator = latex.groups()
        try:
            result = _decimal_fraction(numerator.strip()) / _decimal_fraction(denominator.strip())
        except ZeroDivisionError as exc:
            raise ValueError("zero denominator") from exc
        return -result if sign == "-" else result
    if "/" in value:
        if value.count("/") != 1:
            raise ValueError("multiple fraction separators")
        numerator, denominator = value.split("/")
        try:
            return _decimal_fraction(numerator.strip()) / _decimal_fraction(denominator.strip())
        except ZeroDivisionError as exc:
            raise ValueError("zero denominator") from exc
    return _decimal_fraction(value)


def _boxed_values(text: str) -> list[tuple[str, int, int]]:
    boxes = []
    for match in _BOX_START.finditer(text):
        depth = 1
        cursor = match.end()
        content_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            boxes.append((text[content_start:cursor - 1], match.start(), cursor))
    return boxes


def _unwrap_box(value: str) -> str:
    boxes = _boxed_values(value)
    if len(boxes) == 1 and not value[:boxes[0][1]].strip() and not value[boxes[0][2]:].strip():
        return boxes[0][0].strip()
    return value.strip()


def _normalize_answer(value: str, project: str) -> str:
    value = _unwrap_box(value)
    # Some models preserve the literal <number>/<letter> placeholder delimiters
    # from the prompt. Accept exactly one outer pair, never nested/XML markup.
    angle = re.fullmatch(r"<([^<>]+)>", value)
    if angle:
        value = angle.group(1).strip()
    if project in MATH_PROJECTS:
        number = parse_numeric(value)
        return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"
    raise ValueError("Only GSM8K is included in this export")


def verify_completion(text: str, answer: str, project: str) -> dict[str, Any]:
    """Return binary verified correctness separately from parsing/format status.

    Only an explicit final ``Answer: ...`` line or a final boxed expression is
    eligible. Conflicting explicit answers anywhere in the response are rejected.
    An arbitrary last number in prose can never become a rewarded answer.
    """
    result: dict[str, Any] = {"correctness": 0.0, "parsed": None, "format_ok": False, "reason": ""}
    project = str(project).lower()
    if project not in MATH_PROJECTS:
        return {**result, "reason": "unsupported_project"}
    try:
        reference = _normalize_answer(str(answer), project)
    except (ValueError, ZeroDivisionError, OverflowError):
        return {**result, "reason": "invalid_reference_answer"}
    if not isinstance(text, str) or not text.strip():
        return {**result, "reason": "empty_completion"}
    if len(text) > MAX_COMPLETION_CHARS:
        return {**result, "reason": "completion_too_long"}
    # Generation can begin after an opening tag supplied by the chat template.
    # Score only the final-answer channel, never an answer inside hidden thought.
    tags = list(re.finditer(r"</?think>", text, flags=re.IGNORECASE))
    open_thoughts = 0
    for tag in tags:
        if tag.group().lower() == "<think>":
            open_thoughts += 1
        else:
            open_thoughts = max(0, open_thoughts - 1)
    if open_thoughts:
        return {**result, "reason": "unfinished_thinking"}
    if tags:
        text = text[tags[-1].end():]
    text = text.strip()
    if not text:
        return {**result, "reason": "missing_final_answer"}
    if len(_BOX_START.findall(text)) > 128:
        return {**result, "reason": "too_many_explicit_answers"}
    last_line = text.splitlines()[-1].strip()
    final_line = _ANSWER_LINE.fullmatch(last_line)
    boxes = _boxed_values(text)
    if final_line:
        final_raw = final_line.group(1)
    elif boxes and re.fullmatch(r"[\s.$\\\[\]()]*", text[boxes[-1][2]:]):
        # A boxed expression may finish a sentence, but must really be last.
        final_raw = boxes[-1][0]
    else:
        return {**result, "reason": "missing_final_answer"}
    try:
        parsed = _normalize_answer(final_raw, project)
    except (ValueError, ZeroDivisionError, OverflowError):
        return {**result, "reason": "invalid_final_answer"}
    result["parsed"] = parsed
    explicit = [box[0] for box in boxes]
    explicit.extend(match.group(1) for line in text.splitlines() if (match := _ANSWER_LINE.fullmatch(line)))
    for candidate in explicit:
        try:
            normalized = _normalize_answer(candidate, project)
        except (ValueError, ZeroDivisionError, OverflowError):
            return {**result, "reason": "invalid_explicit_answer"}
        if normalized != parsed:
            return {**result, "reason": "ambiguous_explicit_answers"}
    return {
        "correctness": float(parsed == reference),
        "parsed": parsed,
        "format_ok": True,
        "reason": "correct" if parsed == reference else "wrong_answer",
    }
