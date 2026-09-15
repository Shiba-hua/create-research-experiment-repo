"""Historical author verifier functions; exact source slices, CPU standard library only."""

from __future__ import annotations

from fractions import Fraction
import re
from typing import Any

PARSER_VERSION = 'author_compat_qwen3_final_v2'


def reference_string(canonical_answer: str) -> str:
    """Reconstruct exact finite decimal value; never regex-extract a numerator.

    GSM8K canonicalization discarded formatting/solution text. Current audit
    answers are integers. Finite decimals cover normalization such as 0.5 -> 1/2;
    reject nonterminating fractions because their upstream spelling is unknown.
    """
    value = Fraction(canonical_answer)
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError('Cannot reconstruct upstream finite-decimal reference from this fraction')
    scale = max(twos, fives)
    if not scale:
        return str(value.numerator)
    digits = str(abs(value.numerator) * 2 ** (scale - twos) * 5 ** (scale - fives)).zfill(scale + 1)
    return ('-' if value < 0 else '') + digits[:-scale] + '.' + digits[-scale:]


def _extract_author(text: str) -> str:
    selected = text.split('####')[1] if '####' in text else text
    return selected.strip().replace(',', '').replace('$', '').replace('%', '')


def author_equal(predicted: str, expected: str) -> tuple[bool, str]:
    """Retain even the upstream unsigned/three-leading-digit fallback behavior."""
    if predicted == expected:
        return True, 'exact_extracted_string'
    match = re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', predicted)
    number = match.group().replace(',', '') if match else None
    fallback = str((float(number) if '.' in number else int(number))) if number is not None else 'None'
    if fallback == expected:
        return True, 'first_number_string_fallback'
    try:
        return float(fallback) == float(expected), 'first_number_float_fallback'
    except (ValueError, OverflowError):
        return False, 'no_numeric_fallback'


def score_response(response: str, answer: str, *, thinking_opening_prefilled: bool) -> dict[str, Any]:
    """Run author compatibility and independent strict diagnostics on final only.

    If the actual prompt supplies <think>, generation must supply only its
    closing tag. Otherwise generation must start with <think> and close it.
    Raw generated text is preserved; no missing tag is silently inserted.
    """
    if type(thinking_opening_prefilled) is not bool:
        raise ValueError('The actual prompt must bind a boolean thinking_opening_prefilled')
    expected = reference_string(answer)
    verdict = {'parser_version': PARSER_VERSION, 'reference_string': expected,
               'thinking_opening_prefilled': thinking_opening_prefilled,
               'thinking_closed': False, 'final_text': None, 'author_extracted': None,
               'author_correct': False, 'author_method': 'invalid_thinking_envelope',
               'strict_format_ok': False, 'strict_correct': False, 'strict_parsed': None}
    tags = re.findall(r'</?think\b[^>]*>', response, re.IGNORECASE)
    expected_tags = ['</think>'] if thinking_opening_prefilled else ['<think>', '</think>']
    if tags != expected_tags:
        return verdict
    if not thinking_opening_prefilled and not response.lstrip().startswith('<think>'):
        return verdict
    final = response.split('</think>', 1)[1]
    extracted = _extract_author(final)
    correct, method = author_equal(extracted, expected)
    verdict.update(thinking_closed=True, final_text=final, author_extracted=extracted,
                   author_correct=correct, author_method=method)
    # Exactly one separator followed only by one signed decimal (valid comma
    # grouping allowed). Same-line reasoning before #### is valid, as in the
    # author examples. No units/currency/percent suffix or numeric fallback.
    number = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
    matched = re.fullmatch(number, final.split('####')[1].strip()) if final.count('####') == 1 else None
    if final.count('####') == 1 and matched:
        parsed = matched.group().replace(',', '')
        try:
            correct = Fraction(parsed) == Fraction(answer)
        except (ValueError, OverflowError, ZeroDivisionError):
            # A generated adversarially long integer must not abort every
            # other question in this single-sample evaluation.
            return verdict
        verdict.update(strict_format_ok=True, strict_parsed=parsed, strict_correct=correct)
    return verdict
