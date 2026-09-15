"""Score a UTF-8 completion file with one explicitly selected historical verifier.

Only the standard library is needed. The two historical author experiment arms
use --opening-prefilled false; the flag describes the actual rendered prompt.
The JSON verdict retains every field returned by the selected historical module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from author_math import score_response
from project_math import verify_completion


SCORE_KEYS = {
    "grpo": "correctness",
    "author-strict": "strict_correct",
    "author-compatible": "author_correct",
}


def boolean(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected the literal true or false")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(SCORE_KEYS))
    parser.add_argument("--completion-file", required=True, type=Path,
                        help="UTF-8 file containing the raw generated completion")
    parser.add_argument("--answer", required=True,
                        help="Canonical reference answer, for example 42 or 1/2")
    parser.add_argument("--opening-prefilled", type=boolean, default=None,
                        metavar="{true,false}",
                        help="Required for author modes; false in both historical author arms")
    parser.add_argument("--native-thinking", type=boolean, default=None,
                        metavar="{true,false}",
                        help="Required for grpo: true for native Qwen3 runs, false for Qwen2.5 students")
    args = parser.parse_args(argv)
    if args.mode != "grpo" and args.opening_prefilled is None:
        parser.error("author modes require --opening-prefilled true or false")
    if args.mode == "grpo" and args.opening_prefilled is not None:
        parser.error("--opening-prefilled is only defined for the author modes")
    if args.mode == "grpo" and args.native_thinking is None:
        parser.error("grpo requires --native-thinking true or false")
    if args.mode != "grpo" and args.native_thinking is not None:
        parser.error("--native-thinking is only defined for grpo")
    try:
        # Read bytes first so newline conversion cannot alter the model output.
        completion = args.completion_file.read_bytes().decode("utf-8")
        if args.mode == "grpo":
            checked_text = completion
            if args.native_thinking and not completion.lstrip().startswith("<think>"):
                checked_text = "<think>\n" + completion
            verdict = verify_completion(checked_text, args.answer, "gsm8k")
        else:
            verdict = score_response(completion, args.answer,
                                     thinking_opening_prefilled=args.opening_prefilled)
    except (OSError, UnicodeError, ValueError, OverflowError, ZeroDivisionError) as error:
        parser.error(str(error))
    score_key = SCORE_KEYS[args.mode]
    print(json.dumps({"mode": args.mode, "score_key": score_key,
                      "score": float(verdict[score_key]), "verdict": verdict},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
