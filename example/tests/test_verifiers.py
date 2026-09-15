"""Offline CPU checks of the packaged historical verifiers and their provenance."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "datasets-and-verifiers/gsm8k/code"
AUTHOR_FUNCTIONS = ("reference_string", "_extract_author", "author_equal", "score_response")


def load_module(name):
    spec = importlib.util.spec_from_file_location("packaged_gsm8k_" + name, CODE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


project = load_module("project_math")
author = load_module("author_math")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_lines(source, node):
    return b"".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def ast_fingerprint(node):
    # python-ast-canonical-json-v1: preserve node types and populated fields,
    # without position attributes or version-dependent empty list fields.
    def canonical(value):
        if isinstance(value, ast.AST):
            return {"node_type": type(value).__name__,
                    "fields": {name: canonical(field) for name, field in ast.iter_fields(value)
                               if not (isinstance(field, list) and not field)}}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    encoded = json.dumps(canonical(node), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded)


class ProjectBehaviorTests(unittest.TestCase):

    def test_correct_and_wrong_answers_preserve_format_diagnostics(self):
        for completion, answer, expected in (('<think>draft</think>\nAnswer: 42', '42', (1.0, '42', True, 'correct')), ('Answer: 41', '42', (0.0, '41', True, 'wrong_answer')), ('Answer: 0.5', '1/2', (1.0, '1/2', True, 'correct')), ('Therefore, $\\boxed{\\frac{1}{2}}$.', '0.5', (1.0, '1/2', True, 'correct'))):
            with self.subTest(completion=completion):
                verdict = project.verify_completion(completion, answer, 'gsm8k')
                self.assertEqual(verdict, dict(zip(('correctness', 'parsed', 'format_ok', 'reason'), expected)))

    def test_unfinished_thinking_ambiguous_answers_and_wrong_protocol_fail_closed(self):
        for completion, reason in (('<think>Answer: 42', 'unfinished_thinking'), ('Answer: 41\nAnswer: 42', 'ambiguous_explicit_answers'), ('The result is 42', 'missing_final_answer'), ('#### 42', 'missing_final_answer'), ('#### 41\n#### 42', 'missing_final_answer'), ('Answer: 42\nMore prose', 'missing_final_answer')):
            with self.subTest(completion=completion):
                verdict = project.verify_completion(completion, '42', 'gsm8k')
                self.assertEqual(verdict['correctness'], 0.0)
                self.assertFalse(verdict['format_ok'])
                self.assertEqual(verdict['reason'], reason)

class AuthorBehaviorTests(unittest.TestCase):

    def score(self, final, answer='42'):
        return author.score_response('<think>draft #### 999</think>\n' + final, answer, thinking_opening_prefilled=False)

    def test_correct_wrong_and_exact_decimal_reference(self):
        for final, answer, correct in (('#### 42', '42', True), ('#### 41', '42', False), ('Reasoning. #### 0.5', '1/2', True)):
            with self.subTest(final=final):
                verdict = self.score(final, answer)
                self.assertTrue(verdict['thinking_closed'])
                self.assertTrue(verdict['strict_format_ok'])
                self.assertEqual(verdict['strict_correct'], correct)
                self.assertEqual(verdict['author_correct'], correct)
        self.assertEqual(author.reference_string('-1/8'), '-0.125')
        with self.assertRaisesRegex(ValueError, 'finite-decimal'):
            author.reference_string('1/3')

    def test_missing_or_repeated_separator_is_compatible_only(self):
        for final in ('The answer is 42', '#### 42\n#### 999'):
            with self.subTest(final=final):
                verdict = self.score(final)
                self.assertTrue(verdict['author_correct'])
                self.assertFalse(verdict['strict_format_ok'])
                self.assertFalse(verdict['strict_correct'])
        self.assertFalse(self.score('#### 999\n#### 42')['author_correct'])
        self.assertEqual(self.score('#### 42\n#### 999')['author_extracted'], '42')

    def test_compatible_first_number_fallback_preserves_false_positives(self):
        for final, answer in (('#### -42', '42'), ('#### 1000 apples', '100'), ('42 appears in the draft, but the answer is 999', '42')):
            with self.subTest(final=final):
                verdict = self.score(final, answer)
                self.assertTrue(verdict['author_correct'])
                self.assertEqual(verdict['author_method'], 'first_number_string_fallback')
                self.assertFalse(verdict['strict_correct'])
        sign = self.score('#### -42')
        self.assertTrue(sign['strict_format_ok'])
        self.assertEqual(sign['strict_parsed'], '-42')

    def test_strict_can_pass_when_compatible_fails(self):
        verdict = self.score('#### 00042')
        self.assertTrue(verdict['strict_correct'])
        self.assertTrue(verdict['strict_format_ok'])
        self.assertFalse(verdict['author_correct'])

    def test_actual_opening_prefill_and_unclosed_thinking_are_not_repaired(self):
        for response in ('<think>#### 42', '#### 42', 'draft</think>#### 42', 'prefix<think>draft</think>#### 42', '<think>draft</think></think>#### 42'):
            with self.subTest(response=response):
                verdict = author.score_response(response, '42', thinking_opening_prefilled=False)
                self.assertFalse(verdict['thinking_closed'])
                self.assertFalse(verdict['strict_correct'])
                self.assertFalse(verdict['author_correct'])
                self.assertIsNone(verdict['final_text'])
        self.assertTrue(author.score_response('draft</think>#### 42', '42', thinking_opening_prefilled=True)['strict_correct'])
        self.assertFalse(author.score_response('<think>draft</think>#### 42', '42', thinking_opening_prefilled=True)['strict_correct'])
        with self.assertRaisesRegex(ValueError, 'boolean'):
            author.score_response('draft</think>#### 42', '42', thinking_opening_prefilled=1)

class CommandLineTests(unittest.TestCase):

    def run_cli(self, completion, mode, answer='42', opening=None, native='false'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'completion.txt'
            path.write_bytes(completion.encode('utf-8'))
            command = [sys.executable, '-S', '-B', str(CODE / 'verify.py'), '--mode', mode, '--completion-file', str(path), '--answer', answer]
            if opening is not None:
                command += ['--opening-prefilled', opening]
            if mode == 'grpo' and native is not None:
                command += ['--native-thinking', native]
            return subprocess.run(command, text=True, capture_output=True, cwd=directory, timeout=15)

    def test_cli_returns_unmodified_verdict_and_the_selected_score(self):
        completion = '<think>draft</think>\r\n#### -42'
        expected = author.score_response(completion, '42', thinking_opening_prefilled=False)
        for mode, score, key in (('author-strict', 0.0, 'strict_correct'), ('author-compatible', 1.0, 'author_correct')):
            with self.subTest(mode=mode):
                result = self.run_cli(completion, mode, opening='false')
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload, {'mode': mode, 'score_key': key, 'score': score, 'verdict': expected})
        result = self.run_cli('Answer: 42', 'grpo')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'mode': 'grpo', 'score_key': 'correctness', 'score': 1.0, 'verdict': project.verify_completion('Answer: 42', '42', 'gsm8k')})

    def test_author_modes_require_an_explicit_literal_boolean(self):
        for mode, opening in (('author-strict', None), ('author-compatible', None), ('author-strict', '1'), ('author-compatible', 'False'), ('grpo', 'false')):
            with self.subTest(mode=mode, opening=opening):
                result = self.run_cli('<think>draft</think>#### 42', mode, opening=opening)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')

    def test_model_output_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'must-not-exist'
            expression = "__import__('pathlib').Path(" + repr(str(marker)) + ").write_text('executed')"
            for mode, completion, opening in (('grpo', 'Answer: ' + expression, None), ('author-strict', '<think>draft</think>#### ' + expression, 'false'), ('author-compatible', '<think>draft</think>#### ' + expression, 'false')):
                with self.subTest(mode=mode):
                    result = self.run_cli(completion, mode, opening=opening)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('verdict', json.loads(result.stdout))
                    self.assertFalse(marker.exists())

if __name__=='__main__':unittest.main()
