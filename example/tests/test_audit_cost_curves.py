"""CPU behavior checks, not evidence of a new formal experiment."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'repro/experiments/gsm8k-qwen3-0.6b-grpo/plotting/audit_cost_curves.py'
spec=importlib.util.spec_from_file_location('curves',path)
curves=importlib.util.module_from_spec(spec);spec.loader.exec_module(curves)

def row(length,score,eos=True,truncated=False):
    return dict(response_tokens=length,correctness=score,eos=eos,truncated=truncated)

class ReplayTests(unittest.TestCase):
    def test_failure_costs_denominator_and_endpoints(self):
        result=curves.replay([row(2,1),row(4,0,False,True),row(1,0)],4)
        self.assertEqual(result['correct_counts'],[0,0,1,1,1])
        self.assertEqual(result['charged_decode_totals'],[0,3,5,6,7])
        self.assertEqual(result['display_indices'],[0,2,4])
        self.assertEqual(result['val_acc'],1/3)
        self.assertEqual(result['mean_decode_tokens'],7/3)
    def test_rejects_invalid_or_unfinished_correct_response(self):
        for rows in [[],[row(5,0)],[row(2,0.5)],[row(2,1,False,True)],[row(0,1)]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):curves.replay(rows,4)
    def test_all_wrong_preserves_cost_and_cap(self):
        result=curves.replay([row(4,0,False,True),row(2,0)],4)
        self.assertEqual(result['display_indices'],[0,4])
        self.assertEqual(result['correct_counts'],[0]*5)
        self.assertEqual(result['mean_decode_tokens'],3)

if __name__=='__main__':unittest.main()
