"""Reproducible public-data CPU integration, not formal experiment acceptance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
PLOTS=ROOT/'repro/experiments/gsm8k-qwen3-0.6b-grpo/plotting'
spec=importlib.util.spec_from_file_location('fixture_builder',PLOTS/'build_cpu_fixture.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
spec=importlib.util.spec_from_file_location('cost_curves',PLOTS/'audit_cost_curves.py')
curves=importlib.util.module_from_spec(spec);spec.loader.exec_module(curves)

class PublicIntegration(unittest.TestCase):
    def test_collect_render_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Path(tmp)/'fixture';output=Path(tmp)/'figures'
            receipt=builder.build(ROOT/'results/gsm8k-grpo-formal-001',fixture)
            before=fixture/'audit-before-001/predictions.jsonl'
            after=fixture/'audit-after-001/predictions.jsonl'
            data=curves.collect(before,after)
            self.assertTrue(data['fixture_only']);self.assertFalse(receipt['prediction_bytes_modified'])
            self.assertEqual([data['arms'][a]['correct'] for a in ['before','after']],[837,970])
            subprocess.run([sys.executable,str(PLOTS/'audit_cost_curves.py'),'--before',str(before),'--after',str(after),'--output',str(output)],check=True,capture_output=True)
            manifest=json.loads((output/'plot_manifest.json').read_text())
            self.assertTrue(manifest['fixture_only']);self.assertFalse(manifest['new_inference'])
            self.assertEqual(len(manifest['files']),5)
            meta=before.parent/'evaluation_summary.json';changed=json.loads(meta.read_text());changed['fixture_only']=False;meta.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):curves.collect(before,after)

if __name__=='__main__':unittest.main()
