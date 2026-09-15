"""Recheck only public numeric projections; never replace the private verifier."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rlvr_lab.cod_compare import _paired_bootstrap
from rlvr_lab.evaluate import _digest


def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    script = Path(__file__).with_name('build_public_projection.py')
    spec = importlib.util.spec_from_file_location('public_projection_builder', script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    comparison_path = ROOT / 'results/gsm8k-author-comparison-001.json'
    comparison = json.loads(comparison_path.read_text())
    evidence = {'comparison': identity(comparison_path), 'builder': identity(script), 'verifier': identity(Path(__file__))}
    rows, checks = {}, {}
    for mode, run in (('cot', 'gsm8k-author-cot-audit-002'), ('cod', 'gsm8k-author-cod-audit-001')):
        directory = ROOT / 'results' / run
        path, manifest_path = directory / 'author_metrics.jsonl', directory / 'author_metrics_manifest.json'
        meta = json.loads(manifest_path.read_text())
        if (meta['manifest_sha256'] != _digest({k:v for k,v in meta.items() if k != 'manifest_sha256'})
                or meta['projection_builder_sha256'] != evidence['builder']['sha256']
                or meta['preserved_fields'] != list(builder.FIELDS)
                or meta['projection_field_list_sha256'] != _digest(builder.FIELDS)
                or not set(meta['removed_fields']) <= builder.REMOVED
                or identity(path) != {k:meta['projection'][k] for k in ('bytes','sha256')}):
            raise ValueError('Public projection/whitelist/source identity mismatch')
        rows[mode] = [json.loads(line) for line in path.read_bytes().split(b'\n') if line]
        if len(rows[mode]) != 1319 or len({r['id'] for r in rows[mode]}) != 1319:
            raise ValueError('Incomplete public record coverage')
        for row in rows[mode]:
            if (set(row) != set(builder.FIELDS) | {'original_record_line_sha256'}
                    or any(isinstance(value,(dict,list)) for value in row.values())
                    or row['mode'] != mode or row['parser_version'] != 'author_compat_qwen3_final_v2'
                    or any(type(row[key]) is not bool for key in builder.BOOLEANS)
                    or any(isinstance(value,float) and not math.isfinite(value) for value in row.values())):
                raise ValueError('Unexpected public text/array/type')
        order = [[r[k] for k in ('id','question_hash','project','source_split')] for r in rows[mode]]
        if _digest(order) != comparison['evidence']['row_order_sha256']:
            raise ValueError('Public question identity/order differs from the full verifier')
        counts = {key:sum(r[key] for r in rows[mode]) for key in builder.COUNTS}
        totals = {key:sum(r[key] for r in rows[mode]) for key in ('prompt_tokens','response_tokens','total_tokens')}
        if counts != meta['counts'] or totals != meta['token_totals']:
            raise ValueError('Projection aggregate counters changed')
        for key, total in totals.items():
            if total != comparison['tokens'][key][mode]['total']:
                raise ValueError('Public cost differs from the full private verifier')
        for score in ('author_correct','strict_correct'):
            if counts[score] != comparison['scoring'][score][mode]['correct']:
                raise ValueError('Public accuracy differs from the full private verifier')
        checks[mode] = {'n':1319,'counts':counts,'token_totals':totals,'projection':identity(path),
                        'projection_manifest':identity(manifest_path),'no_text_or_token_id_sequences':True}
    paired = {}
    for score in ('strict_correct','author_correct'):
        before = np.array([row[score] for row in rows['cot']],dtype=float)
        after = np.array([row[score] for row in rows['cod']],dtype=float)
        cot_tokens = np.array([row['response_tokens'] for row in rows['cot']])
        cod_tokens = np.array([row['response_tokens'] for row in rows['cod']])
        ci = _paired_bootstrap(before,after,cot_tokens,cod_tokens,comparison['bootstrap']['samples'],comparison['bootstrap']['seed'])['accuracy_delta']
        wins, losses = int(((before==0)&(after==1)).sum()), int(((before==1)&(after==0)).sum())
        target = comparison['scoring'][score]
        if ((wins,losses) != (target['wrong_to_right'],target['right_to_wrong'])
                or not np.allclose(ci,target['paired_bootstrap_95_ci'],rtol=0,atol=1e-12)):
            raise ValueError('Public paired outcomes/interval differ from the full private verifier')
        paired[score] = {'delta':float((after-before).mean()),'paired_95_ci':ci,'wrong_to_right':wins,
                         'right_to_wrong':losses,'matches_full_private_verifier_comparison':True}
    report = {'kind':'independent_public_numeric_projection_validation','status':'PASS','source_files':evidence,
        'runs':checks,'paired_statistics':paired,'full_tokenizer_parser_verification_replaced':False,
        'scope':'Only public numeric coverage/counts/costs/paired intervals and non-reconstructable field structure. Original text/tokenizer/parser verification remains bound to the separate masked-guest private comparison.'}
    report['report_sha256'] = _digest(report)
    path = Path(__file__).with_name('public_projection_validation.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','runs':{k:v['n'] for k,v in checks.items()},'report_sha256':report['report_sha256']}))


if __name__ == '__main__':
    main()
