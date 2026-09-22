#!/usr/bin/env python3
"""F6/F7 for a new GSM8K audit pair; no model calls or fixed historical paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/evaluate/src'))
from rlvr_lab.statistics import load_predictions, compare_predictions, wilson_interval


def identity(path):
    raw = path.read_bytes()
    return {'name': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def replay(rows, cap):
    if not rows:
        raise ValueError('Empty audit')
    lengths = [0] * (cap + 1)
    correct_lengths = [0] * (cap + 1)
    for row in rows:
        length, score = row['response_tokens'], row['correctness']
        if type(length) is not int or not 0 <= length <= cap:
            raise ValueError('Invalid generated length')
        if type(score) not in (int, float, bool) or score not in (0, 1):
            raise ValueError('Invalid binary score')
        if score and (length == 0 or row.get('eos') is not True or row.get('truncated') is not False):
            raise ValueError('Correct trajectory must terminate without truncation')
        lengths[length] += 1
        correct_lengths[length] += int(score)
    n = len(rows)
    active, charged, correct = n - lengths[0], 0, 0
    correct_counts, charged_totals, jumps = [0], [0], [0]
    for t in range(1, cap + 1):
        charged += active
        active -= lengths[t]
        correct += correct_lengths[t]
        correct_counts.append(correct)
        charged_totals.append(charged)
        if correct_lengths[t]:
            jumps.append(t)
    if jumps[-1] != cap:
        jumps.append(cap)
    assert correct_counts[-1] == sum(row['correctness'] for row in rows)
    assert charged_totals[-1] == sum(row['response_tokens'] for row in rows)
    return {'n': n, 'cap': cap, 'thresholds': list(range(cap + 1)),
            'correct_counts': correct_counts, 'charged_decode_totals': charged_totals,
            'display_indices': jumps, 'val_acc': correct / n, 'mean_decode_tokens': charged / n,
            'correct': correct, 'wilson_95_ci': wilson_interval(correct, n)}


def collect(before_path, after_path):
    a, am = load_predictions(before_path)
    b, bm = load_predictions(after_path)
    if len(a) != 1319 or len(b) != 1319 or any(r.get('project') != 'gsm8k' for r in a + b):
        raise ValueError('Require two complete GSM8K audit1319 records')
    if [r['id'] for r in a] != [r['id'] for r in b]:
        raise ValueError('Pair order differs')
    for meta in (am, bm):
        if meta['evaluation_config'].get('max_new_tokens') != 1536:
            raise ValueError('Expected registered 1536 cap')
    stats = compare_predictions(a, b, before_metadata=am, after_metadata=bm, require_manifests=True)
    sources = []
    for role, path in [('before', before_path), ('after', after_path)]:
        sources.append({'role': role, 'predictions': identity(path),
                        'summary': identity(path.parent / 'evaluation_summary.json')})
    return {'schema': 'gsm8k-audit-cost-curves/v1', 'sources': sources,
            'fixture_only': bool(am.get('fixture_only') or bm.get('fixture_only')),
            'statistics': stats, 'arms': {'before': replay(a, 1536), 'after': replay(b, 1536)},
            'scope': 'Saved audit postprocessing; not new inference, prefix regrading or experiment acceptance',
            'ci_contract': 'Nominal Wilson 95% under hypothetical IID question population; conditional on fixed checkpoint/settings; not training-seed variability',
            'formula': 'Y(t)=sum(c_i*1[L_i<=t])/N; C(t)=sum(min(L_i,t))/N; all failures included',
            'producer': identity(Path(__file__))}


def render(data, output):
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gsm8k-e2e-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    def save(fig, name):
        for suffix in ('png', 'svg'):
            path = output / (name + '.' + suffix)
            fig.savefig(path, bbox_inches='tight')
            files.append(identity(path))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, arm in data['arms'].items():
        y = arm['val_acc'] * 100
        lo, hi = arm['wilson_95_ci']
        ax.errorbar(arm['mean_decode_tokens'], y, yerr=[[y-lo*100], [hi*100-y]], fmt='o', label=label, capsize=4)
    ax.set(xlabel='Mean actual decode tokens (all questions)', ylabel='Accuracy (%)', title='GSM8K audit | observed accuracy and decode cost')
    ax.legend();ax.grid(alpha=.2)
    fig.text(.05, .01, 'N=1319/arm. Includes failures, reasoning, final and returned EOS.\nWilson 95%: hypothetical IID questions; fixed checkpoints, not training-seed uncertainty.', fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));save(fig,'F6_accuracy_decode_cost')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, arm in data['arms'].items():
        ids = arm['display_indices'];y = [100*arm['correct_counts'][i]/arm['n'] for i in ids]
        xs = [[arm['thresholds'][i] for i in ids], [arm['charged_decode_totals'][i]/arm['n'] for i in ids]]
        for ax, x in zip(axes, xs):
            ax.plot(x, y, label=label)
            ax.scatter(x[-1],y[-1],marker='D',s=24)
    for ax, xlabel in zip(axes, ['Per-question token threshold', 'Mean charged decode tokens (all questions)']):
        ax.set(xlabel=xlabel,ylabel='Completed and correct / all questions (%)',ylim=(0,100));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('GSM8K recorded-trajectory completion gates')
    fig.text(.03,.01,'Retrospective replay, not independent budget evaluations or prefix regrading.\nLines connect first-correct jumps; diamonds mark original cap1536. Failures remain in denominator and cost.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.93));save(fig,'F7_completion_gates')
    numeric=output/'plot_data.json';numeric.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    files.append(identity(numeric))
    (output/'plot_manifest.json').write_text(json.dumps({'files':files,'sources':data['sources'],'producer':data['producer'],'new_inference':False,'fixture_only':data['fixture_only'],'experiment_accepted_by_plotter':False},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh figure directory')
    data=collect(args.before,args.after);render(data,args.output)
    print(json.dumps({'status':'rendered','n':1319,'new_inference':False,'correct':{k:v['correct'] for k,v in data['arms'].items()}}))
if __name__=='__main__':main()
