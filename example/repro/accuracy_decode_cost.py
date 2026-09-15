#!/usr/bin/env python3
"""Plot observed accuracy versus decode cost from the two historical audits.

This is CPU postprocessing. One measured budget per arm yields one point, not a
multi-budget curve. Exact generated counts include reasoning/final/returned EOS.
"""
import argparse
import json
import os
from pathlib import Path

from capsule import ROOT, REPO, ident, read, dump
from recompute import module, rows, run as recompute


SPECS = {
    'grpo': {
        'title': 'GSM8K | Qwen3-0.6B | GRPO comparison',
        'score': 'correctness', 'budget': 1536,
        'arms': [('base', 'Base', 'results/gsm8k-grpo-formal-001/audit-before-001/predictions.jsonl'),
                 ('grpo', 'GRPO (step 64)', 'results/gsm8k-grpo-formal-001/audit-after-001/predictions.jsonl')],
        'summary_name': 'evaluation_summary.json',
        'result': 'results/gsm8k-grpo-formal-001/acceptance.json',
        'source_commit': '9359731bc1fcdeeef867ff4c5dd1144ff9cdc11f',
    },
    'author': {
        'title': 'GSM8K | Qwen3-1.7B | Author CoT / CoD prompts',
        'score': 'strict_correct', 'budget': 4096,
        'arms': [('cot', 'CoT', 'results/gsm8k-author-cot-audit-002/author_metrics.jsonl'),
                 ('cod', 'CoD', 'results/gsm8k-author-cod-audit-001/author_metrics.jsonl')],
        'summary_name': 'author_metrics_manifest.json',
        'result': 'results/gsm8k-author-comparison-001.json',
        'source_commit': '61ff6a476c3cacec1fe87c846c9559a010f499dd',
    },
}

CI_CONTRACT = {
    'method': 'Wilson score interval', 'nominal_confidence_level': 0.95,
    'interpretation': 'Model-based interval, conditional on this fixed checkpoint and evaluation setup',
    'estimand': 'Marginal one-sample verifier accuracy on a hypothetical target distribution of questions',
    'sampling_unit': 'One question and its one decoded answer',
    'working_model': 'IID Bernoulli correctness after marginalizing over IID questions and independent decoding',
    'fixed_conditions': ['checkpoint', 'prompt', 'decode settings and budget', 'verifier'],
    'assumptions': ['Questions represent the same target distribution',
                    'Question/answer observations are approximately independent',
                    'No audit-based selection of the reported checkpoint or prompt'],
    'assumptions_verified_by_this_experiment': False,
    'coverage': 'Nominal approximate frequentist coverage under the working model; not demonstrated design-based coverage',
    'not_estimated': ['Training-seed variability', 'Repeated-decoding distribution on the fixed question list',
                      'Paired treatment-effect confidence interval', 'Distribution shift or verifier bias',
                      'Decode-cost uncertainty or a joint accuracy/cost confidence region'],
    'fixed_set_score': 'The realized correct/N value is known exactly; these bars are not uncertainty in that count',
    'assumptions_document': 'docs/shared/evaluation.md',
}


def point(records, score_key, budget, wilson):
    """All rows contribute to both the accuracy denominator and decode cost."""
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Empty records or duplicate question IDs')
    for row in records:
        score, length = row.get(score_key), row.get('response_tokens')
        if type(score) not in (bool, int, float) or score not in (0, 1):
            raise ValueError('Missing or nonbinary recorded score')
        if type(length) is not int or not 0 <= length <= budget:
            raise ValueError('Missing or invalid exact generated token count')
    n = len(records)
    correct = int(sum(r[score_key] for r in records))
    total = sum(r['response_tokens'] for r in records)
    return {'n': n, 'correct': correct, 'val_acc': correct / n,
            'decode_tokens_total': total, 'avg_decode_tokens': total / n,
            'accuracy_wilson_95_ci': wilson(correct, n), 'measured_output_budget': budget,
            'score_field': score_key,
            'correct_and_truncated': sum(bool(r[score_key]) and r['truncated'] for r in records)}


def collect(experiment):
    spec = SPECS[experiment]
    # Reuse the existing source/manifest/paired-statistics validation before plotting.
    validated = recompute(experiment)
    statistics = module(ROOT / 'analysis/statistics.py', 'accuracy_cost_statistics')
    points, sources, order = [], [], None
    for arm, label, relative in spec['arms']:
        path = REPO / relative
        records = rows(path)
        current_order = [(r['id'], r['question_hash']) for r in records]
        if order is not None and current_order != order:
            raise ValueError('Comparison question identity/order differs')
        order = current_order
        p = point(records, spec['score'], spec['budget'], statistics.wilson_interval)
        if p['n'] != 1319:
            raise ValueError('Both historical audits must cover all 1319 questions')
        p.update(arm=arm, label=label, predictions=relative)
        if experiment == 'grpo':
            side = 'before' if arm == 'base' else 'after'
            expected_correct = validated[side + '_correct']
            expected_tokens = validated['token_totals'][side]
        else:
            expected_correct = validated['scores']['strict_correct'][arm]
            expected_tokens = validated['token_totals'][arm]['response_tokens']
        if p['correct'] != expected_correct or p['decode_tokens_total'] != expected_tokens:
            raise ValueError('Point disagrees with validated historical statistics')
        points.append(p)
        for source in (path, path.parent / spec['summary_name']):
            sources.append({'path': str(source.relative_to(REPO)), **ident(source)})
    sources.append({'path': spec['result'], **ident(REPO / spec['result'])})
    return {'schema': 'rlvr.accuracy-decode-cost/v1', 'experiment': experiment,
            'title': spec['title'], 'points': points, 'sources': sources,
            'historical_evaluation_commit': spec['source_commit'],
            'axis_contract': {
                'x': 'sum(response_tokens) / all question count',
                'y': 'sum(recorded strict correctness) / all question count',
                'decode_includes': ['generated reasoning', 'generated final answer', 'returned EOS'],
                'decode_excludes': ['prompt', 'padding', 'teacher generation', 'training'],
                'failure_and_truncation_costs_included': True,
                'thinking_segment_tokens_added_again': False},
            'independent_budgets_per_arm': 1, 'multi_budget_curve_observed': False,
            'new_model_run': False, 'full_text_regraded_in_this_plot': False,
            'confidence_interval_contract': CI_CONTRACT,
            'validation': validated, 'producer': {'path': 'repro/accuracy_decode_cost.py', **ident(Path(__file__))}}


def draw(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=180)
    for p, color, marker in zip(data['points'], ('#296a9d', '#d37528'), ('o', 'D')):
        x, y = p['avg_decode_tokens'], p['val_acc']
        low, high = p['accuracy_wilson_95_ci']
        ax.errorbar([x], [y], yerr=[[y - low], [high - y]], fmt=marker,
                    color=color, markersize=8, capsize=5, linewidth=1.7, linestyle='none')
        ax.annotate(f"{p['label']}\n{x:.2f} tokens; {y:.2%}", (x, y),
                    xytext=(13, 0), textcoords='offset points', va='center', color=color)
    xs = [p['avg_decode_tokens'] for p in data['points']]
    bounds = [value for p in data['points'] for value in p['accuracy_wilson_95_ci']]
    ax.set_xlim(min(xs) * .88, max(xs) * 1.22)
    ax.set_ylim(max(0, min(bounds) - .06), min(1, max(bounds) + .06))
    ax.set_xlabel('Mean actual decode tokens per question\n(reasoning + final + returned EOS; all questions)')
    ax.set_ylabel('Observed accuracy (val_acc)')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.set_title(data['title'] + '\nAccuracy versus actual decode cost', fontsize=13, pad=14)
    budget = data['points'][0]['measured_output_budget']
    n = data['points'][0]['n']
    fig.text(.02, .025,
             f'N={n}; scores and mean decode cost are fixed-set summaries (output cap {budget}).\n'
             'Bars: nominal 95% Wilson CI for a hypothetical IID question population; checkpoint and evaluation setup fixed.\n'
             'Assumes representative independent questions and independent decoding; these assumptions are not verified here.\n'
             'Not a training-seed or paired-gain CI. X has no CI; this is not a joint accuracy/cost confidence region.',
             fontsize=8, va='bottom', color='#454545')
    fig.subplots_adjust(left=.14, right=.98, bottom=.28, top=.82)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=tuple(SPECS))
    parser.add_argument('--output', required=True, help='A new output directory')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new figure directory; existing results are never replaced')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/rlvr-accuracy-decode-matplotlib')
    data = collect(args.experiment)
    fig, _ = draw(data)
    output.mkdir(parents=True)
    dump(output / 'points.json', data)
    files = []
    for suffix in ('png', 'svg'):
        path = output / ('accuracy_decode_cost.' + suffix)
        fig.savefig(path, bbox_inches='tight')
        files.append({'path': path.name, **ident(path)})
    import matplotlib
    import matplotlib.pyplot as plt
    plt.close(fig)
    dump(output / 'plot_manifest.json', {'scope': 'CPU visualization of observed historical audit points',
         'points': {'path': 'points.json', **ident(output / 'points.json')}, 'files': files,
         'matplotlib': matplotlib.__version__, 'producer': data['producer'], 'new_model_run': False})
    print(json.dumps({'experiment': args.experiment, 'points': data['points'], 'new_model_run': False}, indent=2))


if __name__ == '__main__':
    main()
