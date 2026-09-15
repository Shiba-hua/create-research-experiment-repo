#!/usr/bin/env python3
"""Redraw saved author CoT/CoD numerical plot data after public statistics checks.

No model generation, downloads, or private full-text regrading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'repro'))
from capsule import dump, ident, check
from recompute import run as recompute
import hashlib
def _digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _file_identity(path): return ident(path)

MODES = ('cot', 'cod')
COLORS = {'cot': '#286a9c', 'cod': '#d47629'}
LABELS = {'cot': 'CoT', 'cod': 'CoD'}
SCORES = ('strict_correct', 'author_correct')
SCORE_LABELS = {'strict_correct': 'Strict #### numeric score',
                'author_correct': 'Author-compatible score (diagnostic)'}
TOKEN_KEYS = ('prompt_tokens', 'response_tokens', 'total_tokens')
DIAGNOSTICS = ('strict_format_ok', 'thinking_closed', 'eos', 'truncated')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)








def _finish(fig, output: Path, name: str, title: str, note: str) -> list[dict]:
    fig.suptitle(title, fontsize=15, fontweight='semibold', y=.985)
    fig.text(.02, .018, note, fontsize=8.5, va='bottom')
    fig.tight_layout(rect=(0, .08, 1, .95))
    files = []
    for extension in ('png', 'svg', 'pdf'):
        path = output / f'{name}.{extension}'
        fig.savefig(path, dpi=180, facecolor='white')
        files.append({'path': path.name, **_file_identity(path)})
    return files


def render(data: Mapping[str, Any], output: Path, sources: Sequence[Mapping[str, Any]]) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    charts = []
    n = data['n']

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={'width_ratios': [1.15, 1]})
    for row, score in enumerate(SCORES):
        recorded = data['accuracy'][score]
        ax, delta_ax = axes[row]
        for x, mode in enumerate(MODES):
            item = recorded[mode]
            value = 100 * item['accuracy']
            ci = np.asarray(item['wilson_95_ci']) * 100
            ax.bar(x, value, color=COLORS[mode], width=.5)
            ax.errorbar(x, value, yerr=[[value - ci[0]], [ci[1] - value]], color='black', capsize=4, fmt='none')
            ax.annotate(f"{item['correct']}/{n}\n{value:.2f}%", (x, value), xytext=(0, 8), textcoords='offset points', ha='center')
        ax.set(xticks=[0, 1], xticklabels=['CoT', 'CoD'], ylim=(0, 112), ylabel='Accuracy (%)', title=SCORE_LABELS[score])
        delta = 100 * recorded['delta_cod_minus_cot']
        lo, hi = 100 * np.asarray(recorded['paired_bootstrap_95_ci'])
        delta_ax.plot([lo, hi], [0, 0], color='#444444', linewidth=2)
        delta_ax.scatter([delta], [0], color='#333333', s=60, zorder=3)
        delta_ax.axvline(0, color='#888888', linestyle='--', linewidth=1)
        delta_ax.set(yticks=[], ylim=(-1, 1), xlabel='CoD - CoT accuracy (percentage points)', title='Paired question bootstrap, 95% CI')
        delta_ax.text(.5, .76, f'{delta:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]', transform=delta_ax.transAxes, ha='center')
        span = max(abs(lo), abs(hi), abs(delta), .1)
        delta_ax.set_xlim(min(lo, delta, 0) - .2 * span, max(hi, delta, 0) + .2 * span)
    charts.append({'name': 'accuracy', 'files': _finish(fig, output, '01_accuracy', f'Author 8-shot native-Qwen3 comparison | N={n}',
        'Bars: individual Wilson intervals. Deltas: paired percentile intervals. Strict score is primary display; compatibility is diagnostic.\nNo noninferiority or accuracy-preservation claim follows from a nonsignificant difference.')})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, score in zip(axes, SCORES):
        matrix = np.asarray(data['flips'][score]['counts'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, n))
        for i in range(2):
            for j in range(2):
                direction = '\nImproved' if (i, j) == (0, 1) else '\nRegressed' if (i, j) == (1, 0) else ''
                ax.text(j, i, f'{matrix[i, j]}\n{matrix[i, j] / n:.1%}{direction}', ha='center', va='center',
                        color='white' if matrix[i, j] > .55 * n else 'black', fontsize=12)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['CoD wrong', 'CoD correct'],
               yticklabels=['CoT wrong', 'CoT correct'], title=SCORE_LABELS[score])
    charts.append({'name': 'paired_flips', 'files': _finish(fig, output, '02_paired_flips', 'Paired outcomes on every question',
        'Each matrix sums to the same full audit N. Improvement and regression counts use the indicated score separately.')})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, field, scale, ylabel in ((axes[0], 'mean', 1, 'Mean actual tokens per question'),
                                    (axes[1], 'total', 1e6, 'Total actual tokens (millions)')):
        for x, mode in enumerate(MODES):
            prompt = data['cost'][mode]['prompt_tokens'][field] / scale
            generated = data['cost'][mode]['response_tokens'][field] / scale
            total = data['cost'][mode]['total_tokens'][field] / scale
            ax.bar(x, prompt, color=COLORS[mode], alpha=.45, width=.55, label='Input' if x == 0 else None)
            ax.bar(x, generated, bottom=prompt, color=COLORS[mode], width=.55, label='Output' if x == 0 else None)
            ax.text(x, total, f'{total:,.2f}', ha='center', va='bottom')
        ax.set(xticks=[0, 1], xticklabels=['CoT', 'CoD'], ylabel=ylabel)
        ax.margins(y=.15)
        ax.legend()
    charts.append({'name': 'actual_token_cost', 'files': _finish(fig, output, '03_actual_token_cost', 'Actual input + output cost for the full audit',
        'Light = input; dark = generated output. Includes wrong, format-failed and truncated responses, all thinking and returned EOS.\nOutput cap 4096 is a generation limit, not the measured token cost.')})
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, xlabel in ((axes[0, 0], 'response_tokens', 'Actual generated tokens'),
                            (axes[0, 1], 'total_tokens', 'Actual input + output tokens')):
        for mode in MODES:
            values = np.sort(data['lengths'][mode][key])
            ax.step(np.r_[0, values], np.r_[0, np.arange(1, n + 1) / n], where='post',
                    label=LABELS[mode], color=COLORS[mode])
        ax.set(xlabel=xlabel, ylabel='Fraction of all questions', ylim=(0, 1.02))
        ax.legend()
    changes = data['paired_output_token_delta_cod_minus_cot']
    axes[1, 0].hist(changes, bins=min(40, max(1, int(math.sqrt(n)))), color='#596b7a', edgecolor='white')
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set(xlabel='Paired output token change (CoD - CoT)', ylabel='Questions')
    x = np.arange(len(DIAGNOSTICS))
    for offset, mode in ((-.18, 'cot'), (.18, 'cod')):
        items = [data['diagnostics'][mode][key] for key in DIAGNOSTICS]
        bars = axes[1, 1].bar(x + offset, [100 * item['rate'] for item in items], width=.34, color=COLORS[mode], label=LABELS[mode])
        axes[1, 1].bar_label(bars, labels=[str(item['count']) for item in items], padding=2, fontsize=8)
    axes[1, 1].set(xticks=x, xticklabels=['Strict\nformat', 'Thinking\nclosed', 'Natural\nEOS', 'Truncated'],
                   ylabel='All-question rate (%)', ylim=(0, 115))
    axes[1, 1].legend()
    charts.append({'name': 'length_and_termination', 'files': _finish(fig, output, '04_length_and_termination', 'Length distributions and completion diagnostics',
        'Empirical CDFs use every question; no success-only filtering. Negative paired token change means shorter CoD output.\nDiagnostic bars show rates with raw counts; compatibility fallbacks are not strict format passes.')})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    x = np.arange(2)
    for offset, mode in ((-.18, 'cot'), (.18, 'cod')):
        timing = data['timing'][mode]
        values = [timing['evaluation_wall_seconds'], timing['generation_batch_wall_seconds']]
        bars = axes[0].bar(x + offset, values, width=.34, color=COLORS[mode], label=LABELS[mode])
        axes[0].bar_label(bars, fmt='%.1f', padding=3)
        axes[1].plot(timing['batch_index'], timing['batch_wall_seconds'], marker='o', color=COLORS[mode], label=LABELS[mode])
    axes[0].set(xticks=x, xticklabels=['Evaluation wall', 'Generation batches'], ylabel='Seconds')
    axes[0].margins(y=.15)
    axes[1].set(xlabel='Zero-based batch index', ylabel='Measured batch wall seconds')
    for ax in axes:
        ax.legend()
    charts.append({'name': 'timing', 'files': _finish(fig, output, '05_timing', 'Measured sequential-run timing',
        'Evaluation wall excludes model initialization; batch wall time is counted once. Neither measure is individual request latency.\nRun order, warm-up and shared hardware can affect timing. Plot creation does not verify process exit or GPU release.')})
    plt.close(fig)

    numeric_path = output / 'plot_data.json'
    dump(numeric_path, dict(data))
    manifest = {'status': 'rendered', 'kind': 'cod_author_verified_figures', 'created_at': datetime.now(timezone.utc).isoformat(),
        'n': n, 'full_comparison_recomputed': False, 'public_numeric_statistics_recomputed': True, 'score_order': list(SCORES), 'sources': list(sources),
        'plot_data': {'path': numeric_path.name, **_file_identity(numeric_path)}, 'charts': charts,
        'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
        'gpu_release_verified': False, 'experiment_acceptance_assessed': False,
        'content_policy': 'numeric data, labels, paths and hashes only; no author few-shot text or input token IDs'}
    manifest['manifest_sha256'] = _digest(manifest)
    dump(output / 'plot_manifest.json', manifest)
    return manifest




def main():
    p=argparse.ArgumentParser(description='Redraw saved CoT/CoD numerical records; no private text or model required')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    recompute('author')
    path=REPO/'results/gsm8k-author-figures-002/plot_data.json'
    data=json.loads(path.read_text())
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    original=Axes.legend
    def legend(ax,*args,**kwargs):
        if ax.get_ylabel()=='Seconds':kwargs.update(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2)
        return original(ax,*args,**kwargs)
    Axes.legend=legend
    try:result=render(data,a.output.resolve(),[{'path':str(path.relative_to(REPO)),**ident(path)}])
    finally:Axes.legend=original
    print(json.dumps({'status':result['status'],'charts':len(result['charts']),'full_text_regraded':False}))
if __name__=='__main__':main()
