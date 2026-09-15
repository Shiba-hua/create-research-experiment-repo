#!/usr/bin/env python3
"""Redraw one historical figure with its legend in unused space; CPU only."""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from capsule import ROOT, REPO, ident, read, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='A new figure directory')
    output = Path(parser.parse_args().output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to replace an existing figure directory')
    run = REPO / 'results/gsm8k-grpo-formal-001'
    original_manifest = read(run / 'figures/plot_manifest.json')
    renderer = ROOT / 'experiments/gsm8k-qwen3-0.6b-grpo/plotting/training_diagnostics.py'
    if ident(renderer)['sha256'] != 'e1e45a71051448357b5715359bd66ff2bbb01499930ab6ed21af890d7edaea71':
        raise ValueError('Historical renderer differs')
    inputs = []
    for name in ('metrics.jsonl', 'groups.jsonl', 'validation.jsonl'):
        expected = next(x for x in original_manifest['sources'] if x['path'] == name)
        actual = ident(run / name)
        if actual != {k: expected[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Historical input differs: ' + name)
        inputs.append({'path': str((run / name).relative_to(REPO)), **actual})
    with tempfile.TemporaryDirectory(prefix='rlvr-figure-review-') as scratch:
        os.environ['MPLCONFIGDIR'] = str(Path(scratch) / 'matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.axes import Axes
        sys.path.insert(0, str(ROOT / 'experiments/gsm8k-qwen3-0.6b-grpo/code/evaluate/src'))
        spec = importlib.util.spec_from_file_location('rlvr_lab.review_render', renderer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_legend = Axes.legend
        matched = []

        def legend(ax, *args, **kwargs):
            if ax.get_title() == 'Generated completion length':
                matched.append(ax.get_title())
                kwargs.update(loc='center', bbox_to_anchor=(0.53, 0.84), ncol=2)
            return original_legend(ax, *args, **kwargs)

        Axes.legend = legend
        try:
            result = module.render(run, Path(scratch) / 'render',
                before_predictions=run / 'audit-before-001/predictions.jsonl',
                after_predictions=run / 'audit-after-001/predictions.jsonl')
        finally:
            Axes.legend = original_legend
        if matched != ['Generated completion length']:
            raise ValueError('Expected exactly one legend repair')
        chart = next(x for x in result['charts'] if x['name'] == '03_generation')
        old_chart = next(x for x in original_manifest['charts'] if x['name'] == '03_generation')
        for series in chart['series']:
            if series['source']:
                series['source'] = os.path.relpath(series['source'], run)
        if chart['series'] != old_chart['series']:
            raise ValueError('Plotted series differ from original figure')
        output.mkdir(parents=True)
        files = []
        for suffix in ('png', 'svg'):
            path = output / ('03_generation.' + suffix)
            path.write_bytes((Path(scratch) / 'render' / path.name).read_bytes())
            files.append({'path': path.name, **ident(path)})
        dump(output / 'plot_manifest.json', {
            'scope': 'Documentation layout repair only; no new scientific measurements',
            'change': 'Move the completion-length legend into unused upper plot space; two columns',
            'original_figure': {'path': 'results/gsm8k-grpo-formal-001/figures/03_generation.png',
                                **ident(run / 'figures/03_generation.png')},
            'renderer': {'path': str(renderer.relative_to(REPO)), **ident(renderer)},
            'repair_script': {'path': 'repro/redraw_generation.py', **ident(Path(__file__))},
            'inputs': inputs, 'series': chart['series'], 'series_equal_original': True,
            'files': files, 'matplotlib': matplotlib.__version__, 'new_model_run': False})
    print(json.dumps({'status': 'PASS', 'figure_count': 1, 'series_equal_original': True}))


if __name__ == '__main__':
    main()
