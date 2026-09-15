#!/usr/bin/env python3
"""Replay saved trajectories at completion gates; no inference or prefix regrading.

The integer thresholds are retrospective gates on one recorded run per arm,
not independently evaluated generation budgets. Costs include every question.
"""
import argparse
import json
import os
from pathlib import Path

import accuracy_decode_cost as observed
from capsule import REPO, dump, ident
from recompute import rows

SPECS = observed.SPECS


def replay(records, score_key, budget, historical_point=None):
    """Return every integer gate plus unique jumps and the actual cap endpoint."""
    if type(budget) is not int or budget <= 0:
        raise ValueError('The recorded output budget must be a positive integer')
    if not records:
        raise ValueError('Empty records')
    ids = [row.get('id') for row in records]
    if any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate question IDs')
    lengths, correct_lengths = [0] * (budget + 1), [0] * (budget + 1)
    for row in records:
        score, length = row.get(score_key), row.get('response_tokens')
        if type(score) not in (bool, int, float) or score not in (0, 1):
            raise ValueError('Missing or nonbinary recorded strict score')
        if type(length) is not int or not 0 <= length <= budget:
            raise ValueError('Missing or invalid exact generated token count')
        if type(row.get('eos')) is not bool or type(row.get('truncated')) is not bool:
            raise ValueError('Missing or nonboolean EOS/truncation status')
        if score and (row['eos'] is not True or row['truncated'] is True):
            raise ValueError('A recorded correct response must finish with EOS without truncation')
        if score and length == 0:
            raise ValueError('A correct EOS completion cannot have zero generated tokens')
        lengths[length] += 1
        correct_lengths[length] += int(score)

    n = len(records)
    original = {
        'n': n, 'correct': sum(correct_lengths),
        'decode_tokens_total': sum(row['response_tokens'] for row in records),
        'measured_output_budget': budget,
    }
    original['val_acc'] = original['correct'] / n
    original['avg_decode_tokens'] = original['decode_tokens_total'] / n
    if historical_point is not None:
        if any(historical_point.get(key) != value for key, value in original.items()):
            raise ValueError('Records disagree with the validated historical point')

    thresholds, counts, totals, accuracy, means = [0], [0], [0], [0.0], [0.0]
    active, charged, correct = n - lengths[0], 0, 0
    jump_budgets = []
    for threshold in range(1, budget + 1):
        charged += active
        active -= lengths[threshold]
        correct += correct_lengths[threshold]
        thresholds.append(threshold)
        counts.append(correct)
        totals.append(charged)
        accuracy.append(correct / n)
        means.append(charged / n)
        if correct_lengths[threshold]:
            jump_budgets.append(threshold)

    def point(threshold):
        return {
            'budget_threshold': threshold,
            'completed_correct_count': counts[threshold],
            'budget_gated_val_acc': accuracy[threshold],
            'charged_decode_tokens_total': totals[threshold],
            'mean_charged_decode_tokens': means[threshold],
            'is_origin': threshold == 0,
            'is_jump': bool(correct_lengths[threshold]),
            'is_full_horizon': threshold == budget,
        }

    endpoint = point(budget)
    checks = {
        'all_questions_retained': n == original['n'],
        'correct_count_matches': endpoint['completed_correct_count'] == original['correct'],
        'accuracy_matches': endpoint['budget_gated_val_acc'] == original['val_acc'],
        'decode_token_total_matches': endpoint['charged_decode_tokens_total'] == original['decode_tokens_total'],
        'mean_decode_tokens_matches': endpoint['mean_charged_decode_tokens'] == original['avg_decode_tokens'],
        'actual_recorded_cap': endpoint['budget_threshold'] == original['measured_output_budget'],
    }
    if not all(checks.values()):
        raise ValueError('Full-horizon replay endpoint differs from recorded statistics')
    return {
        'n': n, 'score_field': score_key, 'measured_output_budget': budget,
        'recorded_point': historical_point if historical_point is not None else original,
        'budget_thresholds': thresholds,
        'completed_correct_counts': counts,
        'budget_gated_val_acc': accuracy,
        'charged_decode_tokens_totals': totals,
        'mean_charged_decode_tokens': means,
        'jump_points': [point(t) for t in jump_budgets],
        'full_horizon': endpoint,
        'plot_points': [point(t) for t in sorted({0, *jump_budgets, budget})],
        'endpoint_validation': {
            'status': 'PASS', 'checks': checks,
            'validated_historical_point_supplied': historical_point is not None,
        },
    }


def collect(experiment):
    """Validate the historical manifests/statistics before reading replay inputs."""
    original = observed.collect(experiment)
    spec = SPECS[experiment]
    source_by_path = {source['path']: source for source in original['sources']}
    point_by_arm = {point['arm']: point for point in original['points']}
    arms, order = [], None
    for arm, label, relative in spec['arms']:
        path = REPO / relative
        expected = {key: source_by_path[relative][key] for key in ('bytes', 'sha256')}
        if ident(path) != expected:
            raise ValueError('Recorded source changed after historical validation')
        records = rows(path)
        if ident(path) != expected:
            raise ValueError('Recorded source changed while loading replay records')
        current_order = [(row['id'], row['question_hash']) for row in records]
        if order is not None and current_order != order:
            raise ValueError('Comparison question identity/order differs')
        order = current_order
        curve = replay(records, spec['score'], spec['budget'], point_by_arm[arm])
        curve.update(arm=arm, label=label, predictions=relative)
        arms.append(curve)
    return {
        'schema': 'rlvr.recorded-trajectory-jumps/v1',
        'experiment': experiment, 'title': spec['title'], 'arms': arms,
        'sources': original['sources'],
        'historical_evaluation_commit': original['historical_evaluation_commit'],
        'method': {
            'name': 'Recorded-trajectory replay with completion gating',
            'length': 'L_i = recorded response_tokens, including returned EOS',
            'correctness': 'c_i = original recorded strict binary score; never reassigned',
            'denominator': 'N = all recorded questions, including failures and truncations',
            'accuracy_formula': 'Y(t) = sum_i(c_i * I[L_i <= t]) / N',
            'cost_formula': 'C(t) = sum_i(min(L_i, t)) / N',
            'domain': 'Every integer t from 0 through the original per-arm output cap B',
            'correct_completion_rule': 'Reject c_i=1 unless eos is True, truncated is False, and L_i > 0',
            'plot_selection': 'Unique origin, each threshold where Y first rises, and original full-horizon endpoint',
            'ties': 'All correct records with identical L_i enter in a single jump',
            'step_placement': 'post: Y(t) applies from the displayed threshold until the next jump',
            'endpoint_rule': 'At B, Y and C must equal the validated original accuracy and mean decode cost',
            'interpretation': 'Completion-gated hindsight on fixed saved trajectories; not prefix correctness or independent budget evaluation',
        },
        'axis_contract': {
            'left_x': 'per-question decode budget threshold t',
            'right_x': 'mean charged decode tokens C(t)',
            'y': 'Recorded completed-correct fraction / budget-gated val_acc Y(t)',
            **{key: value for key, value in original['axis_contract'].items() if key not in ('x', 'y')},
        },
        'independent_budgets_per_arm': 1,
        'independent_multi_budget_evaluation': False,
        'recorded_trajectory_replay': True,
        'new_model_run': False, 'new_inference': False,
        'prefix_regrading': False, 'full_text_regraded_in_this_plot': False,
        'extrapolation_beyond_recorded_cap': False,
        'validation': original['validation'],
        'historical_validation_producer': original['producer'],
        'producer': {'path': 'repro/trajectory_jumps.py', **ident(Path(__file__))},
    }


def coordinates(arm, x_key):
    """Remove only adjacent identical coordinates, e.g. a zero-cost terminal tail."""
    points = []
    for point in arm['plot_points']:
        xy = (point[x_key], point['budget_gated_val_acc'])
        if not points or xy != points[-1]:
            points.append(xy)
    return [point[0] for point in points], [point[1] for point in points]


def draw(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), dpi=180, sharey=True)
    handles = []
    for index, (ax, x_key) in enumerate(zip(axes, ('budget_threshold', 'mean_charged_decode_tokens'))):
        for arm, color in zip(data['arms'], ('#296a9d', '#d37528')):
            xs, ys = coordinates(arm, x_key)
            line, = ax.step(xs, ys, where='post', color=color, linewidth=1.8, label=arm['label'])
            if index == 0:
                handles.append(line)
            endpoint = arm['full_horizon']
            ax.scatter([endpoint[x_key]], [endpoint['budget_gated_val_acc']],
                       marker='D', s=33, color=color, zorder=5, clip_on=False)
        # One shared origin marker; arm paths each begin at the same measured gate.
        ax.scatter([0], [0], s=13, color='#444444', zorder=6, clip_on=False)
        xmax = max(arm['full_horizon'][x_key] for arm in data['arms'])
        ax.set_xlim(0, max(1, xmax * 1.04))
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=index == 0))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    axes[0].set_title('Completion fraction versus budget threshold', fontsize=11, pad=12)
    axes[1].set_title('Completion fraction versus charged decode cost', fontsize=11, pad=12)
    axes[0].set_xlabel('Per-question decode budget threshold (tokens)')
    axes[1].set_xlabel('Mean charged decode tokens per question\n(reasoning + final + returned EOS; all questions)')
    axes[0].set_ylabel('Recorded completed-correct fraction\n(budget-gated val_acc)', labelpad=12)
    handles.append(Line2D([], [], marker='D', color='#555555', linestyle='none', markersize=5,
                          label='Full-horizon endpoint'))
    fig.suptitle(data['title'], fontsize=14, y=.975)
    fig.text(.56, .927, 'Recorded-trajectory replay | no new inference | no prefix regrading',
             ha='center', fontsize=10, color='#454545')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, .90), ncol=3, frameon=False)
    endpoint_text = ' | '.join(
        f"{arm['label']}: B={arm['measured_output_budget']}, "
        f"C(B)={arm['full_horizon']['mean_charged_decode_tokens']:.2f}, "
        f"Y(B)={arm['full_horizon']['budget_gated_val_acc']:.2%}"
        for arm in data['arms'])
    fig.text(.025, .025,
             'Full horizon (diamond): ' + endpoint_text + '\n'
             f"Each arm: N={data['arms'][0]['n']}; one historical inference cap. These thresholds are not independent budget evaluations.\n"
             'All failures and truncations remain in both the cost and accuracy denominator; prompt tokens are excluded.\n'
             'Paths retain the origin, every first-rise threshold, and the original cap endpoint; no extrapolation beyond that cap.',
             fontsize=8.5, va='bottom', color='#454545', linespacing=1.5)
    # Explicit margins keep the complete shared ylabel and the footnotes on canvas.
    fig.subplots_adjust(left=.14, right=.98, bottom=.29, top=.78, wspace=.18)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=tuple(SPECS))
    parser.add_argument('--output', required=True, help='A new output directory')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new figure directory; existing results are never replaced')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/rlvr-trajectory-jumps-matplotlib')
    data = collect(args.experiment)
    fig, _ = draw(data)
    output.mkdir(parents=True)
    try:
        dump(output / 'curves.json', data)
        files = []
        for suffix in ('png', 'svg'):
            path = output / ('trajectory_jumps.' + suffix)
            fig.savefig(path)
            files.append({'path': path.name, **ident(path)})
        dump(output / 'plot_manifest.json', {
            'schema': 'rlvr.recorded-trajectory-plot/v1',
            'scope': 'CPU completion-gated replay of saved trajectories, not independent budget evaluation',
            'data': {'path': 'curves.json', **ident(output / 'curves.json')},
            'files': files, 'producer': data['producer'],
            'source_files': data['sources'],
            'matplotlib': __import__('matplotlib').__version__,
            'new_model_run': False, 'new_inference': False, 'prefix_regrading': False,
        })
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(json.dumps({'experiment': args.experiment, 'recorded_trajectory_replay': True,
                      'new_inference': False, 'prefix_regrading': False,
                      'endpoints': {arm['arm']: arm['full_horizon'] for arm in data['arms']}}, indent=2))


if __name__ == '__main__':
    main()
