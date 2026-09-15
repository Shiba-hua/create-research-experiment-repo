"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from . import cod_author as author
from .coldstart_data import _artifact_identity, _read_rows, _require_sha, _signed_manifest
from .cod_compare import _length_summary, _paired_bootstrap
from .evaluate import _digest, _file_identity
from .statistics import exact_mcnemar_pvalue, wilson_interval
def _original_model_files() -> dict:
    return json.loads((author.ROOT / 'evidence/assets/qwen3-1.7b.json').read_text(encoding='utf-8'))['files']


def load_original_tokenizer(directory: str | Path):
    """CPU-only original tokenizer, from a full snapshot or tokenizer-only copy."""
    directory = Path(directory).resolve()
    expected = _original_model_files()
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt'}
    for name in required:
        path = directory / name
        if not path.resolve().is_relative_to(directory) or _file_identity(path) != expected[name]:
            raise ValueError(f'Tokenizer differs from the frozen original snapshot: {name}')
    for path in directory.iterdir():
        if path.suffix in ('.json', '.jinja', '.model', '.txt') and path.name != 'snapshot_manifest.json':
            if path.name not in expected or _file_identity(path) != expected[path.name]:
                raise ValueError(f'Unexpected/changed local tokenizer configuration: {path.name}')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)


def _load(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path).resolve()
    rows = _read_rows(path)
    metadata = json.loads((path.parent / 'author_summary.json').read_text(encoding='utf-8'))
    if metadata.get('predictions_file') != path.name or metadata.get('predictions_sha256') != _file_identity(path)['sha256']:
        raise ValueError('Author predictions differ from their complete manifest')
    return rows, metadata


def _validate_side(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], canonical: Sequence[Mapping[str, Any]],
                   mode: str, upstream: Mapping[str, Any]) -> dict[int, float]:
    _signed_manifest(meta, 'manifest_sha256', 'author evaluation manifest')
    if (meta.get('protocol') != author.PROTOCOL or meta.get('status') != 'complete'
            or meta.get('dataset_role') != 'audit' or meta.get('project') != 'gsm8k'
            or meta.get('counts', {}).get('expected') != len(canonical) or meta['counts'].get('evaluated') != len(canonical)
            or len(rows) != len(canonical) or meta.get('data_file_sha256') != author.AUDIT_SHA256
            or meta.get('dataset_content_sha256') != _digest(canonical)):
        raise ValueError(f'{mode}: full locked GSM8K audit coverage and complete evidence required')
    config = meta['evaluation_config']
    if meta.get('evaluation_config_sha256') != _digest(config):
        raise ValueError(f'{mode}: corrupt evaluation configuration hash')
    expected = {**author.FIXED, 'mode': mode, 'protocol': author.PROTOCOL, 'parser_version': author.PARSER_VERSION,
                'author_config_path': f'configs/gsm8k_{mode}.yaml',
                'author_config_sha256': upstream['files'][f'configs/gsm8k_{mode}.yaml']['sha256'],
                'upstream_revision': author.UPSTREAM_REVISION, 'upstream_lock_sha256': upstream['lock_file_sha256']}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{mode}: wrong mode, author config, parser, or fixed generation protocol')
    if type(config.get('thinking_opening_prefilled')) is not bool:
        raise ValueError(f'{mode}: actual native thinking opening contract is missing')
    runtime = meta.get('runtime', {})
    if any(runtime.get(key) != value for key, value in {
            'VLLM_USE_V1': '1', 'engine_generation_verified': 'V1',
            'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'engine_client_mode_verified': 'inprocess',
            'engine_client_class': 'vllm.v1.engine.core_client.InprocClient'}.items()):
        raise ValueError(f'{mode}: actual V1 in-process client was not verified')
    if config.get('sampling_fixed') != author.teacher_generate.teacher_sampling_fixed(config['eos_token_ids']):
        raise ValueError(f'{mode}: changed sampler/repetition/EOS policy')
    if meta.get('upstream') != upstream or meta.get('parser_source_sha256') != _file_identity(Path(author.__file__))['sha256']:
        raise ValueError(f'{mode}: current pinned sources/parser do not match recorded evidence')
    artifact = _artifact_identity(meta)
    base = artifact['base']
    if (base['model_id'] != 'Qwen/Qwen3-1.7B' or base['revision'] != author.MODEL_REVISION
            or base['snapshot_manifest_sha256'] != author.MODEL_SNAPSHOT_SHA256
            or base['files'] != _original_model_files() or artifact.get('adapter') is not None
            or meta.get('adapter') is not None or meta['engine_constructor_kwargs'].get('enable_lora') is not False):
        raise ValueError(f'{mode}: original model identity differs or adapter is present')
    if meta.get('num_generations_per_prompt') != 1 or meta.get('retries_per_prompt') != 0:
        raise ValueError(f'{mode}: expected exactly one generation with no retries')
    reference_audit = meta.get('reference_equivalence_audit', {})
    _signed_manifest(reference_audit, 'report_payload_sha256', 'upstream reference equivalence audit')
    if (reference_audit.get('status') != 'PASS' or reference_audit.get('author_revision') != author.UPSTREAM_REVISION
            or reference_audit.get('source_lock_sha256') != _digest({k: v for k, v in upstream.items() if k != 'lock_file_sha256'})
            or reference_audit.get('canonical_audit', {}).get('sha256') != author.AUDIT_SHA256
            or reference_audit.get('canonical_audit', {}).get('rows') != len(canonical)
            or reference_audit.get('all_original_test_rows_covered_once') is not True
            or reference_audit.get('source', {}).get('sha256') != upstream['reference_dataset']['sha256']
            or reference_audit.get('mismatch_counts') != {key: 0 for key in
                ('source_record_hash', 'source_answer_hash', 'question', 'expected_answer_string')}):
        raise ValueError(f'{mode}: upstream reference string equivalence was not verified')
    if (isinstance(meta.get('elapsed_seconds'), bool) or not isinstance(meta.get('elapsed_seconds'), (int, float))
            or not math.isfinite(meta['elapsed_seconds']) or meta['elapsed_seconds'] <= 0):
        raise ValueError(f'{mode}: invalid evaluation wall time')
    if [row['id'] for row in rows] != [row['id'] for row in canonical]:
        raise ValueError(f'{mode}: missing, duplicate, or reordered canonical questions')
    raw = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows).encode()
    if meta.get('predictions_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError(f'{mode}: prediction content SHA256 mismatch')
    if meta.get('row_order_sha256') != _digest([[r['id'], r['question_hash'], r['project'], r['source_split']] for r in rows]):
        raise ValueError(f'{mode}: row-order SHA256 mismatch')
    if (meta.get('rendered_prompts_sha256') != _digest([row['generation_prompt_sha256'] for row in rows])
            or meta.get('prompt_token_ids_sha256') != _digest([row['generation_prompt_token_ids_sha256'] for row in rows])):
        raise ValueError(f'{mode}: prompt content/token identity mismatch')
    batches: dict[int, float] = {}
    for index, (row, original) in enumerate(zip(rows, canonical)):
        for key in ('question_hash', 'project', 'source_split'):
            if row.get(key) != original[key]:
                raise ValueError(f'{mode}: canonical row {key} mismatch')
        if (row.get('canonical_record_sha256') != original['record_sha256']
                or row.get('canonical_answer') != original['answer'] or row.get('mode') != mode):
            raise ValueError(f'{mode}: canonical answer/record/mode mismatch')
        if not isinstance(row.get('response'), str) or not isinstance(row.get('raw_output_text'), str):
            raise ValueError(f'{mode}: missing full output text')
        for key, expected_value in author.score_response(row['response'], original['answer'],
                thinking_opening_prefilled=config['thinking_opening_prefilled']).items():
            if row.get(key) != expected_value or type(row.get(key)) is not type(expected_value):
                raise ValueError(f'{mode}: recorded author/strict score does not match raw final text: {key}')
        if 'generation_prompt' in row or 'generation_prompt_token_ids' in row:
            raise ValueError('Public predictions must not embed the unlicensed author few-shot input')
        for key in ('generation_prompt_sha256', 'generation_prompt_token_ids_sha256', 'author_payload_sha256'):
            _require_sha(row.get(key), key)
        if type(row.get('prompt_tokens')) is not int or not 1 <= row['prompt_tokens'] <= config['max_model_len'] - config['max_new_tokens']:
            raise ValueError(f'{mode}: invalid exact prompt token count')
        for ids_key, length_key, cap in (('response_token_ids', 'response_tokens', config['max_new_tokens']),):
            ids = row.get(ids_key)
            if (not isinstance(ids, list) or not 1 <= len(ids) <= cap or any(type(token) is not int or token < 0 for token in ids)
                    or type(row.get(length_key)) is not int or row[length_key] != len(ids)):
                raise ValueError(f'{mode}: invalid exact {length_key} including failed/truncated rows')
        if type(row.get('total_tokens')) is not int or row['total_tokens'] != row['response_tokens'] + row['prompt_tokens']:
            raise ValueError(f'{mode}: input/output/total token accounting mismatch')
        if any(row.get(key) != meta.get(key) for key in ('evaluation_config_sha256', 'parser_source_sha256')):
            raise ValueError(f'{mode}: row contract hash mismatch')
        if (type(row.get('eos')) is not bool or type(row.get('truncated')) is not bool
                or row.get('finish_reason') not in ('stop', 'length') or row.get('lora_request_id') is not None
                or row['truncated'] != (row['finish_reason'] == 'length')
                or row['eos'] != (row['response_token_ids'][-1] in config['eos_token_ids'] and row['finish_reason'] == 'stop')):
            raise ValueError(f'{mode}: termination or base-only identity mismatch')
        batch_index = index // config['batch_size']
        if row.get('batch_index') != batch_index or row.get('batch_seed') != (config['seed'] + batch_index) % 2**32:
            raise ValueError(f'{mode}: batch order/seed mismatch')
        for key in ('batch_elapsed', 'elapsed'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{mode}: invalid timing')
        size = min(config['batch_size'], len(rows) - batch_index * config['batch_size'])
        if not math.isclose(row['elapsed'], row['batch_elapsed'] / size, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'{mode}: incorrect batch time allocation')
        if batch_index in batches and row['batch_elapsed'] != batches[batch_index]:
            raise ValueError(f'{mode}: inconsistent shared batch wall time')
        batches[batch_index] = row['batch_elapsed']
    for key in ('author_correct', 'strict_correct', 'strict_format_ok', 'thinking_closed', 'eos', 'truncated'):
        if meta['counts'].get(key) != sum(row[key] for row in rows):
            raise ValueError(f'{mode}: summary {key} count mismatch')
    for key in ('response_tokens', 'prompt_tokens', 'total_tokens', 'thinking_response_tokens', 'final_response_tokens'):
        values = [row[key] for row in rows]
        if (any(type(value) is not int or value < 0 for value in values)
                or meta.get(key) != {**author._quantiles(values), 'total': sum(values)}):
            raise ValueError(f'{mode}: token summary mismatch: {key}')
    return batches


def compare_predictions(cot_rows: Sequence[Mapping[str, Any]], cod_rows: Sequence[Mapping[str, Any]],
                        cot_meta: Mapping[str, Any], cod_meta: Mapping[str, Any], canonical: Sequence[Mapping[str, Any]],
                        configs: Mapping[str, Any], upstream: Mapping[str, Any], tokenizer: Any, *,
                        bootstrap_samples: int = 20_000, seed: int = 20260910) -> dict:
    if type(bootstrap_samples) is not int or bootstrap_samples < 1000 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Require at least 1000 paired bootstrap samples and a uint32 seed')
    batches = {mode: _validate_side(rows, meta, canonical, mode, upstream)
               for mode, rows, meta in (('cot', cot_rows, cot_meta), ('cod', cod_rows, cod_meta))}
    excluded = {'mode', 'author_config_path', 'author_config_sha256'}
    if ({k: v for k, v in cot_meta['evaluation_config'].items() if k not in excluded}
            != {k: v for k, v in cod_meta['evaluation_config'].items() if k not in excluded}):
        raise ValueError('Only the author CoT/CoD mode and corresponding config may differ')
    for key in ('evaluated_artifact', 'dataset_content_sha256', 'row_order_sha256', 'source_code',
                'engine_constructor_kwargs', 'input_files', 'source_model_generation_config'):
        if cot_meta.get(key) != cod_meta.get(key):
            raise ValueError(f'Paired model/data/sampling/source identity mismatch: {key}')
    if cot_meta['rendered_prompts_sha256'] == cod_meta['rendered_prompts_sha256']:
        raise ValueError('Identical CoT/CoD prompts; treatment was not applied')
    for mode, rows in (('cot', cot_rows), ('cod', cod_rows)):
        meta = cot_meta if mode == 'cot' else cod_meta
        config = meta['evaluation_config']
        if (config.get('chat_template_sha256') != author._text_hash(tokenizer.get_chat_template())
                or config.get('tokenizer_vocab_sha256') != _digest(tokenizer.get_vocab())
                or config.get('tokenizer_class') != type(tokenizer).__name__):
            raise ValueError('Original tokenizer/chat template identity differs from recorded generation')
        for row, original in zip(rows, canonical):
            payload = author.compose_payload(configs[mode], author.original_question(original))
            if row.get('author_payload_sha256') != author._text_hash(payload):
                raise ValueError(f'{mode}: full author single-user payload SHA256 differs')
            rendered, opening_prefilled = author.render_native_prompt(tokenizer, [{'role': 'user', 'content': payload}])
            ids = list(tokenizer.encode(rendered, add_special_tokens=False))
            if (opening_prefilled is not config['thinking_opening_prefilled']
                    or row['generation_prompt_sha256'] != author._text_hash(rendered)
                    or row['generation_prompt_token_ids_sha256'] != _digest(ids) or row['prompt_tokens'] != len(ids)):
                raise ValueError(f'{mode}: exact input tokens/count do not match the reconstructed original prompt')
            if tokenizer.decode(row['response_token_ids'], skip_special_tokens=True) != row['response']:
                raise ValueError(f'{mode}: decoded output token IDs do not match the scored response')
            thought, final, method = author._thinking_lengths(row['response'], tokenizer, True)
            if (row['thinking_response_tokens'], row['final_response_tokens'], row.get('segment_token_count_method')) != (thought, final, method):
                raise ValueError(f'{mode}: thinking/final token diagnostics do not match actual decoded segments')
    report: dict[str, Any] = {'status': 'complete', 'analysis': author.PROTOCOL, 'n': len(canonical),
        'scoring': {}, 'tokens': {}, 'timing': {},
        'bootstrap': {'samples': bootstrap_samples, 'seed': seed, 'unit': 'paired question', 'interval': 'percentile 95%'},
        'scope': 'Native-thinking adaptation of author prompt/scoring; no training acceptance or accuracy-preservation claim',
        'gpu_release_verified': False,
        'limitations': ['One sampled response per question/mode; seed matching does not match random-number consumption.',
            'No noninferiority margin or multiple-seed robustness assessment; nonsignificance does not prove equal accuracy.',
            'Five-word steps are a soft prompt instruction, not an enforced format or verified reasoning-quality measure.',
            'Output cost includes all generated thinking/final/EOS tokens and every wrong/format-failed/truncated question.',
            'Native thinking, sampling/backend/model differ from the author non-native API protocol.',
            'Per-question elapsed is shared batch wall time divided by batch size, not individual request latency.',
            'Host exit/GPU release and hardware contention require separate native-supervisor evidence.'],
        'evidence': {'cot_manifest_sha256': cot_meta['manifest_sha256'], 'cod_manifest_sha256': cod_meta['manifest_sha256'],
                     'dataset_content_sha256': cot_meta['dataset_content_sha256'], 'row_order_sha256': cot_meta['row_order_sha256'],
                     'evaluated_artifact': cot_meta['evaluated_artifact'], 'upstream': upstream}}
    cot_out = np.array([row['response_tokens'] for row in cot_rows], dtype=np.int64)
    cod_out = np.array([row['response_tokens'] for row in cod_rows], dtype=np.int64)
    for score in ('author_correct', 'strict_correct'):
        before = np.array([row[score] for row in cot_rows], dtype=float)
        after = np.array([row[score] for row in cod_rows], dtype=float)
        wins, losses = int(((before == 0) & (after == 1)).sum()), int(((before == 1) & (after == 0)).sum())
        ci = _paired_bootstrap(before, after, cot_out, cod_out, bootstrap_samples, seed)['accuracy_delta']
        report['scoring'][score] = {mode: {'correct': int(values.sum()), 'accuracy': float(values.mean()),
                                                 'wilson_95_ci': wilson_interval(int(values.sum()), len(values))}
                                    for mode, values in (('cot', before), ('cod', after))}
        report['scoring'][score].update(delta_cod_minus_cot=float((after - before).mean()), paired_bootstrap_95_ci=ci,
            wrong_to_right=wins, right_to_wrong=losses, mcnemar_exact_two_sided_p_descriptive=exact_mcnemar_pvalue(losses, wins))
    for key in ('prompt_tokens', 'response_tokens', 'total_tokens'):
        before = np.array([row[key] for row in cot_rows], dtype=np.int64)
        after = np.array([row[key] for row in cod_rows], dtype=np.int64)
        report['tokens'][key] = {'cot': _length_summary(before), 'cod': _length_summary(after),
            'mean_reduction_cot_minus_cod': float((before - after).mean()), 'relative_mean_reduction': float(1 - after.mean() / before.mean())}
    for mode, rows, meta in (('cot', cot_rows, cot_meta), ('cod', cod_rows, cod_meta)):
        report['timing'][mode] = {'evaluation_wall_seconds': meta['elapsed_seconds'],
            'generation_batch_wall_seconds': sum(batches[mode].values()),
            'batch_allocated_seconds_per_question': sum(batches[mode].values()) / len(rows), 'batches': len(batches[mode])}
    report['diagnostics'] = {mode: {key: meta['counts'][key] for key in ('strict_format_ok', 'thinking_closed', 'eos', 'truncated')}
                             for mode, meta in (('cot', cot_meta), ('cod', cod_meta))}
    json.dumps(report, allow_nan=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cot', required=True)
    parser.add_argument('--cod', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--tokenizer', required=True, help='Original local Qwen3-1.7B snapshot or hash-identical tokenizer-only copy')
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20260910)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh report path')
    try:
        canonical, _ = author.locked_audit(args.data)
        configs, upstream = author.load_sources(args.source_dir)
        reference = author._source_tools().verify_reference_equivalence(args.source_dir, Path(args.data).resolve().parent)
        if reference.get('status') != 'PASS':
            raise ValueError('Current full canonical reference equivalence failed')
        tokenizer = load_original_tokenizer(args.tokenizer)
        cot, cot_meta = _load(args.cot)
        cod, cod_meta = _load(args.cod)
        report = compare_predictions(cot, cod, cot_meta, cod_meta, canonical, configs, upstream, tokenizer,
                                     bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    except (ValueError, TypeError, KeyError, OSError) as error:
        report = {'status': 'invalid_evidence', 'analysis': author.PROTOCOL, 'error': str(error)}
    output.parent.mkdir(parents=True, exist_ok=True)
    author.teacher_generate._write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
