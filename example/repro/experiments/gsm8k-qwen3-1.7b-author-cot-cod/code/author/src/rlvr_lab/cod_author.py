"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from . import evaluate_vllm, teacher_generate
from .coldstart_data import _read_rows
from .data import question_hash
from .evaluate import _digest, _file_identity, _quantiles, _thinking_lengths
ROOT = Path(__file__).resolve().parents[2]


CODE_PATH = 'src/rlvr_lab/cod_author.py'


UPSTREAM_REVISION = 'a7dbf5dea808b1aa1e12f7a90ea573321581df78'


MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


MODEL_SNAPSHOT_SHA256 = 'd4b07840eec23bc7fe69349ff3706fad4915f8132b4e7bb88c9fe5eee8fd02ab'


AUDIT_SHA256 = '437f2042d9d210e34834d1f596a0b6289f89414672ede7fe47727041dc11da62'


AUDIT_N = 1319


PROTOCOL = 'cod_author_8shot_qwen3_gsm8k_audit_v1'


PARSER_VERSION = 'author_compat_qwen3_final_v2'


FIXED = {'max_new_tokens': 4096, 'batch_size': 128, 'seed': 1729,
         'temperature': .6, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
         'thinking': 'true', 'effective_thinking': True, 'do_sample': True,
         'greedy': False, 'max_model_len': 8192, 'max_num_batched_tokens': 8192,
         'gpu_memory_utilization': .8, 'enforce_eager': False,
         'backend': 'vllm', 'pinned_backend_version': '0.10.2',
         'engine_generation': 'V1', 'engine_client_mode': 'inprocess',
         'dtype': 'bfloat16', 'n': 1, 'enable_lora': False}


def _source_tools():
    spec = importlib.util.spec_from_file_location('cod_author_source_tools', ROOT / 'scripts/fetch_cod_author_sources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sources(source_dir: str | Path) -> tuple[dict, dict]:
    """Verify downloaded inputs against the committed lock before parsing YAML."""
    import yaml
    directory = Path(source_dir).resolve()
    lock_path = ROOT / 'configs/cod_author_upstream.json'
    source_tools = _source_tools()
    lock = source_tools.load_lock()
    source_tools.verify_downloaded(directory, lock)
    configs = {mode: yaml.safe_load((directory / f'configs/gsm8k_{mode}.yaml').read_text(encoding='utf-8'))
               for mode in ('cot', 'cod')}
    for config in configs.values():
        if (set(config) != {'system_prompt', 'format', 'fewshot'} or len(config['fewshot']) != 8
                or not all(set(ex) == {'question', 'answer'} and all(isinstance(v, str) for v in ex.values())
                           for ex in config['fewshot'])):
            raise ValueError('Author config must contain its exact eight question/answer examples')
    if [ex['question'] for ex in configs['cot']['fewshot']] != [ex['question'] for ex in configs['cod']['fewshot']]:
        raise ValueError('Author CoT/CoD eight-shot questions/order differ')
    return configs, {**lock, 'lock_file_sha256': _file_identity(lock_path)['sha256']}


def compose_payload(config: Mapping[str, Any], question: str) -> str:
    """Same newline/8-shot formatting as pinned utils.compose_request(shot=None)."""
    examples = [config['format'].format(question=ex['question'], answer=ex['answer'])
                for ex in config['fewshot']]
    return config['system_prompt'] + '\n' + '\n'.join(examples) + '\n' + config['format'].format(question=question, answer='')


def original_question(row: Mapping[str, Any]) -> str:
    messages = [message for message in row['prompt'] if message['role'] == 'user']
    if len(messages) != 1 or question_hash(messages[0]['content']) != row['question_hash']:
        raise ValueError('Canonical original question does not match its question hash')
    return messages[0]['content']


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def render_native_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> tuple[str, bool]:
    """Render native mode unchanged and identify who supplies its opening tag.

    The locked Qwen3 template leaves only the assistant header in native mode;
    enable_thinking=False instead appends an already-closed thinking envelope.
    A template that prefills only <think> is also valid. Never add tokens here.
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    disabled = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    header = '<|im_start|>assistant\n'
    if rendered == disabled or header not in rendered:
        raise ValueError('Native thinking switch must produce a distinct assistant generation prompt')
    tail = rendered.rsplit(header, 1)[1].strip()
    if tail not in ('', '<think>'):
        raise ValueError('Native assistant prompt must have an empty tail or one unclosed thinking opening')
    return rendered, tail == '<think>'


def reference_string(canonical_answer: str) -> str:
    """Reconstruct exact finite decimal value; never regex-extract a numerator.

    GSM8K canonicalization discarded formatting/solution text. Current audit
    answers are integers. Finite decimals cover normalization such as 0.5 -> 1/2;
    reject nonterminating fractions because their upstream spelling is unknown.
    """
    value = Fraction(canonical_answer)
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError('Cannot reconstruct upstream finite-decimal reference from this fraction')
    scale = max(twos, fives)
    if not scale:
        return str(value.numerator)
    digits = str(abs(value.numerator) * 2 ** (scale - twos) * 5 ** (scale - fives)).zfill(scale + 1)
    return ('-' if value < 0 else '') + digits[:-scale] + '.' + digits[-scale:]


def _extract_author(text: str) -> str:
    selected = text.split('####')[1] if '####' in text else text
    return selected.strip().replace(',', '').replace('$', '').replace('%', '')


def author_equal(predicted: str, expected: str) -> tuple[bool, str]:
    """Retain even the upstream unsigned/three-leading-digit fallback behavior."""
    if predicted == expected:
        return True, 'exact_extracted_string'
    match = re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', predicted)
    number = match.group().replace(',', '') if match else None
    fallback = str((float(number) if '.' in number else int(number))) if number is not None else 'None'
    if fallback == expected:
        return True, 'first_number_string_fallback'
    try:
        return float(fallback) == float(expected), 'first_number_float_fallback'
    except (ValueError, OverflowError):
        return False, 'no_numeric_fallback'


def score_response(response: str, answer: str, *, thinking_opening_prefilled: bool) -> dict[str, Any]:
    """Run author compatibility and independent strict diagnostics on final only.

    If the actual prompt supplies <think>, generation must supply only its
    closing tag. Otherwise generation must start with <think> and close it.
    Raw generated text is preserved; no missing tag is silently inserted.
    """
    if type(thinking_opening_prefilled) is not bool:
        raise ValueError('The actual prompt must bind a boolean thinking_opening_prefilled')
    expected = reference_string(answer)
    verdict = {'parser_version': PARSER_VERSION, 'reference_string': expected,
               'thinking_opening_prefilled': thinking_opening_prefilled,
               'thinking_closed': False, 'final_text': None, 'author_extracted': None,
               'author_correct': False, 'author_method': 'invalid_thinking_envelope',
               'strict_format_ok': False, 'strict_correct': False, 'strict_parsed': None}
    tags = re.findall(r'</?think\b[^>]*>', response, re.IGNORECASE)
    expected_tags = ['</think>'] if thinking_opening_prefilled else ['<think>', '</think>']
    if tags != expected_tags:
        return verdict
    if not thinking_opening_prefilled and not response.lstrip().startswith('<think>'):
        return verdict
    final = response.split('</think>', 1)[1]
    extracted = _extract_author(final)
    correct, method = author_equal(extracted, expected)
    verdict.update(thinking_closed=True, final_text=final, author_extracted=extracted,
                   author_correct=correct, author_method=method)
    # Exactly one separator followed only by one signed decimal (valid comma
    # grouping allowed). Same-line reasoning before #### is valid, as in the
    # author examples. No units/currency/percent suffix or numeric fallback.
    number = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
    matched = re.fullmatch(number, final.split('####')[1].strip()) if final.count('####') == 1 else None
    if final.count('####') == 1 and matched:
        parsed = matched.group().replace(',', '')
        try:
            correct = Fraction(parsed) == Fraction(answer)
        except (ValueError, OverflowError, ZeroDivisionError):
            # A generated adversarially long integer must not abort every
            # other question in this single-sample evaluation.
            return verdict
        verdict.update(strict_format_ok=True, strict_parsed=parsed, strict_correct=correct)
    return verdict


def _code_identity() -> dict:
    source = evaluate_vllm._code_identity()
    for relative in (CODE_PATH, 'configs/cod_author_upstream.json', 'scripts/fetch_cod_author_sources.py'):
        identity = _file_identity(ROOT / relative)
        committed = subprocess.check_output(['git', 'show', f"{source['code_commit']}:{relative}"], cwd=ROOT, timeout=15)
        if hashlib.sha256(committed).hexdigest() != identity['sha256']:
            raise RuntimeError(f'Commit author experiment source before evaluation: {relative}')
        source['source_files'][relative] = identity
    source.update(evaluation_code_path=CODE_PATH, evaluation_code_sha256=source['source_files'][CODE_PATH]['sha256'])
    return source


def locked_audit(data: str | Path) -> tuple[list[dict], dict]:
    rows, role, identities = evaluate_vllm._heldout(Path(data).resolve())
    if (role != 'audit' or len(rows) != AUDIT_N or identities['canonical_audit']['sha256'] != AUDIT_SHA256
            or any(row['project'] != 'gsm8k' or row['source_split'] != 'test' for row in rows)):
        raise ValueError('Author protocol requires the locked full GSM8K audit1319, never dev/train/subsets')
    for row in rows:
        original_question(row)
        reference_string(row['answer'])
    return rows, identities


def prepare_evaluation(data: str | Path, model: str | Path, source_dir: str | Path,
                       output: str | Path, mode: str, tokenizer: Any, model_config: Any) -> dict:
    if mode not in ('cot', 'cod'):
        raise ValueError('Author mode must be cot or cod')
    model, output = Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory; no overwrite/resume/retry')
    configs, upstream = load_sources(source_dir)
    rows, identities = locked_audit(data)
    reference_audit = _source_tools().verify_reference_equivalence(source_dir, Path(data).resolve().parent)
    if reference_audit.get('status') != 'PASS':
        raise ValueError('Author extracted upstream references differ from canonical answers')
    contract = teacher_generate.teacher_model_contract(model, MODEL_SNAPSHOT_SHA256, tokenizer, model_config)
    snapshot = json.loads((model / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    if snapshot.get('type') == 'derived_sft' or contract['artifact']['base']['revision'] != MODEL_REVISION:
        raise ValueError('Only the original frozen Qwen3-1.7B snapshot is allowed')
    if (model / 'adapter_config.json').exists() or any(model.rglob('adapter_model*.safetensors')):
        raise ValueError('Author base-only experiment rejects adapter artifacts')
    messages = [[{'role': 'user', 'content': compose_payload(configs[mode], original_question(row))}] for row in rows]
    if contract['chat_kwargs'] != {'enable_thinking': True}:
        raise ValueError('Author generation requires explicitly enabled native thinking')
    prepared = [render_native_prompt(tokenizer, prompt) for prompt in messages]
    rendered = [text for text, _ in prepared]
    opening_states = {prefilled for _, prefilled in prepared}
    if len(opening_states) != 1:
        raise ValueError('The full evaluation must share one native thinking opening contract')
    opening_prefilled = prepared[0][1]
    tokens = [list(tokenizer.encode(text, add_special_tokens=False)) for text in rendered]
    if any(not ids or any(type(token) is not int or token < 0 for token in ids) for ids in tokens):
        raise ValueError('Empty/invalid exact prompt token IDs')
    if (getattr(model_config, 'max_position_embeddings', 0) < FIXED['max_model_len']
            or max(map(len, tokens)) + FIXED['max_new_tokens'] > FIXED['max_model_len']):
        raise ValueError('Full eight-shot prompt plus 4096 budget exceeds fixed 8192 context; no truncation')
    # Exact/phrase overlap is independently rerunnable via the source-fetch CLI.
    shot_hashes = {question_hash(ex['question']) for ex in configs[mode]['fewshot']}
    for role in ('train', 'dev', 'audit'):
        if shot_hashes.intersection(row['question_hash'] for row in _read_rows(Path(identities['canonical_' + role]['path']))):
            raise ValueError(f'Author few-shot question overlaps canonical {role}')
    code = _code_identity()
    dependencies = {}
    for package in ('vllm', 'torch', 'transformers', 'tokenizers', 'numpy', 'PyYAML'):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = 'unavailable'
    config = {**FIXED, 'mode': mode, 'protocol': PROTOCOL, 'parser_version': PARSER_VERSION,
              'thinking_opening_prefilled': opening_prefilled,
              'author_config_path': f'configs/gsm8k_{mode}.yaml',
              'author_config_sha256': upstream['files'][f'configs/gsm8k_{mode}.yaml']['sha256'],
              'upstream_revision': UPSTREAM_REVISION, 'upstream_lock_sha256': upstream['lock_file_sha256'],
              'prompt_layout': 'single user payload; pinned system text + eight Q/A examples + current Q/empty A',
              'eos_token_ids': contract['eos_ids'], 'sampling_fixed': teacher_generate.teacher_sampling_fixed(contract['eos_ids']),
              'seed_scheme': 'seed plus zero-based batch index modulo 2**32; each mode starts at seed',
              'chat_template_sha256': hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
              'tokenizer_vocab_sha256': _digest(tokenizer.get_vocab()), 'tokenizer_class': type(tokenizer).__name__,
              'source_dependencies_sha256': _digest(code['source_files']), 'dependency_versions': dependencies}
    engine_kwargs = teacher_generate.teacher_engine_config(model, FIXED, model_config, max(map(len, tokens)))
    engine_kwargs.update(max_model_len=FIXED['max_model_len'], enable_lora=False)
    plan = {'status': 'prepared_not_evaluated', 'protocol': PROTOCOL, 'evaluation_config': config,
            'evaluation_config_sha256': _digest(config), 'source_code': code, 'upstream': upstream,
            'model': str(model), 'adapter': None, 'evaluated_artifact': contract['artifact'],
            'artifact_identity_recorded': True, 'source_model_generation_config': contract['generation_config'],
            'engine_constructor_kwargs': engine_kwargs, 'input_files': identities,
            'data_file': str(Path(data).resolve()), 'data_file_sha256': identities['canonical_audit']['sha256'],
            'dataset_role': 'audit', 'project': 'gsm8k', 'expected': len(rows),
            'dataset_content_sha256': _digest(rows),
            'row_order_sha256': _digest([[r['id'], r['question_hash'], r['project'], r['source_split']] for r in rows]),
            'rendered_prompts_sha256': _digest([_text_hash(text) for text in rendered]),
            'prompt_token_ids_sha256': _digest([_digest(ids) for ids in tokens]),
            'input_retention': 'per-question SHA256 and token counts only; upstream source fetched outside Git',
            'parser_source_sha256': _file_identity(Path(__file__))['sha256'],
            'retries_per_prompt': 0, 'num_generations_per_prompt': 1, 'gpu_release_verified': False,
            'reference_policy': 'canonical numeric value reconstructed as exact finite decimal; original solution not used'}
    plan['reference_equivalence_audit'] = reference_audit
    plan['plan_sha256'] = _digest(plan)
    output.mkdir(parents=True, exist_ok=False)
    teacher_generate._write_json(output / 'author_plan.json', plan)
    state = {'status': 'prepared_not_evaluated', 'expected': len(rows), 'requested': 0, 'evaluated': 0,
             'plan_sha256': plan['plan_sha256'], 'gpu_release_verified': False}
    teacher_generate._write_json(output / 'author_status.json', state)
    return {'plan': plan, 'state': state, 'rows': rows, 'rendered': rendered,
            'payload_hashes': [_text_hash(message[0]['content']) for message in messages],
            'prompt_ids': tokens, 'tokenizer': tokenizer, 'output': output}


def _failed(context: Mapping[str, Any], error: BaseException) -> None:
    context['state'].update(status='failed_incomplete', exception_type=type(error).__name__, exception=str(error))
    teacher_generate._write_json(context['output'] / 'author_status.json', context['state'])


def evaluate_prepared(context: Mapping[str, Any], engine: Any, sampling_factory: Any, runtime: Mapping[str, Any]) -> dict:
    plan, state, rows, tokenizer, output = (context[key] for key in ('plan', 'state', 'rows', 'tokenizer', 'output'))
    config = plan['evaluation_config']
    if (state['status'] != 'prepared_not_evaluated' or _digest(rows) != plan['dataset_content_sha256']
            or _digest([_digest(ids) for ids in context['prompt_ids']]) != plan['prompt_token_ids_sha256']
            or _digest([_text_hash(text) for text in context['rendered']]) != plan['rendered_prompts_sha256']
            or plan['plan_sha256'] != _digest({k: v for k, v in plan.items() if k != 'plan_sha256'})):
        raise ValueError('Prepared data/prompts/plan changed or attempted resume')
    pending, predictions = output / 'author_predictions.jsonl.partial', []
    started_at, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    state['status'] = 'running'
    teacher_generate._write_json(output / 'author_status.json', state)
    try:
        with pending.open('x', encoding='utf-8') as handle, (output / 'requests.jsonl').open('x', encoding='utf-8') as requests:
            for batch_index, offset in enumerate(range(0, len(rows), config['batch_size'])):
                batch = rows[offset:offset + config['batch_size']]
                seed = (config['seed'] + batch_index) % 2**32
                params = {**config['sampling_fixed'], 'seed': seed, 'max_tokens': config['max_new_tokens'],
                          **{key: config[key] for key in ('temperature', 'top_p', 'top_k')}}
                requests.write(json.dumps({'event': 'submit', 'batch_index': batch_index, 'mode': config['mode'],
                                           'ids': [row['id'] for row in batch], 'sampling_kwargs': params, 'lora_request': None}) + '\n')
                requests.flush()
                state.update(requested=offset + len(batch), active_batch_index=batch_index)
                teacher_generate._write_json(output / 'author_status.json', state)
                batch_tokens = context['prompt_ids'][offset:offset + len(batch)]
                start = time.perf_counter()
                generated = engine.generate([{'prompt_token_ids': ids} for ids in batch_tokens],
                                            sampling_factory(**params), use_tqdm=False)
                elapsed = time.perf_counter() - start
                if len(generated) != len(batch) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('Incomplete batch or invalid elapsed time')
                for index, (row, result) in enumerate(zip(batch, generated)):
                    if (result.finished is not True or len(result.outputs) != 1
                            or list(result.prompt_token_ids) != batch_tokens[index]
                            or getattr(result, 'lora_request', None) is not None):
                        raise ValueError('Expected exactly one finished base-model response per exact prompt')
                    completion = result.outputs[0]
                    ids = list(completion.token_ids)
                    if (not ids or len(ids) > config['max_new_tokens'] or any(type(token) is not int or token < 0 for token in ids)
                            or completion.finish_reason not in ('stop', 'length') or not isinstance(completion.text, str)):
                        raise ValueError('Invalid raw completion tokens/finish reason')
                    response = tokenizer.decode(ids, skip_special_tokens=True)
                    verdict = score_response(response, row['answer'],
                        thinking_opening_prefilled=config['thinking_opening_prefilled'])
                    thought, final, method = _thinking_lengths(response, tokenizer, True)
                    prediction = {**{key: row[key] for key in ('id', 'project', 'question_hash', 'source_split')},
                        'mode': config['mode'], 'canonical_record_sha256': row['record_sha256'], 'canonical_answer': row['answer'],
                        **verdict, 'response': response, 'raw_output_text': completion.text, 'response_token_ids': ids,
                        'response_tokens': len(ids), 'thinking_response_tokens': thought, 'final_response_tokens': final,
                        'segment_token_count_method': method,
                        'generation_prompt_sha256': _text_hash(context['rendered'][offset + index]),
                        'generation_prompt_token_ids_sha256': _digest(batch_tokens[index]),
                        'author_payload_sha256': context['payload_hashes'][offset + index],
                        'prompt_tokens': len(batch_tokens[index]),
                        'total_tokens': len(ids) + len(batch_tokens[index]),
                        'eos': bool(ids[-1] in config['eos_token_ids'] and completion.finish_reason == 'stop'),
                        'truncated': completion.finish_reason == 'length', 'finish_reason': completion.finish_reason,
                        'stop_reason': completion.stop_reason, 'lora_request_id': None,
                        'batch_index': batch_index, 'batch_seed': seed, 'elapsed': elapsed / len(batch), 'batch_elapsed': elapsed,
                        'evaluation_config_sha256': plan['evaluation_config_sha256'], 'parser_source_sha256': plan['parser_source_sha256']}
                    handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    predictions.append(prediction)
                    state['evaluated'] = len(predictions)
                teacher_generate._write_json(output / 'author_status.json', state)
                print(json.dumps({'event': 'author_evaluation_batch', 'mode': config['mode'], 'evaluated': len(predictions),
                                  'expected': len(rows), 'batch_elapsed': elapsed}), flush=True)
        if [row['id'] for row in predictions] != [row['id'] for row in rows]:
            raise ValueError('Incomplete or reordered full audit')
        summary = {key: value for key, value in plan.items() if key not in ('status', 'expected', 'plan_sha256')}
        counts = {key: sum(row[key] for row in predictions) for key in
                  ('author_correct', 'strict_correct', 'strict_format_ok', 'thinking_closed', 'eos', 'truncated')}
        summary.update(status='complete', started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(),
            runtime=dict(runtime), counts={'expected': len(rows), 'evaluated': len(predictions), **counts},
            author_accuracy=counts['author_correct'] / len(rows), strict_accuracy=counts['strict_correct'] / len(rows),
            elapsed_seconds=time.perf_counter() - started, predictions_file='author_predictions.jsonl',
            predictions_sha256=_file_identity(pending)['sha256'], requests_file_identity=_file_identity(output / 'requests.jsonl'))
        for key in ('response_tokens', 'prompt_tokens', 'total_tokens', 'thinking_response_tokens', 'final_response_tokens'):
            summary[key] = {**_quantiles([row[key] for row in predictions]), 'total': sum(row[key] for row in predictions)}
        summary['manifest_sha256'] = _digest(summary)
        pending.replace(output / 'author_predictions.jsonl')
        teacher_generate._write_json(output / 'author_summary.json', summary)
        state.update(status='complete', manifest_sha256=summary['manifest_sha256'])
        teacher_generate._write_json(output / 'author_status.json', state)
        return summary
    except BaseException as error:
        _failed(context, error)
        raise


def run(args: argparse.Namespace) -> dict:
    evaluate_vllm._require_evaluation_runtime()
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    context = prepare_evaluation(args.data, args.model, args.source_dir, args.output, args.mode, tokenizer, model_config)
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise InterruptedError('Host supervisor terminated author evaluation; preserve partial evidence')
    signal.signal(signal.SIGTERM, terminate)
    try:
        context['state']['gpu_initialization_attempted'] = True
        teacher_generate._write_json(context['output'] / 'author_status.json', context['state'])
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Require one BF16 CUDA GPU; native host supervisor must ensure exclusive access')
        engine = LLM(**context['plan']['engine_constructor_kwargs'])
        engine_class = evaluate_vllm._engine_generation(engine)
        client_class = evaluate_vllm._engine_client(engine)
        runtime = {'python_executable': sys.executable, 'python_version': sys.version, 'pid': os.getpid(),
                   'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                   'VLLM_USE_V1': os.environ['VLLM_USE_V1'], 'engine_class': engine_class,
                   'engine_generation_verified': 'V1',
                   'VLLM_ENABLE_V1_MULTIPROCESSING': os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'],
                   'engine_client_class': client_class, 'engine_client_mode_verified': 'inprocess'}
        return evaluate_prepared(context, engine, SamplingParams, runtime)
    except BaseException as error:
        _failed(context, error)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local original frozen Qwen3-1.7B snapshot; no adapter')
    parser.add_argument('--data', required=True, help='Complete locked GSM8K audit1319 JSONL')
    parser.add_argument('--source-dir', required=True, help='Inputs fetched using scripts/fetch_cod_author_sources.py')
    parser.add_argument('--output', required=True, help='Fresh per-mode output directory')
    parser.add_argument('--mode', required=True, choices=('cot', 'cod'))
    result = run(parser.parse_args(argv))
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'gpu_release_verified': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
