"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from .coldstart_data import _artifact_identity, _require_sha
from .evaluate import _file_identity, _thinking_setting, local_artifact_identity
TEACHER_ID = 'Qwen/Qwen3-1.7B'


CODE_PATH = 'src/rlvr_lab/teacher_generate.py'


def _write_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=15).strip()
    _require_sha(commit, 'teacher code commit', 40)
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=root, text=True, timeout=15).strip()
    if dirty:
        raise RuntimeError('Commit tracked source changes before formal teacher generation')
    # Include imported contract code as well as the real entrypoint. Untracked
    # entrypoints cannot pass just because git status excludes untracked files.
    files = {}
    for name in ('teacher_generate.py', 'coldstart_data.py', 'evaluate.py', 'data.py', 'rewards.py', 'statistics.py'):
        relative = 'src/rlvr_lab/' + name
        identity = _file_identity(root / relative)
        committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=root, timeout=15)
        if hashlib.sha256(committed).hexdigest() != identity['sha256']:
            raise RuntimeError(f'Teacher source does not match recorded commit: {relative}')
        files[relative] = identity
    return {'code_commit': commit, 'evaluation_code_path': CODE_PATH,
            'evaluation_code_sha256': files[CODE_PATH]['sha256'], 'source_files': files}


def teacher_model_contract(model: str | Path, snapshot_sha256: str,
                           tokenizer: Any, model_config: Any) -> dict[str, Any]:
    """Shared CPU model/special-token lock for full generation and dev preflight."""
    model = Path(model).resolve()
    _require_sha(snapshot_sha256, 'locked teacher snapshot SHA256')
    artifact = local_artifact_identity(model)
    _artifact_identity({'artifact_identity_recorded': True, 'evaluated_artifact': artifact})
    if artifact['base']['model_id'] != TEACHER_ID or artifact['adapter'] is not None:
        raise ValueError(f'This teacher protocol requires the original locked {TEACHER_ID} snapshot')
    if artifact['base']['snapshot_manifest_sha256'] != snapshot_sha256:
        raise ValueError('Local teacher differs from the explicitly locked snapshot SHA256')
    effective, chat_kwargs = _thinking_setting(SimpleNamespace(config=model_config), tokenizer, 'true')
    if effective is not True or type(tokenizer.eos_token_id) is not int:
        raise ValueError('Teacher requires a supported native thinking template and tokenizer EOS')
    generation_path = model / 'generation_config.json'
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    extra_eos = generation_config.get('eos_token_id', [])
    extra_eos = [extra_eos] if type(extra_eos) is int else extra_eos
    if extra_eos is None:
        extra_eos = []
    if not isinstance(extra_eos, list) or any(type(token) is not int or token < 0 for token in extra_eos):
        raise ValueError('Invalid teacher EOS IDs in the locked generation config')
    eos_ids = sorted({tokenizer.eos_token_id, *extra_eos})
    return {'artifact': artifact, 'generation_config': generation_config,
            'eos_ids': eos_ids, 'chat_kwargs': chat_kwargs}


def teacher_engine_config(model: str | Path, config: Mapping[str, Any],
                          model_config: Any, max_prompt_tokens: int) -> dict[str, Any]:
    """Public LLM constructor arguments shared with the bounded dev preflight."""
    max_model_len = max_prompt_tokens + config['max_new_tokens']
    max_positions = getattr(model_config, 'max_position_embeddings', None)
    if type(max_positions) is not int or max_model_len > max_positions:
        raise ValueError('Full prompt plus generation exceeds the locked teacher context; prompts are never truncated')
    return {'model': str(model), 'tokenizer': str(model), 'dtype': 'bfloat16',
        'tensor_parallel_size': 1, 'max_model_len': max_model_len,
        'max_num_seqs': config['batch_size'], 'max_num_batched_tokens': config['max_num_batched_tokens'],
        'gpu_memory_utilization': config['gpu_memory_utilization'], 'enforce_eager': config['enforce_eager'],
        'seed': config['seed'], 'trust_remote_code': False, 'generation_config': 'vllm'}


def teacher_sampling_fixed(eos_ids: Sequence[int]) -> dict[str, Any]:
    """Shared single-sample, no-truncation parameters; seed/cap vary by batch."""
    return {'n': 1, 'best_of': None, 'min_p': 0.0, 'repetition_penalty': 1.0,
            'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'ignore_eos': False,
            'stop': None, 'stop_token_ids': list(eos_ids), 'truncate_prompt_tokens': None,
            'detokenize': True, 'skip_special_tokens': True}


