"""Scoped export: helpers used by the GSM8K author prompt comparison."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from . import teacher_generate
from .coldstart_data import _canonical_train, _read_rows, _signed_manifest
from .evaluate import _file_identity
CODE_PATH = 'src/rlvr_lab/evaluate_vllm.py'


def _code_identity() -> dict[str, Any]:
    source = teacher_generate._code_identity()
    root = Path(__file__).resolve().parents[2]
    identity = _file_identity(Path(__file__))
    committed = subprocess.check_output(['git', 'show', f"{source['code_commit']}:{CODE_PATH}"], cwd=root, timeout=15)
    if hashlib.sha256(committed).hexdigest() != identity['sha256']:
        raise RuntimeError('Commit this evaluator before recording an evaluation')
    source['source_files'][CODE_PATH] = identity
    source.update(evaluation_code_path=CODE_PATH, evaluation_code_sha256=identity['sha256'])
    return source


def _heldout(data: Path) -> tuple[list[dict], str, dict]:
    meta = json.loads((data.parent / 'manifest.json').read_text())
    _signed_manifest(meta, 'manifest_payload_sha256', 'canonical manifest')
    entries = meta.get('files', {})
    for role in ('train', 'dev', 'audit'):
        name = entries.get(role, {}).get('path')
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f'Invalid canonical {role} path')
    roles = [role for role in ('dev', 'audit') if (data.parent / entries[role]['path']).resolve() == data]
    if len(roles) != 1:
        raise ValueError('--data must be the complete manifest dev/audit file, never train or an arbitrary subset')
    role = roles[0]
    _, verified, identities = _canonical_train(data.parent / entries['train']['path'])
    rows = _read_rows(data)
    entry = verified['files'][role]
    if (verified['split_policy']['requested_limits'].get(role) is not None
            or entry.get('available_rows') != len(rows) or entry['rows'] != len(rows)
            or _file_identity(data)['sha256'] != identities['canonical_' + role]['sha256']):
        raise ValueError('Evaluation must cover every locked available dev/audit point, without limits')
    return rows, role, identities


def _require_v1_runtime() -> None:
    try:
        version = importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('Use the pinned vLLM 0.10.2 runtime; no automatic installation') from error
    if version != '0.10.2':
        raise RuntimeError('Use the pinned vLLM 0.10.2 runtime; no automatic installation')
    if os.environ.get('VLLM_USE_V1', '1') != '1':
        raise RuntimeError('This evaluator fixes VLLM_USE_V1=1 for strategy/budget parity; V0 is unsupported')
    os.environ['VLLM_USE_V1'] = '1'


def _engine_generation(engine: Any) -> str:
    actual = type(engine.llm_engine).__module__ + '.' + type(engine.llm_engine).__name__
    if actual != 'vllm.v1.engine.llm_engine.LLMEngine':
        raise RuntimeError(f'Actual engine {actual} differs from the frozen vLLM V1 backend')
    return actual


def _require_evaluation_runtime() -> None:
    """Freeze only this evaluator's client mode; existing V1 probe helpers stay unchanged."""
    _require_v1_runtime()
    if os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '0') != '0':
        raise RuntimeError('Evaluation requires VLLM_ENABLE_V1_MULTIPROCESSING=0 for the frozen in-process client')
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'


def _engine_client(engine: Any) -> str:
    actual = type(engine.llm_engine.engine_core).__module__ + '.' + type(engine.llm_engine.engine_core).__name__
    if actual != 'vllm.v1.engine.core_client.InprocClient':
        raise RuntimeError(f'Actual client {actual} differs from the frozen in-process evaluator')
    return actual


