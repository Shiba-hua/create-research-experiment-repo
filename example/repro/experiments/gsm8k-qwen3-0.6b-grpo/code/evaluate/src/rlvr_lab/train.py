"""Pinned stock TRL GRPO loss; instrumentation and periodic held-out dev evaluation."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from .evaluate import align_special_tokens, evaluate_rows
from .rewards import verify_completion


def grpo_backend_kwargs(config):
    """Select a pinned stock backend without importing vLLM or initializing CUDA.

    Sources: https://raw.githubusercontent.com/huggingface/trl/v0.24.0/trl/trainer/grpo_config.py
    and https://raw.githubusercontent.com/huggingface/trl/v0.24.0/trl/trainer/grpo_trainer.py.
    TRL owns generation, LoRA weight synchronization, sleep/wake, and TIS.
    Its colocate path derives max_model_len from the prompt/completion limits;
    vllm_max_model_len is not a GRPOConfig field in this pinned version.
    """
    enabled = config.get('use_vllm', False)
    if not isinstance(enabled, bool):
        raise ValueError('use_vllm must be a JSON boolean')
    if not enabled:
        return {'use_vllm': False}
    if config.get('vllm_mode', 'colocate') != 'colocate':
        raise ValueError('This single-GPU package supports vllm_mode="colocate" only; server mode is unsupported')
    tp = config.get('vllm_tensor_parallel_size', 1)
    if type(tp) is not int or tp != 1:
        raise ValueError('Single-GPU vLLM requires vllm_tensor_parallel_size=1')
    if config.get('vllm_enable_sleep_mode', True) is not True:
        raise ValueError('Colocated vLLM requires vllm_enable_sleep_mode=true for Torch dev/backward memory')
    if config.get('vllm_max_model_len') is not None:
        raise ValueError('TRL 0.24.0 does not support vllm_max_model_len; stock TRL uses max_prompt_length + max_new_tokens')
    memory = config.get('vllm_gpu_memory_utilization', 0.3)
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not 0 < memory < 1:
        raise ValueError('vllm_gpu_memory_utilization must be a finite number in (0, 1)')
    correction = config.get('vllm_importance_sampling_correction', True)
    cap = config.get('vllm_importance_sampling_cap', 2.0)
    if not isinstance(correction, bool):
        raise ValueError('vllm_importance_sampling_correction must be a JSON boolean')
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('vllm_importance_sampling_cap must be a finite positive number')
    for distribution, required in (('trl', '0.24.0'), ('vllm', '0.10.2')):
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f'Optional colocate backend requires {distribution}=={required}; no automatic installation') from error
        if actual != required:
            raise RuntimeError(f'Optional colocate backend requires {distribution}=={required}, found {actual}; select the pinned runtime explicitly')
    for variable in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'):
        if 'expandable_segments:true' in os.environ.get(variable, '').replace(' ', '').lower():
            raise ValueError(f'vLLM 0.10.2 sleep mode is incompatible with {variable}=expandable_segments:True; unset it in this candidate process')
    kwargs = {'use_vllm': True, 'vllm_mode': 'colocate', 'vllm_tensor_parallel_size': 1,
              'vllm_gpu_memory_utilization': float(memory), 'vllm_enable_sleep_mode': True,
              'vllm_importance_sampling_correction': correction, 'vllm_importance_sampling_cap': float(cap)}
    missing = set(kwargs) - set(GRPOConfig.__dataclass_fields__)
    if missing:
        raise RuntimeError(f'Installed GRPOConfig lacks pinned backend fields: {sorted(missing)}')
    return kwargs


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def append_json(path, data):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'at', encoding='utf-8') as handle:
        handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(path):
    # JSONL separates records on physical newlines; str.splitlines also splits
    # legal U+2028/U+0085 characters inside JSON strings from source questions.
    with Path(path).open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty data or duplicate IDs')
    return rows


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_hash(model):
    digest = hashlib.sha256()
    count = 0
    norm_squared = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            value = param.detach().float().cpu().contiguous()
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
            count += value.numel()
            norm_squared += value.square().sum().item()
    return {'sha256': digest.hexdigest(), 'trainable_parameters': count, 'norm': norm_squared ** 0.5}


class JsonTelemetry(TrainerCallback):
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()
        self.nonzero_grad_steps = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = dict(logs or {})
        for name, value in record.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise FloatingPointError(f'Nonfinite training metric {name}={value}')
        if record.get('grad_norm', 0) > 0:
            self.nonzero_grad_steps += 1
        record.update(step=state.global_step, wall_seconds=time.monotonic() - self.started,
                      utc=datetime.now(timezone.utc).isoformat(),
                      gpu_allocated_gb=torch.cuda.memory_allocated() / 2 ** 30,
                      gpu_reserved_gb=torch.cuda.memory_reserved() / 2 ** 30,
                      gpu_peak_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        append_json(self.output / 'metrics.jsonl', record)


class GradientCoverage(TrainerCallback):
    """Measure dense LoRA gradient coverage after clipping, before optimizer.step."""
    def __init__(self, output):
        self.output = output
        self.trainer = None
        self.effective_advantage_steps = 0
        self.missing_gradient_steps = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        detail, layers = [], {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            record = {'name': name, 'numel': parameter.numel(), 'grad_is_none': gradient is None}
            if gradient is not None:
                detached = gradient.detach()
                if not torch.isfinite(detached).all():
                    raise FloatingPointError(f'Nonfinite gradient: {name}')
                record.update(norm=detached.float().norm().item(), nonzero_elements=torch.count_nonzero(detached).item())
            else:
                record.update(norm=None, nonzero_elements=0)
            detail.append(record)
            layer = name.split('.layers.', 1)[1].split('.', 1)[0] if '.layers.' in name else 'outside_decoder'
            bucket = layers.setdefault(layer, {'matrices': 0, 'with_grad': 0, 'with_nonzero_grad': 0, 'norm_squared': 0.0})
            bucket['matrices'] += 1
            bucket['with_grad'] += gradient is not None
            bucket['with_nonzero_grad'] += record['nonzero_elements'] > 0
            bucket['norm_squared'] += (record['norm'] or 0.0) ** 2
        if not detail:
            raise RuntimeError('No trainable parameters for gradient coverage')
        for bucket in layers.values():
            bucket['norm'] = bucket.pop('norm_squared') ** 0.5
        row = {'step': state.global_step + 1, 'stage': 'after_clipping_before_optimizer',
               'trainable_matrices': len(detail), 'matrices_with_grad': sum(not r['grad_is_none'] for r in detail),
               'matrices_with_nonzero_grad': sum(r['nonzero_elements'] > 0 for r in detail), 'layers': layers,
               'note': 'Zero LoRA_A gradients on the first update can be normal because LoRA_B starts at zero.'}
        append_json(self.output / 'gradients.jsonl', row)
        if self.trainer is not None and getattr(self.trainer, '_last_nonzero_advantage', False):
            self.effective_advantage_steps += 1
            self.missing_gradient_steps += row['matrices_with_grad'] != row['trainable_matrices']
        if state.global_step < 2 or state.global_step + 1 == state.max_steps:
            append_json(self.output / 'gradient_details.jsonl.gz', {'step': row['step'], 'parameters': detail})
        if self.trainer is not None:
            self.trainer._metrics['train']['gradient/matrix_coverage'].append(row['matrices_with_grad']/len(detail))
            self.trainer._metrics['train']['gradient/nonzero_matrix_fraction'].append(row['matrices_with_nonzero_grad']/len(detail))


class DevEvaluation(TrainerCallback):
    def __init__(self, rows, tokenizer, output, config, interval):
        self.rows, self.tokenizer, self.output = rows, tokenizer, output
        self.config, self.interval = config, interval
        self.best = -1.0
        self.best_step = None

    def evaluate(self, model, step):
        config = {**self.config, 'artifact_identity': {'trainable_parameters': adapter_hash(model), 'step': step}}
        summary = evaluate_rows(model, self.tokenizer, self.rows,
                                self.output / 'dev' / f'step-{step:05d}', config)
        accuracy = summary['accuracy']
        append_json(self.output / 'validation.jsonl', {'step': step, 'val_acc': accuracy,
                    'summary': str(Path('dev') / f'step-{step:05d}' / 'evaluation_summary.json')})
        # Strict tie handling: retain earliest maximum, avoids length-based hidden selection.
        if accuracy > self.best:
            self.best, self.best_step = accuracy, step
            model.save_pretrained(self.output / 'best_adapter')
            self.tokenizer.save_pretrained(self.output / 'best_adapter')
            write_json(self.output / 'best_checkpoint.json', {'step': step, 'val_acc': accuracy,
                       'selection': 'earliest maximum on fixed dev; no audit access'})
        print(json.dumps({'event': 'dev_evaluation', 'step': step, 'val_acc': accuracy,
                          'best_step': self.best_step, 'best_val_acc': self.best}), flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.interval == 0 or state.global_step == state.max_steps:
            self.evaluate(model, state.global_step)


class AuditedGRPOTrainer(GRPOTrainer):
    def _generate_single_turn(self, prompts, images=None):
        try:
            return super()._generate_single_turn(prompts, images)
        finally:
            # TRL 0.24 unwrap restores checkpointing with default kwargs. Keep
            # the declared mode stable before reference and actor forwards.
            if self.args.gradient_checkpointing:
                model = self.accelerator.unwrap_model(self.model_wrapped)
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=dict(self.args.gradient_checkpointing_kwargs or {}))

    def _generate_and_score_completions(self, inputs):
        start = time.monotonic()
        output = super()._generate_and_score_completions(inputs)
        elapsed = time.monotonic() - start
        mode = 'train' if self.model.training else 'eval'
        advantages = output['advantages'].detach()
        self._last_nonzero_advantage = bool((advantages != 0).any().item())
        if not torch.isfinite(advantages).all():
            raise FloatingPointError('Nonfinite advantages')
        self._metrics[mode]['advantage/nonzero_fraction'].append((advantages != 0).float().mean().item())
        self._metrics[mode]['advantage/absolute_mean'].append(advantages.abs().mean().item())
        self._metrics[mode]['rollout_and_reference_seconds'].append(elapsed)
        self._metrics[mode]['rollout_and_reference_tokens_per_second'].append(
            output['completion_mask'].sum().item() / max(elapsed, 1e-6))
        checkpointing = {}
        for name, module in self.model.named_modules():
            if getattr(module, 'gradient_checkpointing', False):
                function = getattr(module, '_gradient_checkpointing_func', None)
                if function is not None:
                    checkpointing[name] = dict(getattr(function, 'keywords', {}))
        if not getattr(self, '_checkpointing_recorded', False):
            append_json(Path(self.args.output_dir).parent / 'effective_checkpointing.jsonl',
                        {'step': self.state.global_step, 'enabled': self.args.gradient_checkpointing,
                         'declared_kwargs': self.args.gradient_checkpointing_kwargs, 'modules': checkpointing})
            self._checkpointing_recorded = True
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite GRPO loss')
        return loss


def run(config_path, output_path, resume=None):
    if resume:
        raise NotImplementedError('Resume requires a separate audited recovery path; start a new recorded run instead')
    config = json.loads(Path(config_path).read_text())
    backend_kwargs = grpo_backend_kwargs(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() and not resume:
        raise FileExistsError('Refusing to overwrite an existing experiment')
    assert torch.cuda.device_count() == 1, 'This experiment contract is single GPU'
    torch.set_num_threads(config.get('cpu_threads', 8))
    set_seed(config['seed'])
    model_path = config['model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='sdpa')
    special_token_alignment = align_special_tokens(model, tokenizer)
    train_rows, dev_rows = read_rows(config['train_data']), read_rows(config['dev_data'])
    if {r['question_hash'] for r in train_rows} & {r['question_hash'] for r in dev_rows}:
        raise ValueError('Train/dev overlap')
    dev_rows = dev_rows[:config.get('dev_limit', 128)]
    thinking = config.get('thinking', False)
    dataset_rows = copy.deepcopy(train_rows)
    for row in dataset_rows:
        row['prompt'] = tokenizer.apply_chat_template(row['prompt'], tokenize=False,
                         add_generation_prompt=True, enable_thinking=thinking)
    prompt_lengths = [len(tokenizer.encode(row['prompt'], add_special_tokens=False)) for row in dataset_rows]
    max_prompt_length = config.get('max_prompt_length', max(prompt_lengths))
    if max(prompt_lengths) > max_prompt_length:
        raise ValueError('Configured prompt length would silently truncate training questions')
    g, b = config['G'], config['B']
    micro = config['microbatch']
    if g < 2 or b * g % micro:
        raise ValueError('G >= 2 and B*G divisible by microbatch required')
    args = GRPOConfig(output_dir=str(output / 'checkpoints'),
        per_device_train_batch_size=micro, gradient_accumulation_steps=b * g // micro,
        generation_batch_size=b * g, num_generations=g,
        max_steps=config['max_steps'], learning_rate=config['learning_rate'],
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant_with_warmup'),
        warmup_steps=config.get('warmup_steps', 10),
        optim='adamw_torch', adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        weight_decay=0.0, max_grad_norm=config.get('max_grad_norm', 1.0),
        bf16=True, tf32=True, gradient_checkpointing=config.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_completion_length=config['max_new_tokens'], max_prompt_length=max_prompt_length,
        temperature=config.get('temperature', 0.6), top_p=config.get('top_p', 0.95),
        top_k=config.get('top_k', 20), repetition_penalty=1.0,
        beta=config.get('beta', 0.01), epsilon=config.get('epsilon', 0.2),
        num_iterations=1, loss_type=config.get('loss_type', 'grpo'), scale_rewards='group',
        mask_truncated_completions=False, use_liger_loss=False,
        logging_steps=1, save_steps=config.get('save_steps', 25), save_total_limit=1,
        save_strategy='steps', eval_strategy='no', report_to='none',
        seed=config['seed'], data_seed=config['seed'], remove_unused_columns=False,
        dataloader_num_workers=0, log_completions=False, disable_tqdm=True, **backend_kwargs)
    adapter_config = LoraConfig(r=config.get('lora_r', 32), lora_alpha=config.get('lora_alpha', 64),
                      target_modules='all-linear', lora_dropout=0.0, bias='none', task_type='CAUSAL_LM')
    telemetry = JsonTelemetry(output)
    gradient_coverage = GradientCoverage(output)
    eval_config = {'max_new_tokens': config['max_new_tokens'], 'batch_size': config.get('eval_batch_size', 16),
                   'seed': config.get('eval_seed', 1729), 'temperature': config.get('temperature', 0.6),
                   'top_p': config.get('top_p', 0.95), 'top_k': config.get('top_k', 20),
                   'thinking': 'true' if thinking else 'false', 'mode': 'cot',
                   'greedy': False, 'model': config.get('model_id', model_path)}
    eval_config['special_token_alignment'] = special_token_alignment
    dev_callback = DevEvaluation(dev_rows, tokenizer, output, eval_config, config.get('eval_steps', 25))
    trainer_holder = {}

    def verified_reward(completions, answer, project, id, question_hash, completion_ids=None, trainer_state=None, **kwargs):
        if len(completions) != b * g:
            raise ValueError(f'Incomplete generation batch: {len(completions)} != {b*g}')
        rewards, verified, texts = [], [], []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else completion[0]['content']
            checked = '<think>\n' + text if thinking and not text.lstrip().startswith('<think>') else text
            result = verify_completion(checked, answer[i], project[i])
            rewards.append(float(result['correctness']))
            verified.append(result)
            texts.append(text)
        group_correct, group_answers = [], []
        for start in range(0, len(rewards), g):
            if len(set(id[start:start+g])) != 1:
                raise ValueError('Group contains more than one prompt ID')
            group_correct.append(sum(rewards[start:start+g]))
            group_answers.append(len({v['parsed'] for v in verified[start:start+g]}))
        stats = {'step': trainer_state.global_step if trainer_state else 0,
                 'G': g, 'B': b, 'reward': sum(rewards) / len(rewards),
                 'zero_variance_rate': sum(k in (0, g) for k in group_correct) / b,
                 'all_wrong_rate': group_correct.count(0) / b, 'all_correct_rate': group_correct.count(g) / b,
                 'format_pass_rate': sum(v['format_ok'] for v in verified) / len(verified),
                 'unique_answers_per_group': sum(group_answers) / b,
                 'success_count_histogram': dict(Counter(group_correct))}
        append_json(output / 'groups.jsonl', stats)
        if 'trainer' in trainer_holder:
            trainer = trainer_holder['trainer']
            for key in ('zero_variance_rate', 'all_wrong_rate', 'all_correct_rate', 'format_pass_rate', 'unique_answers_per_group'):
                trainer._metrics['train'][key].append(stats[key])
        for i, text in enumerate(texts):
            tokens = completion_ids[i] if completion_ids is not None else tokenizer.encode(text, add_special_tokens=False)
            append_json(output / 'rollouts.jsonl.gz', {'step': stats['step'], 'id': id[i],
                        'question_hash': question_hash[i], 'group_index': i // g, 'sample_index': i % g,
                        'response': text, 'answer': answer[i], 'response_tokens': len(tokens),
                        'eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id), **verified[i]})
        return rewards

    trainer = AuditedGRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(dataset_rows),
              reward_funcs=verified_reward, processing_class=tokenizer, peft_config=adapter_config,
              callbacks=[telemetry, gradient_coverage, dev_callback])
    gradient_coverage.trainer = trainer
    trainer_holder['trainer'] = trainer
    initial = adapter_hash(trainer.model)
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise RuntimeError('Commit source changes before training')
    manifest = {'status': 'starting', 'utc': datetime.now(timezone.utc).isoformat(),
                'config': config, 'trl_args': args.to_dict(), 'lora': adapter_config.to_dict(),
                'git_commit': git_commit, 'train_sha256': file_hash(config['train_data']),
                'dev_sha256': file_hash(config['dev_data']), 'verifier_sha256': file_hash(Path(__file__).with_name('rewards.py')),
                'initial_adapter': initial, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                'special_token_alignment': special_token_alignment,
                'model_snapshot': json.loads((Path(model_path) / 'snapshot_manifest.json').read_text()),
                'train_prompt_token_lengths': {'min': min(prompt_lengths), 'max': max(prompt_lengths),
                                               'mean': sum(prompt_lengths)/len(prompt_lengths)},
                'gradient_accumulation_steps': b*g//micro,
                'algorithm': 'TRL GRPO; explicitly chosen loss normalization in config'}
    if args.use_vllm:
        manifest['rollout_backend'] = {
            'trl_version': metadata.version('trl'), 'vllm_version': metadata.version('vllm'),
            'rollouts_weight_sync_and_importance_correction': 'stock TRL GRPOTrainer 0.24.0',
            'max_model_len': max_prompt_length + config['max_new_tokens'],
            'max_model_len_source': 'stock TRL derives max_prompt_length + max_completion_length',
            'sleep_policy': 'stock TRL level-1 sleep after initialization and each completed generation; wake before weight sync/generation',
            'release_policy': 'dedicated process must exit; externally verify process termination and GPU memory release before the next GPU job',
            'release_verified': False,
            'release_note': 'vLLM 0.10.2 LLM has no public shutdown method; sleep is not process shutdown. No private cleanup API is called.',
            'source': 'https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/entrypoints/llm.py'}
    # peft target_modules is a set; normalize before JSON serialization.
    manifest['lora'] = json.loads(json.dumps(manifest['lora'], default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    try:
        if not resume:
            dev_callback.evaluate(trainer.model, 0)
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output / 'final_adapter'))
        tokenizer.save_pretrained(output / 'final_adapter')
        final = adapter_hash(trainer.model)
        if initial['sha256'] == final['sha256'] or telemetry.nonzero_grad_steps == 0:
            raise RuntimeError('No authentic parameter update or nonzero gradient evidence')
        if gradient_coverage.effective_advantage_steps == 0 or gradient_coverage.missing_gradient_steps:
            raise RuntimeError('Dense LoRA gradient coverage is incomplete on effective-advantage steps')
        manifest.update(status='training_completed_not_yet_accepted', final_adapter=final,
                        optimizer_steps=trainer.state.global_step, nonzero_grad_steps=telemetry.nonzero_grad_steps,
                        effective_advantage_steps=gradient_coverage.effective_advantage_steps,
                        missing_gradient_steps=gradient_coverage.missing_gradient_steps,
                        elapsed_seconds=time.monotonic()-started, training_metrics=result.metrics,
                        best_dev_step=dev_callback.best_step, best_dev_accuracy=dev_callback.best)
        write_json(output / 'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', exception_type=type(error).__name__, exception=str(error),
                        optimizer_steps=trainer.state.global_step, elapsed_seconds=time.monotonic()-started)
        write_json(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    run(args.config, args.output, args.resume)


if __name__ == '__main__':
    main()
