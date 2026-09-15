#!/usr/bin/env python3
"""Rebuild recorded prompt messages and Qwen3 text; never load model weights."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def messages(template, question, author_source=None):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('A nonempty original question is required')
    question = question.strip()
    if template == 'gsm8k-grpo':
        spec = read(ROOT / 'gsm8k-grpo/template.json')
        return [{'role': 'system', 'content': spec['messages'][0]['content']},
                {'role': 'user', 'content': question}]
    if template not in ('gsm8k-author-cot', 'gsm8k-author-cod') or author_source is None:
        raise ValueError('Author mode requires its fixed external eight-shot source directory')
    import yaml
    mode = template.rsplit('-', 1)[1]
    spec = read(ROOT / 'gsm8k-author-8shot/template.json')
    source = Path(author_source) / f'configs/gsm8k_{mode}.yaml'
    raw = source.read_bytes()
    lock = spec['fewshot_sources'][mode]
    if len(raw) != lock['bytes'] or hashlib.sha256(raw).hexdigest() != lock['sha256']:
        raise ValueError('Author input differs from its pinned source')
    config = yaml.safe_load(raw)
    instruction = spec['styles'][mode] + '\n' + spec['answer_instruction'] + '\n'
    if (config['system_prompt'] != instruction or config['format'] != spec['qa_format']
            or len(config['fewshot']) != spec['fewshot_count']):
        raise ValueError('Template differs from the actual historical configuration')
    examples = [spec['qa_format'].format(**example) for example in config['fewshot']]
    payload = instruction + '\n' + '\n'.join(examples) + '\n' + spec['qa_format'].format(question=question, answer='')
    return [{'role': 'user', 'content': payload}]


@lru_cache(maxsize=2)
def compiled_template(source):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=['jinja2.ext.loopcontrols'])
    def reject(message): raise ValueError(message)
    env.globals['raise_exception'] = reject
    return env.from_string(source.decode())


def native_text(chat_messages):
    source = (ROOT / 'qwen3/chat_template.jinja').read_bytes()
    if hashlib.sha256(source).hexdigest() != read(ROOT / 'qwen3/source.json')['chat_template_sha256']:
        raise ValueError('Qwen3 chat template differs from the measured template')
    return compiled_template(source).render(messages=chat_messages, tools=None,
        add_generation_prompt=True, enable_thinking=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', required=True, choices=['gsm8k-grpo', 'gsm8k-author-cot', 'gsm8k-author-cod'])
    p.add_argument('--question-file', required=True)
    p.add_argument('--author-source-dir')
    p.add_argument('--output', required=True, help='New JSON file; expanded author inputs stay outside the repository')
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Output must be new')
    if args.template.startswith('gsm8k-author') and output.is_relative_to(ROOT.parent):
        raise ValueError('Keep expanded external eight-shot inputs outside the repository')
    chat = messages(args.template, Path(args.question_file).read_text(), args.author_source_dir)
    result = {'template': args.template, 'messages': chat, 'chat_kwargs': {'add_generation_prompt': True, 'enable_thinking': True},
              'rendered_prompt': native_text(chat), 'model_loaded': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'PASS', 'message_count': len(chat), 'model_loaded': False}))


if __name__ == '__main__': main()
