"""Project private, hash-verified author outputs into non-reconstructable metrics.

No source/model text or token ID sequences are published. The result is not an
original prediction file and cannot replace private inputs to the full verifier.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path


FIELDS = ('id', 'project', 'question_hash', 'source_split', 'mode', 'canonical_record_sha256',
          'parser_version', 'thinking_opening_prefilled', 'thinking_closed', 'author_correct',
          'strict_format_ok', 'strict_correct', 'response_tokens', 'thinking_response_tokens',
          'final_response_tokens', 'generation_prompt_sha256', 'generation_prompt_token_ids_sha256',
          'author_payload_sha256', 'prompt_tokens', 'total_tokens', 'eos', 'truncated', 'finish_reason',
          'batch_index', 'batch_seed', 'elapsed', 'batch_elapsed',
          'evaluation_config_sha256', 'parser_source_sha256')
BOOLEANS = ('thinking_opening_prefilled', 'thinking_closed', 'author_correct', 'strict_format_ok',
            'strict_correct', 'eos', 'truncated')
COUNTS = ('author_correct', 'strict_correct', 'strict_format_ok', 'thinking_closed', 'eos', 'truncated')
REMOVED = {'author_extracted', 'author_method', 'canonical_answer', 'final_text', 'lora_request_id',
           'raw_output_text', 'reference_string', 'response', 'response_token_ids',
           'segment_token_count_method', 'stop_reason', 'strict_parsed'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build(raw_path, summary_path, receipt_path, output_path, manifest_path):
    paths = [Path(output_path), Path(manifest_path)]
    if any(path.exists() for path in paths):
        raise FileExistsError('Use fresh numeric projection paths')
    meta = json.loads(Path(summary_path).read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    original = {key: receipt['original'][key] for key in ('name', 'bytes', 'physical_rows', 'sha256')}
    compressed = {key: receipt['gzip'][key] for key in ('name', 'bytes', 'sha256')}
    for item, name in ((original, 'author_predictions.jsonl'), (compressed, 'author_predictions.jsonl.gz')):
        if (item['name'] != name or type(item['bytes']) is not int or item['bytes'] <= 0
                or not isinstance(item['sha256'],str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Invalid fixed archive receipt identity')
    if (type(original['physical_rows']) is not int or original['physical_rows'] != 1319
            or meta['evaluation_config']['parser_version'] != 'author_compat_qwen3_final_v2'
            or meta['evaluation_config']['mode'] not in ('cot','cod')
            or meta['evaluation_config_sha256'] != digest(meta['evaluation_config'])):
        raise ValueError('Unexpected frozen author projection contract')
    raw = Path(raw_path).read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    if (meta['status'] != 'complete' or meta['manifest_sha256'] != digest({k:v for k,v in meta.items() if k != 'manifest_sha256'})
            or original_sha != meta['predictions_sha256'] or original_sha != receipt['original']['sha256']
            or len(raw) != receipt['original']['bytes']):
        raise ValueError('Private raw evidence differs from complete summary/original receipt')
    rows, removed = [], set()
    for line in io.BytesIO(raw):
        source = json.loads(line)
        if set(source) - set(FIELDS) - REMOVED:
            raise ValueError('Unknown raw fields cannot enter a public projection manifest')
        row = {key: source[key] for key in FIELDS}
        if (not row['id'].startswith('gsm8k:test:') or not row['id'].removeprefix('gsm8k:test:').isdigit()
                or row['project'] != 'gsm8k' or row['source_split'] != 'test' or row['mode'] not in ('cot','cod')
                or row['finish_reason'] not in ('stop','length') or row['parser_version'] != 'author_compat_qwen3_final_v2'
                or row['mode'] != meta['evaluation_config']['mode']
                or row['evaluation_config_sha256'] != meta['evaluation_config_sha256']
                or row['parser_source_sha256'] != meta['parser_source_sha256']):
            raise ValueError('Unexpected text identifier in public metric fields')
        for key in FIELDS:
            value = row[key]
            if key in BOOLEANS and type(value) is not bool:
                raise ValueError('Nonboolean metric')
            if key.endswith('sha256') or key == 'question_hash':
                if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Invalid non-reconstructable hash field')
            if key.endswith('_tokens') or key in ('batch_index','batch_seed'):
                if type(value) is not int or value < 0:
                    raise ValueError('Invalid token/count metric')
            if key in ('elapsed','batch_elapsed') and (type(value) not in (int,float) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid time metric')
        if row['response_tokens'] != len(source['response_token_ids']) or row['total_tokens'] != row['prompt_tokens'] + row['response_tokens']:
            raise ValueError('Token accounting changed in the projection')
        row['original_record_line_sha256'] = hashlib.sha256(line).hexdigest()
        rows.append(row)
        removed.update(set(source) - set(FIELDS))
    if len(rows) != 1319 or len({row['id'] for row in rows}) != 1319 or len(rows) != receipt['original']['physical_rows']:
        raise ValueError('Projection must preserve all 1319 unique records')
    counts = {key:sum(row[key] for row in rows) for key in COUNTS}
    if any(counts[key] != meta['counts'][key] for key in COUNTS):
        raise ValueError('Projection scores/diagnostics differ from original summary')
    encoded = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows).encode()
    manifest = {'kind':'numeric_only_author_projection_not_original_predictions', 'mode':rows[0]['mode'],
        'rows':len(rows), 'original_private_jsonl':original, 'original_private_gzip':compressed,
        'original_summary_manifest_sha256':meta['manifest_sha256'], 'preserved_fields':list(FIELDS),
        'projection_builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'projection_field_list_sha256':digest(FIELDS),
        'additional_field':'original_record_line_sha256 binds the exact LF-inclusive private raw record',
        'removed_fields':sorted(removed), 'counts':counts,
        'token_totals':{key:sum(row[key] for row in rows) for key in ('prompt_tokens','response_tokens','total_tokens')},
        'projection':{'name':paths[0].name,'bytes':len(encoded),'sha256':hashlib.sha256(encoded).hexdigest()},
        'generated_text_published':False, 'token_id_sequences_published':False,
        'limitation':'This projection supports public numeric inspection only. Full tokenizer/parser verification requires the unchanged private raw archive or authorized server original.'}
    manifest['manifest_sha256'] = digest(manifest)
    paths[0].write_bytes(encoded)
    paths[1].write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','summary','receipt','output','manifest'):
        parser.add_argument('--'+key, required=True, type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.raw,args.summary,args.receipt,args.output,args.manifest),ensure_ascii=False))


if __name__ == '__main__':
    main()
