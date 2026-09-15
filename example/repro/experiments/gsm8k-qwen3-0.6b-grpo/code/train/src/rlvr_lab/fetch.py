"""Fetch public model snapshots through a recorded endpoint, then lock hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    info = HfApi(endpoint=endpoint).model_info(args.model, revision=args.revision)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.model, revision=info.sha, local_dir=output,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'README.md', 'LICENSE'],
                      max_workers=4, endpoint=endpoint)
    files = {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(output.rglob('*')) if p.is_file() and '.cache' not in p.parts and p.name != 'snapshot_manifest.json'}
    if not any(name.endswith('.safetensors') for name in files):
        raise RuntimeError('No model weights downloaded')
    manifest = {'model_id': args.model, 'revision': info.sha, 'requested_revision': args.revision,
                'source_url': f'https://huggingface.co/{args.model}/tree/{info.sha}',
                'transport_endpoint': endpoint, 'files': files}
    (output / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
