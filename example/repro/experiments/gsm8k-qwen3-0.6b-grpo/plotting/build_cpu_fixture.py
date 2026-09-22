"""Build an explicitly derivative CPU fixture from the public historical export.

Only summary metadata is re-signed after sanitization. Prediction bytes and scores
are unchanged. This is NOT original signed evidence or a new experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source, output):
    if output.exists():
        raise FileExistsError('Use a fresh CPU fixture directory')
    loaded = []
    for arm in ['audit-before-001', 'audit-after-001']:
        predictions = (source / arm / 'predictions.jsonl').read_bytes()
        summary_raw = (source / arm / 'evaluation_summary.json').read_bytes()
        summary = json.loads(summary_raw)
        rows = [json.loads(line) for line in predictions.splitlines()]
        if len(rows) != 1319 or len({r['id'] for r in rows}) != 1319:
            raise ValueError('Require complete historical audit fixture')
        if digest(predictions) != summary['predictions_sha256']:
            raise ValueError('Public prediction bytes differ from historical summary')
        loaded.append((arm, predictions, summary_raw, summary))
    output.mkdir(parents=True)
    provenance = []
    for arm, predictions, summary_raw, summary in loaded:
        folder = output / arm
        folder.mkdir()
        (folder / 'predictions.jsonl').write_bytes(predictions)
        original_signature = summary.pop('manifest_sha256')
        summary['fixture_only'] = True
        summary['fixture_origin'] = {'public_summary_sha256': digest(summary_raw),
                                    'historical_manifest_sha256': original_signature,
                                    'transformation': 'Re-sign already-sanitized metadata for CPU integration only; prediction bytes untouched'}
        summary['manifest_sha256'] = digest(json.dumps(summary, sort_keys=True, ensure_ascii=False,
                                            separators=(',', ':'), allow_nan=False).encode())
        raw = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        (folder / 'evaluation_summary.json').write_bytes(raw)
        provenance.append({'arm': arm, 'predictions_sha256': digest(predictions),
                           'source_summary_sha256': digest(summary_raw),
                           'derived_summary_sha256': digest(raw)})
    receipt = {'fixture_only': True, 'new_experiment': False, 'original_signature_verified': False,
               'prediction_bytes_modified': False, 'files': provenance}
    (output / 'fixture-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Public exported historical run directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
