"""Read-only helpers for the scoped export; historical hashes remain historical."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
SPECS=('gsm8k-qwen3-0.6b-grpo','gsm8k-qwen3-1.7b-author-cot-cod')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(path.read_text())
def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def ident(path):
 raw=path.read_bytes();return {'bytes':len(raw),'sha256':sha(raw)}
def check():
 errors=[]
 for row in read(REPO/'EXPORT-MANIFEST.json')['files']:
  path=REPO/row['path']
  if not path.is_file() or sha(path.read_bytes())!=row['export_sha256']:errors.append(row['path'])
 return {'status':'FAIL' if errors else 'PASS','errors':errors,'scope':'Export file integrity only; not scientific acceptance'}
