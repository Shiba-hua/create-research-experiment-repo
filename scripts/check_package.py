"""Read-only packaging checks. Does not approve research or run experiments."""
import ast,gzip,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
errors=[];count=0
for p in ROOT.rglob('*'):
 if not p.is_file() or any(x in p.parts for x in ('.git','__pycache__','.venv')):continue
 count+=1;raw=p.read_bytes()
 if p.suffix=='.gz':raw=gzip.decompress(raw)
 try:text=raw.decode('utf-8')
 except UnicodeDecodeError:continue
 if p.suffix=='.py':
  try:ast.parse(text)
  except SyntaxError:errors.append({'file':str(p.relative_to(ROOT)),'error':'syntax'})
 if p.suffix=='.json':
  try:json.loads(text)
  except ValueError:errors.append({'file':str(p.relative_to(ROOT)),'error':'json'})
 for pattern in [r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'sk-(?:proj-)?[A-Za-z0-9_-]{40,}']:
  if re.search(pattern,text):errors.append({'file':str(p.relative_to(ROOT)),'error':'secret pattern'})
 if p.suffix=='.md':
  for url in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
   url=url.split('#')[0]
   if url and not re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',url) and not (p.parent/url).exists():errors.append({'file':str(p.relative_to(ROOT)),'error':'broken link','target':url})
 if p.is_relative_to(ROOT/'example') and p.name!='EXPORT-MANIFEST.json':
  if re.search(r'\b(?:ARC-Challenge|SciQ|MATH-500|MBPP|HumanEval|APPS)\b',text):errors.append({'file':str(p.relative_to(ROOT)),'error':'out-of-scope dataset'})
  if '/Users/' in text or '/root/' in text:errors.append({'file':str(p.relative_to(ROOT)),'error':'machine path'})
sys.path.insert(0,str(ROOT/'example/repro'))
from capsule import check
integrity=check()
errors.extend({'error':'export hash','file':p} for p in integrity['errors'])
result={'status':'FAIL' if errors else 'PASS','files_checked':count,'errors':errors,'scope':'Packaging only; not scientific acceptance'}
print(json.dumps(result,ensure_ascii=False,indent=2));sys.exit(bool(errors))
