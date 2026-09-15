#!/usr/bin/env python3
"""Recompute paired statistics from saved records; optionally regrade GSM8K text."""
import argparse,ast,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from capsule import ROOT,REPO,SPECS,check,dump,read,sha

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def rows(path):return [json.loads(x) for x in path.read_bytes().split(b'\n') if x]
def same(value,expected,label):
 if isinstance(value,(int,float)) and not isinstance(value,bool):
  if not np.isclose(value,expected,rtol=0,atol=1e-12):raise ValueError(label+' differs')
 elif value!=expected:raise ValueError(label+' differs')
def run(which,data=None):
 checked=check()
 if checked['status']!='PASS':raise ValueError('Capsule validation failed: '+str(checked['errors']))
 stats=module(ROOT/'analysis/statistics.py','review_paired_statistics')
 if which=='grpo':
  runroot=REPO/'results/gsm8k-grpo-formal-001'
  a=rows(runroot/'audit-before-001/predictions.jsonl')
  b=rows(runroot/'audit-after-001/predictions.jsonl')
  result=stats.compare_predictions(a,b,require_manifests=False)
  target=read(runroot/'acceptance.json')['statistics']
  for key in ['n','before','after','absolute_gain','paired_flips','paired_bootstrap_95_ci','mcnemar_exact_two_sided_p','adjusted_p']:
   same(result[key],target[key],key)
  regraded=False
  if data is not None:
   path=Path(data)
   if sha(path.read_bytes())!='437f2042d9d210e34834d1f596a0b6289f89414672ede7fe47727041dc11da62':
    raise ValueError('Canonical GSM8K data identity mismatch')
   source=rows(path);gold={r['id']:r for r in source}
   if len(gold)!=1319 or [r['id'] for r in source]!=[r['id'] for r in a]:raise ValueError('Canonical coverage/order mismatch')
   rewards=module(ROOT/'experiments/gsm8k-qwen3-0.6b-grpo/code/evaluate/src/rlvr_lab/rewards.py','historical_rewards')
   for r in a+b:
    score=rewards.verify_completion(r['response'],gold[r['id']]['answer'],'gsm8k')
    if any(score[k]!=r[k] for k in ('correctness','format_ok','parsed','reason')):raise ValueError('Fresh text regrade differs for '+r['id'])
   regraded=True
  return {'status':'PASS','experiment':'grpo','scope':'Saved paired statistics'+(' and original-verifier text regrading' if regraded else ' only; supply --data for text regrading'),
   'text_regraded':regraded,'new_model_run':False,'n':len(a),'before_correct':result['before']['correct'],'after_correct':result['after']['correct'],
   'delta':result['absolute_gain'],'paired_95_ci':result['paired_bootstrap_95_ci'],'mcnemar_p':result['mcnemar_exact_two_sided_p'],
   'adjusted_p':result['adjusted_p'],'token_totals':{'before':sum(r['response_tokens'] for r in a),'after':sum(r['response_tokens'] for r in b)},'numpy':np.__version__}
 if data is not None:raise ValueError('Numeric author projection cannot be text-regraded; use the full private historical pipeline')
 comparison=read(REPO/'results/gsm8k-author-comparison-001.json');modes={}
 for mode,runid in [('cot','gsm8k-author-cot-audit-002'),('cod','gsm8k-author-cod-audit-001')]:
  d=REPO/'results'/runid;p=d/'author_metrics.jsonl';meta=read(d/'author_metrics_manifest.json')
  if stats._digest({k:v for k,v in meta.items() if k!='manifest_sha256'})!=meta['manifest_sha256']:
   raise ValueError('Projection manifest signature mismatch')
  if sha(p.read_bytes())!=meta['projection']['sha256'] or len(p.read_bytes())!=meta['projection']['bytes']:
   raise ValueError('Projection identity mismatch')
  rr=rows(p)
  if len(rr)!=1319 or len({r['id'] for r in rr})!=1319 or any(r['mode']!=mode for r in rr):raise ValueError('Projection coverage/mode mismatch')
  if any(type(r[k]) is not bool for r in rr for k in ('strict_correct','author_correct','eos','truncated')):raise ValueError('Invalid score type')
  order=[[r[k] for k in ('id','question_hash','project','source_split')] for r in rr]
  if stats._digest(order)!=comparison['evidence']['row_order_sha256']:raise ValueError('Projection order mismatch')
  modes[mode]=rr
 tree=ast.parse((ROOT/'analysis/cod_compare.py').read_text())
 function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_paired_bootstrap')
 namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'historical_cod_bootstrap','exec'),namespace)
 scores={};costs={}
 for key in ('strict_correct','author_correct'):
  a=np.array([r[key] for r in modes['cot']],dtype=float);b=np.array([r[key] for r in modes['cod']],dtype=float)
  x=np.array([r['response_tokens'] for r in modes['cot']]);y=np.array([r['response_tokens'] for r in modes['cod']])
  ci=namespace['_paired_bootstrap'](a,b,x,y,20000,20260910)['accuracy_delta']
  wins=int(((a==0)&(b==1)).sum());losses=int(((a==1)&(b==0)).sum());target=comparison['scoring'][key]
  same(int(a.sum()),target['cot']['correct'],key+' cot');same(int(b.sum()),target['cod']['correct'],key+' cod')
  same(ci,target['paired_bootstrap_95_ci'],key+' CI');same(wins,target['wrong_to_right'],'wins');same(losses,target['right_to_wrong'],'losses')
  p=stats.exact_mcnemar_pvalue(losses,wins);same(p,target['mcnemar_exact_two_sided_p_descriptive'],key+' p')
  scores[key]={'cot':int(a.sum()),'cod':int(b.sum()),'delta':float((b-a).mean()),'paired_95_ci':ci,'mcnemar_p_descriptive':p}
 for mode,rr in modes.items():
  costs[mode]={key:sum(r[key] for r in rr) for key in ('prompt_tokens','response_tokens','total_tokens')}
  for key,total in costs[mode].items():same(total,comparison['tokens'][key][mode]['total'],mode+' '+key)
 return {'status':'PASS','experiment':'author','scope':'Public numeric projection statistics only; no full-text regrading',
  'text_regraded':False,'new_model_run':False,'n':1319,'scores':scores,'token_totals':costs,'numpy':np.__version__}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',choices=('grpo','author'));p.add_argument('--data');p.add_argument('--output')
 args=p.parse_args()
 if args.output and Path(args.output).exists():raise FileExistsError('Use a new output file')
 result=run(args.experiment,args.data)
 if args.output:dump(Path(args.output),result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
