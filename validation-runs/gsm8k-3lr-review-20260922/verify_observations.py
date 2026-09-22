#!/usr/bin/env python3
"""Read-only independent evidence audit. Writes only an explicitly new output file.

Usage: python verify_observations.py --repo LAB --canonical AUDIT_JSONL --reference HISTORICAL_RUN --output NEW_JSON
This uses the registered verifier, independently recomputes statistics, and does
not train, alter the lab, or treat saved pass flags as sufficient evidence.
"""
import argparse,base64,collections,datetime,gzip,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 op=gzip.open if p.suffix=='.gz' else open
 with op(p,'rt') as f:return [json.loads(x) for x in f if x.strip()]
def load(p):return json.loads(p.read_text())
def module(p):
 spec=importlib.util.spec_from_file_location('registered_verifier',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def audit(repo,canonical,reference):
 expected='437f2042d9d210e34834d1f596a0b6289f89414672ede7fe47727041dc11da62'
 assert sha(canonical)==expected
 truth=rows(canonical);gold={r['id']:r for r in truth};assert len(gold)==1319
 exp='gsm8k-qwen3-0.6b-grpo-lr-sweep-20260920'
 rewards_path=repo/f'repro/experiments/{exp}/code/evaluate/src/rlvr_lab/rewards.py'
 verifier=module(rewards_path)
 result={'scope':'Independent CPU audit of existing formal run evidence; no new experiments',
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
         'parents':subprocess.check_output(['git','show','-s','--format=%P','HEAD'],cwd=repo,text=True).strip().split(),
         'canonical_sha256':expected,'verifier_sha256':sha(rewards_path),'runs':{}}
 pvalues=[];poll_counts=collections.Counter();all_intervals=[]
 for label in ['1e-4','5e-5','2e-4']:
  name=f'gsm8k-grpo-lr{label}-20260920-001';r=repo/'results'/name;e=repo/'evidence'/name
  a,b=rows(r/'audit-before-001/predictions.jsonl'),rows(r/'audit-after-001/predictions.jsonl')
  assert [x['id'] for x in a]==[x['id'] for x in b]==[x['id'] for x in truth]
  mismatches=[];summary=[]
  for arm,rr in [('before',a),('after',b)]:
   for row in rr:
    assert row['question_hash']==gold[row['id']]['question_hash']
    text=row['response']
    if not text.lstrip().startswith('<think>'):text='<think>\n'+text
    verdict=verifier.verify_completion(text,gold[row['id']]['answer'],'gsm8k')
    if any(row[k]!=verdict[k] for k in ['correctness','format_ok','parsed','reason']):mismatches.append({'arm':arm,'id':row['id']})
   material=[{k:x[k] for k in ['id','response','response_tokens','correctness']} for x in rr]
   summary.append({'arm':arm,'n':len(rr),'correct':sum(int(x['correctness']) for x in rr),
     'mean_decode_tokens':sum(x['response_tokens'] for x in rr)/len(rr),
     'predictions_sha256':sha(r/f'audit-{arm}-001/predictions.jsonl'),
     'response_content_sha256':hashlib.sha256(json.dumps(material,sort_keys=True,ensure_ascii=False).encode()).hexdigest()})
  wins=sum(x['correctness']==0 and y['correctness']==1 for x,y in zip(a,b));losses=sum(x['correctness']==1 and y['correctness']==0 for x,y in zip(a,b));n=len(a);discordant=wins+losses
  p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
  pvalues.append(p)
  draws=np.random.default_rng(20260910).multinomial(n,[losses/n,(n-discordant)/n,wins/n],size=20000)
  ci=np.quantile((draws[:,2]-draws[:,0])/n,[.025,.975]).tolist()
  vals=rows(r/'validation.jsonl');best=min(vals,key=lambda v:(-v['val_acc'],v['step']))['step']
  manifest=load(r/'training_manifest.json');assert best==manifest['best_dev_step'];assert manifest['optimizer_steps']==64
  remote_rows=[]
  for line in (e/'remote_file_hashes.txt').read_text().splitlines():
   h,rel=line.split(maxsplit=1);path=r/'training_manifest.json' if rel==f'results/{name}/manifest.json' else repo/rel
   assert sha(path)==h;remote_rows.append(rel)
  poll=rows(e/'monitoring/polling.jsonl.gz');cats=collections.Counter(x['category'] for x in poll);poll_counts.update(cats)
  bystage=collections.defaultdict(list)
  for x in poll:
   stage=Path(x['source_path']).parent.name
   if x.get('checked_utc'):bystage[stage].append(datetime.datetime.strptime(x['checked_utc'],'%Y%m%dT%H%M%SZ').timestamp())
  intervals=[]
  for times in bystage.values():
   times=sorted(set(times));intervals += [b-a for a,b in zip(times,times[1:])]
  all_intervals+=intervals
  c=load(r/'weight_cleanup_receipt.json');assert c['status']=='completed' and not any(c['post_cleanup_exists'].values())
  jobs=[]
  for stage in ['audit-before-001','train','audit-after-001']:
   j=load(e/stage/'job.json');assert j['exit_code']==0 and j['status']=='completed';jobs.append({'stage':stage,'pid':j.get('pid'),'elapsed_seconds':j.get('elapsed_seconds'),'git_commit':j['git_commit']})
  current_correct=summary[1]['correct']
  result['runs'][label]={'summaries':summary,'text_regraded_rows':len(a)+len(b),'verdict_mismatches':mismatches,
   'gain_pp':100*(wins-losses)/n,'paired_flips':{'wrong_to_right':wins,'right_to_wrong':losses},'paired_bootstrap_ci':ci,'mcnemar_exact_p':p,
   'best_dev_step':best,'optimizer_steps':manifest['optimizer_steps'],'remote_hashes_matched':len(remote_rows),
   'cleanup_receipt_verified':True,'new_local_weight_files':sum(p.suffix in ('.safetensors','.pt','.ckpt','.pth') for p in r.rglob('*') if p.is_file()),
   'polling_categories':dict(cats),'polling_intervals_seconds':{'count':len(intervals),'median':float(np.median(intervals)),'under_120':sum(v<120 for v in intervals)},
   'jobs':jobs,'reference_after_correct':970,'absolute_reference_delta_pp':100*abs(current_correct-970)/1319,
   'reference_under_0_5pp':abs(current_correct-970)*100 < .5*1319}
  assert not mismatches
 order=sorted(range(3),key=pvalues.__getitem__);adjusted=[0.]*3;bound=0.
 for rank,index in enumerate(order):bound=max(bound,min(1.,(3-rank)*pvalues[index]));adjusted[index]=bound
 for label,p in zip(['1e-4','5e-5','2e-4'],adjusted):result['runs'][label]['holm3_p']=p
 old=load(reference/'manifest.json');new=load(repo/'results/gsm8k-grpo-lr1e-4-20260920-001/training_manifest.json')
 oldroll=rows(reference/'rollouts.jsonl.gz');newroll=rows(repo/'results/gsm8k-grpo-lr1e-4-20260920-001/rollouts.jsonl.gz')
 same_prefix=0;first=None
 for index,(left,right) in enumerate(zip(oldroll,newroll)):
  keys=['step','id','sample_index','response','response_tokens','correctness']
  if all(left.get(k)==right.get(k) for k in keys):same_prefix+=1
  elif first is None:first={'row_index':index,'old_step':left['step'],'new_step':right['step'],'id_equal':left['id']==right['id'],'fields_differ':[k for k in keys if left.get(k)!=right.get(k)]};break
 result['reference_comparison']={'initial_adapter_equal':old['initial_adapter']==new['initial_adapter'],
  'old_selected_step':old.get('best_dev_step'),'new_selected_step':new['best_dev_step'],
  'train_args_differences':[k for k in set(old['trl_args'])|set(new['trl_args']) if old['trl_args'].get(k)!=new['trl_args'].get(k)],
  'matching_rollout_prefix_rows':same_prefix,'first_rollout_difference':first,'cause_of_divergence':'Not established; dataset/Arrow compatibility shims and runtime are material differences requiring separate equivalence evidence'}
 result['polling_totals']=dict(poll_counts)
 result['acceptance']={'recorded_experimental_numbers_and_text_regrade':'PASS','three_formal_runs_completed':'PASS',
  'approved_polling_schedule':'FAIL','original_reference_under_0_5pp':'FAIL','full_original_e2e_acceptance':'NOT_PASS',
  'report_contract_and_model_provenance':'REQUIRES_SEPARATE_REVIEW'}
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--canonical',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a new audit output')
 output=audit(a.repo,a.canonical,a.reference)
 a.output.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps(output['acceptance']))
