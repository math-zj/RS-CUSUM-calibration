from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np, yaml
from sklearn.linear_model import LogisticRegression, Ridge
from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum.support import screen_candidates
from src.rs_cusum.types import SupportRule
from src.rs_cusum_phase2.grids import candidate_grid, evaluation_grid
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.types import SyntheticDataset
from src.rs_cusum_phase2rb.engine import PreparedAnalysis
from src.rs_cusum_phase2r.core import fit_candidate, finite_p
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
ROOT=Path(__file__).resolve().parents[2]; CFG=ROOT/'configs'/'m4_realdata_bridge_v0_validation.yaml'; OUT=ROOT/'results_rs_cusum'/'m4_realdata_bridge_v0'; CHECK=OUT/'checkpoints'; M4=ROOT/'src'/'rs_cusum_phase2rd'/'engine.py'; M4HASH='EC7E7DA7D8625736CEAC2B872CD739DEBA29792CD535B8AD533D5266251AF977'
LAYERS=('D0','D1','D2','D3','D4')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def seed(*x): return int.from_bytes(hashlib.sha256(':'.join(map(str,x)).encode()).digest()[:8],'little')
def cfg():
 c=yaml.safe_load(CFG.read_text()); assert sha(M4)==M4HASH==c['protocol']['locked_m4_source_sha256']; return c
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,sort_keys=True)+'\n');os.replace(t,p)
def lengths(kind,h):
 return np.full(12,h,int) if kind=='all_240' else np.array([144]*4+[192]*4+[240]*4,int)
def make_data(scen:str, sd:int, c:dict):
 v=c['validation']; q=v['scenarios'][scen]; d=v['dgp']; n,p,h=d['patients'],d['state_dimension'],q['horizon']; rng=np.random.default_rng(sd); T=lengths(q['termination'],h); st=np.zeros((n,h+1,p));ac=np.zeros((n,h),int);rw=np.zeros((n,h)); ph=np.r_[-np.ones(6),np.ones(6)]; st[:,0]=rng.normal(0,.7,(n,p));st[:,0,0]+=.5*ph;st[:,0,1]=ph; pe=rng.normal(0,q['patient_effect_sd'],n); inter=np.log(q['action_target']/(1-q['action_target']))
 for t in range(h):
  pr=1/(1+np.exp(-np.clip(inter+pe+d['propensity_state']*st[:,t,0]+.2*st[:,t,1],-25,25)));ac[:,t]=rng.binomial(1,pr)
  st[:,t+1,0]=d['state_ar']*st[:,t,0]+.12*st[:,t,1]+d['state_action']*ac[:,t]+rng.normal(0,d['state_noise_sd'],n);st[:,t+1,1]=ph;st[:,t+1,2:]=.55*st[:,t,2:]+.04*st[:,t,[0]]+rng.normal(0,d['state_noise_sd'],(n,p-2))
  rp=1/(1+np.exp(-np.clip(d['reward_intercept']+d['reward_state']*st[:,t,0]+.1*st[:,t,1]+d['reward_action']*ac[:,t],-25,25)));rw[:,t]=rng.binomial(1,rp)
 mask=np.arange(h)[None,:]<T[:,None]
 # Values outside a patient's fixed termination time are deliberately not
 # observations.  Zeroing action/reward makes that invariant explicit even
 # before the mask is applied by downstream panel construction.
 ac[~mask]=0; rw[~mask]=0
 data=SyntheticDataset('N0_complete_balanced_null',f'bridge::{scen}',True,'none',0.,sd,st,ac,rw,mask,tuple(f'P{i:02d}' for i in range(n)),ph,d['analysis_start'],h,None);data.validate();return data,T
def rule(c):
 r=c['validation']['class_ab_support'];return SupportRule('bridge_class_ab',r['minimum_action_count_each_side'],0.,r['minimum_active_patients_each_side'],r['minimum_both_side_patients'],r['minimum_unique_patients_each_side_action'],r['maximum_patient_share'],r['maximum_patient_share'])
def prepare(data,c,evals=None,fixed_candidates=None):
 panel=build_panel(data,'strict_first_gap'); cand=candidate_grid(data.analysis_start,data.analysis_end,c['validation']['candidates']);sc=screen_candidates(panel.support_view(),cand,data.analysis_start,data.analysis_end,rule(c)); sup={x.candidate:x for x in sc.candidates if x.admissible}
 if fixed_candidates is not None:
  required=tuple(fixed_candidates)
  missing=sorted(set(required)-set(sup))
  if missing: raise ValueError('INVALID_SUPPORT_DRAW:'+','.join(map(str,missing)))
  # A bootstrap draw may make additional grid points eligible, but the
  # observed outer dataset alone fixes U_adm.  They must not enter a draw.
  sup={u:sup[u] for u in required}
 if not sup: raise ValueError('NO_ADMISSIBLE_CANDIDATE')
 f=c['validation']['fitting'];fits={u:fit_candidate(panel,u,data.analysis_start,data.analysis_end,float(.9825931938526898),f['ridge_alpha'],f['fqi_max_iterations'],f['fqi_tolerance'],'patient_balanced','patient') for u in sup}
 if evals is None:
  es,ea,_=evaluation_grid(panel,data.analysis_start,data.analysis_end,c['validation']['evaluation_states'],seed(data.seed,'grid'))
 else: es,ea=evals
 ids=tuple(sorted(panel.patient_ids));bounds={u:boundary_factor(sup[u],data.analysis_start,data.analysis_end,len(ids),'harmonic_active_patients') for u in sup};mid=min(sup,key=lambda x:abs(x-(data.analysis_start+(data.analysis_end-data.analysis_start)//2)))
 return PreparedAnalysis(data,panel,cand,sup,fits,action_feature(es,ea),mid,ids,bounds),(es,ea)
@dataclass
class Model: mean:np.ndarray; sd:np.ndarray; ab:Any; trans:list; tsd:np.ndarray; rb:Any; re_sd:float; T:np.ndarray
def fit_null(data,T,c):
 m=data.observed_mask;st=data.states;ac=data.actions;rw=data.rewards;n,h=ac.shape;p=st.shape[2];ix=np.where(m);X=st[:,:-1][m];NX=st[:,1:][m];A=ac[m];Y=rw[m];
 ab=LogisticRegression(C=c['validation']['fitting']['logistic_C'],max_iter=1000,solver='lbfgs').fit(X,A); base=ab.predict_proba(X)[:,1]; pid=ix[0]; rate=np.array([np.mean(A[pid==i]) if np.any(pid==i) else .5 for i in range(n)]); pred=np.array([np.mean(base[pid==i]) if np.any(pid==i) else .5 for i in range(n)]); re=np.log(np.clip(rate,.02,.98)/(1-np.clip(rate,.02,.98)))-np.log(np.clip(pred,.02,.98)/(1-np.clip(pred,.02,.98))); re_sd=max(float(np.std(re,ddof=1)),.05)
 Z=np.c_[X,A];trans=[];res=[]
 for j in range(p):
  rr=Ridge(alpha=c['validation']['fitting']['ridge_alpha']).fit(Z,NX[:,j]);trans.append(rr);res.append(max(float(np.std(NX[:,j]-rr.predict(Z),ddof=1)),.05))
 rb=LogisticRegression(C=c['validation']['fitting']['logistic_C'],max_iter=1000,solver='lbfgs').fit(Z,Y.astype(int));return Model(np.mean(st[:,0],0),np.maximum(np.std(st[:,0],0,ddof=1),.05),ab,trans,np.array(res),rb,re_sd,T.copy())
def simulate(model,sd,c):
 rng=np.random.default_rng(sd);n=len(model.T);h=int(max(model.T));p=len(model.mean);st=np.zeros((n,h+1,p));ac=np.zeros((n,h),int);rw=np.zeros((n,h));st[:,0]=rng.normal(model.mean,model.sd,(n,p));re=rng.normal(0,model.re_sd,n)
 for t in range(h):
  live=t<model.T; lp=model.ab.decision_function(st[:,t])+re;prob=1/(1+np.exp(-np.clip(lp,-25,25)));ac[:,t]=rng.binomial(1,prob)*live;Z=np.c_[st[:,t],ac[:,t]];st[:,t+1]=np.column_stack([x.predict(Z) for x in model.trans])+rng.normal(0,model.tsd,(n,p));rp=1/(1+np.exp(-np.clip(model.rb.decision_function(Z),-25,25)));rw[:,t]=rng.binomial(1,rp)*live
 mask=np.arange(h)[None,:]<model.T[:,None];d=c['validation']['dgp'];data=SyntheticDataset('N0_complete_balanced_null','bridge_bootstrap',True,'none',0.,sd,st,ac,rw,mask,tuple(f'P{i:02d}' for i in range(n)),np.zeros(n),d['analysis_start'],h,None);data.validate();return data
def layers(raw,norm,joint,mid): return np.array([abs(raw[mid,0]),abs(norm[mid,0]),np.max(abs(norm[mid])),np.max(abs(joint[:,0])),np.max(abs(joint))])
def failure_category(exc):
 """Classify a recorded draw error without changing the draw or candidate set."""
 msg=f'{type(exc).__name__}: {exc}'.lower()
 if 'invalid_support_draw' in msg or 'no_admissible_candidate' in msg: return 'SUPPORT_REJECTION'
 if 'nonfinite' in msg or 'nan' in msg or 'inf' in msg: return 'NONFINITE_STATISTIC'
 if 'panel' in msg or 'mask' in msg: return 'PANEL_REBUILD_FAILURE'
 if any(x in msg for x in ('linalg','singular','converg','ridge','fqi')): return 'FQI_NUMERICAL_FAILURE'
 if any(x in msg for x in ('logistic','bernoulli','transition','generative')): return 'GENERATIVE_MODEL_FAILURE'
 return 'OTHER_EXCEPTION'

def draw_failure(task, draw_id, draw_seed, exc):
 return {'scenario':task['scenario'],'outer_id':task['outer_id'],'outer_seed':task['dataset_seed'],
         'draw_id':draw_id,'draw_seed':draw_seed,'failure_category':failure_category(exc),
         'exception_type':type(exc).__name__,'exception_message':str(exc),
         'traceback':traceback.format_exc()}

def draw_statistics(model, prep, evals, task, c, B=None, simulate_fn=simulate):
 """Run fixed-seed bridge draws; failures are data, not a runner-abort signal."""
 B=c['validation']['bootstrap_draws_per_outer'] if B is None else B
 fixed=tuple(sorted(prep.fits));draw=np.full((B,5),np.nan);failures=[];refits=0
 for b in range(B):
  ds=seed(task['bootstrap_seed'],'draw',b)
  try:
   bd=simulate_fn(model,ds,c);bp,_=prepare(bd,c,evals,fixed)
   br,bn,bj=observed_arrays(bp);draw[b]=layers(br,bn,bj,sorted(bp.fits).index(bp.midpoint));refits+=2*len(bp.fits)
  except Exception as exc:
   failures.append(draw_failure(task,b,ds,exc))
 return draw,failures,refits

def invalid_outer(task, start, exc, failures=None, valid_draws=0, refits=0):
 failures=[] if failures is None else failures
 return {'scenario':task['scenario'],'outer_id':task['outer_id'],'seed':task['dataset_seed'],
         'bootstrap_seed':task['bootstrap_seed'],'N':12,'T_min':None,'T_median':None,'T_max':None,
         'state_dimension':21,'reward_family':'bernoulli_TIR','action_rate':None,'patient_effect_sd':None,
         'U_adm':[],'candidate_count':0,'observed':[float('nan')]*5,'critical':[float('nan')]*5,
         'pvalue':[float('nan')]*5,'reject':[False]*5,'outer_valid':False,
         'bootstrap_valid_draws':valid_draws,'bootstrap_invalid_draws':len(failures),
         'support_invalid_draws':sum(x['failure_category']=='SUPPORT_REJECTION' for x in failures),
         'fit_failure_count':sum(x['failure_category']=='FQI_NUMERICAL_FAILURE' for x in failures),
         'critical_value_finite':False,'refit_count':refits,'draw_failures':failures,
         'outer_failure_category':failure_category(exc),'error':f'{type(exc).__name__}: {exc}',
         'runtime':time.perf_counter()-start}

def one(task):
 c=cfg();s=task['scenario'];start=time.perf_counter()
 try:
  data,T=make_data(s,task['dataset_seed'],c);prep,ev=prepare(data,c);raw,norm,joint=observed_arrays(prep);cs=sorted(prep.fits);mid=cs.index(prep.midpoint);obs=layers(raw,norm,joint,mid);model=fit_null(data,T,c)
  draw,failures,refits=draw_statistics(model,prep,ev,task,c);good=np.all(np.isfinite(draw),1);vals=draw[good]
  # Execution clarification: the frozen protocol permits draw/outer failure
  # rates but did not specify a replacement draw.  We never replace one: if
  # at least one finite draw remains, use that fixed-seed empirical sample;
  # otherwise record an untestable outer for the pre-registered rate gates.
  if not len(vals): return invalid_outer(task,start,RuntimeError('NO_VALID_BOOTSTRAP_DRAWS'),failures,0,refits)
  crit=np.quantile(vals,.95,axis=0);finite=bool(np.all(np.isfinite(crit)&(crit>0)))
  if not finite: return invalid_outer(task,start,RuntimeError('NONFINITE_CRITICAL_VALUE'),failures,len(vals),refits)
  p=np.array([finite_p(obs[i],vals[:,i]) for i in range(5)])
  return {'scenario':s,'outer_id':task['outer_id'],'seed':task['dataset_seed'],'bootstrap_seed':task['bootstrap_seed'],'N':12,'T_min':int(T.min()),'T_median':float(np.median(T)),'T_max':int(T.max()),'state_dimension':21,'reward_family':'bernoulli_TIR','action_rate':float(np.sum(data.actions*data.observed_mask)/np.sum(data.observed_mask)),'patient_effect_sd':float(model.re_sd),'U_adm':cs,'candidate_count':len(cs),'observed':obs.tolist(),'critical':crit.tolist(),'pvalue':p.tolist(),'reject':(p<=.05).tolist(),'outer_valid':True,'bootstrap_valid_draws':int(len(vals)),'bootstrap_invalid_draws':len(failures),'support_invalid_draws':sum(x['failure_category']=='SUPPORT_REJECTION' for x in failures),'fit_failure_count':sum(x['failure_category']=='FQI_NUMERICAL_FAILURE' for x in failures),'critical_value_finite':True,'refit_count':refits,'draw_failures':failures,'outer_failure_category':'','runtime':time.perf_counter()-start,'error':''}
 except Exception as exc:
  return invalid_outer(task,start,exc)
def path(s,i):return CHECK/s/f'{i:03d}.json'
def valid(x,t):
 identity=(x.get('scenario')==t['scenario'] and x.get('outer_id')==t['outer_id'] and
           x.get('seed')==t['dataset_seed'] and x.get('bootstrap_seed')==t['bootstrap_seed'])
 if not identity: return False
 if x.get('outer_valid'):
  return (x.get('bootstrap_valid_draws',0)>0 and x.get('critical_value_finite') is True and
          len(x.get('observed',[]))==5 and len(x.get('critical',[]))==5 and len(x.get('pvalue',[]))==5)
 return ('outer_failure_category' in x and 'draw_failures' in x and
         isinstance(x.get('bootstrap_valid_draws'),int) and isinstance(x.get('bootstrap_invalid_draws'),int))
def freeze():
 c=cfg();OUT.mkdir(parents=True,exist_ok=True)
 if CHECK.exists() and any(CHECK.rglob('*.json')):
  raise RuntimeError('REFUSE_TO_REFREEZE_AFTER_FORMAL_CHECKPOINT')
 regs={};
 for si,s in enumerate(c['validation']['scenarios']): regs[s]={'dataset':[seed(c['seeds']['master_seed'],s,'data',i) for i in range(200)],'bootstrap':[seed(c['seeds']['master_seed'],s,'boot',i) for i in range(200)]}
 seed_path=OUT/'seed_registry.json'
 if seed_path.exists():
  if json.loads(seed_path.read_text()) != regs: raise RuntimeError('FROZEN_SEED_REGISTRY_MISMATCH')
 else: atomic(seed_path,regs)
 files=[CFG,Path(__file__),ROOT/'tests'/'test_m4_realdata_bridge_v0.py',M4,ROOT/'src'/'rs_cusum_phase2r'/'core.py',ROOT/'src'/'rs_cusum_phase2'/'panels.py']
 atomic(OUT/'hash_registry_pre_formal.json',{
  'freeze':'pre-formal-after-deterministic-tests-and-smoke',
  'supersedes_pre_smoke_registry':'hash_registry_pre_run.json' if (OUT/'hash_registry_pre_run.json').exists() else None,
  'files':{str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in files},
  'm4_core_unchanged':True,'M4_hash':sha(M4),'R':200,'B':199,
  'component_seed_scheme':'sha256(master_seed:scenario:data|boot:outer_id[:draw]) first 8 bytes little-endian'})

def resume_freeze():
 """Hash the authorised execution-only runner revision before resume."""
 c=cfg();OUT.mkdir(parents=True,exist_ok=True)
 if not (OUT/'seed_registry.json').exists() or not (OUT/'hash_registry_pre_formal.json').exists(): raise RuntimeError('MISSING_ORIGINAL_FREEZE')
 expected={s:{'dataset':[seed(c['seeds']['master_seed'],s,'data',i) for i in range(200)],'bootstrap':[seed(c['seeds']['master_seed'],s,'boot',i) for i in range(200)]} for s in c['validation']['scenarios']}
 if json.loads((OUT/'seed_registry.json').read_text())!=expected: raise RuntimeError('FROZEN_SEED_REGISTRY_MISMATCH')
 files=[CFG,Path(__file__),ROOT/'tests'/'test_m4_realdata_bridge_v0.py',M4,ROOT/'src'/'rs_cusum_phase2r'/'core.py',ROOT/'src'/'rs_cusum_phase2'/'panels.py']
 atomic(OUT/'hash_registry_resume_pre_run.json',{'freeze':'execution-only-runner-failure-accounting-fix-before-resume','parent_freeze':'hash_registry_pre_formal.json','files':{str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in files},'m4_core_unchanged':True,'M4_hash':sha(M4),'R':200,'B':199,'component_seed_scheme':'sha256(master_seed:scenario:data|boot:outer_id[:draw]) first 8 bytes little-endian','execution_clarification':'A failed frozen draw is recorded and never replaced. A finite positive critical value uses remaining fixed-seed valid draws; no valid draws or a nonfinite critical value makes the outer untestable and is counted by the pre-registered engineering rates.'})
def tasks():
 r=json.loads((OUT/'seed_registry.json').read_text());return [{'scenario':s,'outer_id':i,'dataset_seed':r[s]['dataset'][i],'bootstrap_seed':r[s]['bootstrap'][i]} for s in r for i in range(200)]

def write_jsonl(path, records):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for x in records: f.write(json.dumps(x,sort_keys=True)+'\n')
 os.replace(tmp,p)

def append_jsonl(path, records):
 if not records: return
 with Path(path).open('a',encoding='utf-8') as f:
  for x in records: f.write(json.dumps(x,sort_keys=True)+'\n')

def emit_failure_logs(rows):
 draw=[];outer=[]
 for x in rows:
  draw.extend(x.get('draw_failures',[]))
  if not x.get('outer_valid'): outer.append({'scenario':x['scenario'],'outer_id':x['outer_id'],'outer_seed':x['seed'],'bootstrap_seed':x['bootstrap_seed'],'outer_failure_category':x.get('outer_failure_category',''),'error':x.get('error',''),'bootstrap_valid_draws':x.get('bootstrap_valid_draws',0),'bootstrap_invalid_draws':x.get('bootstrap_invalid_draws',0),'support_invalid_draws':x.get('support_invalid_draws',0),'fit_failure_count':x.get('fit_failure_count',0)})
 write_jsonl(OUT/'draw_failures.jsonl',draw);write_jsonl(OUT/'outer_failures.jsonl',outer)
def summarize(rows):
 import pandas as pd
 if len(rows)!=800 or len({(x['scenario'],x['outer_id']) for x in rows})!=800:
  raise RuntimeError('INCOMPLETE_OR_DUPLICATE_FORMAL_RESULTS')
 f=pd.DataFrame(rows).sort_values(['scenario','outer_id']);f.to_json(OUT/'raw_outer_results.jsonl',orient='records',lines=True);f.to_csv(OUT/'raw_outer_results.csv',index=False);emit_failure_logs(rows); flat=[]
 # Type-I is reported over testable outers; their fraction and all failure
 # rates are separately subjected to the frozen engineering gate.
 for _,x in f[f.outer_valid].iterrows():
  for j,k in enumerate(LAYERS): flat.append({'scenario':x.scenario,'layer':k,'reject':x.reject[j] if x.outer_valid else False,'outer_valid':x.outer_valid,'critical':x.critical[j] if x.outer_valid else np.nan})
 z=pd.DataFrame(flat); ty=z.groupby(['scenario','layer']).agg(size=('reject','mean'),N=('reject','size'),critical_median=('critical','median')).reset_index();ty.to_csv(OUT/'type1_summary.csv',index=False);en=f.groupby('scenario').agg(outer_failure_rate=('outer_valid',lambda x:1-x.mean()),draw_failure_rate=('bootstrap_invalid_draws',lambda x:x.sum()/(199*len(x))),support_rejection_rate=('support_invalid_draws',lambda x:x.sum()/(199*len(x))),testable_fraction=('outer_valid','mean')).reset_index();en.to_csv(OUT/'engineering_summary.csv',index=False);f[['scenario','runtime','candidate_count','action_rate','T_min','T_median','T_max','outer_valid','bootstrap_valid_draws','bootstrap_invalid_draws','support_invalid_draws','fit_failure_count']].to_csv(OUT/'scenario_diagnostics.csv',index=False);f.groupby('scenario').runtime.agg(['median','mean']).reset_index().to_csv(OUT/'runtime_summary.csv',index=False)
 critical=[];gate=[]
 for s,g in ty.groupby('scenario'):
  d4=float(g[g.layer=='D4']['size'].iloc[0]); allok=bool((g['size']<=.13).all() and d4<=.10); e=en[en.scenario==s].iloc[0]
  gate.append(allok and e.outer_failure_rate<=.01 and e.draw_failure_rate<=.01 and e.support_rejection_rate<=.01 and e.testable_fraction>=.99)
  for _,z in g.iterrows(): critical.append({'scenario':s,'layer':z['layer'],'critical_value_median':z['critical_median']})
 pd.DataFrame(critical).to_csv(OUT/'critical_value_summary.csv',index=False)
 label='BRIDGE_V0_NULL_PASS' if all(gate) else ('BRIDGE_V0_NULL_PARTIAL' if any(gate) else 'BRIDGE_V0_NULL_FAIL')
 frozen_path=OUT/('hash_registry_resume_pre_run.json' if (OUT/'hash_registry_resume_pre_run.json').exists() else 'hash_registry_pre_formal.json');frozen=json.loads(frozen_path.read_text())
 metadata={'gate':label,'completed_outer':len(f),'m4_core_unchanged':sha(M4)==M4HASH,'R':200,'B':199,'alpha':.05,'config_sha256':sha(CFG),'frozen_preformal_hashes':frozen['files'],'component_seed_scheme':frozen['component_seed_scheme'],'python':sys.version,'platform':platform.platform(),'worker_count':'recorded in formal stdout'}
 atomic(OUT/'run_metadata.json',metadata)
 files=[x for x in OUT.rglob('*') if x.is_file() and x.name not in {'hash_registry.json','SUCCESS','FAILURE'} and not x.suffix=='.tmp']
 atomic(OUT/'hash_registry.json',{'m4_core_unchanged':sha(M4)==M4HASH,'files':{str(x.relative_to(OUT)).replace('\\','/'):sha(x) for x in files}})
 (OUT/'manifest.json').write_text(json.dumps({'gate':label,'completed_outer':800,'expected_outer':800,'expected_draws':159200,'files':sorted(str(x.relative_to(OUT)).replace('\\','/') for x in OUT.rglob('*') if x.is_file() and x.suffix!='.tmp')},indent=2)+'\n')
 (OUT/'FINAL_BRIDGE_VALIDATION_SUMMARY.md').write_text(f'# Bridge v0 validation\n\nGate: `{label}`\n\nAll four pre-registered null scenarios used R=200 and B=199.  See `type1_summary.csv` and `engineering_summary.csv`.\n',encoding='utf-8');(OUT/'SUCCESS').write_text(label+'\n')
def formal(workers):
 c=cfg()
 freeze_path=OUT/('hash_registry_resume_pre_run.json' if (OUT/'hash_registry_resume_pre_run.json').exists() else 'hash_registry_pre_formal.json')
 if not freeze_path.exists(): raise RuntimeError('MISSING_PREFORMAL_FREEZE')
 frozen=json.loads(freeze_path.read_text())
 for k,v in frozen['files'].items():
  if sha(ROOT/k)!=v: raise RuntimeError('POST_FREEZE_SOURCE_OR_CONFIG_CHANGE:'+k)
 old=OUT/'FAILURE';archived=OUT/'FAILURE_RUNNER_EXCEPTION_PRE_FIX.txt'
 if old.exists() and not archived.exists(): archived.write_text(old.read_text(),encoding='utf-8');old.unlink()
 ts=tasks();done=[];todo=[]
 for t in ts:
  p=path(t['scenario'],t['outer_id'])
  if p.exists():
   x=json.loads(p.read_text());
   if not valid(x,t):raise RuntimeError('invalid checkpoint '+str(p))
   done.append(x)
  else:todo.append(t)
 print(f'Bridge v0 formal started: resumed={len(done)} missing={len(todo)} workers={workers}',flush=True)
 with ProcessPoolExecutor(max_workers=workers) as ex:
  for fu in as_completed([ex.submit(one,t) for t in todo]):
   x=fu.result()
   atomic(path(x['scenario'],x['outer_id']),x);append_jsonl(OUT/'draw_failures.jsonl',x.get('draw_failures',[]));done.append(x);print(f"Bridge v0: {len(done)}/800 [{x['scenario']}:{x['outer_id']}; valid={x['outer_valid']}; invalid_draws={x.get('bootstrap_invalid_draws',0)}]",flush=True)
 summarize(done)
def smoke():
 c=cfg();d,T=make_data('A_complete_balanced_baseline',12345671,c);p,e=prepare(d,c);m=fit_null(d,T,c);valid=0
 for k in range(3):
  bp,_=prepare(simulate(m,seed(12345672,'smoke',k),c),c,e,tuple(sorted(p.fits)))
  assert set(p.fits).issubset(bp.fits);valid+=1
 atomic(OUT/'smoke_result.json',{'status':'PASS','smoke_only':True,'candidate_count':len(p.fits),'state_dimension':21,'bridge_refit_draws':valid})
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','resume-freeze','smoke','formal']);a.add_argument('--workers',type=int,default=4);z=a.parse_args()
 try:
  if z.stage=='freeze':freeze()
  elif z.stage=='resume-freeze':resume_freeze()
  elif z.stage=='smoke':smoke()
  else:formal(z.workers)
 except Exception:
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'FAILURE').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
