"""One frozen, exploratory OhioT1DM empirical application.

This module adapts already-frozen five-minute arrays to the existing Bridge-v0
interfaces.  It neither changes the observed statistic nor the Bridge/M4 core.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum.support import screen_candidates
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.types import SyntheticDataset
from src.rs_cusum_phase2rb.engine import PreparedAnalysis, run_influence_arm
from src.rs_cusum_phase2r.core import finite_p, fit_candidate
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_realdata_bridge_v0.runner import M4, M4HASH, draw_failure, fit_null, rule, seed, sha, simulate

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs'/'ohio_final_empirical_application.yaml'
BRIDGE=ROOT/'src'/'rs_cusum_realdata_bridge_v0'/'runner.py'
OUT=ROOT/'results_rs_cusum'/'ohio_final_empirical_application'

LAYERS=('D0','D1','D2','D3','D4')

class SupportDrawError(ValueError):
 """A fixed observed candidate lost Bridge support in one generated draw."""
 def __init__(self, missing, details):
  self.missing=tuple(missing); self.details=details
  super().__init__('INVALID_SUPPORT_DRAW:'+','.join(map(str,self.missing)))

def atomic(path: Path, obj: Any):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def load_cfg():
 c=yaml.safe_load(CFG.read_text(encoding='utf-8'))
 assert c['protocol']['locked_m4_source_sha256']==M4HASH==sha(M4)
 return c

def load_data(c: dict[str,Any]):
 ids=tuple(c['data']['patient_ids']); lmin=int(c['trajectory']['landmark_hours']*60); rows=[]; parts=[]
 for pid in ids:
  p=ROOT/'results_mdp_gate'/'arrays'/f'training_{pid}_5min.npz'; z=np.load(p,allow_pickle=True)
  taus=[np.asarray(x).astype('datetime64[ns]') for x in z['tau']]; starts=[x[0] for x in taus if len(x)]
  recording_start=min(starts); target=recording_start+np.timedelta64(lmin,'m'); found=None
  for sid,tau in enumerate(taus):
   hit=np.flatnonzero((tau>=target)&(tau<target+np.timedelta64(5,'m')))
   if len(hit) and int(hit[0])<len(tau)-1:
    j=int(hit[0]); found=(sid,j,tau[j:]); break
  if found is None: raise RuntimeError(f'NO_OUTCOME_BLIND_ENTRY:{pid}')
  sid,j,tau=found; st=np.asarray(z['states'][sid][j:],float); ac=np.asarray(z['actions'][sid][j:],int); rw=np.asarray(z['rewards_binary'][sid][j:],float)
  if len(st)!=len(ac)+1 or len(ac)!=len(rw): raise RuntimeError(f'BELL_MAN_SHAPE:{pid}')
  delay=float((tau[0]-target)/np.timedelta64(1,'m')); parts.append((pid,st,ac,rw,tau))
  rows.append({'patient_id':pid,'source_array':str(p.relative_to(ROOT)).replace('\\','/'),'segment_id':int(z['segment_ids'][sid]),'entry_timestamp':str(tau[0]),'entry_delay_min':delay,'T_i':len(ac),'action_positive':int(ac.sum()),'reward_mean':float(rw.mean()),'end_timestamp':str(tau[-1])})
 if len(parts)!=int(c['trajectory']['required_patients']): raise RuntimeError('REQUIRED_PATIENT_COUNT')
 h=max(len(x[2]) for x in parts); n=len(parts); p=parts[0][1].shape[1]; states=np.zeros((n,h+1,p));actions=np.zeros((n,h),int);rewards=np.zeros((n,h));mask=np.zeros((n,h),bool)
 for i,(_,st,ac,rw,_) in enumerate(parts):
  t=len(ac); states[i,:t+1]=st; actions[i,:t]=ac; rewards[i,:t]=rw; mask[i,:t]=True
 # SyntheticDataset's is_null flag is a container-validation invariant only;
 # this observed empirical panel has no known true change point.
 data=SyntheticDataset('OhioT1DM_training_empirical','OhioT1DM_5min_monotone',True,'observed',0.0,int(c['seeds']['master_seed']),states,actions,rewards,mask,ids,np.zeros(n),0,h,None); data.validate()
 return data,np.asarray([len(x[2]) for x in parts],int),pd.DataFrame(rows)

def candidate_grid(c,data):
 step=int(c['candidate']['elapsed_grid_hours']*12); end=data.analysis_end-step
 return tuple(range(step,end+1,step))

def support_only(data,c):
 bridge=load_bridge_cfg(); panel=build_panel(data,'strict_first_gap'); cand=candidate_grid(c,data)
 sc=screen_candidates(panel.support_view(),cand,data.analysis_start,data.analysis_end,rule(bridge)); sup={x.candidate:x for x in sc.candidates if x.admissible}
 table=[]
 for x in sc.candidates:
  d={'candidate_transition':x.candidate,'candidate_elapsed_hours':x.candidate/12,'admissible':x.admissible,'reasons':'|'.join(x.failure_reasons),'required_count_each_side_action':x.required_count_each_side_action,'patients_both_sides':len(x.patients_both_sides)}
  for side,label in ((x.left,'left'),(x.right,'right')):
   d.update({f'{label}_transitions':side.transition_count,f'{label}_A0':side.action_counts[0],f'{label}_A1':side.action_counts[1],f'{label}_active_patients':len(side.active_patients),f'{label}_A1_unique_patients':len(side.unique_action_patients[1]),f'{label}_max_A1_share':side.maximum_action_patient_share[1],f'{label}_max_total_share':side.maximum_total_patient_share,f'{label}_action1_ess':side.action_count_ess[1]})
  table.append(d)
 return panel,sup,pd.DataFrame(table)

def load_bridge_cfg():
 return yaml.safe_load((ROOT/'configs'/'m4_realdata_bridge_v0_validation.yaml').read_text(encoding='utf-8'))

def prepare_fixed(data,c,candidates,evals=None):
 bridge=load_bridge_cfg(); panel=build_panel(data,'strict_first_gap'); sc=screen_candidates(panel.support_view(),tuple(candidates),data.analysis_start,data.analysis_end,rule(bridge)); sup={x.candidate:x for x in sc.candidates if x.admissible}
 missing=sorted(set(candidates)-set(sup))
 if missing:
  details=[]
  for x in sc.candidates:
   if x.candidate not in missing: continue
   for side,label in ((x.left,'left'),(x.right,'right')):
    details.append({'candidate_transition':int(x.candidate),'candidate_elapsed_hours':float(x.candidate/12),'side':label,'active_patient_count':len(side.active_patients),'A0_count':int(side.action_counts[0]),'A1_count':int(side.action_counts[1]),'A0_unique_patient_count':len(side.unique_action_patients[0]),'A1_unique_patient_count':len(side.unique_action_patients[1]),'max_A1_patient_share':float(side.maximum_action_patient_share[1]),'max_total_patient_share':float(side.maximum_total_patient_share),'action1_ess':float(side.action_count_ess[1]),'failure_reasons':list(x.failure_reasons)})
  raise SupportDrawError(missing,details)
 f=c['fitting']; fits={u:fit_candidate(panel,u,data.analysis_start,data.analysis_end,float(f['gamma_5min']),float(f['ridge_alpha']),int(f['fqi_max_iterations']),float(f['fqi_tolerance']),'patient_balanced','patient') for u in candidates}
 if evals is None:
  rng=np.random.default_rng(seed(c['seeds']['master_seed'],'ohio','evaluation-grid')); records=panel.records; idx=rng.choice(len(records),size=min(int(f['evaluation_states']),len(records)),replace=False); es=np.vstack([records[i].state for i in idx]); ea=np.asarray([records[i].action for i in idx])
 else: es,ea=evals
 ids=tuple(sorted(panel.patient_ids)); bounds={u:boundary_factor(sup[u],data.analysis_start,data.analysis_end,len(ids),'harmonic_active_patients') for u in candidates}; mid=min(candidates,key=lambda u:abs(u-data.analysis_end/2))
 return PreparedAnalysis(data,panel,tuple(candidates),sup,fits,action_feature(es,ea),mid,ids,bounds),(es,ea)

def layer_values(raw,norm,joint,mid_index): return np.array([abs(raw[mid_index,0]),abs(norm[mid_index,0]),np.max(abs(norm[mid_index])),np.max(abs(joint[:,0])),np.max(abs(joint))])

def bridge_draws(model, prep, evals, candidates, task, c, draws):
 values=np.full((draws,5),np.nan); failures=[]
 for b in range(draws):
  draw_seed=seed(task['bootstrap_seed'],'draw',b)
  try:
   boot=simulate(model,draw_seed,c);bp,_=prepare_fixed(boot,task['ohio_cfg'],candidates,evals)
   raw,norm,joint=observed_arrays(bp);values[b]=layer_values(raw,norm,joint,sorted(bp.fits).index(bp.midpoint))
  except Exception as exc:
   record=draw_failure(task,b,draw_seed,exc)
   if isinstance(exc,SupportDrawError):
    record['missing_candidates']=list(exc.missing);record['support_diagnostics']=exc.details
   failures.append(record)
 return values,failures

def write_jsonl(path, rows):
 with path.open('w',encoding='utf-8') as f:
  for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')

def candidate_process(raw,norm,joint,candidates):
 rows=[]
 for i,u in enumerate(candidates):
  for z in range(joint.shape[1]): rows.append({'candidate_id':int(u),'candidate_transition':int(u),'elapsed_time_hours':float(u/12),'candidate_position':i,'evaluation_index':z,'D0':float(raw[i,z]),'D1':float(norm[i,z]),'D2':float(abs(norm[i,z])),'D3':float(abs(joint[i,0])),'D4':float(joint[i,z])})
 return pd.DataFrame(rows)

def persist_pre_bridge(observed, tau_hat, raw, norm, joint, candidates, original_draws, original_critical, original_pvalues, alpha, support, patients):
 """Persist all valid observed/Original quantities before any Bridge draw."""
 observed_frame=pd.DataFrame({'Statistic':LAYERS,'Observed value':observed,'Estimated change point transition':tau_hat,'Estimated change point elapsed hours':tau_hat/12})
 observed_frame.to_csv(OUT/'ohio_final_observed_results.csv',index=False)
 pd.DataFrame({'Statistic':LAYERS,'Observed value':observed,'Original critical value':original_critical,'Original p-value':original_pvalues,'Original reject':original_pvalues<=alpha,'Estimated change point transition':tau_hat,'Estimated change point elapsed hours':tau_hat/12,'Original bootstrap draws':len(original_draws)}).to_csv(OUT/'original_ohio_result.csv',index=False)
 candidate_process(raw,norm,joint,candidates).to_csv(OUT/'candidate_process.csv',index=False)
 support.to_csv(OUT/'ohio_support_diagnostics.csv',index=False)
 patients.to_csv(OUT/'patient_trajectory_summary.csv',index=False)

def bridge_status(attempted, valid, failures):
 categories={str(k):int(v) for k,v in pd.Series([x['failure_category'] for x in failures]).value_counts().items()} if failures else {}
 return {'attempted_draws':int(attempted),'valid_draws':int(valid),'invalid_draws':int(attempted-valid),'invalid_fraction':float((attempted-valid)/attempted) if attempted else 0.0,'calibration_available':bool(valid),'failure_reason':None if valid else 'CALIBRATION_UNAVAILABLE_NO_VALID_DRAWS','failure_categories':categories}

def write_summary(observed, tau_hat, candidates, original_critical, original_pvalues, status):
 lines=['# OhioT1DM final empirical application','', 'Exploratory empirical application only. Bridge-v0 B/D null-validation failures remain limitations; results are not confirmatory inference.','', '## Observed analysis','',f'- completed: yes',f'- candidate count: {len(candidates)}',f'- estimated tau: transition {tau_hat} ({tau_hat/12:g} h)',f'- observed statistics: '+', '.join(f'{k}={v:.8g}' for k,v in zip(LAYERS,observed)),'','## Original calibration','',f'- completed: yes',f'- critical values: '+', '.join(f'{k}={v:.8g}' for k,v in zip(LAYERS,original_critical)),f'- p-values: '+', '.join(f'{k}={v:.8g}' for k,v in zip(LAYERS,original_pvalues)),'','## Bridge/M4 calibration','',f"- attempted draws: {status['attempted_draws']}",f"- valid draws: {status['valid_draws']}",f"- invalid draws: {status['invalid_draws']}"]
 if not status['calibration_available']: lines.extend(['','Bridge/M4 calibration unavailable because no valid generative-null draw preserved the frozen candidate support.'])
 (OUT/'FINAL_OHIO_EMPIRICAL_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def freeze():
 c=load_cfg(); data,T,patients=load_data(c); _,sup,support=support_only(data,c)
 if not sup: raise RuntimeError('NO_ADMISSIBLE_CANDIDATE')
 OUT.mkdir(parents=True,exist_ok=True); patients.to_csv(OUT/'patient_trajectory_summary.csv',index=False);support.to_csv(OUT/'support_diagnostics.csv',index=False)
 atomic(OUT/'protocol_freeze.json',{'config_sha256':sha(CFG),'m4_hash':sha(M4),'bridge_hash':sha(BRIDGE),'candidate_grid':list(candidate_grid(c,data)),'U_adm':sorted(sup),'patient_ids':list(data.patient_ids),'T_i':T.tolist(),'no_outcome_used_for_candidate_selection':True,'original_bootstrap_seed':seed(c['seeds']['master_seed'],'ohio','original'),'bridge_bootstrap_seed':seed(c['seeds']['master_seed'],'ohio','bridge')})
 files=[CFG,BRIDGE,M4,ROOT/'configs'/'mdp_5min_protocol.yaml',Path(__file__)]+[ROOT/'results_mdp_gate'/'arrays'/f'training_{pid}_5min.npz' for pid in c['data']['patient_ids']]
 atomic(OUT/'hash_registry_pre_run.json',{'files':{str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in files},'m4_core_unchanged':True,'m4_hash':sha(M4)})

def formal():
 c=load_cfg(); frozen=json.loads((OUT/'protocol_freeze.json').read_text()); hashes=json.loads((OUT/'hash_registry_pre_run.json').read_text())
 for rel,want in hashes['files'].items():
  if sha(ROOT/rel)!=want: raise RuntimeError('SOURCE_OR_DATA_HASH_DRIFT:'+rel)
 started=time.perf_counter(); data,T,patients=load_data(c); candidates=tuple(frozen['U_adm']); _,_,support=support_only(data,c); prep,ev=prepare_fixed(data,c,candidates)
 raw,norm,joint=observed_arrays(prep); cs=sorted(prep.fits); mi=cs.index(prep.midpoint); observed=layer_values(raw,norm,joint,mi)
 tau_hat=cs[int(np.argmax([np.max(abs(joint[i])) for i in range(len(cs))]))]
 B=int(c['calibration']['bootstrap_draws']); alpha=float(c['calibration']['alpha']); ose=int(frozen['original_bootstrap_seed']); bse=int(frozen['bridge_bootstrap_seed'])
 original=run_influence_arm(prep,'M0',B,ose,'gaussian'); o_draw=np.column_stack([np.asarray(original.bootstrap[k],float) for k in LAYERS]); ocrit=np.quantile(o_draw,.95,axis=0); op=np.asarray([finite_p(observed[i],o_draw[:,i]) for i in range(5)])
 persist_pre_bridge(observed,tau_hat,raw,norm,joint,cs,o_draw,ocrit,op,alpha,support,patients)
 bridge_cfg=load_bridge_cfg(); model=fit_null(data,T,bridge_cfg); task={'scenario':'OhioT1DM_training_empirical','outer_id':0,'dataset_seed':int(c['seeds']['master_seed']),'bootstrap_seed':bse,'ohio_cfg':c}; bdraw,failures=bridge_draws(model,prep,ev,candidates,task,bridge_cfg,B)
 good=np.all(np.isfinite(bdraw),axis=1); valid=bdraw[good]
 status=bridge_status(B,len(valid),failures); write_jsonl(OUT/'draw_failures.jsonl',failures); atomic(OUT/'bridge_calibration_status.json',status)
 if len(valid): bcrit=np.quantile(valid,.95,axis=0); bp=np.asarray([finite_p(observed[i],valid[:,i]) for i in range(5)]); breject=bp<=alpha
 else: bcrit=np.full(5,np.nan);bp=np.full(5,np.nan);breject=np.full(5,False)
 result=pd.DataFrame({'Statistic':LAYERS,'Observed value':observed,'Original critical value':ocrit,'Original p-value':op,'Original reject':op<=alpha,'Bridge critical value':bcrit,'Bridge p-value':bp,'Bridge reject':breject,'Bridge calibration status':'available' if status['calibration_available'] else 'CALIBRATION_UNAVAILABLE_NO_VALID_DRAWS','estimated_tau_transition':tau_hat,'estimated_tau_elapsed_hours':tau_hat/12})
 result.to_csv(OUT/'ohio_final_results.csv',index=False)
 pd.DataFrame({'Statistic':LAYERS,'original_valid_draws':B,'bridge_valid_draws':int(good.sum()),'bridge_invalid_draws':int((~good).sum()),'bridge_invalid_fraction':float((~good).mean()),'original_critical':ocrit,'bridge_critical':bcrit}).to_csv(OUT/'bootstrap_diagnostics.csv',index=False)
 # Figures are descriptive renderings of the fixed observed process/calibration.
 fig,ax=plt.subplots(figsize=(9,4)); ax.plot(np.asarray(cs)/12,np.max(abs(joint),axis=1),marker='o');ax.axvline(tau_hat/12,color='crimson',ls='--',label=f'tau={tau_hat/12:g} h');ax.set(xlabel='Elapsed follow-up (hours from landmark)',ylabel='Observed D4 max over evaluation states',title='Ohio observed patient-balanced RS-CUSUM process');ax.legend();fig.tight_layout();fig.savefig(OUT/'figure1_observed_cusum_process.png',dpi=180);plt.close(fig)
 if status['calibration_available']:
  fig,ax=plt.subplots(figsize=(8,4));x=np.arange(5);ax.bar(x-.18,ocrit,.36,label='Original');ax.bar(x+.18,bcrit,.36,label='Bridge');ax.set_xticks(x,LAYERS);ax.set_ylabel('0.95 critical value');ax.set_title('Ohio calibration comparison');ax.legend();fig.tight_layout();fig.savefig(OUT/'figure2_calibration_comparison.png',dpi=180);plt.close(fig)
 atomic(OUT/'run_metadata.json',{'completed':True,'completion_status':'COMPLETED_WITH_BRIDGE_CALIBRATION_UNAVAILABLE' if not status['calibration_available'] else 'COMPLETED','classification':c['protocol']['classification'],'m4_core_unchanged':sha(M4)==M4HASH,'m4_hash':sha(M4),'bridge_hash':sha(BRIDGE),'config_sha256':sha(CFG),'patients':len(data.patient_ids),'T_i':T.tolist(),'candidate_count':len(cs),'U_adm':cs,'tau_hat_transition':tau_hat,'tau_hat_elapsed_hours':tau_hat/12,'B':B,'alpha':alpha,'runtime_seconds':time.perf_counter()-started,'bridge_valid_draws':int(good.sum()),'bridge_invalid_draws':int((~good).sum()),'bridge_failure_categories':status['failure_categories']})
 write_summary(observed,tau_hat,cs,ocrit,op,status)
 files=[x for x in OUT.rglob('*') if x.is_file() and x.name not in {'hash_registry.json','SUCCESS','FAILURE','COMPLETED_WITH_BRIDGE_CALIBRATION_UNAVAILABLE'} and x.suffix!='.tmp'];atomic(OUT/'hash_registry.json',{'files':{str(x.relative_to(OUT)).replace('\\','/'):sha(x) for x in files},'m4_core_unchanged':sha(M4)==M4HASH})
 marker='COMPLETED_WITH_BRIDGE_CALIBRATION_UNAVAILABLE' if not status['calibration_available'] else 'SUCCESS';(OUT/marker).write_text(marker+'\n')

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','formal']);z=a.parse_args()
 try:
  if z.stage=='freeze': freeze()
  else: formal()
 except Exception:
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'FAILURE').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__': main()
