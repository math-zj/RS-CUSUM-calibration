from __future__ import annotations
import argparse,csv,json,os,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from typing import Any
import numpy as np
from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2rb.engine import run_influence_arm
from src.rs_cusum_phase2rd.engine import run_m4
from src.rs_cusum_phase2rd.protocol import load_config as m4_config
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_phase2r.core import finite_p
from .analysis import prepare_alternative_analysis
from .generator import generate
from .protocol import OUTPUT,WORKSPACE,freeze,load,seeds,verify,sha
CHECK=OUTPUT/'checkpoints'
def _atomic(path:Path,obj:dict[str,Any]):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def _grid(cfg):
 with np.load(WORKSPACE/'results_rs_cusum'/'phase2r'/'fixed_evaluation_grid.npz',allow_pickle=False) as d:return np.asarray(d['states'],float),np.asarray(d['actions'],int)
def _path(s:str,i:int)->Path:return CHECK/s/f'rep_{i:03d}.json'
def _valid(x:dict[str,Any],scenario:dict[str,Any],i:int,ds:int,osd:int,ms:int)->bool:
 keys=('original_statistic','original_critical_value','original_pvalue','m4_statistic','m4_critical_value','m4_pvalue')
 return x.get('scenario')==scenario['id'] and x.get('replication_id')==i and x.get('seed')==ds and x.get('original_bootstrap_seed')==osd and x.get('m4_bootstrap_seed')==ms and x.get('status')=='OK' and all(np.isfinite(float(x[k])) for k in keys) and x.get('bootstrap_success') is True
def _one(task:dict[str,Any])->dict[str,Any]:
 started=time.perf_counter(); cfg=load(); rd=m4_config('D2'); parent=load_phase2_config(); s=task['scenario']; states,actions=_grid(cfg)
 try:
  data=generate(s,int(task['seed']),parent,float(cfg['design']['true_change_fraction']),float(cfg['design']['transition_shift_multiplier']))
  prepared=prepare_alternative_analysis(data,rd,parent,states,actions); raw,norm,joint=observed_arrays(prepared); observed=float(np.max(np.abs(joint))); tau=int(sorted(prepared.fits)[int(np.unravel_index(np.argmax(np.abs(joint)),joint.shape)[0])])
  t=time.perf_counter(); original=run_influence_arm(prepared,'M0',int(cfg['design']['bootstrap_draws_per_method']),int(task['original_bootstrap_seed']),'gaussian'); ort=time.perf_counter()-t
  od=np.asarray(original.bootstrap['D4'],float); oc=float(np.quantile(od,.95)); op=float(finite_p(observed,od))
  t=time.perf_counter(); m4=run_m4(prepared,rd,parent,states,actions,int(cfg['design']['bootstrap_draws_per_method']),int(task['m4_bootstrap_seed'])); mrt=time.perf_counter()-t
  md=np.asarray(m4.joint,float); valid=np.asarray(m4.valid,bool); layers=np.max(np.abs(md),axis=(1,2)); layers=layers[valid & np.isfinite(layers)]
  if len(layers)!=int(cfg['design']['bootstrap_draws_per_method']): raise RuntimeError(f'M4 invalid draws {len(layers)}')
  mc=float(np.quantile(layers,.95)); mp=float(finite_p(observed,layers)); alpha=float(cfg['design']['alpha'])
  return {'scenario':s['id'],'change_type':s['change_type'],'effect_level':s['effect_level'],'effect_size':s['effect_size'],'replication_id':int(task['index']),'seed':int(task['seed']),'original_bootstrap_seed':int(task['original_bootstrap_seed']),'m4_bootstrap_seed':int(task['m4_bootstrap_seed']),'tau_true':int(data.true_change_point),'original_statistic':observed,'original_critical_value':oc,'original_pvalue':op,'original_detect':bool(op<=alpha),'original_tau_hat':tau,'original_runtime':ort,'m4_statistic':observed,'m4_critical_value':mc,'m4_pvalue':mp,'m4_detect':bool(mp<=alpha),'m4_tau_hat':tau,'m4_runtime':mrt,'bootstrap_success':True,'status':'OK','error':'','total_runtime':time.perf_counter()-started}
 except Exception as e:
  return {'scenario':s['id'],'change_type':s['change_type'],'effect_level':s['effect_level'],'effect_size':s['effect_size'],'replication_id':int(task['index']),'seed':int(task['seed']),'original_bootstrap_seed':int(task['original_bootstrap_seed']),'m4_bootstrap_seed':int(task['m4_bootstrap_seed']),'tau_true':144,'status':'FAIL','bootstrap_success':False,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'total_runtime':time.perf_counter()-started}
def _summarize(rows:list[dict[str,Any]]):
 import pandas as pd
 import matplotlib.pyplot as plt
 frame=pd.DataFrame(rows); frame.to_csv(OUTPUT/'raw_replication_results.csv',index=False); ok=frame[frame.status=='OK'].copy(); det=[]; loc=[]; run=[]
 for s,g in frame.groupby('scenario',sort=False):
  for method in ('original','m4'):
   gg=g[g.status=='OK']; n=len(g); d=int(gg[f'{method}_detect'].sum()) if len(gg) else 0; rate=d/n if n else np.nan; se=np.sqrt(rate*(1-rate)/n) if n else np.nan
   det.append({'Scenario':s,'Method':'Original RS-CUSUM' if method=='original' else 'Locked M4','N':n,'Detections':d,'Detection Rate':rate,'MCSE':se,'Failure Rate':1-len(gg)/n if n else np.nan,'Median Runtime':gg[f'{method}_runtime'].median() if len(gg) else np.nan})
   for scope,lg in [('unconditional',gg),('conditional_on_detection',gg[gg[f'{method}_detect']])]:
    ae=(lg[f'{method}_tau_hat']-lg.tau_true).abs(); loc.append({'Scenario':s,'Method':'Original RS-CUSUM' if method=='original' else 'Locked M4','Scope':scope,'Detected N':int(gg[f'{method}_detect'].sum()) if len(gg) else 0,'Valid Localization N':len(ae),'MAE':ae.mean(),'Median AE':ae.median(),'Q25':ae.quantile(.25),'Q75':ae.quantile(.75),'P90':ae.quantile(.90),'Within +/-5':(ae<=5).mean() if len(ae) else np.nan,'Within +/-10':(ae<=10).mean() if len(ae) else np.nan,'Within +/-20':(ae<=20).mean() if len(ae) else np.nan})
   run.append({'Scenario':s,'Method':method,'median_seconds':gg[f'{method}_runtime'].median() if len(gg) else np.nan,'mean_seconds':gg[f'{method}_runtime'].mean() if len(gg) else np.nan})
 pd.DataFrame(det).to_csv(OUTPUT/'detection_summary.csv',index=False); pd.DataFrame(loc).to_csv(OUTPUT/'localization_summary.csv',index=False); pd.DataFrame(run).to_csv(OUTPUT/'runtime_summary.csv',index=False)
 d=pd.DataFrame(det); fig,ax=plt.subplots(figsize=(9,4));
 for k,(m,g) in enumerate(d.groupby('Method')): ax.bar(np.arange(len(g))+(.18 if k else -.18),g['Detection Rate'],.36,label=m)
 ax.set_xticks(np.arange(len(d.Scenario.unique())),d.Scenario.unique(),rotation=20); ax.set_ylabel('Detection rate'); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT/'figure_detection_rate.png',dpi=160); plt.close(fig)
 l=pd.DataFrame(loc); l=l[l.Scope=='conditional_on_detection']; fig,ax=plt.subplots(figsize=(9,4));
 for k,(m,g) in enumerate(l.groupby('Method')): ax.plot(g.Scenario,g['Median AE'],marker='o',label=m)
 ax.set_ylabel('Median absolute localization error'); ax.tick_params(axis='x',rotation=20); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT/'figure_localization_error.png',dpi=160); plt.close(fig)
 text=['# FINAL EXPERIMENT SUMMARY','','Limited paired representative-alternative comparison; not a full power validation.','',f'Completed valid rows: {len(ok)}/{len(frame)}.','', '## Detection',pd.DataFrame(det).to_markdown(index=False),'','## Localization',pd.DataFrame(loc).to_markdown(index=False),'','M0 and M4 use the same observed process; only their calibration differs, so identical tau_hat values are expected.']
 (OUTPUT/'FINAL_EXPERIMENT_SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
def formal(workers:int):
 verify(); cfg=load(); ds,seedo,seedsm=seeds('dataset'),seeds('original_bootstrap'),seeds('m4_bootstrap'); tasks=[]; completed=[]
 for si,s in enumerate(cfg['design']['scenarios']):
  for i in range(ds.shape[1]):
   path=_path(s['id'],i)
   if path.exists():
    try: x=json.loads(path.read_text(encoding='utf-8'))
    except Exception: raise RuntimeError(f'corrupt checkpoint {path}')
    if not _valid(x,s,i,int(ds[si,i]),int(seedo[si,i]),int(seedsm[si,i])): raise RuntimeError(f'invalid checkpoint {path}')
    completed.append(x); continue
   tasks.append({'scenario':s,'index':i,'seed':int(ds[si,i]),'original_bootstrap_seed':int(seedo[si,i]),'m4_bootstrap_seed':int(seedsm[si,i])})
 print(f'Representative AltLoc formal started: resumed={len(completed)} missing={len(tasks)} workers={workers}',flush=True)
 with ProcessPoolExecutor(max_workers=workers) as ex:
  for future in as_completed([ex.submit(_one,t) for t in tasks]):
   r=future.result();
   if r['status']!='OK': raise RuntimeError('replication failure: '+r['error'])
   _atomic(_path(r['scenario'],r['replication_id']),r); completed.append(r); print(f"Representative AltLoc: {len(completed)}/400 [{r['scenario']}:{r['replication_id']}]",flush=True)
 if len(completed)!=400: raise RuntimeError('incomplete formal run')
 _summarize(completed); meta={'planned_outer':400,'completed_outer':400,'draws_per_method':199,'completed_m4_draws':79600,'completed_original_draws':79600,'m4_engine_sha256':sha(WORKSPACE/'src'/'rs_cusum_phase2rd'/'engine.py'),'source_freeze':json.loads((OUTPUT/'hash_registry_pre_run.json').read_text())}; _atomic(OUTPUT/'run_metadata.json',meta); _atomic(OUTPUT/'manifest.json',{'status':'SUCCESS','files':[x.name for x in OUTPUT.iterdir() if x.is_file()]}); (OUTPUT/'SUCCESS').write_text('SUCCESS\n',encoding='utf-8'); print('Representative AltLoc SUCCESS',flush=True)
def smoke():
 cfg=load(); s=cfg['design']['scenarios'][0]; task={'scenario':s,'index':-1,'seed':991827,'original_bootstrap_seed':991828,'m4_bootstrap_seed':991829}; r=_one_smoke(task); (OUTPUT/'smoke_result.json').parent.mkdir(parents=True,exist_ok=True); _atomic(OUTPUT/'smoke_result.json',r)
 if r['status']!='OK': raise RuntimeError(r['error'])
def _one_smoke(task):
 # Same code path but temporarily freeze only three computational draws; never merged with formal results.
 import copy
 cfg=load(); original=cfg['design']['bootstrap_draws_per_method']; cfg['design']['bootstrap_draws_per_method']=3
 # use direct reduced calculation to avoid changing frozen on-disk config
 try:
  rd=m4_config('D2'); parent=load_phase2_config(); st,ac=_grid(cfg); data=generate(task['scenario'],task['seed'],parent,.5,.25); p=prepare_alternative_analysis(data,rd,parent,st,ac); _,_,j=observed_arrays(p); o=float(np.max(abs(j))); a=run_influence_arm(p,'M0',3,task['original_bootstrap_seed']); m=run_m4(p,rd,parent,st,ac,3,task['m4_bootstrap_seed']); return {'status':'OK','smoke_only':True,'fields_present':True,'observed':o,'m4_valid_draws':int(np.sum(m.valid)),'original_valid_draws':a.valid_draws}
 except Exception as e:return {'status':'FAIL','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['freeze','smoke','formal']); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
 try:
  if a.stage=='freeze': freeze()
  elif a.stage=='smoke': smoke()
  else: formal(a.workers)
 except Exception:
  OUTPUT.mkdir(parents=True,exist_ok=True); (OUTPUT/'FAILURE').write_text(traceback.format_exc(),encoding='utf-8'); raise
if __name__=='__main__':main()
