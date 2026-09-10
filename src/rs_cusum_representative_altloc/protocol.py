from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np, yaml
WORKSPACE=Path(__file__).resolve().parents[2]; CONFIG=WORKSPACE/'configs'/'rs_cusum_representative_altloc.yaml'; PROTOCOL=WORKSPACE/'report'/'representative_altloc_protocol.md'; OUTPUT=WORKSPACE/'results_rs_cusum'/'representative_altloc'; SEEDS=OUTPUT/'seed_registry.npz'; META=OUTPUT/'seed_registry.json'; FREEZE=OUTPUT/'hash_registry_pre_run.json'
def sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest().upper()
def stable(*parts:object)->int: return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')
def load()->dict[str,Any]:
 cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8')); p=cfg['protocol']; assert p['label']=='representative-altloc' and not p['m4_modified'] and not p['posthoc_tuning']; assert cfg['design']['outer_replicates_per_scenario']==100 and cfg['design']['bootstrap_draws_per_method']==199; return cfg
def _atomic(path:Path,obj:dict[str,Any]):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def freeze()->None:
 if FREEZE.exists(): return
 cfg=load(); sc=cfg['design']['scenarios']; n=int(cfg['design']['outer_replicates_per_scenario']); arrays={}
 for label in ('dataset','original_bootstrap','m4_bootstrap'):
  arrays[label]=np.asarray([[stable(cfg['seeds']['master_seed'], row['id'],label,i) for i in range(n)] for row in sc],dtype=np.uint64)
 flat=np.concatenate([v.ravel() for v in arrays.values()]);
 if len(np.unique(flat))!=len(flat): raise RuntimeError('seed collision')
 prior=set()
 for path in (WORKSPACE/'results_rs_cusum').glob('**/*seed*.npz'):
  if path.resolve()==SEEDS.resolve(): continue
  try:
   with np.load(path,allow_pickle=False) as z:
    for key in z.files:
     value=np.asarray(z[key])
     if np.issubdtype(value.dtype,np.integer): prior.update(map(int,value.ravel()))
  except (OSError,ValueError): pass
 if set(map(int,flat)) & prior: raise RuntimeError('historical seed collision')
 if SEEDS.exists() or META.exists(): raise RuntimeError('partial seed registry exists without freeze')
 OUTPUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(SEEDS,**arrays)
 _atomic(META,{'master_seed':cfg['seeds']['master_seed'],'algorithm':cfg['seeds']['algorithm'],'arrays':{k:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest().upper()} for k,v in arrays.items()},'unique':True,'historical_seed_values_scanned':len(prior),'historical_collision_count':0,'formal_results_before_freeze':False})
 paths=[CONFIG,PROTOCOL,SEEDS,META,WORKSPACE/'src'/'rs_cusum_representative_altloc'/'analysis.py',WORKSPACE/'src'/'rs_cusum_representative_altloc'/'generator.py',WORKSPACE/'src'/'rs_cusum_representative_altloc'/'runner.py',WORKSPACE/'src'/'rs_cusum_phase2rd'/'engine.py',WORKSPACE/'src'/'rs_cusum_phase2rb'/'engine.py']
 _atomic(FREEZE,{'stage':'REPRESENTATIVE_ALTLOC_PRE_RUN_FREEZE','files':{str(x.relative_to(WORKSPACE)).replace('\\','/'):sha(x) for x in paths},'m4_unchanged':True,'m4_engine_sha256':sha(WORKSPACE/'src'/'rs_cusum_phase2rd'/'engine.py'),'total_outer':400,'bootstrap_draws_per_method':199})
def verify():
 payload=json.loads(FREEZE.read_text(encoding='utf-8'))
 bad=[name for name,value in payload['files'].items() if not (WORKSPACE/name).is_file() or sha(WORKSPACE/name)!=value]
 if bad: raise RuntimeError('frozen file hash mismatch: '+str(bad))
 if payload['m4_engine_sha256']!=load()['design']['m4_engine_sha256']: raise RuntimeError('M4 hash mismatch')
def seeds(name:str)->np.ndarray:
 verify()
 with np.load(SEEDS,allow_pickle=False) as x:return np.asarray(x[name]).copy()
