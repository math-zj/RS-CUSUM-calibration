"""Outcome-blind static eligibility record for the Ohio empirical protocol gate.

This deliberately reads no reward, Q, CUSUM, bootstrap, p-value, or changepoint field.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs'/'ohio_rs_cusum_empirical_frozen.yaml'
OUT=ROOT/'results_rs_cusum'/'ohio_empirical_protocol_freeze'
FEAS=ROOT/'results_mdp_gate'/'landmark_feasibility'/'landmark_feasibility_summary.csv'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 cfg=yaml.safe_load(CFG.read_text(encoding='utf-8')); rows=[]
 x=pd.read_csv(FEAS)
 for split,clock,landmark in [('TRAINING','training_reset_clock',42.0),('TESTING','testing_reset_clock',24.0)]:
  q=x[(x.dataset_clock==clock)&(x.landmark_h==landmark)]
  if len(q)!=1: raise RuntimeError(f'missing pre-CUSUM feasibility row for {split}')
  r=q.iloc[0]
  support_pass=bool(r.minimum_active_patients_20_80>=6 and r.worst_side_action_20_80>=66 and r.maximum_patient_dominance_20_80<=.60)
  rows.append({'split':split,'availability_clock':clock,'availability_landmark_h':landmark,'eligible_patients_pre_cusum_4to6':int(r.eligible_patients),'minimum_active_patients_20to80_pre_cusum_4to6':int(r.minimum_active_patients_20_80),'worst_side_A1_pre_cusum_4to6':int(r.worst_side_action_20_80),'max_A1_dominance_pre_cusum_4to6':float(r.maximum_patient_dominance_20_80),'primary_support_pass_under_permissive_4to6_summary':support_pass,'formal_eligible':False,'formal_status':'FAIL','reasons':'locked_M4_has_no_real_data_interface; primary_4.5to5.5_trajectory_eligibility_not_certified' + ('; insufficient_active_patient_and_action_support' if not support_pass else '')})
 OUT.mkdir(parents=True,exist_ok=True); frame=pd.DataFrame(rows); frame.to_csv(OUT/'ohio_empirical_eligibility.csv',index=False)
 manifest={'formal_cusum_called':False,'bootstrap_called':False,'p_values_computed':False,'changepoints_computed':False,'selection_inputs':['availability','patient counts','action counts','support','missingness'],'forbidden_inputs':['reward','Q','CUSUM statistic','p-value','tau_hat'],'training_formal_eligible':False,'testing_formal_eligible':False,'gate':'OHIO_PROTOCOL_NOT_FEASIBLE','config_sha256':sha(CFG),'m4_engine_sha256':sha(ROOT/'src'/'rs_cusum_phase2rd'/'engine.py')}
 (OUT/'eligibility_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 frozen=[CFG,ROOT/'report'/'ohio_rs_cusum_empirical_protocol_freeze.md',Path(__file__),ROOT/'configs'/'mdp_5min_protocol.yaml',ROOT/'configs'/'rs_cusum_main.yaml',ROOT/'configs'/'rs_cusum_sensitivity.yaml',ROOT/'configs'/'common_landmark_feasibility.yaml',ROOT/'report'/'common_landmark_monotone_censoring_feasibility.md',ROOT/'src'/'rs_cusum_phase2rd'/'engine.py',FEAS]
 (OUT/'hash_registry_protocol.json').write_text(json.dumps({'stage':'OHIO_EMPIRICAL_PROTOCOL_OUTCOME_BLIND_FREEZE','hash_algorithm':'SHA-256','files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in frozen},'gate':'OHIO_PROTOCOL_NOT_FEASIBLE','formal_results_before_freeze':False},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
