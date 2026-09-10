"""Unified Phase 2R-D size, covariance, tail, joint-exceedance and gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .protocol import OUTPUT,load_config


def _wilson(successes: int,total: int,z: float=1.959963984540054) -> tuple[float,float]:
    p=successes/total; denominator=1+z*z/total; center=(p+z*z/(2*total))/denominator; half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return float(center-half),float(center+half)


def _ks(a: np.ndarray,b: np.ndarray) -> float:
    a=np.sort(np.asarray(a).ravel()); b=np.sort(np.asarray(b).ravel()); values=np.sort(np.concatenate([a,b])); return float(np.max(np.abs(np.searchsorted(a,values,side="right")/len(a)-np.searchsorted(b,values,side="right")/len(b))))


def size_summary(csv_path: Path,stage: str) -> pd.DataFrame:
    frame=pd.read_csv(csv_path); rows=[]
    for (arm,layer),group in frame.groupby(["arm","layer"],sort=True):
        valid=group[group.status=="TESTABLE"]; n=len(valid); observed=valid.observed.to_numpy(float); row={"stage":stage,"arm":arm,"layer":layer,"replicates":len(group),"testable_replicates":n,"testable_fraction":n/len(group)}
        for label,column in (("001","reject_001"),("005","reject_005"),("010","reject_010")):
            count=int(valid[column].astype(str).str.lower().eq("true").sum()); rate=count/n; low,high=_wilson(count,n); row.update({f"rejections_{label}":count,f"size_{label}":rate,f"mcse_{label}":float(np.sqrt(rate*(1-rate)/n)),f"wilson_low_{label}":low,f"wilson_high_{label}":high})
        row.update({"median_p_value":float(valid.p_value.median()),"mean_bootstrap_q90":float(valid.bootstrap_q90.mean()),"mean_bootstrap_q95":float(valid.bootstrap_q95.mean()),"mean_bootstrap_q99":float(valid.bootstrap_q99.mean()),
                    "oracle_q90":float(np.quantile(observed,.90)),"oracle_q95":float(np.quantile(observed,.95)),"oracle_q99":float(np.quantile(observed,.99)),
                    "q90_ratio":float(valid.bootstrap_q90.mean()/np.quantile(observed,.90)),"q95_ratio":float(valid.bootstrap_q95.mean()/np.quantile(observed,.95)),"q99_ratio":float(valid.bootstrap_q99.mean()/np.quantile(observed,.99)),
                    "draw_failure_rate":float(group.failed_draws.sum()/(group.valid_draws.sum()+group.failed_draws.sum())),"mean_runtime_seconds":float(group.runtime_seconds.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def covariance_summary(npz_path: Path,stage: str) -> pd.DataFrame:
    rows=[]
    with np.load(npz_path,allow_pickle=False) as data:
        observed_joint=np.asarray(data["observed_joint"],float); observed_mid=np.asarray(data["observed_midpoint"],float)
        for name in data.files:
            if not name.endswith("__mean_joint_cov"): continue
            arm=name.removesuffix("__mean_joint_cov")
            for scope,truth,cov,means in (("midpoint_64",observed_mid,np.asarray(data[f"{arm}__mean_mid_cov"],float),np.asarray(data[f"{arm}__conditional_mid_means"],float)),
                                          ("joint_448",observed_joint,np.asarray(data[name],float),np.asarray(data[f"{arm}__conditional_joint_means"],float))):
                truth_cov=np.cov(truth,rowvar=False,ddof=1); truth_sd=np.sqrt(np.maximum(np.diag(truth_cov),0)); boot_sd=np.sqrt(np.maximum(np.diag(cov),0)); te=np.linalg.eigvalsh(truth_cov); be=np.linalg.eigvalsh(cov)
                tc=truth_cov/np.maximum(truth_sd[:,None]*truth_sd[None,:],1e-15); bc=cov/np.maximum(boot_sd[:,None]*boot_sd[None,:],1e-15); mask=~np.eye(len(truth_sd),dtype=bool); standardized=means/np.maximum(truth_sd[None,:],1e-12)
                rows.append({"stage":stage,"arm":arm,"scope":scope,"pointwise_SD_ratio":float(np.mean(boot_sd)/np.mean(truth_sd)),"leading_eigenvalue_ratio":float(be[-1]/te[-1]),
                             "leading_10_eigenvalue_ratios":json.dumps((be[-10:]/np.maximum(te[-10:],1e-15))[::-1].tolist()),"effective_rank_ratio":float((be.sum()**2/np.sum(be**2))/(te.sum()**2/np.sum(te**2))),
                             "correlation_frobenius_relative_error":float(np.linalg.norm(bc-tc)/np.linalg.norm(tc)),"offdiagonal_correlation_MAE":float(np.mean(np.abs((bc-tc)[mask]))),
                             "conditional_mean_RMS_over_truth_SD":float(np.sqrt(np.mean(standardized**2))),"conditional_mean_max_abs_over_truth_SD":float(np.max(np.abs(standardized)))})
    return pd.DataFrame(rows)


def tail_summary(npz_path: Path,stage: str) -> pd.DataFrame:
    rows=[]
    with np.load(npz_path,allow_pickle=False) as data:
        truth=np.asarray(data["observed_joint"],float); midpoint=np.asarray(data["observed_midpoint"],float); tq95=np.quantile(np.abs(truth),.95,axis=0); tq99=np.quantile(np.abs(truth),.99,axis=0)
        centered=truth-truth.mean(0); z=centered/np.maximum(centered.std(0),1e-15); tskew=np.mean(z**3,0); tkurt=np.mean(z**4,0)-3; td2=np.max(np.abs(midpoint),1); td4=np.max(np.abs(truth),1); oracle95=float(np.quantile(td4,.95))
        for name in data.files:
            if not name.endswith("__mean_coordinate_abs_q95"): continue
            arm=name.removesuffix("__mean_coordinate_abs_q95"); q95=np.asarray(data[name],float); q99=np.asarray(data[f"{arm}__mean_coordinate_abs_q99"],float); skew=np.asarray(data[f"{arm}__mean_coordinate_skew"],float); kurt=np.asarray(data[f"{arm}__mean_coordinate_excess_kurtosis"],float)
            d2=np.asarray(data[f"{arm}__D2_draws"],float).ravel(); d2=d2[np.isfinite(d2)]; d4=np.asarray(data[f"{arm}__D4_draws"],float).ravel(); d4=d4[np.isfinite(d4)]
            rows.append({"stage":stage,"arm":arm,"mean_coordinate_abs_q95_ratio":float(np.mean(q95/np.maximum(tq95,1e-12))),"median_coordinate_abs_q95_ratio":float(np.median(q95/np.maximum(tq95,1e-12))),
                         "mean_coordinate_abs_q99_ratio":float(np.mean(q99/np.maximum(tq99,1e-12))),"skewness_RMSE":float(np.sqrt(np.mean((skew-tskew)**2))),"excess_kurtosis_RMSE":float(np.sqrt(np.mean((kurt-tkurt)**2))),
                         "D2_KS_distance":_ks(d2,td2),"D4_KS_distance":_ks(d4,td4),"probability_bootstrap_max_exceeds_oracle_q95":float(np.mean(d4>oracle95)),"bootstrap_D4_q95":float(np.quantile(d4,.95)),"oracle_D4_q95":oracle95})
    return pd.DataFrame(rows)


def joint_exceedance_summary(npz_path: Path,stage: str) -> pd.DataFrame:
    rows=[]
    with np.load(npz_path,allow_pickle=False) as data:
        truth=np.asarray(data["observed_joint"],float); threshold=np.quantile(np.abs(truth),.95,axis=0); te=(np.abs(truth)>threshold).astype(np.float32); tj=te.T@te/len(te); tm=te.mean(0); tc=tj/np.maximum(tm[:,None],1e-15); mask=~np.eye(truth.shape[1],dtype=bool)
        for name in data.files:
            if not name.endswith("__saved_direct_process"): continue
            arm=name.removesuffix("__saved_direct_process"); process=np.asarray(data[name],float).reshape(-1,truth.shape[1]); process=process[np.all(np.isfinite(process),axis=1)]; be=(np.abs(process)>threshold).astype(np.float32); bj=be.T@be/len(be); bm=be.mean(0); bc=bj/np.maximum(bm[:,None],1e-15)
            rows.append({"stage":stage,"arm":arm,"truth_replicates":len(truth),"saved_bootstrap_processes":len(process),"joint_exceedance_frobenius_relative_error":float(np.linalg.norm(bj-tj)/np.linalg.norm(tj)),
                         "joint_exceedance_offdiagonal_MAE":float(np.mean(np.abs((bj-tj)[mask]))),"tail_dependence_offdiagonal_MAE":float(np.mean(np.abs((bc-tc)[mask]))),
                         "truth_tail_dependence_p95":float(np.quantile(tc[mask],.95)),"bootstrap_tail_dependence_p95":float(np.quantile(bc[mask],.95)),"tail_dependence_p95_abs_error":float(abs(np.quantile(bc[mask],.95)-np.quantile(tc[mask],.95))),
                         "truth_tail_dependence_max":float(np.max(tc[mask])),"bootstrap_tail_dependence_max":float(np.max(bc[mask]))})
    return pd.DataFrame(rows)


def runtime_summary(csv_path: Path,stage: str) -> pd.DataFrame:
    frame=pd.read_csv(csv_path); d4=frame[frame.layer=="D4"]; rows=[]
    for arm,group in d4.groupby("arm",sort=True): rows.append({"stage":stage,"arm":arm,"replicates":len(group),"mean_seconds":float(group.runtime_seconds.mean()),"median_seconds":float(group.runtime_seconds.median()),"p90_seconds":float(group.runtime_seconds.quantile(.9)),"total_worker_seconds":float(group.runtime_seconds.sum()),"draw_failure_rate":float(group.failed_draws.sum()/(group.valid_draws.sum()+group.failed_draws.sum()))})
    return pd.DataFrame(rows)


def _range(value: float,bounds: list[float]) -> bool: return float(bounds[0])<=value<=float(bounds[1])


def _gate_rows(stage: str,size: pd.DataFrame,cov: pd.DataFrame,tail: pd.DataFrame,joint: pd.DataFrame,cfg: dict[str,Any]) -> tuple[list[dict[str,Any]],dict[str,bool]]:
    g=cfg[f"{stage.lower()}_gate"]; s=size[size.arm=="M4"].set_index("layer"); c=cov[(cov.arm=="M4")&(cov.scope=="joint_448")].iloc[0]; t=tail[tail.arm=="M4"].iloc[0]; j=joint[joint.arm=="M4"].iloc[0]; checks=[]
    def add(domain,metric,value,passed,threshold): checks.append({"stage":stage,"domain":domain,"metric":metric,"value":float(value),"threshold":str(threshold),"pass":bool(passed)})
    if stage=="D2":
        for layer in ("D0","D1","D2","D3","D4"): add("type1",f"{layer}_size_005",s.loc[layer,"size_005"],s.loc[layer,"size_005"]<=g["type1"]["size_005_max_each_D0_D4"],f"<= {g['type1']['size_005_max_each_D0_D4']}")
    else:
        for layer in ("D0","D1","D2","D3","D4"): add("type1",f"{layer}_size_005",s.loc[layer,"size_005"],s.loc[layer,"size_005"]<=g["type1"][f"{layer}_size_005_max"],f"<= {g['type1'][f'{layer}_size_005_max']}")
    add("type1","D4_size_001",s.loc["D4","size_001"],s.loc["D4","size_001"]<=g["type1"]["D4_size_001_max"],f"<= {g['type1']['D4_size_001_max']}")
    for layer in ("D0","D1","D2","D3","D4"): add("type1",f"{layer}_q95_ratio",s.loc[layer,"q95_ratio"],_range(s.loc[layer,"q95_ratio"],g["type1"]["q95_ratio_range_each_D0_D4"]),g["type1"]["q95_ratio_range_each_D0_D4"])
    add("type1","D4_q95_primary",s.loc["D4","q95_ratio"],_range(s.loc["D4","q95_ratio"],g["type1"]["D4_q95_ratio_range"]),g["type1"]["D4_q95_ratio_range"])
    if stage=="D3":
        nominal=(s.loc["D4","wilson_low_005"]<=.05<=s.loc["D4","wilson_high_005"]) or abs(s.loc["D4","size_005"]-.05)<=s.loc["D4","mcse_005"]
        add("type1","D4_nominal_within_Wilson_or_one_MCSE",float(nominal),nominal,"true")
    cg=g["center_covariance"]; add("center","conditional_mean_RMS_over_truth_SD",c.conditional_mean_RMS_over_truth_SD,c.conditional_mean_RMS_over_truth_SD<=cg["conditional_mean_RMS_over_truth_SD_max"],f"<= {cg['conditional_mean_RMS_over_truth_SD_max']}")
    for metric,key in (("pointwise_SD_ratio","pointwise_SD_ratio_range"),("leading_eigenvalue_ratio","leading_eigenvalue_ratio_range"),("effective_rank_ratio","effective_rank_ratio_range")): add("covariance",metric,c[metric],_range(c[metric],cg[key]),cg[key])
    for metric,key in (("correlation_frobenius_relative_error","correlation_frobenius_relative_error_max"),("offdiagonal_correlation_MAE","offdiagonal_correlation_MAE_max")): add("covariance",metric,c[metric],c[metric]<=cg[key],f"<= {cg[key]}")
    tg=g["joint_tail"]
    for metric,key in (("mean_coordinate_abs_q95_ratio","mean_coordinate_abs_q95_ratio_range"),("mean_coordinate_abs_q99_ratio","mean_coordinate_abs_q99_ratio_range"),("probability_bootstrap_max_exceeds_oracle_q95","probability_max_exceeds_oracle_q95_range")): add("joint_tail",metric,t[metric],_range(t[metric],tg[key]),tg[key])
    for metric,key in (("skewness_RMSE","skewness_RMSE_max"),("excess_kurtosis_RMSE","excess_kurtosis_RMSE_max"),("D2_KS_distance","D2_KS_max"),("D4_KS_distance","D4_KS_max")): add("joint_tail",metric,t[metric],t[metric]<=tg[key],f"<= {tg[key]}")
    for metric,key in (("joint_exceedance_frobenius_relative_error","joint_exceedance_frobenius_relative_error_max"),("joint_exceedance_offdiagonal_MAE","joint_exceedance_offdiagonal_MAE_max"),("tail_dependence_offdiagonal_MAE","tail_dependence_offdiagonal_MAE_max"),("tail_dependence_p95_abs_error","tail_dependence_p95_abs_error_max")): add("joint_tail",metric,j[metric],j[metric]<=tg[key],f"<= {tg[key]}")
    eg=g["engineering"]; add("engineering","draw_failure_rate",s.loc["D4","draw_failure_rate"],s.loc["D4","draw_failure_rate"]<=eg["draw_failure_rate_max"],f"<= {eg['draw_failure_rate_max']}"); add("engineering","testable_fraction",s.loc["D4","testable_fraction"],s.loc["D4","testable_fraction"]>=eg["testable_fraction_min"],f">= {eg['testable_fraction_min']}")
    domains={domain:all(row["pass"] for row in checks if row["domain"]==domain) for domain in ("type1","center","covariance","joint_tail","engineering")}; return checks,domains


def summarize(stage: str) -> dict[str,Any]:
    cfg=load_config(stage); stem=stage.lower(); size=size_summary(OUTPUT/f"{stem}_replicate_results.csv",stage); cov=covariance_summary(OUTPUT/f"{stem}_process_forensics.npz",stage); tail=tail_summary(OUTPUT/f"{stem}_process_forensics.npz",stage); joint=joint_exceedance_summary(OUTPUT/f"{stem}_process_forensics.npz",stage); runtime=runtime_summary(OUTPUT/f"{stem}_replicate_results.csv",stage)
    if stage=="D2": names=("m4_size_summary.csv","m4_covariance_process.csv","m4_tail_diagnostics.csv","m4_joint_exceedance.csv","m4_runtime_summary.csv","m4_mechanism_summary.csv","d2_gate.json")
    else: names=("m4_holdout_summary.csv","m4_holdout_covariance.csv","m4_holdout_tail_diagnostics.csv","m4_holdout_joint_exceedance.csv","m4_holdout_runtime.csv","m4_holdout_mechanism.csv","m4_holdout_gate.json")
    checks,domains=_gate_rows(stage,size,cov,tail,joint,cfg); pd.DataFrame(checks).to_csv(OUTPUT/names[5],index=False); size.to_csv(OUTPUT/names[0],index=False); cov.to_csv(OUTPUT/names[1],index=False); tail.to_csv(OUTPUT/names[2],index=False); joint.to_csv(OUTPUT/names[3],index=False); runtime.to_csv(OUTPUT/names[4],index=False)
    if stage=="D2":
        failed=[key for key,value in domains.items() if not value]
        if not failed: label=cfg["stop_rules"]["D2_pass"]
        elif len(failed)>1: label=cfg["stop_rules"]["D2_multi_fail"]
        else: label=cfg["stop_rules"][{"type1":"D2_type1_fail","center":"D2_center_fail","covariance":"D2_covariance_fail","joint_tail":"D2_tail_fail","engineering":"D1_fail"}[failed[0]]]
        payload={"gate":label,"D3_allowed":not failed,"domain_checks":domains,"failed_domains":failed,"posthoc_tuning":False,"holdout_seeds_loaded":False}
        (OUTPUT/names[6]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    else:
        passed=all(domains.values()); payload={"gate":cfg["stop_rules"]["D3_pass" if passed else "D3_fail"],"domain_checks":domains,"establish_1_0_2":False,"phase3_allowed":False,"transfer_null_allowed":False,"ohiot1dm_allowed":False,"posthoc_tuning":False}
        (OUTPUT/names[6]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["D2","D3"]); args=parser.parse_args(); print(json.dumps(summarize(args.stage),sort_keys=True))


if __name__=="__main__": main()
