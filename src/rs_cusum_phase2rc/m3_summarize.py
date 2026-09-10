"""C2/C3 calibration, covariance, tail-shape diagnostics, and gates."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from .m3_runner import load_supplement
from .protocol import OUTPUT,load_config,stable_seed


def _wilson(successes: int,total: int,z: float=1.959963984540054) -> tuple[float,float]:
    p=successes/total; denominator=1+z*z/total; center=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator; return float(center-half),float(center+half)


def summarize_rows(path,stage: str) -> pd.DataFrame:
    frame=pd.read_csv(path); rows=[]
    for (arm,law,layer),group in frame.groupby(["arm","weight_law","layer"],sort=True):
        valid=group[group.status=="TESTABLE"]; n=len(valid); observed=valid.observed.to_numpy(float)
        row={"stage":stage,"arm":arm,"weight_law":law,"layer":layer,"replicates":len(group),"testable_replicates":n}
        for label,column in (("001","reject_001"),("005","reject_005"),("010","reject_010")):
            rejection=valid[column]; count=int(rejection.astype(str).str.lower().eq("true").sum()); rate=count/n; low,high=_wilson(count,n)
            row.update({f"size_{label}":rate,f"mcse_{label}":float(np.sqrt(rate*(1-rate)/n)),f"wilson_low_{label}":low,f"wilson_high_{label}":high})
        row.update({"median_p_value":float(valid.p_value.median()),"mean_bootstrap_q90":float(valid.bootstrap_q90.mean()),"mean_bootstrap_q95":float(valid.bootstrap_q95.mean()),"mean_bootstrap_q99":float(valid.bootstrap_q99.mean()),
                    "oracle_q90":float(np.quantile(observed,.90)),"oracle_q95":float(np.quantile(observed,.95)),"oracle_q99":float(np.quantile(observed,.99)),
                    "q90_ratio":float(valid.bootstrap_q90.mean()/np.quantile(observed,.90)),"q95_ratio":float(valid.bootstrap_q95.mean()/np.quantile(observed,.95)),"q99_ratio":float(valid.bootstrap_q99.mean()/np.quantile(observed,.99)),
                    "draw_failure_rate":float(group.failed_draws.sum()/(group.valid_draws.sum()+group.failed_draws.sum())),"mean_runtime_seconds":float(valid.runtime_seconds.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def covariance_table(npz_path,stage: str) -> pd.DataFrame:
    rows=[]
    with np.load(npz_path,allow_pickle=False) as data:
        observed_joint=np.asarray(data["observed_joint"],float); observed_mid=np.asarray(data["observed_midpoint"],float)
        for name in data.files:
            if not name.endswith("__mean_joint_cov"): continue
            key=name.removesuffix("__mean_joint_cov"); arm,law=key.split("__",1)
            for scope,observed,cov,means in (
                ("midpoint_64",observed_mid,np.asarray(data[f"{key}__mean_mid_cov"],float),np.asarray(data[f"{key}__conditional_mid_means"],float)),
                ("joint_448",observed_joint,np.asarray(data[name],float),np.asarray(data[f"{key}__conditional_joint_means"],float)),
            ):
                truth=np.cov(observed,rowvar=False,ddof=1); truth_sd=np.sqrt(np.maximum(np.diag(truth),0)); boot_sd=np.sqrt(np.maximum(np.diag(cov),0)); eig_truth=np.linalg.eigvalsh(truth); eig_boot=np.linalg.eigvalsh(cov)
                corr_truth=truth/np.maximum(truth_sd[:,None]*truth_sd[None,:],1e-15); corr_boot=cov/np.maximum(boot_sd[:,None]*boot_sd[None,:],1e-15); mask=~np.eye(len(truth),dtype=bool)
                standardized_means=means/np.maximum(truth_sd[None,:],1e-12)
                rows.append({"stage":stage,"arm":arm,"weight_law":law,"scope":scope,"pointwise_SD_ratio":float(np.mean(boot_sd)/np.mean(truth_sd)),
                             "leading_eigenvalue_ratio":float(eig_boot[-1]/eig_truth[-1]),"leading_10_eigenvalue_ratios":json.dumps((eig_boot[-10:]/np.maximum(eig_truth[-10:],1e-15))[::-1].tolist()),
                             "effective_rank_ratio":float((np.sum(eig_boot)**2/np.sum(eig_boot**2))/(np.sum(eig_truth)**2/np.sum(eig_truth**2))),
                             "correlation_frobenius_relative_error":float(np.linalg.norm(corr_truth-corr_boot)/np.linalg.norm(corr_truth)),"offdiagonal_correlation_MAE":float(np.mean(np.abs((corr_truth-corr_boot)[mask]))),
                             "conditional_mean_RMS_over_truth_SD":float(np.sqrt(np.mean(standardized_means**2))),"conditional_mean_max_abs_over_truth_SD":float(np.max(np.abs(standardized_means)))})
    return pd.DataFrame(rows)


def _ks(a: np.ndarray,b: np.ndarray) -> float:
    a=np.sort(np.asarray(a).ravel()); b=np.sort(np.asarray(b).ravel()); values=np.sort(np.concatenate([a,b]))
    return float(np.max(np.abs(np.searchsorted(a,values,side="right")/len(a)-np.searchsorted(b,values,side="right")/len(b))))


def tail_table(npz_path,stage: str) -> pd.DataFrame:
    rows=[]
    with np.load(npz_path,allow_pickle=False) as data:
        observed=np.asarray(data["observed_joint"],float); truth_abs_q95=np.quantile(np.abs(observed),.95,axis=0); truth_abs_q99=np.quantile(np.abs(observed),.99,axis=0)
        truth_centered=observed-np.mean(observed,axis=0); truth_sd=np.std(truth_centered,axis=0,ddof=0); truth_z=truth_centered/np.maximum(truth_sd,1e-15)
        truth_skew=np.mean(truth_z**3,axis=0); truth_kurt=np.mean(truth_z**4,axis=0)-3
        truth_d4=np.max(np.abs(observed),axis=1); truth_d2=np.max(np.abs(np.asarray(data["observed_midpoint"],float)),axis=1)
        for name in data.files:
            if not name.endswith("__mean_coordinate_abs_q95"): continue
            key=name.removesuffix("__mean_coordinate_abs_q95"); arm,law=key.split("__",1)
            q95=np.asarray(data[name],float); q99=np.asarray(data[f"{key}__mean_coordinate_abs_q99"],float); skew=np.asarray(data[f"{key}__mean_coordinate_skew"],float); kurt=np.asarray(data[f"{key}__mean_coordinate_excess_kurtosis"],float)
            d2=np.asarray(data[f"{key}__D2_draws"],float).ravel(); d4=np.asarray(data[f"{key}__D4_draws"],float).ravel(); oracle95=float(np.quantile(truth_d4,.95))
            rows.append({"stage":stage,"arm":arm,"weight_law":law,"mean_coordinate_abs_q95_ratio":float(np.mean(q95/np.maximum(truth_abs_q95,1e-12))),"median_coordinate_abs_q95_ratio":float(np.median(q95/np.maximum(truth_abs_q95,1e-12))),
                         "mean_coordinate_abs_q99_ratio":float(np.mean(q99/np.maximum(truth_abs_q99,1e-12))),"skewness_RMSE":float(np.sqrt(np.mean((skew-truth_skew)**2))),"excess_kurtosis_RMSE":float(np.sqrt(np.mean((kurt-truth_kurt)**2))),
                         "D2_KS_distance":_ks(d2,truth_d2),"D4_KS_distance":_ks(d4,truth_d4),"bootstrap_probability_D4_exceeds_oracle_q95":float(np.mean(d4>oracle95))})
    return pd.DataFrame(rows)


def m1_oracle_forensic(npz_path) -> dict[str,Any]:
    supplement=load_supplement(); cfg=load_config("m3")
    with np.load(npz_path,allow_pickle=False) as data:
        truth=np.asarray(data["observed_joint"],float); boot=np.asarray(data["M1__gaussian__saved_centered_process"],float).reshape(-1,truth.shape[1])
    thresholds=np.quantile(np.abs(truth),.95,axis=0); truth_ex=(np.abs(truth)>thresholds).astype(np.float32); boot_ex=(np.abs(boot)>thresholds).astype(np.float32)
    truth_pair=truth_ex.T@truth_ex/len(truth_ex); boot_pair=boot_ex.T@boot_ex/len(boot_ex); mask=~np.eye(truth.shape[1],dtype=bool)
    truth_max=np.max(np.abs(truth),axis=1); boot_max=np.max(np.abs(boot),axis=1); oracle_q95=float(np.quantile(truth_max,.95))
    covariance=np.cov(truth,rowvar=False,ddof=1); eigval,eigvec=np.linalg.eigh(covariance); positive=eigval>max(eigval[-1]*1e-12,0); root=eigvec[:,positive]*np.sqrt(eigval[positive])[None,:]
    draws=int(supplement["c2_forensics"]["covariance_matched_gaussian_draws"]); rng=np.random.default_rng(stable_seed(cfg["fresh_seeds"]["master_seed"],supplement["c2_forensics"]["covariance_matched_gaussian_seed_label"])); gaussian_max=[]
    for start in range(0,draws,5000):
        count=min(5000,draws-start); sample=rng.standard_normal((count,root.shape[1]))@root.T; gaussian_max.append(np.max(np.abs(sample),axis=1))
    gaussian_max=np.concatenate(gaussian_max)
    truth_cond=truth_pair[mask]/.05; boot_cond=boot_pair[mask]/.05
    result={"truth_D4_q95":oracle_q95,"M1_mixture_D4_q95":float(np.quantile(boot_max,.95)),"covariance_matched_gaussian_D4_q95":float(np.quantile(gaussian_max,.95)),
            "M1_q95_over_truth":float(np.quantile(boot_max,.95)/oracle_q95),"gaussian_q95_over_truth":float(np.quantile(gaussian_max,.95)/oracle_q95),
            "M1_probability_max_exceeds_truth_q95":float(np.mean(boot_max>oracle_q95)),"gaussian_probability_max_exceeds_truth_q95":float(np.mean(gaussian_max>oracle_q95)),
            "M1_vs_truth_max_KS":_ks(boot_max,truth_max),"gaussian_vs_truth_max_KS":_ks(gaussian_max,truth_max),
            "pairwise_joint_exceedance_frobenius_relative_error":float(np.linalg.norm(boot_pair-truth_pair)/np.linalg.norm(truth_pair)),"pairwise_joint_exceedance_offdiagonal_MAE":float(np.mean(np.abs((boot_pair-truth_pair)[mask]))),
            "truth_mean_offdiagonal_tail_dependence":float(np.mean(truth_cond)),"M1_mean_offdiagonal_tail_dependence":float(np.mean(boot_cond)),"truth_p95_offdiagonal_tail_dependence":float(np.quantile(truth_cond,.95)),"M1_p95_offdiagonal_tail_dependence":float(np.quantile(boot_cond,.95)),
            "gaussian_draws":draws,"M1_saved_mixture_draws":len(boot)}
    (OUTPUT/"c2_m1_gaussian_oracle.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return result


def summarize_c2() -> dict[str,Any]:
    cfg=load_config("m3"); supplement=load_supplement(); summary=summarize_rows(OUTPUT/"c2_replicate_results.csv","C2"); summary.to_csv(OUTPUT/"c2_summary.csv",index=False)
    cov=covariance_table(OUTPUT/"c2_process_forensics.npz","C2"); cov.to_csv(OUTPUT/"c2_covariance.csv",index=False)
    tails=tail_table(OUTPUT/"c2_process_forensics.npz","C2"); tails.to_csv(OUTPUT/"c2_tail_shape.csv",index=False); m1=m1_oracle_forensic(OUTPUT/"c2_process_forensics.npz")
    m3=summary[(summary.arm=="M3")&(summary.weight_law=="normalized_exponential")].set_index("layer"); m3cov=cov[(cov.arm=="M3")&(cov.weight_law=="normalized_exponential")&(cov.scope=="joint_448")].iloc[0]; gate_cfg=cfg["m3_c2_continuation"]
    checks={"D0":float(m3.loc["D0","size_005"])<=gate_cfg["D0_size_005_max"],"D2":float(m3.loc["D2","size_005"])<=gate_cfg["D2_size_005_max"],"D4":float(m3.loc["D4","size_005"])<=gate_cfg["D4_size_005_max"],
            "D4_q95":gate_cfg["D4_q95_ratio_range"][0]<=float(m3.loc["D4","q95_ratio"])<=gate_cfg["D4_q95_ratio_range"][1],"conditional_mean":float(m3cov.conditional_mean_RMS_over_truth_SD)<=gate_cfg["conditional_mean_RMS_over_truth_SD_max"],
            "pointwise_SD":gate_cfg["pointwise_SD_ratio_range"][0]<=float(m3cov.pointwise_SD_ratio)<=gate_cfg["pointwise_SD_ratio_range"][1],"leading_eigenvalue":gate_cfg["leading_eigenvalue_ratio_range"][0]<=float(m3cov.leading_eigenvalue_ratio)<=gate_cfg["leading_eigenvalue_ratio_range"][1],
            "failure":float(m3.loc["D4","draw_failure_rate"])<=gate_cfg["failure_rate_max"]}
    partial_cfg=supplement["selection"]["partial_evidence_if_C2_fails"]
    partial=(float(m3.loc["D4","size_005"])<=partial_cfg["D4_size_005_max"] and partial_cfg["D4_q95_ratio_range"][0]<=float(m3.loc["D4","q95_ratio"])<=partial_cfg["D4_q95_ratio_range"][1]
             and float(m3cov.conditional_mean_RMS_over_truth_SD)<=partial_cfg["conditional_mean_RMS_over_truth_SD_max"] and partial_cfg["pointwise_SD_ratio_range"][0]<=float(m3cov.pointwise_SD_ratio)<=partial_cfg["pointwise_SD_ratio_range"][1]
             and partial_cfg["leading_eigenvalue_ratio_range"][0]<=float(m3cov.leading_eigenvalue_ratio)<=partial_cfg["leading_eigenvalue_ratio_range"][1])
    gate={"gate":"C2_CONTINUE" if all(checks.values()) else ("DESIGN_VARIATION_HYPOTHESIS_SUPPORTED_BUT_M3_PARTIAL" if partial else "PHASE_2RC_FAILED"),"M3_continue":all(checks.values()),"checks":checks,"partial_evidence":partial,"M1_oracle_forensic":m1}
    (OUTPUT/"c2_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return gate


def summarize_mult() -> None:
    summary=summarize_rows(OUTPUT/"c2_mult_replicate_results.csv","C2_MULT"); summary.to_csv(OUTPUT/"c2_mult_summary.csv",index=False)
    covariance_table(OUTPUT/"c2_mult_process_forensics.npz","C2_MULT").to_csv(OUTPUT/"c2_mult_covariance.csv",index=False)
    tail_table(OUTPUT/"c2_mult_process_forensics.npz","C2_MULT").to_csv(OUTPUT/"c2_mult_tail_shape.csv",index=False)


def summarize_c3() -> dict[str,Any]:
    cfg=load_config("m3"); supplement=load_supplement(); summary=summarize_rows(OUTPUT/"c3_replicate_results.csv","C3"); summary.to_csv(OUTPUT/"c3_summary.csv",index=False)
    cov=covariance_table(OUTPUT/"c3_process_forensics.npz","C3"); cov.to_csv(OUTPUT/"c3_covariance.csv",index=False); tail_table(OUTPUT/"c3_process_forensics.npz","C3").to_csv(OUTPUT/"c3_tail_shape.csv",index=False)
    m3=summary.set_index("layer"); joint=cov[cov.scope=="joint_448"].iloc[0]; g=cfg["m3_c3_acceptance"]; broad=supplement["selection"]["c3_broad_joint_spread_required"]
    size=float(m3.loc["D4","size_005"]); mcse=float(m3.loc["D4","mcse_005"]); low=float(m3.loc["D4","wilson_low_005"]); high=float(m3.loc["D4","wilson_high_005"])
    checks={"D4":size<=g["D4_size_005_max"],"D4_q95":g["D4_q95_ratio_range"][0]<=float(m3.loc["D4","q95_ratio"])<=g["D4_q95_ratio_range"][1],"conditional_mean":float(joint.conditional_mean_RMS_over_truth_SD)<=g["conditional_mean_RMS_over_truth_SD_max"],
            "pointwise_SD_broad":broad["pointwise_SD_ratio_range"][0]<=float(joint.pointwise_SD_ratio)<=broad["pointwise_SD_ratio_range"][1],"leading_eigenvalue_broad":broad["leading_eigenvalue_ratio_range"][0]<=float(joint.leading_eigenvalue_ratio)<=broad["leading_eigenvalue_ratio_range"][1],
            "near_zero":float(m3.loc["D4","size_001"])<=g["size_001_max"],"failure":float(m3.loc["D4","draw_failure_rate"])<=g["failure_rate_max"],"nominal_within_or_near_MC":low<=.05<=high or abs(size-.05)<=mcse}
    preferred={"pointwise_SD":g["pointwise_SD_ratio_preferred"][0]<=float(joint.pointwise_SD_ratio)<=g["pointwise_SD_ratio_preferred"][1],"leading_eigenvalue":g["leading_eigenvalue_ratio_preferred"][0]<=float(joint.leading_eigenvalue_ratio)<=g["leading_eigenvalue_ratio_preferred"][1]}
    gate={"gate":"WHOLE_CLUSTER_INFERENCE_CANDIDATE_CONFIRMED_ON_N0" if all(checks.values()) else "DESIGN_VARIATION_HYPOTHESIS_SUPPORTED_BUT_M3_PARTIAL","checks":checks,"preferred_joint_spread":preferred,"establish_1_0_2":False,"phase3_allowed":False,"ohiot1dm_allowed":False,"transfer_null_next_phase_allowed":all(checks.values())}
    (OUTPUT/"phase2rc_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return gate


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["c2","mult","c3"]); args=parser.parse_args(); result={"c2":summarize_c2,"mult":summarize_mult,"c3":summarize_c3}[args.stage](); print(json.dumps(result,sort_keys=True) if result is not None else "done")


if __name__=="__main__": main()
