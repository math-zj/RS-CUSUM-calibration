"""Summarize C0 mean/center/covariance, nested variance, and patient evidence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .protocol import OUTPUT,WORKSPACE,load_config


def _safe_standardized(values: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return values/np.maximum(sd,1e-12)


def _covariance_rows(data: Any, truth_joint: np.ndarray, truth_mid: np.ndarray, arms: list[str]) -> list[dict]:
    rows=[]
    for arm in arms:
        for scope,truth,cov,second in (
            ("midpoint_64",truth_mid,np.asarray(data[f"{arm}_mean_mid_cov"],float),np.asarray(data[f"{arm}_mean_mid_second"],float)),
            ("joint_448",truth_joint,np.asarray(data[f"{arm}_mean_joint_cov"],float),np.asarray(data[f"{arm}_mean_joint_second"],float)),
        ):
            truth_sd=np.sqrt(np.maximum(np.diag(truth),0)); centered_sd=np.sqrt(np.maximum(np.diag(cov),0)); uncentered_rms=np.sqrt(np.maximum(np.diag(second),0))
            eig_truth=np.linalg.eigvalsh(truth); eig_centered=np.linalg.eigvalsh(cov); eig_second=np.linalg.eigvalsh(second)
            rows.append({"arm":arm,"scope":scope,
                         "centered_pointwise_SD_ratio":float(np.mean(centered_sd)/np.mean(truth_sd)),
                         "uncentered_root_second_moment_ratio":float(np.mean(uncentered_rms)/np.mean(truth_sd)),
                         "centered_leading_eigenvalue_ratio":float(eig_centered[-1]/eig_truth[-1]),
                         "uncentered_leading_second_moment_ratio":float(eig_second[-1]/eig_truth[-1]),
                         "centered_leading_10_eigenvalue_ratios":json.dumps((eig_centered[-10:]/np.maximum(eig_truth[-10:],1e-15))[::-1].tolist()),
                         "conditional_mean_second_moment_fraction":float(np.trace(second-cov)/np.trace(second))})
    return rows


def summarize() -> dict:
    cfg=load_config("c0"); arms=list(cfg["c0"]["arms"])
    with np.load(OUTPUT/"c0_process_forensics.npz",allow_pickle=False) as data:
        observed_raw=np.asarray(data["observed_raw"],float); observed_norm=np.asarray(data["observed_normalized"],float); observed_joint=np.asarray(data["observed_joint"],float)
        truth_raw_sd=np.std(observed_raw,axis=0,ddof=1); truth_norm_sd=np.std(observed_norm,axis=0,ddof=1); truth_joint_flat=observed_joint.reshape(len(observed_joint),-1)
        truth_joint_cov=np.cov(truth_joint_flat,rowvar=False,ddof=1); truth_mid_cov=np.cov(observed_norm[:,3],rowvar=False,ddof=1)
        conditional_rows=[]; candidate_rows=[]; state_rows=[]
        for arm in arms:
            for scale,truth_sd in (("raw",truth_raw_sd),("normalized",truth_norm_sd)):
                mean=np.asarray(data[f"{arm}_{scale}_conditional_mean"],float); standardized=_safe_standardized(mean,truth_sd[None])
                conditional_rows.append({"arm":arm,"scale":scale,"mean_absolute_conditional_mean":float(np.mean(np.abs(mean))),
                                         "RMS_conditional_mean":float(np.sqrt(np.mean(mean**2))),
                                         "RMS_conditional_mean_over_truth_pointwise_SD":float(np.sqrt(np.mean(standardized**2))),
                                         "maximum_absolute_conditional_mean_over_truth_SD":float(np.max(np.abs(standardized)))})
                for candidate_index,candidate in enumerate(cfg["frozen_observed_method"]["base_candidates"]):
                    candidate_rows.append({"arm":arm,"scale":scale,"candidate":candidate,
                                           "RMS_standardized_conditional_mean":float(np.sqrt(np.mean(standardized[:,candidate_index]**2))),
                                           "mean_signed_standardized_conditional_mean":float(np.mean(standardized[:,candidate_index]))})
                for state_index in range(mean.shape[2]):
                    state_rows.append({"arm":arm,"scale":scale,"state_grid_index":state_index,
                                       "RMS_standardized_conditional_mean":float(np.sqrt(np.mean(standardized[:,:,state_index]**2))),
                                       "mean_signed_standardized_conditional_mean":float(np.mean(standardized[:,:,state_index]))})
        covariance_rows=_covariance_rows(data,truth_joint_cov,truth_mid_cov,arms)
        m2_mean=np.asarray(data["M2_raw_conditional_mean"],float); zero=np.asarray(data["zero_raw"],float)
        x=zero.ravel(); y=m2_mean.ravel(); correlation=float(np.corrcoef(x,y)[0,1]); slope=float(np.dot(x,y)/np.dot(x,x)); rmse=float(np.sqrt(np.mean((y-x)**2))); relative_rmse=rmse/float(np.sqrt(np.mean(y**2)))
        standardized_zero=_safe_standardized(zero,truth_raw_sd[None])
        zero_row={"RMS_zero_null_contrast_over_truth_SD":float(np.sqrt(np.mean(standardized_zero**2))),
                  "max_abs_zero_null_contrast":float(np.max(np.abs(zero))),"zero_vs_M2_conditional_mean_correlation":correlation,
                  "zero_to_conditional_mean_slope_through_origin":slope,"RMSE":rmse,"relative_RMSE":relative_rmse}
    pd.DataFrame(conditional_rows).to_csv(OUTPUT/"c0_conditional_mean_summary.csv",index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUT/"c0_candidate_mean_pattern.csv",index=False)
    pd.DataFrame(state_rows).to_csv(OUTPUT/"c0_state_grid_mean_pattern.csv",index=False)
    pd.DataFrame(covariance_rows).to_csv(OUTPUT/"c0_centered_covariance.csv",index=False)
    pd.DataFrame([zero_row]).to_csv(OUTPUT/"c0_zero_center_forensic.csv",index=False)

    replicates=pd.read_csv(OUTPUT/"c0_replicates.csv"); max_rows=[]
    for arm,group in replicates.groupby("arm"):
        for layer in ("D2","D4"):
            oracle=float(np.quantile(group[f"observed_{layer}"],0.95))
            for version in ("uncentered","centered"):
                max_rows.append({"arm":arm,"layer":layer,"version":version,"size_005":float(group[f"{version}_{layer}_reject005"].astype(str).str.lower().eq("true").mean()),
                                 "mean_bootstrap_q90":float(group[f"{version}_{layer}_q90"].mean()),"mean_bootstrap_q95":float(group[f"{version}_{layer}_q95"].mean()),
                                 "mean_bootstrap_q99":float(group[f"{version}_{layer}_q99"].mean()),"oracle_q95":oracle,
                                 "q95_ratio":float(group[f"{version}_{layer}_q95"].mean()/oracle),"median_p":float(group[f"{version}_{layer}_p"].median())})
    pd.DataFrame(max_rows).to_csv(OUTPUT/"c0_centering_maxima.csv",index=False)

    with np.load(OUTPUT/"c0_nested_process.npz",allow_pickle=False) as nested_data: process=np.asarray(nested_data["joint_process"],float)
    O,I,C,Z=process.shape; flat=process.reshape(O,I,C*Z); global_mean=np.mean(flat,axis=(0,1)); outer_mean=np.mean(flat,axis=1)
    total_centered=flat-global_mean; within_centered=flat-outer_mean[:,None,:]; between_centered=outer_mean-np.mean(outer_mean,axis=0)
    total_cov=np.einsum("oik,oil->kl",total_centered,total_centered)/(O*I)
    within_cov=np.einsum("oik,oil->kl",within_centered,within_centered)/(O*I)
    between_cov=between_centered.T@between_centered/O
    decomposition_error=float(np.linalg.norm(total_cov-within_cov-between_cov)/np.linalg.norm(total_cov))
    eig_total=np.linalg.eigvalsh(total_cov); eig_within=np.linalg.eigvalsh(within_cov); eig_between=np.linalg.eigvalsh(between_cov)
    point_fraction=float(np.mean(np.diag(within_cov))/np.mean(np.diag(total_cov))); leading_fraction=float(eig_within[-1]/eig_total[-1])
    boundary_mid=float(np.sqrt((144-48)*(240-144)/(240-48)))
    versions={"total":flat,"within_centered":within_centered,"design_mean":outer_mean[:,None,:]}
    nested_rows=[]
    for name,values in versions.items():
        values=values.reshape(-1,C,Z); d4=np.max(np.abs(values),axis=(1,2)); d2=np.max(np.abs(values[:,3]/boundary_mid),axis=1)
        nested_rows.append({"component":name,"draws":len(values),"D2_q90":float(np.quantile(d2,.90)),"D2_q95":float(np.quantile(d2,.95)),"D2_q99":float(np.quantile(d2,.99)),
                            "D4_q90":float(np.quantile(d4,.90)),"D4_q95":float(np.quantile(d4,.95)),"D4_q99":float(np.quantile(d4,.99))})
    nested_summary={"outer_designs":O,"inner_rewards":I,"pointwise_within_variance_fraction":point_fraction,
                    "leading_eigenvalue_within_fraction":leading_fraction,"leading_eigenvalue_between_fraction":float(eig_between[-1]/eig_total[-1]),
                    "covariance_decomposition_frobenius_relative_error":decomposition_error}
    pd.DataFrame([nested_summary]).to_csv(OUTPUT/"c0_nested_variance_decomposition.csv",index=False)
    pd.DataFrame(nested_rows).to_csv(OUTPUT/"c0_nested_oracle_quantiles.csv",index=False)
    np.savez_compressed(OUTPUT/"c0_nested_covariance.npz",total=total_cov.astype(np.float32),within=within_cov.astype(np.float32),between=between_cov.astype(np.float32))

    patients=pd.read_csv(OUTPUT/"c0_patient_decomposition.csv"); design_columns=["phenotype","state_cov_trace","state_mean_distance","state_range_mean","action_rate","candidate_action_composition_range","gram_frobenius_distance","gram_trace"]
    association_rows=[]; attenuated=0
    for column in design_columns:
        m1=float(np.corrcoef(patients[column],np.log1p(patients.M1_patient_process_norm))[0,1]); m2=float(np.corrcoef(patients[column],np.log1p(patients.M2_patient_projection_norm))[0,1]); attenuation=abs(m1)-abs(m2)
        if attenuation>=0.15: attenuated+=1
        association_rows.append({"design_statistic":column,"M1_correlation":m1,"M2_correlation":m2,"absolute_correlation_attenuation":attenuation})
    norm_ratio=float(patients.M2_patient_projection_norm.mean()/patients.M1_patient_process_norm.mean())
    pd.DataFrame(association_rows).to_csv(OUTPUT/"c0_patient_design_associations.csv",index=False)
    pd.DataFrame([{"M2_over_M1_mean_patient_process_norm":norm_ratio,"design_associations_attenuated_by_at_least_0_15":attenuated}]).to_csv(OUTPUT/"c0_patient_norm_summary.csv",index=False)

    gate_cfg=cfg["c0"]["evidence_gate"]; m2_cond=[row for row in conditional_rows if row["arm"]=="M2" and row["scale"]=="raw"][0]
    A=gate_cfg["A"]; evidence_a=(m2_cond["RMS_conditional_mean_over_truth_pointwise_SD"]>=A["conditional_mean_RMS_over_truth_SD_min"] and abs(correlation)>=A["zero_center_abs_correlation_min"] and A["zero_center_slope_range"][0]<=slope<=A["zero_center_slope_range"][1] and relative_rmse<=A["zero_center_relative_RMSE_max"])
    B=gate_cfg["B"]; evidence_b=((point_fraction<=B["within_pointwise_variance_fraction_max"] or leading_fraction<=B["within_leading_eigenvalue_fraction_max"]) and decomposition_error<=B["decomposition_frobenius_relative_error_max"])
    Cgate=gate_cfg["C"]; evidence_c=(norm_ratio<=Cgate["M2_over_M1_mean_patient_process_norm_max"] and attenuated>=Cgate["minimum_design_associations_attenuated_by_0_15"])
    gate={"gate":"DESIGN_VARIATION_HYPOTHESIS_SUPPORTED" if any((evidence_a,evidence_b,evidence_c)) else "DESIGN_VARIATION_HYPOTHESIS_NOT_SUPPORTED",
          "evidence_A":evidence_a,"evidence_B":evidence_b,"evidence_C":evidence_c,"metrics":{"M2_raw_conditional_mean_ratio":m2_cond["RMS_conditional_mean_over_truth_pointwise_SD"],**zero_row,**nested_summary,"M2_over_M1_patient_norm":norm_ratio,"attenuated_associations":attenuated},
          "M3_implementation_allowed":bool(any((evidence_a,evidence_b,evidence_c)))}
    (OUTPUT/"c0_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    maxima=pd.DataFrame(max_rows); covariance=pd.DataFrame(covariance_rows)
    report=["# Phase 2R-C C0 Design-Variation Localization","",f"C0 gate：`{gate['gate']}`","",
            "## Conditional mean与centered maximum","",pd.DataFrame(conditional_rows).to_string(index=False),"",maxima.to_string(index=False),"",
            "## Zero-perturbation finite-design center","",pd.DataFrame([zero_row]).to_string(index=False),"",
            "## Centered covariance vs zero-centered second moment","",covariance.to_string(index=False),"",
            "## Nested variance decomposition","",pd.DataFrame([nested_summary]).to_string(index=False),"",pd.DataFrame(nested_rows).to_string(index=False),"",
            "## Patient trajectory contribution","",pd.DataFrame([{"norm_ratio":norm_ratio,"attenuated_associations":attenuated}]).to_string(index=False),"",pd.DataFrame(association_rows).to_string(index=False),"",
            "## Gate evidence","",json.dumps(gate,indent=2,sort_keys=True),""]
    (WORKSPACE/"report"/"rs_cusum_phase2rc_c0_report.md").write_text("\n".join(report),encoding="utf-8")
    return gate


if __name__=="__main__": print(json.dumps(summarize(),sort_keys=True))
