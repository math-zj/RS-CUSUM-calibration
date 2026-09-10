"""Confirmatory summaries, covariance diagnostics, continuation gates, and final gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .protocol import OUTPUT, load_config


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total==0: return np.nan, np.nan
    p=successes/total; denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return float(center-half),float(center+half)


def summarize_rows(path: Path, stage: str) -> pd.DataFrame:
    frame=pd.read_csv(path)
    rows=[]
    for (scenario,arm,distribution,layer),group in frame.groupby(["scenario","arm","distribution","layer"],sort=True):
        valid=group[group.status=="TESTABLE"].copy(); n=len(valid)
        # Same observed statistic is repeated across arms; one value per dataset is the oracle sample.
        observed=valid.observed.to_numpy(float)
        row={"stage":stage,"scenario":scenario,"arm":arm,"distribution":distribution,"layer":layer,
             "generated_replicates":len(group),"testable_replicates":n,"testable_fraction":n/len(group)}
        for label,column in (("001","reject_001"),("005","reject_005"),("010","reject_010")):
            if n:
                rejection=valid[column]
                if pd.api.types.is_bool_dtype(rejection):
                    count=int(rejection.sum())
                else:
                    count=int(rejection.astype(str).str.strip().str.lower().eq("true").sum())
            else:
                count=0
            rate=count/n if n else np.nan
            low,high=_wilson(count,n); row.update({f"rejection_count_{label}":count,f"size_{label}":rate,
                f"mcse_{label}":float(np.sqrt(rate*(1-rate)/n)) if n else np.nan,
                f"wilson_low_{label}":low,f"wilson_high_{label}":high})
        row.update({
            "median_p_value":float(valid.p_value.median()) if n else np.nan,
            "pvalue_le_001_fraction":float(np.mean(valid.p_value<=0.01)) if n else np.nan,
            "pvalue_le_005_fraction":float(np.mean(valid.p_value<=0.05)) if n else np.nan,
            "pvalue_le_010_fraction":float(np.mean(valid.p_value<=0.10)) if n else np.nan,
            "mean_bootstrap_q90":float(valid.bootstrap_q90.mean()) if n else np.nan,
            "mean_bootstrap_q95":float(valid.bootstrap_q95.mean()) if n else np.nan,
            "mean_bootstrap_q99":float(valid.bootstrap_q99.mean()) if n else np.nan,
            "oracle_q90":float(np.quantile(observed,0.90)) if n else np.nan,
            "oracle_q95":float(np.quantile(observed,0.95)) if n else np.nan,
            "oracle_q99":float(np.quantile(observed,0.99)) if n else np.nan,
            "mean_bootstrap_q90_over_oracle":float(valid.bootstrap_q90.mean()/np.quantile(observed,0.90)) if n else np.nan,
            "mean_bootstrap_q95_over_oracle":float(valid.bootstrap_q95.mean()/np.quantile(observed,0.95)) if n else np.nan,
            "mean_bootstrap_q99_over_oracle":float(valid.bootstrap_q99.mean()/np.quantile(observed,0.99)) if n else np.nan,
            "mean_runtime_seconds":float(valid.runtime_seconds.mean()) if n else np.nan,
            "total_runtime_seconds":float(valid.runtime_seconds.sum()) if n else np.nan,
            "draw_failure_rate":float(group.failed_draws.sum()/(group.valid_draws.sum()+group.failed_draws.sum())) if (group.valid_draws.sum()+group.failed_draws.sum()) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def covariance_summary(moment_path: Path, stage: str) -> pd.DataFrame:
    with np.load(moment_path,allow_pickle=False) as data:
        observed=np.asarray(data["observed_joint"],dtype=float)
        if observed.size==0: return pd.DataFrame()
        truth_joint=np.cov(observed,rowvar=False,ddof=1)
        observed_midpoint=np.asarray(data["observed_midpoint"],dtype=float)
        if observed_midpoint.size==0:
            raise RuntimeError("midpoint covariance truth is missing")
        truth_mid=np.cov(observed_midpoint,rowvar=False,ddof=1)
        rows=[]
        for name in data.files:
            if not name.endswith("__mean_joint_cov"): continue
            key=name.removesuffix("__mean_joint_cov"); arm,distribution=key.split("__",1)
            joint=np.asarray(data[name],dtype=float); mid=np.asarray(data[f"{key}__mean_mid_cov"],dtype=float)
            joint_mean=np.asarray(data[f"{key}__mean_joint_mean"],dtype=float)
            mid_mean=np.asarray(data[f"{key}__mean_mid_mean"],dtype=float)
            conditional_joint_means=np.asarray(data[f"{key}__conditional_joint_means"],dtype=float)
            conditional_mid_means=np.asarray(data[f"{key}__conditional_mid_means"],dtype=float)
            for scope,truth,boot,mean,conditional_means in (
                ("midpoint_64",truth_mid,mid,mid_mean,conditional_mid_means),
                ("joint_448",truth_joint,joint,joint_mean,conditional_joint_means),
            ):
                sd_truth=np.sqrt(np.maximum(np.diag(truth),0)); sd_boot=np.sqrt(np.maximum(np.diag(boot),0))
                corr_truth=truth/np.maximum(sd_truth[:,None]*sd_truth[None,:],1e-15)
                corr_boot=boot/np.maximum(sd_boot[:,None]*sd_boot[None,:],1e-15)
                eig_truth=np.linalg.eigvalsh(truth); eig_boot=np.linalg.eigvalsh(boot)
                effective=lambda values:float(np.sum(values)**2/np.sum(values**2))
                mask=~np.eye(len(truth),dtype=bool)
                ratios=(eig_boot[-10:]/np.maximum(eig_truth[-10:],1e-15))[::-1]
                rows.append({
                    "stage":stage,"arm":arm,"distribution":distribution,"scope":scope,
                    "mean_pointwise_SD_ratio":float(np.mean(sd_boot)/np.mean(sd_truth)),
                    "leading_eigenvalue_ratio":float(eig_boot[-1]/eig_truth[-1]),
                    "leading_10_eigenvalue_ratios":json.dumps(ratios.tolist()),
                    "truth_effective_rank":effective(eig_truth),"bootstrap_effective_rank":effective(eig_boot),
                    "effective_rank_ratio":effective(eig_boot)/effective(eig_truth),
                    "correlation_frobenius_relative_error":float(np.linalg.norm(corr_truth-corr_boot)/np.linalg.norm(corr_truth)),
                    "offdiagonal_correlation_MAE":float(np.mean(np.abs((corr_truth-corr_boot)[mask]))),
                    "conditional_process_mean_norm":float(np.linalg.norm(mean)),
                    "conditional_process_mean_RMS":float(np.sqrt(np.mean(conditional_means**2))),
                    "conditional_mean_RMS_over_truth_mean_SD":float(
                        np.sqrt(np.mean(conditional_means**2))/np.mean(sd_truth)
                    ),
                })
        return pd.DataFrame(rows)


def pvalue_distributions(path: Path, stage: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    frame=pd.read_csv(path); frame=frame[frame.status=="TESTABLE"].copy()
    bins=np.asarray([0,0.01,0.025,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0000001])
    grid=np.asarray([0,0.01,0.025,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.975,0.99,1.0])
    hist_rows=[]; ecdf_rows=[]
    keys=["scenario","arm","distribution","layer"]
    for key,group in frame.groupby(keys,sort=True):
        values=group.p_value.to_numpy(float); counts,_=np.histogram(values,bins=bins)
        base={"stage":stage,**dict(zip(keys,key)),"n":len(values)}
        for index,count in enumerate(counts):
            hist_rows.append({**base,"bin_left":bins[index],"bin_right":min(bins[index+1],1.0),
                              "count":int(count),"fraction":float(count/len(values))})
        for threshold in grid:
            ecdf_rows.append({**base,"p_threshold":threshold,"ecdf":float(np.mean(values<=threshold))})
    return pd.DataFrame(hist_rows),pd.DataFrame(ecdf_rows)


def restricted_null_audit(path: Path, stage: str) -> pd.DataFrame:
    frame=pd.read_csv(path)
    frame=frame[(frame.arm=="M2")&(frame.layer=="D4")].copy()
    rows=[]
    for (scenario,distribution),group in frame.groupby(["scenario","distribution"],sort=True):
        diagnostics=[json.loads(value) for value in group.diagnostics]
        extract=lambda name:np.asarray([item[name] for item in diagnostics],dtype=float)
        rows.append({
            "stage":stage,"scenario":scenario,"distribution":distribution,"datasets":len(group),
            "max_zero_refit_error":float(np.max(extract("zero_refit_error"))),
            "max_ridge_score_error":float(np.max(extract("ridge_score_error"))),
            "mean_left_right_beta_variance_ratio":float(np.mean(extract("left_right_beta_variance_ratio"))),
            "median_left_right_beta_variance_ratio":float(np.median(extract("left_right_beta_variance_ratio"))),
            "mean_conditional_joint_mean_norm":float(np.mean(extract("conditional_joint_mean_norm"))),
            "minimum_refit_count":int(np.min(extract("refit_count"))),
            "fraction_positive_W_checksum_SD":float(np.mean(extract("W_checksum_sd")>0)),
            "draw_failure_rate":float(group.failed_draws.sum()/(group.valid_draws.sum()+group.failed_draws.sum())),
        })
    return pd.DataFrame(rows)


def _write_combined() -> None:
    mappings=[
        ("rb0_replicate_results.csv","RB0"),("rb1_replicate_results.csv","RB1"),
        ("multiplier_sensitivity_replicate_results.csv","RB1_SENSITIVITY"),
        ("rb2_holdout_replicate_results.csv","RB2"),("transfer_null_replicate_results.csv","TRANSFER"),
    ]
    frames=[]
    for filename,_ in mappings:
        path=OUTPUT/filename
        if path.is_file(): frames.append(pd.read_csv(path))
    if frames: pd.concat(frames,ignore_index=True).to_csv(OUTPUT/"replicate_results.csv",index=False)


def summarize_rb1() -> dict[str, Any]:
    cfg=load_config(); summary=summarize_rows(OUTPUT/"rb1_replicate_results.csv","RB1")
    summary.to_csv(OUTPUT/"rb1_summary.csv",index=False)
    cov=covariance_summary(OUTPUT/"rb1_process_moments.npz","RB1")
    cov.to_csv(OUTPUT/"covariance_process.csv",index=False)
    summary.to_csv(OUTPUT/"decomposition.csv",index=False)
    summary[[c for c in summary.columns if "q" in c or c in {"stage","scenario","arm","distribution","layer"}]].to_csv(OUTPUT/"quantile_comparison.csv",index=False)
    pcols=["stage","scenario","arm","distribution","layer","median_p_value","pvalue_le_001_fraction","pvalue_le_005_fraction","pvalue_le_010_fraction"]
    summary[pcols].to_csv(OUTPUT/"pvalue_diagnostics.csv",index=False)
    histogram,ecdf=pvalue_distributions(OUTPUT/"rb1_replicate_results.csv","RB1")
    histogram.to_csv(OUTPUT/"pvalue_histogram.csv",index=False)
    ecdf.to_csv(OUTPUT/"pvalue_ecdf.csv",index=False)
    restricted_null_audit(OUTPUT/"rb1_replicate_results.csv","RB1").to_csv(
        OUTPUT/"restricted_null_audit.csv",index=False
    )
    continuing=[]; details={}
    gate_cfg=cfg["rb1_continuation"]
    for arm in ("M1","M2"):
        subset=summary[(summary.arm==arm)&(summary.distribution=="gaussian")].set_index("layer")
        checks={
            "D0_size":float(subset.loc["D0","size_005"])<=float(gate_cfg["D0_size_005_max"]),
            "D2_size":float(subset.loc["D2","size_005"])<=float(gate_cfg["D2_size_005_max"]),
            "D4_size":float(subset.loc["D4","size_005"])<=float(gate_cfg["D4_size_005_max"]),
            "D4_q95_ratio":float(subset.loc["D4","mean_bootstrap_q95_over_oracle"])>=float(gate_cfg["D4_q95_ratio_min"]),
            "failure":float(subset.loc["D4","draw_failure_rate"])<=float(gate_cfg["failure_rate_max"]),
        }
        details[arm]={"checks":checks,"pass":all(checks.values())}
        if all(checks.values()): continuing.append(arm)
    gate={"gate":"RB1_CONTINUE" if continuing else "PHASE_2RB_REVISION_FAILED",
          "continuing_candidates":continuing,"candidate_checks":details}
    (OUTPUT/"rb1_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _write_combined(); return gate


def summarize_sensitivity() -> None:
    summary=summarize_rows(OUTPUT/"multiplier_sensitivity_replicate_results.csv","RB1_SENSITIVITY")
    summary.to_csv(OUTPUT/"multiplier_sensitivity.csv",index=False)
    existing=pd.read_csv(OUTPUT/"decomposition.csv"); pd.concat([existing,summary],ignore_index=True).to_csv(OUTPUT/"decomposition.csv",index=False)
    cov=covariance_summary(OUTPUT/"multiplier_sensitivity_process_moments.npz","RB1_SENSITIVITY")
    pd.concat([pd.read_csv(OUTPUT/"covariance_process.csv"),cov],ignore_index=True).to_csv(OUTPUT/"covariance_process.csv",index=False)
    histogram,ecdf=pvalue_distributions(
        OUTPUT/"multiplier_sensitivity_replicate_results.csv","RB1_SENSITIVITY"
    )
    pd.concat([pd.read_csv(OUTPUT/"pvalue_histogram.csv"),histogram],ignore_index=True).to_csv(OUTPUT/"pvalue_histogram.csv",index=False)
    pd.concat([pd.read_csv(OUTPUT/"pvalue_ecdf.csv"),ecdf],ignore_index=True).to_csv(OUTPUT/"pvalue_ecdf.csv",index=False)
    audit=restricted_null_audit(OUTPUT/"multiplier_sensitivity_replicate_results.csv","RB1_SENSITIVITY")
    pd.concat([pd.read_csv(OUTPUT/"restricted_null_audit.csv"),audit],ignore_index=True).to_csv(OUTPUT/"restricted_null_audit.csv",index=False)
    _write_combined()


def summarize_rb2() -> dict[str,Any]:
    cfg=load_config(); summary=summarize_rows(OUTPUT/"rb2_holdout_replicate_results.csv","RB2")
    summary.to_csv(OUTPUT/"rb2_holdout_summary.csv",index=False)
    cov=covariance_summary(OUTPUT/"rb2_holdout_process_moments.npz","RB2")
    pd.concat([pd.read_csv(OUTPUT/"covariance_process.csv"),cov],ignore_index=True).to_csv(OUTPUT/"covariance_process.csv",index=False)
    pd.concat([pd.read_csv(OUTPUT/"decomposition.csv"),summary],ignore_index=True).to_csv(OUTPUT/"decomposition.csv",index=False)
    histogram,ecdf=pvalue_distributions(OUTPUT/"rb2_holdout_replicate_results.csv","RB2")
    pd.concat([pd.read_csv(OUTPUT/"pvalue_histogram.csv"),histogram],ignore_index=True).to_csv(OUTPUT/"pvalue_histogram.csv",index=False)
    pd.concat([pd.read_csv(OUTPUT/"pvalue_ecdf.csv"),ecdf],ignore_index=True).to_csv(OUTPUT/"pvalue_ecdf.csv",index=False)
    audit=restricted_null_audit(OUTPUT/"rb2_holdout_replicate_results.csv","RB2")
    pd.concat([pd.read_csv(OUTPUT/"restricted_null_audit.csv"),audit],ignore_index=True).to_csv(OUTPUT/"restricted_null_audit.csv",index=False)
    gate_cfg=cfg["rb2_acceptance"]; covariance_cfg=cfg["covariance_confirmation"]
    covariance_table=pd.read_csv(OUTPUT/"covariance_process.csv")
    confirmed=[]; details={}
    for arm in sorted(summary.arm.unique()):
        subset=summary[(summary.arm==arm)&(summary.distribution=="gaussian")].set_index("layer")
        d4_size=float(subset.loc["D4","size_005"])
        d4_mcse=float(subset.loc["D4","mcse_005"])
        d4_wilson_low=float(subset.loc["D4","wilson_low_005"])
        d4_wilson_high=float(subset.loc["D4","wilson_high_005"])
        covariance_checks={}; relative_checks={}
        for scope in covariance_cfg["scopes"]:
            rb2_cov=covariance_table[
                (covariance_table.stage=="RB2")&(covariance_table.arm==arm)&
                (covariance_table.distribution=="gaussian")&(covariance_table.scope==scope)
            ].iloc[0]
            absolute=covariance_cfg["rb2_absolute"]
            covariance_checks[scope]={
                "pointwise_SD":float(absolute["pointwise_SD_ratio_range"][0])<=float(rb2_cov.mean_pointwise_SD_ratio)<=float(absolute["pointwise_SD_ratio_range"][1]),
                "leading_eigenvalue":float(absolute["leading_eigenvalue_ratio_range"][0])<=float(rb2_cov.leading_eigenvalue_ratio)<=float(absolute["leading_eigenvalue_ratio_range"][1]),
                "effective_rank":float(absolute["effective_rank_ratio_range"][0])<=float(rb2_cov.effective_rank_ratio)<=float(absolute["effective_rank_ratio_range"][1]),
                "correlation_error":float(rb2_cov.correlation_frobenius_relative_error)<=float(absolute["correlation_frobenius_error_max"]),
                "offdiagonal_MAE":float(rb2_cov.offdiagonal_correlation_MAE)<=float(absolute["offdiagonal_correlation_MAE_max"]),
                "conditional_mean":float(rb2_cov.conditional_mean_RMS_over_truth_mean_SD)<=float(absolute["conditional_mean_RMS_over_truth_mean_SD_max"]),
            }
            rb1_arm=covariance_table[
                (covariance_table.stage=="RB1")&(covariance_table.arm==arm)&
                (covariance_table.distribution=="gaussian")&(covariance_table.scope==scope)
            ].iloc[0]
            rb1_m0=covariance_table[
                (covariance_table.stage=="RB1")&(covariance_table.arm=="M0")&
                (covariance_table.distribution=="gaussian")&(covariance_table.scope==scope)
            ].iloc[0]
            candidate_errors=np.asarray([
                abs(np.log(float(rb1_arm.mean_pointwise_SD_ratio))),
                abs(np.log(float(rb1_arm.leading_eigenvalue_ratio))),
                abs(np.log(float(rb1_arm.effective_rank_ratio))),
                float(rb1_arm.correlation_frobenius_relative_error),
                float(rb1_arm.offdiagonal_correlation_MAE),
            ])
            reference_errors=np.asarray([
                abs(np.log(float(rb1_m0.mean_pointwise_SD_ratio))),
                abs(np.log(float(rb1_m0.leading_eigenvalue_ratio))),
                abs(np.log(float(rb1_m0.effective_rank_ratio))),
                float(rb1_m0.correlation_frobenius_relative_error),
                float(rb1_m0.offdiagonal_correlation_MAE),
            ])
            ratios=candidate_errors/np.maximum(reference_errors,1e-15)
            relative=covariance_cfg["rb1_relative_to_M0"]
            relative_checks[scope]={
                "metrics_closer_or_lower":int(np.sum(candidate_errors<=reference_errors)),
                "minimum_met":int(np.sum(candidate_errors<=reference_errors))>=int(relative["minimum_metrics_closer_or_lower"]),
                "maximum_relative_degradation":float(np.max(ratios-1.0)),
                "degradation_limit_met":float(np.max(ratios-1.0))<=float(relative["maximum_relative_degradation_any_metric"]),
            }
        covariance_pass=all(
            all(values.values()) for values in covariance_checks.values()
        ) and all(
            values["minimum_met"] and values["degradation_limit_met"]
            for values in relative_checks.values()
        )
        checks={
            "D0_range":float(gate_cfg["D0_size_range"][0])<=float(subset.loc["D0","size_005"])<=float(gate_cfg["D0_size_range"][1]),
            "D1":float(subset.loc["D1","size_005"])<=float(gate_cfg["D1_size_max"]),
            "D2":float(subset.loc["D2","size_005"])<=float(gate_cfg["D2_size_max"]),
            "D3":float(subset.loc["D3","size_005"])<=float(gate_cfg["D3_size_max"]),
            "D4":float(subset.loc["D4","size_005"])<=float(gate_cfg["D4_size_max"]),
            "D4_q95_ratio":float(gate_cfg["D4_q95_ratio_range"][0])<=float(subset.loc["D4","mean_bootstrap_q95_over_oracle"])<=float(gate_cfg["D4_q95_ratio_range"][1]),
            "near_zero":float(subset.loc["D4","size_001"])<=float(gate_cfg["size_001_max"]),
            "failure":float(subset.loc["D4","draw_failure_rate"])<=float(gate_cfg["failure_rate_max"]),
            "nominal_within_or_near_MC_uncertainty":(
                d4_wilson_low<=0.05<=d4_wilson_high or abs(d4_size-0.05)<=d4_mcse
            ),
            "covariance_confirmation":covariance_pass,
        }
        details[arm]={"checks":checks,"covariance_absolute":covariance_checks,
                      "covariance_relative_to_M0":relative_checks,"pass":all(checks.values())}
        if all(checks.values()): confirmed.append(arm)
    gate={"gate":"RB2_HOLDOUT_PASS" if confirmed else "RB2_HOLDOUT_FAIL",
          "confirmed_candidates":confirmed,"candidate_checks":details,"selected_candidate":confirmed[0] if len(confirmed)==1 else None}
    (OUTPUT/"rb2_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _write_combined(); return gate


def summarize_transfer() -> dict[str,Any]:
    cfg=load_config(); summary=summarize_rows(OUTPUT/"transfer_null_replicate_results.csv","TRANSFER")
    d4=summary[summary.layer=="D4"].copy(); d4.to_csv(OUTPUT/"transfer_null_summary.csv",index=False)
    histogram,ecdf=pvalue_distributions(OUTPUT/"transfer_null_replicate_results.csv","TRANSFER")
    pd.concat([pd.read_csv(OUTPUT/"pvalue_histogram.csv"),histogram],ignore_index=True).to_csv(OUTPUT/"pvalue_histogram.csv",index=False)
    pd.concat([pd.read_csv(OUTPUT/"pvalue_ecdf.csv"),ecdf],ignore_index=True).to_csv(OUTPUT/"pvalue_ecdf.csv",index=False)
    audit=restricted_null_audit(OUTPUT/"transfer_null_replicate_results.csv","TRANSFER")
    pd.concat([pd.read_csv(OUTPUT/"restricted_null_audit.csv"),audit],ignore_index=True).to_csv(OUTPUT/"restricted_null_audit.csv",index=False)
    confirmed=json.loads((OUTPUT/"rb2_gate.json").read_text(encoding="utf-8"))["confirmed_candidates"]
    passing=[]; details={}
    for arm in confirmed:
        subset=d4[(d4.arm==arm)&(d4.distribution=="gaussian")]
        scenario_sizes={row.scenario:float(row.size_005) for row in subset.itertuples()}
        passes=all(value<=0.15 for value in scenario_sizes.values()) and len(scenario_sizes)==len(cfg["stages"]["transfer_null"]["scenarios"])
        details[arm]={"scenario_sizes":scenario_sizes,"pass":passes,
                      "yellow_flags":[scenario for scenario,value in scenario_sizes.items() if 0.08<value<=0.15],
                      "failures":[scenario for scenario,value in scenario_sizes.items() if value>0.15]}
        if passes: passing.append(arm)
    selected=None
    if len(passing)==1: selected=passing[0]
    elif set(passing)=={"M1","M2"}:
        rb2=pd.read_csv(OUTPUT/"rb2_holdout_summary.csv")
        cov=pd.read_csv(OUTPUT/"covariance_process.csv")
        m1q=float(rb2[(rb2.arm=="M1")&(rb2.layer=="D4")].mean_bootstrap_q95_over_oracle.iloc[0])
        m1err=float(cov[(cov.stage=="RB2")&(cov.arm=="M1")&(cov.scope=="joint_448")].correlation_frobenius_relative_error.iloc[0])
        m2err=float(cov[(cov.stage=="RB2")&(cov.arm=="M2")&(cov.scope=="joint_448")].correlation_frobenius_relative_error.iloc[0])
        m1sizes=details["M1"]["scenario_sizes"]; m2sizes=details["M2"]["scenario_sizes"]
        simple=(0.90<=m1q<=1.10 and m1err<=1.10*m2err and all(m1sizes[s]<=m2sizes[s]+0.03 for s in m1sizes))
        selected="M1" if simple else "M2"
    if selected:
        final_gate="INFERENCE_REVISION_CONFIRMED"
    elif passing:
        final_gate="INFERENCE_REVISION_PARTIAL"
    else:
        final_gate="INFERENCE_REVISION_PARTIAL" if confirmed else "PHASE_2RB_REVISION_FAILED"
    gate={"gate":final_gate,"transfer_passing_candidates":passing,"selected_candidate":selected,
          "candidate_transfer_checks":details,"establish_1_0_2":bool(selected),
          "restart_new_phase2_calibration_allowed":bool(selected),
          "phase3_allowed":False,"ohiot1dm_allowed":False}
    (OUTPUT/"phase2rb_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _write_combined(); return gate


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["rb1","sensitivity","rb2","transfer"]); args=parser.parse_args()
    {"rb1":summarize_rb1,"sensitivity":summarize_sensitivity,"rb2":summarize_rb2,"transfer":summarize_transfer}[args.stage]()


if __name__=="__main__": main()
