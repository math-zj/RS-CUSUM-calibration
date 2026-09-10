"""Summarize frozen hybrid diagnostics and assign the localization gate."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .forensics import _tail_matrices
from .protocol import OUTPUT, load_config


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    values = np.sort(np.unique(np.r_[left, right])); a = np.searchsorted(np.sort(left), values, side="right") / len(left); b = np.searchsorted(np.sort(right), values, side="right") / len(right)
    return float(np.max(np.abs(a - b)))


def _arm_metrics(truth: np.ndarray, arm: np.ndarray, name: str) -> dict[str, Any]:
    truth = truth[np.all(np.isfinite(truth), axis=1)]; total_arm = len(arm); arm = arm[np.all(np.isfinite(arm), axis=1)]
    threshold = np.quantile(np.abs(truth), 0.95, axis=0); te = np.abs(truth) > threshold; ae = np.abs(arm) > threshold
    tm, tj, tc = _tail_matrices(te); am, aj, ac = _tail_matrices(ae); mask = ~np.eye(truth.shape[1], dtype=bool)
    truth_cov = np.cov(truth, rowvar=False, ddof=1); arm_cov = np.cov(arm, rowvar=False, ddof=1)
    truth_sd = np.sqrt(np.maximum(np.diag(truth_cov), 1e-15)); arm_sd = np.sqrt(np.maximum(np.diag(arm_cov), 0))
    truth_eigen = np.linalg.eigvalsh(truth_cov); arm_eigen = np.linalg.eigvalsh(arm_cov)
    tq95 = np.quantile(np.abs(truth), 0.95, axis=0); aq95 = np.quantile(np.abs(arm), 0.95, axis=0)
    tq99 = np.quantile(np.abs(truth), 0.99, axis=0); aq99 = np.quantile(np.abs(arm), 0.99, axis=0)
    truth_max = np.max(np.abs(truth), axis=1); arm_max = np.max(np.abs(arm), axis=1)
    tcenter = truth - truth.mean(0); acenter = arm - arm.mean(0); tz = tcenter / np.maximum(tcenter.std(0), 1e-15); az = acenter / np.maximum(acenter.std(0), 1e-15)
    return {
        "arm": name, "truth_draws": len(truth), "arm_draws": len(arm),
        "failure_rate": 1.0 - len(arm) / max(1, total_arm),
        "mean_difference_RMS_over_truth_SD": float(np.sqrt(np.mean(((arm.mean(0) - truth.mean(0)) / truth_sd) ** 2))),
        "pointwise_SD_ratio": float(np.mean(arm_sd) / np.mean(truth_sd)),
        "leading_eigenvalue_ratio": float(arm_eigen[-1] / truth_eigen[-1]),
        "mean_coordinate_abs_q95_ratio": float(np.mean(aq95 / np.maximum(tq95, 1e-15))),
        "mean_coordinate_abs_q99_ratio": float(np.mean(aq99 / np.maximum(tq99, 1e-15))),
        "skewness_RMSE": float(np.sqrt(np.mean((np.mean(az**3, 0) - np.mean(tz**3, 0)) ** 2))),
        "excess_kurtosis_RMSE": float(np.sqrt(np.mean(((np.mean(az**4, 0) - 3) - (np.mean(tz**4, 0) - 3)) ** 2))),
        "joint_exceedance_frobenius_relative_error": float(np.linalg.norm(aj - tj) / np.linalg.norm(tj)),
        "joint_exceedance_offdiagonal_MAE": float(np.mean(np.abs((aj - tj)[mask]))),
        "tail_dependence_offdiagonal_MAE": float(np.mean(np.abs((ac - tc)[mask]))),
        "truth_tail_dependence_p95": float(np.quantile(tc[mask], 0.95)),
        "arm_tail_dependence_p95": float(np.quantile(ac[mask], 0.95)),
        "tail_dependence_p95_abs_error": float(abs(np.quantile(ac[mask], 0.95) - np.quantile(tc[mask], 0.95))),
        "D4_KS": _ks(truth_max, arm_max),
        "D4_q95_ratio": float(np.quantile(arm_max, 0.95) / np.quantile(truth_max, 0.95)),
        "marginal_probability_MAE": float(np.mean(np.abs(am - tm))),
    }


def summarize_hybrid() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(); status = json.loads((OUTPUT / "hybrid_run_status.json").read_text(encoding="utf-8"))
    with np.load(OUTPUT / "hybrid_process_draws.npz", allow_pickle=False) as data:
        truth = np.asarray(data["true_all"], float).reshape(-1, 448)
        rows = [_arm_metrics(truth, np.asarray(data[arm], float).reshape(-1, 448), arm) for arm in cfg["hybrid"]["arms"]]
    frame = pd.DataFrame(rows); baseline = float(frame.loc[frame.arm == "fitted_all", "tail_dependence_p95_abs_error"].iloc[0])
    frame["p95_error_reduction_vs_fitted_all"] = 1.0 - frame.tail_dependence_p95_abs_error / max(baseline, 1e-15)
    frame["failed_draws_from_runner"] = int(status["failed_draws"])
    frame.to_csv(OUTPUT / "hybrid_component_diagnostics.csv", index=False)
    raw = pd.read_csv(OUTPUT / "hybrid_component_fit_raw.csv")
    group = raw.groupby(["diagnostic_type", "component", "subcomponent"], dropna=False)
    summary = group.agg(
        replicates=("outer_replicate", "nunique"), estimate_mean=("estimate", "mean"), estimate_sd=("estimate", "std"),
        truth=("truth", "mean"), bias=("error", "mean"), rmse=("error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        skewness_mean=("skewness", "mean"), excess_kurtosis_mean=("excess_kurtosis", "mean"),
        lag1_correlation_mean=("lag1_correlation", "mean"), extreme_fraction_mean=("extreme_fraction_abs_gt_2sd", "mean"),
        adjacent_extreme_ratio_mean=("adjacent_extreme_ratio_to_independence", "mean"),
        cross_coordinate_correlation_mean=("mean_abs_cross_coordinate_correlation", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT / "component_fit_diagnostics.csv", index=False)
    return frame, summary


def assign_gate(hybrid: pd.DataFrame) -> dict[str, Any]:
    cfg = load_config(); mc = pd.read_csv(OUTPUT / "mc_uncertainty.csv"); stability = pd.read_csv(OUTPUT / "subsample_stability.csv")
    combined = mc[mc.scheme == "both"].iloc[0]; rules = cfg["mc_uncertainty"]["unstable_if"]
    axis_ranges = stability.groupby("subsample_axis").median_p95_gap.agg(lambda x: float(np.max(x) - np.min(x)))
    max_range = float(axis_ranges.max())
    instability = {
        "mcse": float(combined.mc_standard_error) > float(rules["mc_standard_error_greater_than"]),
        "ci_width": float(combined.ci_high - combined.ci_low) > float(rules["or_confidence_interval_width_greater_than"]),
        "subsample_range": max_range > float(rules["or_nested_subsample_range_greater_than"]),
    }
    baseline = float(hybrid.loc[hybrid.arm == "fitted_all", "tail_dependence_p95_abs_error"].iloc[0])
    singles = hybrid[hybrid.arm.isin(["true_initial", "true_policy", "true_random_effect", "true_transition", "true_reward"])].copy()
    singles = singles.sort_values("p95_error_reduction_vs_fitted_all", ascending=False)
    best = singles.iloc[0]; second = singles.iloc[1]
    rules_local = cfg["localization_rules"]
    if any(instability.values()):
        gate = "TAIL_METRIC_UNSTABLE"
    elif (
        best.p95_error_reduction_vs_fitted_all >= rules_local["A_TAIL_SOURCE_LOCALIZED"]["single_component_error_reduction_min"]
        and best.tail_dependence_p95_abs_error <= rules_local["A_TAIL_SOURCE_LOCALIZED"]["rescued_p95_error_max"]
        and best.p95_error_reduction_vs_fitted_all - second.p95_error_reduction_vs_fitted_all >= rules_local["A_TAIL_SOURCE_LOCALIZED"]["advantage_over_next_single_component_min"]
    ):
        gate = "TAIL_SOURCE_LOCALIZED"
    elif int(np.sum(singles.p95_error_reduction_vs_fitted_all >= 0.20)) >= 2:
        gate = "TAIL_SOURCE_DIFFUSE"
    elif float(hybrid.p95_error_reduction_vs_fitted_all.max()) >= rules_local["B_TAIL_SOURCE_PARTIALLY_LOCALIZED"]["component_or_family_reduction_min"]:
        gate = "TAIL_SOURCE_PARTIALLY_LOCALIZED"
    else:
        gate = "FORENSIC_INCONCLUSIVE"
    payload = {
        "gate": gate, "phase2rd_gate_unchanged": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "mc_instability_checks": instability, "max_subsample_median_range": max_range,
        "hybrid_fitted_all_p95_error": baseline, "best_single_component": str(best.arm),
        "best_single_component_reduction": float(best.p95_error_reduction_vs_fitted_all),
        "D3_allowed": False, "OhioT1DM_allowed": False, "Phase3_allowed": False,
        "transfer_null_allowed": False, "establish_1_0_2": False, "M5_implemented": False,
    }
    (OUTPUT / "phase2re_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    hybrid, _ = summarize_hybrid(); print(json.dumps(assign_gate(hybrid), sort_keys=True))
