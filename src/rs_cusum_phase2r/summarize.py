"""Create all predeclared Phase 2R-A forensic summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.rs_cusum.patient_balance import action_feature

from .protocol import OUTPUT, load_config, load_fixed_grid


def _bootstrap_ci(values: np.ndarray, statistic, seed: int, draws: int = 2000) -> tuple[float, float]:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for index in range(draws):
        estimates[index] = statistic(values[rng.integers(0, len(values), len(values))])
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.read_csv(OUTPUT / "truth_replicates.csv")
    D = truth["D"].to_numpy(float); V = truth["V_hat"].to_numpy(float)
    Z = truth["Z"].to_numpy(float)
    sd_mc = float(np.std(D, ddof=1)); var_mc = float(np.var(D, ddof=1))
    ratio = float(np.sqrt(np.mean(V))/sd_mc)
    ratio_ci = _bootstrap_ci(
        np.column_stack([D, V]),
        lambda x: np.sqrt(np.mean(x[:, 1]))/np.std(x[:, 0], ddof=1),
        2026082910,
    )
    sd_ci = _bootstrap_ci(D, lambda x: np.std(x, ddof=1), 2026082911)
    rows = [{
        "layer": "D0_single_candidate_single_point",
        "replicates": len(D), "mean_D": float(np.mean(D)), "mcse_mean_D": float(np.std(D, ddof=1)/np.sqrt(len(D))),
        "Var_MC_D": var_mc, "SD_MC_D": sd_mc, "SD_MC_ci_low": sd_ci[0], "SD_MC_ci_high": sd_ci[1],
        "mean_V_hat": float(np.mean(V)), "median_V_hat": float(np.median(V)),
        "mcse_mean_V_hat": float(np.std(V, ddof=1)/np.sqrt(len(V))),
        "sqrt_mean_V_over_SD_MC": ratio,
        "ratio_ci_low": ratio_ci[0], "ratio_ci_high": ratio_ci[1],
        "mean_sqrt_V_over_SD_MC": float(np.mean(np.sqrt(V))/sd_mc),
        "oracle_abs_q90": float(np.quantile(np.abs(D), 0.90)),
        "oracle_abs_q95": float(np.quantile(np.abs(D), 0.95)),
        "oracle_abs_q99": float(np.quantile(np.abs(D), 0.99)),
    }]
    variance = pd.DataFrame(rows)
    variance.to_csv(OUTPUT / "variance_forensics.csv", index=False)
    zrow = pd.DataFrame([{
        "statistic": "D1_Z",
        "replicates": len(Z), "mean": float(np.mean(Z)), "sd": float(np.std(Z, ddof=1)),
        "median": float(np.median(Z)), "MAD": float(stats.median_abs_deviation(Z, scale=1.0)),
        "skewness": float(stats.skew(Z, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(Z, fisher=True, bias=False)),
        "q025": float(np.quantile(Z, 0.025)), "q05": float(np.quantile(Z, 0.05)),
        "q50": float(np.quantile(Z, 0.50)), "q95": float(np.quantile(Z, 0.95)),
        "q975": float(np.quantile(Z, 0.975)),
        "abs_q90": float(np.quantile(np.abs(Z), 0.90)),
        "abs_q95": float(np.quantile(np.abs(Z), 0.95)),
        "abs_q99": float(np.quantile(np.abs(Z), 0.99)),
    }])
    zrow.to_csv(OUTPUT / "studentized_truth.csv", index=False)
    return variance, zrow


def _linear_metrics(actual: np.ndarray, approximation: np.ndarray, label: str) -> dict:
    actual = np.asarray(actual, dtype=float).ravel(); approximation = np.asarray(approximation, dtype=float).ravel()
    correlation = float(np.corrcoef(actual, approximation)[0, 1])
    slope, intercept = np.polyfit(approximation, actual, 1)
    rmse = float(np.sqrt(np.mean((actual-approximation)**2)))
    return {
        "target": label, "observations": len(actual), "correlation": correlation,
        "slope_actual_on_linearized": float(slope), "intercept": float(intercept),
        "RMSE": rmse, "relative_RMSE": float(rmse/np.sqrt(np.mean(actual**2))),
        "variance_ratio_linearized_over_actual": float(np.var(approximation, ddof=1)/np.var(actual, ddof=1)),
        "sign_agreement": float(np.mean(np.sign(actual)==np.sign(approximation))),
    }


def summarize_linearization() -> pd.DataFrame:
    with np.load(OUTPUT / "truth_forensic_arrays.npz", allow_pickle=False) as data:
        actual = np.asarray(data["actual_error"], dtype=float)
        approximation = np.asarray(data["linearized_error"], dtype=float)
    states, actions = load_fixed_grid(); design = action_feature(states[[0]], actions[[0]])[0]
    rows = [
        _linear_metrics(actual[:, 0], approximation[:, 0], "left_beta_vector"),
        _linear_metrics(actual[:, 1], approximation[:, 1], "right_beta_vector"),
        _linear_metrics(actual[:, 0]-actual[:, 1], approximation[:, 0]-approximation[:, 1], "left_right_beta_contrast_vector"),
        _linear_metrics(actual[:, 0]@design, approximation[:, 0]@design, "left_single_point"),
        _linear_metrics(actual[:, 1]@design, approximation[:, 1]@design, "right_single_point"),
        _linear_metrics((actual[:, 0]-actual[:, 1])@design, (approximation[:, 0]-approximation[:, 1])@design, "left_right_single_point_contrast"),
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "linearization_forensics.csv", index=False)
    return frame


def summarize_decomposition() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUT / "decomposition_replicates.csv")
    rows = []
    for (method, layer), group in frame.groupby(["method", "layer"], sort=True):
        rows.append({
            "method": method, "layer": layer, "replicates": len(group),
            "testable_fraction": float(np.mean(group["status"]=="TESTABLE")),
            "rejection_rate_005": float(group["reject_005"].mean()),
            "mcse_size": float(np.sqrt(group["reject_005"].mean()*(1-group["reject_005"].mean())/len(group))),
            "mean_observed": float(group["observed"].mean()),
            "median_observed": float(group["observed"].median()),
            "mean_bootstrap_q95": float(group["bootstrap_q95"].mean()),
            "median_bootstrap_q95": float(group["bootstrap_q95"].median()),
            "median_p_value": float(group["p_value"].median()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "decomposition_summary.csv", index=False)
    pivot = result.pivot(index="layer", columns="method", values="rejection_rate_005").reset_index()
    pivot.to_csv(OUTPUT / "original_vs_rs.csv", index=False)
    return result


def _scaling_summary(group: pd.DataFrame) -> dict:
    valid = group[group["status"]=="TESTABLE"].copy()
    D = valid["D"].to_numpy(float); V = valid["V_hat"].to_numpy(float)
    result = {
        "replicates": len(group), "testable": len(valid), "testable_fraction": len(valid)/len(group),
        "numerical_failure_rate": 1-len(valid)/len(group),
    }
    if len(valid):
        sd = float(np.std(D, ddof=1)) if len(valid)>1 else np.nan
        result.update({
            "mean_D": float(np.mean(D)), "Var_MC_D": float(np.var(D, ddof=1)), "SD_MC_D": sd,
            "mean_V_hat": float(np.mean(V)), "sqrt_mean_V_over_SD_MC": float(np.sqrt(np.mean(V))/sd),
            "studentized_mean": float(valid["Z"].mean()), "studentized_sd": float(valid["Z"].std(ddof=1)),
            "size_D0_005": float(valid["reject_D0_005"].mean()),
            "size_D1_005": float(valid["reject_D1_005"].mean()),
            "mean_bootstrap_signed_SD": float(valid["bootstrap_raw_sd"].mean()),
            "bootstrap_SD_over_MC_SD": float(valid["bootstrap_raw_sd"].mean()/sd),
            "mean_bootstrap_abs_q95": float(valid["bootstrap_raw_q95_abs"].mean()),
            "mean_W_condition": float(valid["mean_W_condition"].mean()),
            "mean_ridge_condition": float(valid["mean_ridge_condition"].mean()),
            "mean_beta_norm": float(valid["mean_beta_norm"].mean()),
            "max_abs_Q": float(valid["max_abs_Q"].max()),
            "mean_fqi_iterations": float(np.mean(pd.concat([valid["left_iterations"], valid["right_iterations"]]))),
        })
    return result


def summarize_scaling() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(OUTPUT / "scaling_replicates.csv")
    outputs = []
    mapping = {"N": "n_scaling.csv", "T": "t_scaling.csv", "alpha": "alpha_diagnostic.csv"}
    for diagnostic, filename in mapping.items():
        rows = []
        subset = frame[frame["diagnostic"]==diagnostic]
        for value, group in subset.groupby("value", sort=True):
            rows.append({"diagnostic": diagnostic, "value": value, **_scaling_summary(group)})
        result = pd.DataFrame(rows)
        result.to_csv(OUTPUT / filename, index=False)
        outputs.append(result)
    return tuple(outputs)


def summarize_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(OUTPUT / "bootstrap_forensic_replicates.csv")
    truth = pd.read_csv(OUTPUT / "truth_replicates.csv")
    oracle_raw_sd = float(truth["D"].std(ddof=1))
    oracle_raw_q95 = float(np.quantile(np.abs(truth["D"]), 0.95))
    oracle_z_sd = float(truth["Z"].std(ddof=1))
    oracle_z_q95 = float(np.quantile(np.abs(truth["Z"]), 0.95))
    rows = []; oracle_rows = []
    for (distribution, version), group in frame.groupby(["distribution", "variance_version"], sort=True):
        size = float(group["reject_005"].mean())
        target_sd = oracle_raw_sd if version=="D0_raw" else oracle_z_sd
        if version=="D0_raw":
            target_q95 = oracle_raw_q95; oracle_scope = "1000 independent truth datasets"
        elif version in {"B0_fixed", "B1_draw_specific"}:
            target_q95 = oracle_z_q95; oracle_scope = "1000 independent truth datasets"
        else:
            target_q95 = float(np.quantile(group["observed"], 0.95)); oracle_scope = "50 diagnostic datasets (alternate denominator)"
        rows.append({
            "distribution": distribution, "variance_version": version,
            "datasets": len(group), "draws_per_dataset": 1999,
            "rejection_rate_005": size,
            "mean_p_value": float(group["p_value"].mean()), "median_p_value": float(group["p_value"].median()),
            "mean_bootstrap_signed_sd": float(group["bootstrap_signed_sd"].mean()),
            "bootstrap_SD_over_truth_SD": float(group["bootstrap_signed_sd"].mean()/target_sd),
            "mean_bootstrap_q95": float(group["bootstrap_q95"].mean()),
            "mean_unique_draw_values": float(group["unique_values"].mean()),
            "mean_observed_rank": float(group["observed_rank"].mean()),
        })
        oracle_rows.append({
            "distribution": distribution, "variance_version": version,
            "oracle_scope": oracle_scope, "oracle_q95": target_q95,
            "mean_conditional_bootstrap_q95": float(group["bootstrap_q95"].mean()),
            "bootstrap_q95_over_oracle_q95": float(group["bootstrap_q95"].mean()/target_q95),
            "oracle_SD": target_sd,
            "mean_conditional_bootstrap_signed_SD": float(group["bootstrap_signed_sd"].mean()),
            "bootstrap_SD_over_oracle_SD": float(group["bootstrap_signed_sd"].mean()/target_sd),
        })
    result = pd.DataFrame(rows); result.to_csv(OUTPUT / "multiplier_diagnostic.csv", index=False)
    oracle = pd.DataFrame(oracle_rows); oracle.to_csv(OUTPUT / "bootstrap_vs_oracle.csv", index=False)
    return result, oracle


def summarize_full_refit() -> pd.DataFrame | None:
    path = OUTPUT / "full_refit_replicates.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    row = pd.DataFrame([{
        "prototype": "restricted_null_common_Q_full_refit",
        "datasets": len(frame), "draws_each": int(frame["valid_draws"].median()),
        "failed_draw_fraction": float(frame["failed_draws"].sum()/(frame["valid_draws"].sum()+frame["failed_draws"].sum())),
        "rejection_rate_005": float(frame["reject_005"].mean()),
        "mean_bootstrap_sd": float(frame["bootstrap_sd"].mean()),
        "mean_bootstrap_q95": float(frame["bootstrap_q95"].mean()),
        "median_p_value": float(frame["p_value"].median()),
        "max_imposed_null_common_beta_error": float(frame["imposed_null_common_beta_error"].max()),
    }])
    row.to_csv(OUTPUT / "full_refit_summary.csv", index=False)
    return row


def main() -> None:
    load_config()
    summarize_truth()
    summarize_linearization()
    summarize_decomposition()
    summarize_scaling()
    summarize_bootstrap()
    summarize_full_refit()
    print("Phase 2R-A summaries written")


if __name__ == "__main__":
    main()
