"""Deterministic post-run forensics from already frozen and saved Phase 2R arrays.

This module generates no new datasets and performs no model fitting.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.rs_cusum.patient_balance import action_feature

from .core import finite_p, multiplier_draws
from .protocol import OUTPUT, load_config, load_fixed_grid, stable_seed


def truth_calibration(cfg: dict) -> pd.DataFrame:
    truth = pd.read_csv(OUTPUT / "truth_replicates.csv")
    with np.load(OUTPUT / "truth_forensic_arrays.npz", allow_pickle=False) as data:
        contributions = np.asarray(data["single_contributions"], dtype=float)
    B = int(cfg["decomposition"]["decomposition_bootstrap_draws"])
    rows = []
    for replicate, row in truth.iterrows():
        xi = multiplier_draws(
            "gaussian", B, contributions.shape[1],
            stable_seed(cfg["dgp"]["master_seed"], "truth_calibration", replicate),
        )
        raw = np.abs(xi @ contributions[replicate])
        observed0 = abs(float(row.D)); observed1 = abs(float(row.Z))
        standardized = raw / np.sqrt(float(row.V_hat))
        p0 = finite_p(observed0, raw); p1 = finite_p(observed1, standardized)
        rows.append({
            "replicate": replicate, "seed": int(row.seed),
            "D0_observed": observed0, "D1_observed": observed1,
            "p_D0": p0, "p_D1": p1,
            "reject_D0_005": p0 <= 0.05, "reject_D1_005": p1 <= 0.05,
            "bootstrap_raw_signed_sd": float(np.std(xi @ contributions[replicate], ddof=1)),
            "bootstrap_D0_q95": float(np.quantile(raw, 0.95)),
            "bootstrap_D1_q95": float(np.quantile(standardized, 0.95)),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "d0_d1_truth_calibration_replicates.csv", index=False)
    summary = pd.DataFrame([{
        "replicates": len(result), "bootstrap_draws": B,
        "size_D0_005": float(result.reject_D0_005.mean()),
        "size_D1_005": float(result.reject_D1_005.mean()),
        "mcse_size": float(np.sqrt(result.reject_D0_005.mean()*(1-result.reject_D0_005.mean())/len(result))),
        "mean_bootstrap_raw_signed_sd": float(result.bootstrap_raw_signed_sd.mean()),
        "mean_bootstrap_D0_q95": float(result.bootstrap_D0_q95.mean()),
        "mean_bootstrap_D1_q95": float(result.bootstrap_D1_q95.mean()),
        "oracle_D0_q95": float(np.quantile(np.abs(truth.D), 0.95)),
        "oracle_D1_q95": float(np.quantile(np.abs(truth.Z), 0.95)),
    }])
    summary["bootstrap_D0_q95_over_oracle"] = summary.mean_bootstrap_D0_q95/summary.oracle_D0_q95
    summary["bootstrap_D1_q95_over_oracle"] = summary.mean_bootstrap_D1_q95/summary.oracle_D1_q95
    summary.to_csv(OUTPUT / "d0_d1_truth_calibration.csv", index=False)
    return summary


def reconstruct_hessians(cfg: dict) -> None:
    left = np.load(OUTPUT / "truth_design_left.npy", mmap_mode="r")
    right = np.load(OUTPUT / "truth_design_right.npy", mmap_mode="r")
    with np.load(OUTPUT / "truth_forensic_arrays.npz", allow_pickle=False) as data:
        W = np.asarray(data["W"], dtype=float)
    alpha = float(cfg["dgp"]["ridge_alpha_primary"])
    shape = W.shape
    hessian = np.lib.format.open_memmap(
        OUTPUT / "truth_ridge_hessian.npy", mode="w+", dtype=np.float64, shape=shape
    )
    difference = np.lib.format.open_memmap(
        OUTPUT / "truth_W_minus_hessian.npy", mode="w+", dtype=np.float64, shape=shape
    )
    for replicate in range(shape[0]):
        for side_index, design in enumerate((left[replicate], right[replicate])):
            matrix = np.asarray(design, dtype=float)
            H = matrix.T @ matrix / len(matrix) + alpha*np.eye(matrix.shape[1])
            hessian[replicate, side_index] = H
            difference[replicate, side_index] = W[replicate, side_index]-H
    hessian.flush(); difference.flush()


def covariance_forensics() -> pd.DataFrame:
    states, actions = load_fixed_grid(); evaluation = action_feature(states, actions)
    with np.load(OUTPUT / "truth_forensic_arrays.npz", allow_pickle=False) as data:
        beta = np.asarray(data["beta"], dtype=float)
        coefficient = np.asarray(data["coefficient_contributions"], dtype=float)
    beta_diff = beta[:, 0]-beta[:, 1]
    contrast = beta_diff @ evaluation.T
    eval_contribution = np.einsum("rgp,zp->rgz", coefficient, evaluation)
    G = eval_contribution.shape[1]
    variance = G/(G-1.0)*np.sum(eval_contribution**2, axis=1)
    Z = contrast/np.sqrt(variance)
    empirical_cov = np.cov(Z, rowvar=False, ddof=1)
    bootstrap_cov = np.mean(
        np.einsum("rgz,rgw->rzw", eval_contribution, eval_contribution)
        / np.sqrt(variance[:, :, None]*variance[:, None, :]), axis=0
    )
    empirical_corr = np.corrcoef(Z, rowvar=False)
    boot_diag = np.sqrt(np.diag(bootstrap_cov))
    bootstrap_corr = bootstrap_cov/(boot_diag[:, None]*boot_diag[None, :])
    eig_emp = np.linalg.eigvalsh(empirical_cov)
    eig_boot = np.linalg.eigvalsh(bootstrap_cov)
    effective_rank = lambda values: float((np.sum(values)**2)/np.sum(values**2))
    row = pd.DataFrame([{
        "replicates": len(Z), "evaluation_points": Z.shape[1],
        "mean_empirical_point_SD": float(np.mean(np.std(Z, axis=0, ddof=1))),
        "mean_conditional_bootstrap_point_SD": float(np.mean(boot_diag)),
        "mean_point_SD_ratio": float(np.mean(boot_diag)/np.mean(np.std(Z, axis=0, ddof=1))),
        "correlation_frobenius_relative_error": float(np.linalg.norm(empirical_corr-bootstrap_corr)/np.linalg.norm(empirical_corr)),
        "correlation_offdiagonal_MAE": float(np.mean(np.abs((empirical_corr-bootstrap_corr)[~np.eye(Z.shape[1], dtype=bool)]))),
        "empirical_leading_cov_eigenvalue": float(eig_emp[-1]),
        "bootstrap_leading_cov_eigenvalue": float(eig_boot[-1]),
        "leading_eigenvalue_ratio": float(eig_boot[-1]/eig_emp[-1]),
        "empirical_effective_rank": effective_rank(eig_emp),
        "bootstrap_effective_rank": effective_rank(eig_boot),
        "empirical_single_candidate_max_abs_q95": float(np.quantile(np.max(np.abs(Z), axis=1), 0.95)),
    }])
    row.to_csv(OUTPUT / "max_covariance_forensics.csv", index=False)
    np.savez_compressed(
        OUTPUT / "max_covariance_matrices.npz",
        empirical_covariance=empirical_cov,
        mean_conditional_bootstrap_covariance=bootstrap_cov,
        empirical_correlation=empirical_corr,
        mean_conditional_bootstrap_correlation=bootstrap_corr,
    )
    return row


def max_oracle_comparison() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUT / "decomposition_replicates.csv")
    rows = []
    for (method, layer), group in frame.groupby(["method", "layer"], sort=True):
        oracle = float(np.quantile(group.observed, 0.95))
        bootstrap = float(group.bootstrap_q95.mean())
        rows.append({
            "method": method, "layer": layer, "replicates": len(group),
            "size_005": float(group.reject_005.mean()), "oracle_MC_q95": oracle,
            "mean_conditional_bootstrap_q95": bootstrap,
            "bootstrap_q95_over_oracle_q95": bootstrap/oracle,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "max_amplification_forensics.csv", index=False)
    return result


def distributional_distance() -> pd.DataFrame:
    truth = pd.read_csv(OUTPUT / "truth_replicates.csv")
    with np.load(OUTPUT / "highB_bootstrap_draws.npz", allow_pickle=False) as data:
        draws = np.asarray(data["draws"], dtype=float)
        distributions = [str(value) for value in data["distributions"]]
        versions = [str(value) for value in data["versions"]]
    rows = []
    forensic = pd.read_csv(OUTPUT / "bootstrap_forensic_replicates.csv")
    for d_index, distribution in enumerate(distributions):
        for v_index, version in enumerate(versions):
            pooled = draws[:, d_index, v_index].ravel()
            if version == "D0_raw":
                target = np.abs(truth.D.to_numpy(float)); scope = "1000 truth D0"
            elif version in {"B0_fixed", "B1_draw_specific"}:
                target = np.abs(truth.Z.to_numpy(float)); scope = "1000 truth D1"
            else:
                target = forensic[(forensic.distribution==distribution)&(forensic.variance_version==version)].observed.to_numpy(float)
                scope = "50 alternate-denominator observed statistics"
            ks = stats.ks_2samp(target, pooled)
            rows.append({
                "distribution": distribution, "variance_version": version,
                "target_scope": scope, "KS_distance": float(ks.statistic), "KS_p_value": float(ks.pvalue),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "bootstrap_distribution_distance.csv", index=False)
    return result


def forensic_index() -> None:
    files = []
    for path in sorted(OUTPUT.iterdir()):
        if not path.is_file():
            continue
        record = {"file": path.name, "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                record["rows"] = sum(1 for _ in csv.reader(handle))-1
        files.append(record)
    (OUTPUT / "replicate_forensic_index.json").write_text(
        json.dumps({"files": files}, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )


def main() -> None:
    cfg = load_config()
    truth_calibration(cfg)
    reconstruct_hessians(cfg)
    covariance_forensics()
    max_oracle_comparison()
    distributional_distance()
    forensic_index()
    print("Deterministic Phase 2R-A postprocessing complete")


if __name__ == "__main__":
    main()
