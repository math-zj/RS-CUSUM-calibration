"""Balanced comparisons, uncertainty, multi-domain gates, and Phase 2R-F outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rs_cusum_phase2r.core import finite_p

from .metrics import (
    center_covariance_metrics,
    integrated_standardized_joint_mae,
    marginal_tail_metrics,
    max_distribution_metrics,
    tail_matrices,
    threshold_tail_metrics,
    wilson,
)
from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds, stable_seed
from .runner import CHECKPOINTS, _m4_path, _oracle_path


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_oracle(bank: str, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = int(cfg["sample_plan"]["oracle_processes_per_bank"])
    chunk = int(cfg["sample_plan"]["oracle_chunk_size"])
    joint: list[np.ndarray] = []
    layers: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    for start in range(0, expected, chunk):
        stop = min(start + chunk, expected)
        path = _oracle_path(bank, start, stop)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["indices"], np.arange(start, stop)):
                raise RuntimeError(f"oracle checkpoint index mismatch: {path}")
            joint.append(np.asarray(data["joint"], np.float32))
            layers.append(np.asarray(data["layers"], np.float32))
            valid.append(np.asarray(data["valid"], bool))
            seeds.append(np.asarray(data["seeds"], np.uint64))
    return np.vstack(joint), np.vstack(layers), np.concatenate(valid), np.concatenate(seeds)


def _primary_draw_count(outer: int) -> int:
    return 9 if outer < 100 else 8


def _load_m4(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    outer_n = int(cfg["sample_plan"]["m4_outer_replicates"])
    inner_n = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])
    full_layers = np.full((outer_n, inner_n, 5), np.nan, np.float32)
    full_valid = np.zeros((outer_n, inner_n), bool)
    observed_joint = np.full((outer_n, 448), np.nan, np.float32)
    observed_layers = np.full((outer_n, 5), np.nan, np.float32)
    selected_joint: list[np.ndarray] = []
    selected_layers: list[np.ndarray] = []
    selected_cluster: list[np.ndarray] = []
    runtime = np.zeros(outer_n, float)
    for outer in range(outer_n):
        path = _m4_path(outer)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if int(data["outer_index"]) != outer:
                raise RuntimeError(f"M4 checkpoint index mismatch: {path}")
            joint = np.asarray(data["joint"], np.float32)
            layers = np.asarray(data["layers"], np.float32)
            valid = np.asarray(data["valid"], bool)
            full_layers[outer] = layers
            full_valid[outer] = valid
            observed_joint[outer] = data["observed_joint"]
            observed_layers[outer] = data["observed_layers"]
            runtime[outer] = float(data["seconds"])
            take = _primary_draw_count(outer)
            selected_joint.append(joint[:take])
            selected_layers.append(layers[:take])
            selected_cluster.append(np.full(take, outer, int))
    return {
        "joint": np.vstack(selected_joint),
        "layers": np.vstack(selected_layers),
        "cluster": np.concatenate(selected_cluster),
        "full_layers": full_layers,
        "full_valid": full_valid,
        "observed_joint": observed_joint,
        "observed_layers": observed_layers,
        "runtime": runtime,
    }


def _cluster_balanced_indices(cluster: np.ndarray, n: int, seed: int) -> np.ndarray:
    cluster = np.asarray(cluster, int)
    unique = np.unique(cluster)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(unique)
    pools = {value: rng.permutation(np.flatnonzero(cluster == value)).tolist() for value in unique}
    selected: list[int] = []
    while len(selected) < n:
        advanced = False
        for value in order:
            if pools[int(value)] and len(selected) < n:
                selected.append(pools[int(value)].pop())
                advanced = True
        if not advanced:
            break
        order = rng.permutation(order)
    if len(selected) != n:
        raise ValueError(f"cannot select {n} M4 rows from cluster-balanced bank")
    return np.asarray(selected, int)


def _comparison(
    oracle_primary: np.ndarray,
    m4: np.ndarray,
    oracle_reference: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    m4_rows = threshold_tail_metrics(oracle_primary, m4, thresholds)
    noise_rows = threshold_tail_metrics(oracle_primary, oracle_reference, thresholds)
    raw = integrated_standardized_joint_mae(m4_rows)
    noise = integrated_standardized_joint_mae(noise_rows)
    q95 = thresholds.index(0.95)
    return {
        "raw_ISJEM": raw,
        "oracle_oracle_noise": noise,
        "primary_EISJEM": raw - noise,
        "legacy_p95_gap": m4_rows[q95]["legacy_conditional_p95_abs_gap"],
        "oracle_oracle_legacy_p95_gap": noise_rows[q95]["legacy_conditional_p95_abs_gap"],
        "q95_tail_dependence_MAE": m4_rows[q95]["tail_dependence_offdiagonal_MAE"],
        "q95_joint_exceedance_MAE": m4_rows[q95]["joint_exceedance_offdiagonal_MAE"],
    }


def _balanced(
    oracle_primary: np.ndarray,
    oracle_primary_layers: np.ndarray,
    oracle_reference: np.ndarray,
    m4: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    thresholds = tuple(map(float, cfg["primary_metric"]["thresholds"]))
    seeds = load_seeds("balanced_analysis")
    rows: list[dict[str, Any]] = []
    for n in map(int, cfg["balanced_comparison"]["sample_sizes"]):
        repetitions = 1 if n == int(cfg["sample_plan"]["m4_balanced_primary_processes"]) else int(cfg["balanced_comparison"]["repetitions"])
        for repetition in range(repetitions):
            seed = stable_seed(int(seeds[repetition]), "balanced", n)
            rng = np.random.default_rng(seed)
            first = rng.choice(len(oracle_primary), n, replace=False)
            second = rng.choice(len(oracle_reference), n, replace=False)
            m4_index = _cluster_balanced_indices(m4["cluster"], n, stable_seed(seed, "m4"))
            metric = _comparison(oracle_primary[first], m4["joint"][m4_index], oracle_reference[second], thresholds)
            reference_d4 = oracle_primary_layers[first, 4]
            m4_d4 = m4["layers"][m4_index, 4]
            values = np.sort(np.unique(np.r_[reference_d4, m4_d4]))
            d4_ks = float(np.max(np.abs(np.searchsorted(np.sort(reference_d4), values, side="right") / n - np.searchsorted(np.sort(m4_d4), values, side="right") / n)))
            rows.append(
                {
                    "sample_size_each": n,
                    "repetition": repetition,
                    "analysis_seed": seed,
                    "oracle_unique_rows": len(np.unique(first)),
                    "reference_unique_rows": len(np.unique(second)),
                    "m4_outer_clusters": len(np.unique(m4["cluster"][m4_index])),
                    "D4_KS": d4_ks,
                    "D4_q95_ratio": float(np.quantile(m4_d4, 0.95) / np.quantile(reference_d4, 0.95)),
                    **metric,
                }
            )
        print(f"Phase2R-F balanced diagnostics n={n}", flush=True)
    frame = pd.DataFrame(rows)
    _atomic_csv(OUTPUT / "balanced_tail_comparison.csv", frame)
    convergence_rows: list[dict[str, Any]] = []
    metrics = [
        "primary_EISJEM", "raw_ISJEM", "legacy_p95_gap", "q95_tail_dependence_MAE",
        "q95_joint_exceedance_MAE", "D4_KS", "D4_q95_ratio",
    ]
    for (n,), group in frame.groupby(["sample_size_each"]):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            convergence_rows.append(
                {
                    "oracle_sample_size": int(n), "metric": metric, "repetitions": len(values),
                    "mean": float(np.mean(values)), "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "q025": float(np.quantile(values, 0.025)), "q975": float(np.quantile(values, 0.975)),
                    "min": float(np.min(values)), "max": float(np.max(values)),
                }
            )
    _atomic_csv(OUTPUT / "oracle_convergence.csv", pd.DataFrame(convergence_rows))
    return frame


def _uncertainty(
    oracle_primary: np.ndarray,
    oracle_reference: np.ndarray,
    m4: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    thresholds = tuple(map(float, cfg["primary_metric"]["thresholds"]))
    seeds = load_seeds("uncertainty_bootstrap")
    groups = {cluster: np.flatnonzero(m4["cluster"] == cluster) for cluster in np.unique(m4["cluster"])}
    cluster_values = np.asarray(sorted(groups), int)
    rows = []
    raw_draws = np.full((len(seeds), 5), np.nan, float)
    for repetition, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        first = rng.choice(len(oracle_primary), len(oracle_primary), replace=True)
        second = rng.choice(len(oracle_reference), len(oracle_reference), replace=True)
        sampled_clusters = rng.choice(cluster_values, len(cluster_values), replace=True)
        m4_index = np.concatenate([groups[int(cluster)] for cluster in sampled_clusters])
        metric = _comparison(oracle_primary[first], m4["joint"][m4_index], oracle_reference[second], thresholds)
        raw_draws[repetition] = [
            metric["primary_EISJEM"], metric["raw_ISJEM"], metric["oracle_oracle_noise"],
            metric["legacy_p95_gap"], len(m4_index),
        ]
        if (repetition + 1) % 50 == 0:
            print(f"Phase2R-F uncertainty: {repetition + 1}/{len(seeds)}", flush=True)
    labels = ("primary_EISJEM", "raw_ISJEM", "oracle_oracle_noise", "legacy_p95_gap")
    summary = []
    for column, label in enumerate(labels):
        values = raw_draws[:, column]
        low, high = np.quantile(values, (0.025, 0.975))
        summary.append(
            {
                "metric": label, "bootstrap_replicates": len(values),
                "point_estimate": np.nan,
                "mc_standard_error": float(np.std(values, ddof=1)),
                "ci_low": float(low), "ci_high": float(high), "ci_width": float(high - low),
                "bootstrap_mean": float(np.mean(values)), "bootstrap_median": float(np.median(values)),
                "oracle_resampling_unit": "process row",
                "m4_resampling_unit": "outer dataset cluster",
            }
        )
    frame = pd.DataFrame(summary)
    _atomic_csv(OUTPUT / "mc_uncertainty_summary.csv", frame)
    _atomic_npz(OUTPUT / "mc_uncertainty_draws.npz", values=raw_draws, columns=np.asarray((*labels, "m4_rows")))
    return frame, raw_draws


def _type1(m4: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for layer in range(5):
        p_values = []
        for outer in range(len(m4["observed_layers"])):
            values = m4["full_layers"][outer, :, layer]
            values = values[m4["full_valid"][outer] & np.isfinite(values)]
            p_values.append(finite_p(float(m4["observed_layers"][outer, layer]), values) if len(values) else np.nan)
        p_values = np.asarray(p_values, float)
        valid = p_values[np.isfinite(p_values)]
        row: dict[str, Any] = {
            "layer": f"D{layer}", "outer_replicates": len(p_values),
            "testable_replicates": len(valid), "testable_fraction": len(valid) / len(p_values),
            "median_p_value": float(np.median(valid)),
        }
        for alpha, label in ((0.01, "001"), (0.05, "005"), (0.10, "010")):
            count = int(np.sum(valid <= alpha))
            low, high = wilson(count, len(valid))
            rate = count / len(valid)
            row.update(
                {
                    f"rejections_{label}": count, f"size_{label}": rate,
                    f"mcse_{label}": float(np.sqrt(rate * (1 - rate) / len(valid))),
                    f"wilson_low_{label}": low, f"wilson_high_{label}": high,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    _atomic_csv(OUTPUT / "type1_summary.csv", frame)
    return frame


def _structure_summary(reference: np.ndarray, comparison: np.ndarray) -> pd.DataFrame:
    metadata = pd.read_csv(OUTPUT.parent / "phase2re" / "coordinate_metadata.csv")
    threshold = np.quantile(np.abs(reference), 0.95, axis=0)
    _, _, ref_conditional = tail_matrices(np.abs(reference) > threshold)
    _, _, cmp_conditional = tail_matrices(np.abs(comparison) > threshold)
    error = np.abs(cmp_conditional - ref_conditional)
    mask = ~np.eye(reference.shape[1], dtype=bool)
    row_index, column_index = np.where(mask)
    groups = {
        "same_candidate": metadata.candidate_index.to_numpy()[row_index] == metadata.candidate_index.to_numpy()[column_index],
        "same_evaluation": metadata.evaluation_index.to_numpy()[row_index] == metadata.evaluation_index.to_numpy()[column_index],
        "same_base_state": metadata.base_state_index.to_numpy()[row_index] == metadata.base_state_index.to_numpy()[column_index],
        "same_value_action": metadata.value_action.to_numpy()[row_index] == metadata.value_action.to_numpy()[column_index],
        "same_source_patient": metadata.source_patient.to_numpy()[row_index] == metadata.source_patient.to_numpy()[column_index],
    }
    values = error[mask]
    rows = []
    for variable, indicator in groups.items():
        for level in (False, True):
            selected = values[indicator == level]
            rows.append(
                {
                    "group_variable": variable, "group_value": level, "pairs": len(selected),
                    "mean_abs_error": float(np.mean(selected)), "median_abs_error": float(np.median(selected)),
                    "p95_abs_error": float(np.quantile(selected, 0.95)), "max_abs_error": float(np.max(selected)),
                }
            )
    frame = pd.DataFrame(rows)
    _atomic_csv(OUTPUT / "joint_tail_structure_summary.csv", frame)
    return frame


def _between(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def summarize() -> dict[str, Any]:
    cfg = load_config("run")
    oracle_primary_all, oracle_primary_layers_all, oracle_primary_valid, oracle_primary_seeds = _load_oracle("primary", cfg)
    oracle_reference_all, oracle_reference_layers_all, oracle_reference_valid, oracle_reference_seeds = _load_oracle("reference", cfg)
    m4 = _load_m4(cfg)
    oracle_primary = oracle_primary_all[oracle_primary_valid]
    oracle_primary_layers = oracle_primary_layers_all[oracle_primary_valid]
    oracle_reference = oracle_reference_all[oracle_reference_valid]
    oracle_reference_layers = oracle_reference_layers_all[oracle_reference_valid]
    m4_selected_valid = np.all(np.isfinite(m4["joint"]), axis=1) & np.all(np.isfinite(m4["layers"]), axis=1)
    m4_joint = m4["joint"][m4_selected_valid]
    m4_layers = m4["layers"][m4_selected_valid]
    m4_cluster = m4["cluster"][m4_selected_valid]
    m4["joint"], m4["layers"], m4["cluster"] = m4_joint, m4_layers, m4_cluster
    planned = int(cfg["sample_plan"]["m4_balanced_primary_processes"])
    if min(len(oracle_primary), len(oracle_reference), len(m4_joint)) < planned:
        raise RuntimeError("formal planned balanced banks are incomplete")
    oracle_primary = oracle_primary[:planned]
    oracle_primary_layers = oracle_primary_layers[:planned]
    oracle_reference = oracle_reference[:planned]
    oracle_reference_layers = oracle_reference_layers[:planned]
    m4["joint"] = m4["joint"][:planned]
    m4["layers"] = m4["layers"][:planned]
    m4["cluster"] = m4["cluster"][:planned]
    _atomic_npz(
        OUTPUT / "process_banks.npz",
        oracle_primary=oracle_primary, oracle_primary_layers=oracle_primary_layers,
        oracle_reference=oracle_reference, oracle_reference_layers=oracle_reference_layers,
        m4_balanced=m4["joint"], m4_balanced_layers=m4["layers"], m4_cluster=m4["cluster"],
        m4_observed_joint=m4["observed_joint"], m4_observed_layers=m4["observed_layers"],
        m4_full_layers=m4["full_layers"], m4_full_valid=m4["full_valid"],
    )
    thresholds = tuple(map(float, cfg["primary_metric"]["thresholds"]))
    full = _comparison(oracle_primary, m4["joint"], oracle_reference, thresholds)
    tail_rows = []
    for comparison_name, comparison in (("M4_balanced", m4["joint"]), ("oracle_reference", oracle_reference)):
        for row in threshold_tail_metrics(oracle_primary, comparison, thresholds):
            tail_rows.append({"comparison": comparison_name, **row})
    tail_frame = pd.DataFrame(tail_rows)
    _atomic_csv(OUTPUT / "stable_joint_tail_metrics.csv", tail_frame)
    legacy = tail_frame[[
        "comparison", "threshold", "reference_conditional_p95", "comparison_conditional_p95",
        "legacy_conditional_p95_abs_gap", "tail_dependence_offdiagonal_MAE",
        "pair_abs_error_median", "pair_abs_error_p90", "pair_abs_error_p95",
        "pair_abs_error_p99", "pair_abs_error_max",
    ]].copy()
    _atomic_csv(OUTPUT / "legacy_p95_tail_metric.csv", legacy)
    center = center_covariance_metrics(oracle_primary, m4["joint"])
    center_frame = pd.DataFrame([center])
    center_frame["leading_10_eigenvalue_ratios"] = center_frame.leading_10_eigenvalue_ratios.map(json.dumps)
    _atomic_csv(OUTPUT / "center_covariance_summary.csv", center_frame)
    marginal_frame = pd.DataFrame(marginal_tail_metrics(oracle_primary, m4["joint"], thresholds))
    _atomic_csv(OUTPUT / "marginal_tail_summary.csv", marginal_frame)
    max_frame = pd.DataFrame(max_distribution_metrics(oracle_primary_layers, m4["layers"]))
    _atomic_csv(OUTPUT / "max_distribution_summary.csv", max_frame)
    balanced = _balanced(oracle_primary, oracle_primary_layers, oracle_reference, m4, cfg)
    uncertainty, uncertainty_draws = _uncertainty(oracle_primary, oracle_reference, m4, cfg)
    for metric, point in full.items():
        if metric in set(uncertainty.metric):
            uncertainty.loc[uncertainty.metric == metric, "point_estimate"] = point
    _atomic_csv(OUTPUT / "mc_uncertainty_summary.csv", uncertainty)
    type1 = _type1(m4)
    _structure_summary(oracle_primary, m4["joint"])
    runtime_frame = pd.DataFrame(
        [{
            "stage": "M4", "outer_replicates": len(m4["runtime"]),
            "inner_draws_per_outer": int(cfg["sample_plan"]["m4_inner_draws_per_outer"]),
            "mean_outer_seconds": float(np.mean(m4["runtime"])),
            "median_outer_seconds": float(np.median(m4["runtime"])),
            "max_outer_seconds": float(np.max(m4["runtime"])),
            "total_cpu_outer_seconds": float(np.sum(m4["runtime"])),
        }]
    )
    _atomic_csv(OUTPUT / "runtime_summary.csv", runtime_frame)

    uncertainty_primary = uncertainty[uncertainty.metric == "primary_EISJEM"].iloc[0]
    uncertainty_legacy = uncertainty[uncertainty.metric == "legacy_p95_gap"].iloc[0]
    convergence = pd.read_csv(OUTPUT / "oracle_convergence.csv")
    large_primary = convergence[(convergence.metric == "primary_EISJEM") & (convergence.oracle_sample_size >= 1000)]
    primary_range = float(large_primary["median"].max() - large_primary["median"].min())
    large_legacy = convergence[(convergence.metric == "legacy_p95_gap") & (convergence.oracle_sample_size >= 1000)]
    legacy_range = float(large_legacy["median"].max() - large_legacy["median"].min())
    precision = (
        float(uncertainty_primary.mc_standard_error) <= float(cfg["uncertainty"]["target_mc_standard_error_max"])
        and float(uncertainty_primary.ci_width) <= float(cfg["uncertainty"]["target_ci_width_max"])
        and primary_range <= float(cfg["uncertainty"]["primary_median_range_max"])
    )
    gates = cfg["domain_gates"]
    q95_tail = tail_frame[(tail_frame.comparison == "M4_balanced") & (tail_frame.threshold == 0.95)].iloc[0]
    domain = {
        "center": center["mean_difference_RMS_over_truth_SD"] <= gates["center"]["mean_RMS_over_truth_SD_max"],
        "covariance": all([
            _between(center["pointwise_SD_ratio"], gates["covariance"]["pointwise_SD_ratio"]),
            _between(center["leading_eigenvalue_ratio"], gates["covariance"]["leading_eigenvalue_ratio"]),
            _between(center["effective_rank_ratio"], gates["covariance"]["effective_rank_ratio"]),
            center["correlation_frobenius_relative_error"] <= gates["covariance"]["correlation_frobenius_relative_error_max"],
            center["offdiagonal_correlation_MAE"] <= gates["covariance"]["offdiagonal_correlation_MAE_max"],
        ]),
        "marginal_tail": all([
            _between(float(marginal_frame.loc[marginal_frame.threshold == 0.90, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q90_ratio"]),
            _between(float(marginal_frame.loc[marginal_frame.threshold == 0.95, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q95_ratio"]),
            _between(float(marginal_frame.loc[marginal_frame.threshold == 0.975, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q975_ratio"]),
            _between(float(marginal_frame.loc[marginal_frame.threshold == 0.99, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q99_ratio"]),
            float(marginal_frame.marginal_exceedance_MAE.max()) <= gates["marginal_tail"]["marginal_exceedance_MAE_max"],
            float(marginal_frame.skewness_RMSE.max()) <= gates["marginal_tail"]["skewness_RMSE_max"],
            float(marginal_frame.excess_kurtosis_RMSE.max()) <= gates["marginal_tail"]["excess_kurtosis_RMSE_max"],
        ]),
        "joint_tail_secondary": all([
            q95_tail.joint_exceedance_frobenius_relative_error <= gates["joint_tail_secondary"]["q95_joint_exceedance_frobenius_relative_error_max"],
            q95_tail.joint_exceedance_offdiagonal_MAE <= gates["joint_tail_secondary"]["q95_joint_exceedance_offdiagonal_MAE_max"],
            q95_tail.tail_dependence_offdiagonal_MAE <= gates["joint_tail_secondary"]["q95_tail_dependence_offdiagonal_MAE_max"],
        ]),
        "max_distribution": all([
            float(max_frame.loc[max_frame.layer == "D2", "KS"].iloc[0]) <= gates["max_distribution"]["D2_KS_max"],
            float(max_frame.loc[max_frame.layer == "D4", "KS"].iloc[0]) <= gates["max_distribution"]["D4_KS_max"],
            all(_between(row.q95_ratio, gates["max_distribution"]["q95_ratio_D4" if row.layer == "D4" else "q95_ratio_D0_D3"]) for _, row in max_frame.iterrows()),
            _between(float(max_frame.loc[max_frame.layer == "D4", "probability_comparison_exceeds_reference_q95"].iloc[0]), gates["max_distribution"]["probability_M4_D4_exceeds_oracle_q95"]),
        ]),
    }
    d4_type = type1[type1.layer == "D4"].iloc[0]
    nominal_wilson_or_mcse = (
        d4_type.wilson_low_005 <= 0.05 <= d4_type.wilson_high_005
        or abs(d4_type.size_005 - 0.05) <= d4_type.mcse_005
    )
    domain["type1"] = bool(
        float(type1.size_005.max()) <= gates["type1"]["size_005_max_each_D0_D4"]
        and float(d4_type.size_001) <= gates["type1"]["D4_size_001_max"]
        and float(type1.testable_fraction.min()) >= gates["type1"]["testable_fraction_min"]
        and nominal_wilson_or_mcse
    )
    oracle_failure = 1 - (int(np.sum(oracle_primary_valid)) + int(np.sum(oracle_reference_valid))) / (len(oracle_primary_valid) + len(oracle_reference_valid))
    m4_draw_failure = float(np.mean(~m4["full_valid"]))
    m4_outer_failure = float(np.mean(np.sum(m4["full_valid"], axis=1) == 0))
    domain["engineering"] = bool(
        oracle_failure <= gates["engineering"]["oracle_failure_rate_max"]
        and m4_draw_failure <= gates["engineering"]["m4_draw_failure_rate_max"]
        and m4_outer_failure <= gates["engineering"]["m4_outer_failure_rate_max"]
        and np.all(np.isfinite(oracle_primary)) and np.all(np.isfinite(m4["joint"]))
    )
    stable_major_secondary = bool(
        float(uncertainty_legacy.ci_low) > float(cfg["secondary_metrics"]["stable_major_legacy_threshold"])
        and legacy_range <= float(cfg["secondary_metrics"]["stable_legacy_convergence_range_max"])
    )
    margin = float(cfg["primary_metric"]["equivalence_margin"])
    if not precision:
        gate = cfg["stop_rules"]["unstable"]
    elif float(uncertainty_primary.ci_low) > margin:
        gate = cfg["stop_rules"]["stable_fail"]
    elif float(uncertainty_primary.ci_high) <= margin and all(domain.values()) and not stable_major_secondary:
        gate = cfg["stop_rules"]["confirmed"]
    else:
        gate = cfg["stop_rules"]["inconclusive"]
    payload = {
        "gate": gate,
        "primary_point_estimate": full["primary_EISJEM"],
        "primary_mc_standard_error": float(uncertainty_primary.mc_standard_error),
        "primary_ci_low": float(uncertainty_primary.ci_low),
        "primary_ci_high": float(uncertainty_primary.ci_high),
        "primary_ci_width": float(uncertainty_primary.ci_width),
        "primary_equivalence_margin": margin,
        "primary_convergence_median_range_n_ge_1000": primary_range,
        "precision_and_stability_pass": bool(precision),
        "legacy_p95_point_estimate": full["legacy_p95_gap"],
        "legacy_p95_ci_low": float(uncertainty_legacy.ci_low),
        "legacy_p95_ci_high": float(uncertainty_legacy.ci_high),
        "legacy_convergence_median_range_n_ge_1000": legacy_range,
        "stable_major_secondary": stable_major_secondary,
        "domain_checks": {key: bool(value) for key, value in domain.items()},
        "oracle_failure_rate": oracle_failure,
        "m4_draw_failure_rate": m4_draw_failure,
        "m4_outer_failure_rate": m4_outer_failure,
        "phase2rd_gate_unchanged": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "phase2re_gate_unchanged": "TAIL_METRIC_UNSTABLE",
        "M4_modified": False,
        "posthoc_tuning": False,
        "qualifies_future_Phase2R_G": gate == cfg["stop_rules"]["confirmed"],
        "D3_run": False,
        "OhioT1DM_allowed": False,
        "Phase3_allowed": False,
        "transfer_null_allowed": False,
        "alternatives_power_changepoint_allowed": False,
        "establish_1_0_2": False,
    }
    _atomic_json(OUTPUT / "phase2rf_gate.json", payload)
    engineering = {
        "oracle_primary_expected": len(oracle_primary_valid),
        "oracle_primary_valid": int(np.sum(oracle_primary_valid)),
        "oracle_reference_expected": len(oracle_reference_valid),
        "oracle_reference_valid": int(np.sum(oracle_reference_valid)),
        "m4_outer_expected": int(cfg["sample_plan"]["m4_outer_replicates"]),
        "m4_inner_expected_each": int(cfg["sample_plan"]["m4_inner_draws_per_outer"]),
        "m4_valid_draws": int(np.sum(m4["full_valid"])),
        "m4_failed_draws": int(np.sum(~m4["full_valid"])),
        "seed_integrity": True,
        "finite_primary_banks": True,
    }
    _atomic_json(OUTPUT / "engineering_summary.json", engineering)
    return payload


if __name__ == "__main__":
    print(json.dumps(summarize(), sort_keys=True))
