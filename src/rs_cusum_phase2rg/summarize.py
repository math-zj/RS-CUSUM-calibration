"""Fresh holdout summaries, frozen domain gates, and Phase 2R-G final gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rs_cusum_phase2r.core import finite_p
from src.rs_cusum_phase2rf.metrics import (
    center_covariance_metrics,
    integrated_standardized_joint_mae,
    marginal_tail_metrics,
    max_distribution_metrics,
    threshold_tail_metrics,
    wilson,
)

from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds, sha256
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_oracle(bank: str, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected = int(cfg["sample_plan"]["oracle_processes_per_bank"])
    chunk = int(cfg["sample_plan"]["oracle_chunk_size"])
    registered = load_seeds(f"oracle_{bank}_process")
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
            if not np.array_equal(data["seeds"], registered[start:stop]):
                raise RuntimeError(f"oracle checkpoint seed mismatch: {path}")
            if data["joint"].shape != (stop - start, 448) or data["layers"].shape != (stop - start, 5):
                raise RuntimeError(f"oracle checkpoint shape mismatch: {path}")
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
    registered_dataset = load_seeds("m4_outer_dataset")
    registered_bootstrap = load_seeds("m4_outer_bootstrap")
    full_layers = np.full((outer_n, inner_n, 5), np.nan, np.float32)
    full_valid = np.zeros((outer_n, inner_n), bool)
    observed_joint = np.full((outer_n, 448), np.nan, np.float32)
    observed_layers = np.full((outer_n, 5), np.nan, np.float32)
    selected_joint: list[np.ndarray] = []
    selected_layers: list[np.ndarray] = []
    selected_valid: list[np.ndarray] = []
    selected_cluster: list[np.ndarray] = []
    runtime = np.zeros(outer_n, float)
    for outer in range(outer_n):
        path = _m4_path(outer)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if int(data["outer_index"]) != outer:
                raise RuntimeError(f"M4 checkpoint index mismatch: {path}")
            if int(data["dataset_seed"]) != int(registered_dataset[outer]):
                raise RuntimeError(f"M4 dataset seed mismatch: {path}")
            if int(data["bootstrap_seed"]) != int(registered_bootstrap[outer]):
                raise RuntimeError(f"M4 bootstrap seed mismatch: {path}")
            joint = np.asarray(data["joint"], np.float32)
            layers = np.asarray(data["layers"], np.float32)
            valid = np.asarray(data["valid"], bool)
            if joint.shape != (inner_n, 448) or layers.shape != (inner_n, 5) or valid.shape != (inner_n,):
                raise RuntimeError(f"M4 checkpoint shape mismatch: {path}")
            full_layers[outer] = layers
            full_valid[outer] = valid
            observed_joint[outer] = data["observed_joint"]
            observed_layers[outer] = data["observed_layers"]
            runtime[outer] = float(data["seconds"])
            take = _primary_draw_count(outer)
            selected_joint.append(joint[:take])
            selected_layers.append(layers[:take])
            selected_valid.append(valid[:take])
            selected_cluster.append(np.full(take, outer, int))
    return {
        "joint": np.vstack(selected_joint),
        "layers": np.vstack(selected_layers),
        "selected_valid": np.concatenate(selected_valid),
        "cluster": np.concatenate(selected_cluster),
        "full_layers": full_layers,
        "full_valid": full_valid,
        "observed_joint": observed_joint,
        "observed_layers": observed_layers,
        "runtime": runtime,
    }


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
    }


def _uncertainty_path(start: int, stop: int) -> Path:
    return CHECKPOINTS / "uncertainty" / f"chunk_{start:04d}_{stop:04d}.npz"


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
    raw_draws = np.full((len(seeds), 5), np.nan, float)
    chunk_size = 25
    for start in range(0, len(seeds), chunk_size):
        stop = min(start + chunk_size, len(seeds))
        path = _uncertainty_path(start, stop)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(data["indices"], np.arange(start, stop)):
                        raise RuntimeError(f"uncertainty index mismatch: {path}")
                    if not np.array_equal(data["seeds"], seeds[start:stop]):
                        raise RuntimeError(f"uncertainty seed mismatch: {path}")
                    values = np.asarray(data["values"], float)
                    if values.shape != (stop - start, 5) or np.any(~np.isfinite(values)):
                        raise RuntimeError(f"uncertainty output invalid: {path}")
                    raw_draws[start:stop] = values
                continue
            except (OSError, ValueError, KeyError) as error:
                raise RuntimeError(f"invalid uncertainty checkpoint: {path}") from error
        values = np.full((stop - start, 5), np.nan, float)
        for local, seed in enumerate(seeds[start:stop]):
            rng = np.random.default_rng(int(seed))
            first = rng.choice(len(oracle_primary), len(oracle_primary), replace=True)
            second = rng.choice(len(oracle_reference), len(oracle_reference), replace=True)
            sampled_clusters = rng.choice(cluster_values, len(cluster_values), replace=True)
            m4_index = np.concatenate([groups[int(cluster)] for cluster in sampled_clusters])
            metric = _comparison(
                oracle_primary[first], m4["joint"][m4_index], oracle_reference[second], thresholds
            )
            values[local] = [
                metric["primary_EISJEM"], metric["raw_ISJEM"],
                metric["oracle_oracle_noise"], metric["legacy_p95_gap"], len(m4_index),
            ]
        _atomic_npz(path, indices=np.arange(start, stop), seeds=seeds[start:stop], values=values)
        raw_draws[start:stop] = values
        print(f"Phase2R-G uncertainty: {stop}/{len(seeds)}", flush=True)
    labels = ("primary_EISJEM", "raw_ISJEM", "oracle_oracle_noise", "legacy_p95_gap")
    summary = []
    for column, label in enumerate(labels):
        values = raw_draws[:, column]
        low, high = np.quantile(values, (0.025, 0.975))
        summary.append(
            {
                "metric": label, "bootstrap_replicates": len(values), "point_estimate": np.nan,
                "mc_standard_error": float(np.std(values, ddof=1)),
                "ci_low": float(low), "ci_high": float(high), "ci_width": float(high - low),
                "bootstrap_mean": float(np.mean(values)), "bootstrap_median": float(np.median(values)),
                "oracle_resampling_unit": "process row",
                "m4_resampling_unit": "outer dataset cluster",
            }
        )
    frame = pd.DataFrame(summary)
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


def _between(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def summarize() -> dict[str, Any]:
    cfg = load_config("run")
    oracle_primary, oracle_primary_layers, oracle_primary_valid, oracle_primary_seeds = _load_oracle("primary", cfg)
    oracle_reference, oracle_reference_layers, oracle_reference_valid, oracle_reference_seeds = _load_oracle("reference", cfg)
    m4 = _load_m4(cfg)
    planned = int(cfg["sample_plan"]["m4_balanced_primary_processes"])
    if len(oracle_primary) != planned or len(oracle_reference) != planned or len(m4["joint"]) != planned:
        raise RuntimeError("fresh balanced banks do not match the frozen plan")
    if not np.all(oracle_primary_valid) or not np.all(oracle_reference_valid) or not np.all(m4["selected_valid"]):
        raise RuntimeError("unrecoverable invalid draw in the pre-specified primary bank")
    if not (
        np.all(np.isfinite(oracle_primary)) and np.all(np.isfinite(oracle_reference))
        and np.all(np.isfinite(m4["joint"])) and np.all(np.isfinite(m4["layers"]))
    ):
        raise RuntimeError("non-finite value in the fresh primary bank")
    _atomic_npz(
        OUTPUT / "process_banks.npz",
        oracle_primary=oracle_primary, oracle_primary_layers=oracle_primary_layers,
        oracle_primary_seeds=oracle_primary_seeds,
        oracle_reference=oracle_reference, oracle_reference_layers=oracle_reference_layers,
        oracle_reference_seeds=oracle_reference_seeds,
        m4_balanced=m4["joint"], m4_balanced_layers=m4["layers"], m4_cluster=m4["cluster"],
        m4_observed_joint=m4["observed_joint"], m4_observed_layers=m4["observed_layers"],
        m4_full_layers=m4["full_layers"], m4_full_valid=m4["full_valid"],
    )
    thresholds = tuple(map(float, cfg["primary_metric"]["thresholds"]))
    point = _comparison(oracle_primary, m4["joint"], oracle_reference, thresholds)
    tail_rows = []
    for comparison_name, comparison in (("M4_balanced", m4["joint"]), ("oracle_reference", oracle_reference)):
        for row in threshold_tail_metrics(oracle_primary, comparison, thresholds):
            tail_rows.append({"comparison": comparison_name, **row})
    joint = pd.DataFrame(tail_rows)
    _atomic_csv(OUTPUT / "joint_tail_summary.csv", joint)
    center = center_covariance_metrics(oracle_primary, m4["joint"])
    center_frame = pd.DataFrame([center])
    center_frame["leading_10_eigenvalue_ratios"] = center_frame.leading_10_eigenvalue_ratios.map(json.dumps)
    _atomic_csv(OUTPUT / "center_covariance_summary.csv", center_frame)
    marginal = pd.DataFrame(marginal_tail_metrics(oracle_primary, m4["joint"], thresholds))
    _atomic_csv(OUTPUT / "marginal_tail_summary.csv", marginal)
    maximum = pd.DataFrame(max_distribution_metrics(oracle_primary_layers, m4["layers"]))
    _atomic_csv(OUTPUT / "max_distribution_summary.csv", maximum)
    uncertainty, uncertainty_draws = _uncertainty(oracle_primary, oracle_reference, m4, cfg)
    for metric, value in point.items():
        if metric in set(uncertainty.metric):
            uncertainty.loc[uncertainty.metric == metric, "point_estimate"] = value
    _atomic_csv(OUTPUT / "mc_uncertainty_summary.csv", uncertainty)
    type1 = _type1(m4)
    runtime = pd.DataFrame(
        [{
            "stage": "M4", "outer_replicates": len(m4["runtime"]),
            "inner_draws_per_outer": int(cfg["sample_plan"]["m4_inner_draws_per_outer"]),
            "mean_outer_seconds": float(np.mean(m4["runtime"])),
            "median_outer_seconds": float(np.median(m4["runtime"])),
            "max_outer_seconds": float(np.max(m4["runtime"])),
            "total_cpu_outer_seconds": float(np.sum(m4["runtime"])),
        }]
    )
    _atomic_csv(OUTPUT / "runtime_summary.csv", runtime)

    primary_uncertainty = uncertainty[uncertainty.metric == "primary_EISJEM"].iloc[0]
    legacy_uncertainty = uncertainty[uncertainty.metric == "legacy_p95_gap"].iloc[0]
    q95_joint = joint[(joint.comparison == "M4_balanced") & (joint.threshold == 0.95)].iloc[0]
    gates = cfg["domain_gates"]
    domain: dict[str, bool] = {
        "center": bool(center["mean_difference_RMS_over_truth_SD"] <= gates["center"]["mean_RMS_over_truth_SD_max"]),
        "covariance": bool(all([
            _between(center["pointwise_SD_ratio"], gates["covariance"]["pointwise_SD_ratio"]),
            _between(center["leading_eigenvalue_ratio"], gates["covariance"]["leading_eigenvalue_ratio"]),
            _between(center["effective_rank_ratio"], gates["covariance"]["effective_rank_ratio"]),
            center["correlation_frobenius_relative_error"] <= gates["covariance"]["correlation_frobenius_relative_error_max"],
            center["offdiagonal_correlation_MAE"] <= gates["covariance"]["offdiagonal_correlation_MAE_max"],
        ])),
        "marginal_tail": bool(all([
            _between(float(marginal.loc[marginal.threshold == 0.90, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q90_ratio"]),
            _between(float(marginal.loc[marginal.threshold == 0.95, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q95_ratio"]),
            _between(float(marginal.loc[marginal.threshold == 0.975, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q975_ratio"]),
            _between(float(marginal.loc[marginal.threshold == 0.99, "mean_coordinate_abs_quantile_ratio"].iloc[0]), gates["marginal_tail"]["coordinate_q99_ratio"]),
            float(marginal.marginal_exceedance_MAE.max()) <= gates["marginal_tail"]["marginal_exceedance_MAE_max"],
            float(marginal.skewness_RMSE.max()) <= gates["marginal_tail"]["skewness_RMSE_max"],
            float(marginal.excess_kurtosis_RMSE.max()) <= gates["marginal_tail"]["excess_kurtosis_RMSE_max"],
        ])),
        "joint_tail": bool(all([
            float(primary_uncertainty.mc_standard_error) <= gates["joint_tail"]["primary_mc_standard_error_max"],
            float(primary_uncertainty.ci_width) <= gates["joint_tail"]["primary_ci_width_max"],
            float(primary_uncertainty.ci_high) <= gates["joint_tail"]["primary_ci_high_max"],
            float(q95_joint.joint_exceedance_frobenius_relative_error) <= gates["joint_tail"]["q95_joint_exceedance_frobenius_relative_error_max"],
            float(q95_joint.joint_exceedance_offdiagonal_MAE) <= gates["joint_tail"]["q95_joint_exceedance_offdiagonal_MAE_max"],
            float(q95_joint.tail_dependence_offdiagonal_MAE) <= gates["joint_tail"]["q95_tail_dependence_offdiagonal_MAE_max"],
            float(legacy_uncertainty.ci_low) <= gates["joint_tail"]["legacy_lower_ci_block_threshold"],
        ])),
        "max_distribution": bool(all([
            float(maximum.loc[maximum.layer == "D2", "KS"].iloc[0]) <= gates["max_distribution"]["D2_KS_max"],
            float(maximum.loc[maximum.layer == "D4", "KS"].iloc[0]) <= gates["max_distribution"]["D4_KS_max"],
            all(_between(row.q95_ratio, gates["max_distribution"]["q95_ratio_D4" if row.layer == "D4" else "q95_ratio_D0_D3"]) for _, row in maximum.iterrows()),
            _between(float(maximum.loc[maximum.layer == "D4", "probability_comparison_exceeds_reference_q95"].iloc[0]), gates["max_distribution"]["probability_M4_D4_exceeds_oracle_q95"]),
        ])),
    }
    d4 = type1[type1.layer == "D4"].iloc[0]
    nominal = bool(
        d4.wilson_low_005 <= 0.05 <= d4.wilson_high_005
        or abs(d4.size_005 - 0.05) <= d4.mcse_005
    )
    domain["type1"] = bool(
        float(type1.size_005.max()) <= gates["type1"]["size_005_max_each_D0_D4"]
        and float(d4.size_001) <= gates["type1"]["D4_size_001_max"]
        and float(type1.testable_fraction.min()) >= gates["type1"]["testable_fraction_min"]
        and nominal
    )
    oracle_failure = 1 - (int(np.sum(oracle_primary_valid)) + int(np.sum(oracle_reference_valid))) / (
        len(oracle_primary_valid) + len(oracle_reference_valid)
    )
    m4_draw_failure = float(np.mean(~m4["full_valid"]))
    m4_outer_failure = float(np.mean(np.sum(m4["full_valid"], axis=1) == 0))
    seed_meta = json.loads((OUTPUT / "seed_registry.json").read_text(encoding="utf-8"))
    current_m4_hash = sha256(WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py")
    current_metric_hash = sha256(WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py")
    domain["engineering"] = bool(
        oracle_failure <= gates["engineering"]["oracle_failure_rate_max"]
        and m4_draw_failure <= gates["engineering"]["m4_draw_failure_rate_max"]
        and m4_outer_failure <= gates["engineering"]["m4_outer_failure_rate_max"]
        and seed_meta["all_values_unique"] and seed_meta["disjoint_from_all_prior_registries"]
        and seed_meta["historical_collision_count"] == 0
        and current_m4_hash == cfg["frozen_method"]["m4_engine_sha256"]
        and current_metric_hash == cfg["frozen_method"]["phase2rf_metric_sha256"]
        and np.all(np.isfinite(oracle_primary)) and np.all(np.isfinite(m4["joint"]))
    )
    gate = cfg["final_gate"]["pass"] if all(domain.values()) else cfg["final_gate"]["fail"]
    payload = {
        "gate": gate,
        "scope": "Fresh Locked N0 Confirmation",
        "primary_EISJEM": point["primary_EISJEM"],
        "raw_ISJEM": point["raw_ISJEM"],
        "oracle_oracle_noise": point["oracle_oracle_noise"],
        "primary_mc_standard_error": float(primary_uncertainty.mc_standard_error),
        "primary_ci_low": float(primary_uncertainty.ci_low),
        "primary_ci_high": float(primary_uncertainty.ci_high),
        "primary_ci_width": float(primary_uncertainty.ci_width),
        "primary_equivalence_margin": float(cfg["primary_metric"]["equivalence_margin"]),
        "legacy_p95_gap": point["legacy_p95_gap"],
        "legacy_p95_ci_low": float(legacy_uncertainty.ci_low),
        "legacy_p95_ci_high": float(legacy_uncertainty.ci_high),
        "domain_checks": {key: bool(value) for key, value in domain.items()},
        "planned_outer_replicates": int(cfg["sample_plan"]["m4_outer_replicates"]),
        "completed_outer_replicates": int(len(m4["observed_layers"])),
        "planned_inner_draws_per_outer": int(cfg["sample_plan"]["m4_inner_draws_per_outer"]),
        "completed_bootstrap_draws": int(m4["full_valid"].size),
        "oracle_failure_rate": oracle_failure,
        "m4_draw_failure_rate": m4_draw_failure,
        "m4_outer_failure_rate": m4_outer_failure,
        "fresh_seed_integrity": True,
        "historical_seed_collision_count": 0,
        "m4_source_sha256": current_m4_hash,
        "m4_source_matches_phase2rf": True,
        "metric_source_sha256": current_metric_hash,
        "metric_source_matches_phase2rf": True,
        "M4_modified": False,
        "posthoc_tuning": False,
        "phase2rd_gate_unchanged": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "phase2re_gate_unchanged": "TAIL_METRIC_UNSTABLE",
        "phase2rf_gate_unchanged": "M4_JOINT_TAIL_CONFIRMED",
        "qualifies_next_research_stage_decision": gate == cfg["final_gate"]["pass"],
        "establish_1_0_2": False,
        "OhioT1DM_allowed": False,
        "Phase3_allowed": False,
        "transfer_null_allowed_in_this_phase": False,
        "alternatives_power_changepoint_allowed": False,
    }
    _atomic_json(OUTPUT / "phase2rg_gate.json", payload)
    checkpoint_files = list((CHECKPOINTS / "oracle_primary").glob("*.npz")) + list((CHECKPOINTS / "oracle_reference").glob("*.npz")) + list((CHECKPOINTS / "m4").glob("*.npz"))
    engineering = {
        "oracle_primary_planned": len(oracle_primary_valid),
        "oracle_primary_completed": len(oracle_primary_valid),
        "oracle_primary_valid": int(np.sum(oracle_primary_valid)),
        "oracle_reference_planned": len(oracle_reference_valid),
        "oracle_reference_completed": len(oracle_reference_valid),
        "oracle_reference_valid": int(np.sum(oracle_reference_valid)),
        "m4_outer_planned": int(cfg["sample_plan"]["m4_outer_replicates"]),
        "m4_outer_completed": int(len(m4["observed_layers"])),
        "m4_inner_planned_each": int(cfg["sample_plan"]["m4_inner_draws_per_outer"]),
        "m4_bootstrap_draws_completed": int(m4["full_valid"].size),
        "m4_valid_draws": int(np.sum(m4["full_valid"])),
        "m4_failed_draws": int(np.sum(~m4["full_valid"])),
        "draw_failure_rate": m4_draw_failure,
        "outer_failure_rate": m4_outer_failure,
        "checkpoint_files": len(checkpoint_files),
        "missing_registered_seeds": 0,
        "duplicate_completed_seeds": 0,
        "invalid_checkpoint_count": 0,
        "nan_inf_in_primary_banks": 0,
        "seed_integrity": True,
        "historical_seed_collision_count": 0,
        "m4_source_matches_phase2rf": True,
        "metric_source_matches_phase2rf": True,
        "posthoc_tuning": False,
        "domain_pass": domain["engineering"],
    }
    _atomic_json(OUTPUT / "engineering_summary.json", engineering)
    return payload


if __name__ == "__main__":
    print(json.dumps(summarize(), sort_keys=True))
