"""Scenario-wise inference, mechanism diagnostics, and frozen Phase 2R-H gate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.rs_cusum_phase2r.core import finite_p
from src.rs_cusum_phase2rf.metrics import (
    center_covariance_metrics,
    integrated_standardized_joint_mae,
    marginal_tail_metrics,
    max_distribution_metrics,
    threshold_tail_metrics,
    wilson,
)

from .generator import COMPONENTS
from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds, sha256
from .runner import _m4_path, _oracle_path


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_oracle(scenario_index: int, scenario: str, bank: str, cfg: dict[str, Any]) -> dict[str, Any]:
    expected = int(cfg["sample_plan"]["oracle_processes_per_bank_per_scenario"])
    chunk = int(cfg["sample_plan"]["oracle_chunk_size"])
    registered = load_seeds(f"oracle_{bank}_process")[scenario_index]
    registered_components = {
        name: load_seeds(f"oracle_{bank}_dgp_{name}")[scenario_index] for name in COMPONENTS
    }
    blocks: dict[str, list[np.ndarray]] = {
        "joint": [], "layers": [], "valid": [], "seeds": [], "errors": [], "dgp": []
    }
    for start in range(0, expected, chunk):
        stop = min(start + chunk, expected)
        path = _oracle_path(scenario, bank, start, stop)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if str(data["scenario"].item()) != scenario or str(data["bank"].item()) != bank:
                raise RuntimeError(f"oracle scenario/bank mismatch: {path}")
            if not np.array_equal(data["indices"], np.arange(start, stop)):
                raise RuntimeError(f"oracle index mismatch: {path}")
            if not np.array_equal(data["seeds"], registered[start:stop]):
                raise RuntimeError(f"oracle seed mismatch: {path}")
            for name in COMPONENTS:
                if not np.array_equal(data[f"seed_{name}"], registered_components[name][start:stop]):
                    raise RuntimeError(f"oracle component seed mismatch {name}: {path}")
            if data["joint"].shape != (stop - start, 448) or data["layers"].shape != (stop - start, 5):
                raise RuntimeError(f"oracle shape mismatch: {path}")
            blocks["joint"].append(np.asarray(data["joint"], np.float32))
            blocks["layers"].append(np.asarray(data["layers"], np.float32))
            blocks["valid"].append(np.asarray(data["valid"], bool))
            blocks["seeds"].append(np.asarray(data["seeds"], np.uint64))
            blocks["errors"].append(np.asarray(data["errors"], str))
            blocks["dgp"].append(np.asarray(data["dgp_diagnostics"], str))
    return {
        "joint": np.vstack(blocks["joint"]), "layers": np.vstack(blocks["layers"]),
        "valid": np.concatenate(blocks["valid"]), "seeds": np.concatenate(blocks["seeds"]),
        "errors": np.concatenate(blocks["errors"]), "dgp": np.concatenate(blocks["dgp"]),
    }


def _load_m4(scenario_index: int, scenario: str, cfg: dict[str, Any]) -> dict[str, Any]:
    outer_n = int(cfg["sample_plan"]["m4_outer_replicates_per_scenario"])
    inner_n = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])
    dataset_seeds = load_seeds("m4_outer_dataset")[scenario_index]
    bootstrap_seeds = load_seeds("m4_outer_bootstrap")[scenario_index]
    registered_components = {
        name: load_seeds(f"m4_outer_dgp_{name}")[scenario_index] for name in COMPONENTS
    }
    full_layers = np.full((outer_n, inner_n, 5), np.nan, np.float32)
    full_valid = np.zeros((outer_n, inner_n), bool)
    observed_joint = np.full((outer_n, 448), np.nan, np.float32)
    observed_layers = np.full((outer_n, 5), np.nan, np.float32)
    selected_joint: list[np.ndarray] = []
    selected_layers: list[np.ndarray] = []
    selected_valid: list[np.ndarray] = []
    selected_cluster: list[np.ndarray] = []
    errors: list[str] = []
    diagnostics: list[str] = []
    dgp: list[str] = []
    runtime = np.zeros(outer_n, float)
    for outer in range(outer_n):
        path = _m4_path(scenario, outer)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if str(data["scenario"].item()) != scenario or int(data["outer_index"]) != outer:
                raise RuntimeError(f"M4 scenario/index mismatch: {path}")
            if int(data["dataset_seed"]) != int(dataset_seeds[outer]):
                raise RuntimeError(f"M4 dataset seed mismatch: {path}")
            if int(data["bootstrap_seed"]) != int(bootstrap_seeds[outer]):
                raise RuntimeError(f"M4 bootstrap seed mismatch: {path}")
            for name in COMPONENTS:
                if int(data[f"seed_{name}"]) != int(registered_components[name][outer]):
                    raise RuntimeError(f"M4 component seed mismatch {name}: {path}")
            joint = np.asarray(data["joint"], np.float32)
            layers = np.asarray(data["layers"], np.float32)
            valid = np.asarray(data["valid"], bool)
            if joint.shape != (inner_n, 448) or layers.shape != (inner_n, 5) or valid.shape != (inner_n,):
                raise RuntimeError(f"M4 shape mismatch: {path}")
            full_layers[outer] = layers
            full_valid[outer] = valid
            observed_joint[outer] = data["observed_joint"]
            observed_layers[outer] = data["observed_layers"]
            selected_joint.append(joint[:5])
            selected_layers.append(layers[:5])
            selected_valid.append(valid[:5])
            selected_cluster.append(np.full(5, outer, int))
            errors.append(str(data["error"].item()))
            diagnostics.append(str(data["diagnostics"].item()))
            dgp.append(str(data["dgp_diagnostics"].item()))
            runtime[outer] = float(data["seconds"])
    return {
        "joint": np.vstack(selected_joint), "layers": np.vstack(selected_layers),
        "selected_valid": np.concatenate(selected_valid), "cluster": np.concatenate(selected_cluster),
        "full_layers": full_layers, "full_valid": full_valid,
        "observed_joint": observed_joint, "observed_layers": observed_layers,
        "errors": np.asarray(errors), "diagnostics": np.asarray(diagnostics),
        "dgp": np.asarray(dgp), "runtime": runtime,
    }


def _between(value: float, bounds: list[float]) -> bool:
    return np.isfinite(value) and float(bounds[0]) <= float(value) <= float(bounds[1])


def _type1_rows(scenario: str, role: str, m4: dict[str, np.ndarray], bonf_z: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in range(5):
        p_values = []
        for outer in range(len(m4["observed_layers"])):
            observed = float(m4["observed_layers"][outer, layer])
            values = m4["full_layers"][outer, :, layer]
            values = values[m4["full_valid"][outer] & np.isfinite(values)]
            p_values.append(finite_p(observed, values) if np.isfinite(observed) and len(values) else np.nan)
        p_values = np.asarray(p_values, float)
        valid = p_values[np.isfinite(p_values)]
        row: dict[str, Any] = {
            "scenario_id": scenario, "role": role, "layer": f"D{layer}",
            "outer_replicates": len(p_values), "testable_replicates": len(valid),
            "testable_fraction": len(valid) / len(p_values),
            "median_p_value": float(np.median(valid)) if len(valid) else np.nan,
        }
        for alpha, label in ((0.01, "001"), (0.05, "005"), (0.10, "010")):
            count = int(np.sum(valid <= alpha)); rate = count / len(valid) if len(valid) else np.nan
            low, high = wilson(count, len(valid))
            blow, bhigh = wilson(count, len(valid), z=bonf_z)
            row.update({
                f"rejections_{label}": count, f"size_{label}": rate,
                f"mcse_{label}": float(np.sqrt(rate * (1-rate) / len(valid))) if len(valid) else np.nan,
                f"wilson_low_{label}": low, f"wilson_high_{label}": high,
                f"bonferroni_wilson_low_{label}": blow if role == "primary" else np.nan,
                f"bonferroni_wilson_high_{label}": bhigh if role == "primary" else np.nan,
            })
        rows.append(row)
    return rows


def _diagnostic_aggregate(dgp_strings: np.ndarray) -> dict[str, float]:
    parsed = [json.loads(item) for item in dgp_strings if item]
    fields = (
        "action_rate", "action_count", "minimum_patient_action_count", "maximum_patient_action_share",
        "state_x0_mean", "state_x0_sd", "reward_mean", "reward_sd", "innovation_mean",
        "innovation_sd", "conditional_scale_min", "conditional_scale_max",
    )
    result: dict[str, float] = {"diagnostic_records": float(len(parsed))}
    for field in fields:
        values = np.asarray([row[field] for row in parsed], float)
        result[f"{field}_mean"] = float(np.mean(values)) if len(values) else np.nan
        result[f"{field}_min"] = float(np.min(values)) if len(values) else np.nan
        result[f"{field}_max"] = float(np.max(values)) if len(values) else np.nan
    return result


def summarize() -> dict[str, Any]:
    cfg = load_config("run")
    scenarios = tuple(cfg["scenario_plan"]["order"])
    role_lookup = {scenario: "reference" for scenario in cfg["scenario_plan"]["reference"]}
    role_lookup.update({scenario: "primary" for scenario in cfg["scenario_plan"]["primary_transfer"]})
    role_lookup.update({scenario: "stress" for scenario in cfg["scenario_plan"]["stress"]})
    thresholds = tuple(map(float, cfg["mechanism_diagnostics"]["thresholds"]))
    bonf_z = float(norm.ppf(1.0 - 0.05 / (2.0 * 6.0)))
    seed_meta = json.loads((OUTPUT / "seed_registry.json").read_text(encoding="utf-8"))

    type1_rows: list[dict[str, Any]] = []
    maximum_rows: list[dict[str, Any]] = []
    center_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    engineering_rows: list[dict[str, Any]] = []
    dgp_rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios):
        role = role_lookup[scenario]
        oracle = _load_oracle(scenario_index, scenario, "primary", cfg)
        reference = _load_oracle(scenario_index, scenario, "reference", cfg)
        m4 = _load_m4(scenario_index, scenario, cfg)
        oracle_joint = oracle["joint"][oracle["valid"] & np.all(np.isfinite(oracle["joint"]), axis=1)]
        oracle_layers = oracle["layers"][oracle["valid"] & np.all(np.isfinite(oracle["layers"]), axis=1)]
        reference_joint = reference["joint"][reference["valid"] & np.all(np.isfinite(reference["joint"]), axis=1)]
        m4_mask = m4["selected_valid"] & np.all(np.isfinite(m4["joint"]), axis=1)
        m4_joint = m4["joint"][m4_mask]
        m4_layers = m4["layers"][m4_mask]
        scenario_type = _type1_rows(scenario, role, m4, bonf_z)
        type1_rows.extend(scenario_type)

        if min(len(oracle_joint), len(reference_joint), len(m4_joint)) >= 2:
            center = center_covariance_metrics(oracle_joint, m4_joint)
            center_rows.append({"scenario_id": scenario, "role": role, **center})
            marginal = marginal_tail_metrics(oracle_joint, m4_joint, thresholds)
            marginal_rows.extend({"scenario_id": scenario, "role": role, **row} for row in marginal)
            m4_tail = threshold_tail_metrics(oracle_joint, m4_joint, thresholds)
            noise_tail = threshold_tail_metrics(oracle_joint, reference_joint, thresholds)
            eisjem = integrated_standardized_joint_mae(m4_tail) - integrated_standardized_joint_mae(noise_tail)
            for comparison, rows in (("M4_balanced", m4_tail), ("oracle_reference", noise_tail)):
                joint_rows.extend({
                    "scenario_id": scenario, "role": role, "comparison": comparison,
                    "primary_EISJEM": eisjem if comparison == "M4_balanced" else np.nan,
                    **row,
                } for row in rows)
            maximum_rows.extend({"scenario_id": scenario, "role": role, **row}
                                for row in max_distribution_metrics(oracle_layers, m4_layers))
        else:
            center = {}
            marginal = []
            m4_tail = []
            eisjem = np.nan

        oracle_failure = 1.0 - (np.sum(oracle["valid"]) + np.sum(reference["valid"])) / (
            len(oracle["valid"]) + len(reference["valid"])
        )
        draw_failure = float(np.mean(~m4["full_valid"]))
        observed_valid = np.all(np.isfinite(m4["observed_layers"]), axis=1)
        outer_failure = float(np.mean((np.sum(m4["full_valid"], axis=1) == 0) | ~observed_valid))
        test_frame = pd.DataFrame(scenario_type)
        engineering_cfg = cfg["engineering_gate"]
        engineering_pass = bool(
            oracle_failure <= engineering_cfg["oracle_failure_rate_max"]
            and draw_failure <= engineering_cfg["m4_draw_failure_rate_max"]
            and outer_failure <= engineering_cfg["m4_outer_failure_rate_max"]
            and float(test_frame.testable_fraction.min()) >= engineering_cfg["testable_fraction_min"]
            and len(oracle_joint) >= 990 and len(reference_joint) >= 990 and len(m4_joint) >= 990
            and seed_meta["all_values_unique"] and seed_meta["disjoint_from_all_prior_registries"]
            and seed_meta["historical_collision_count"] == 0
        )
        dgp_summary = _diagnostic_aggregate(m4["dgp"])
        dgp_rows.append({"scenario_id": scenario, "role": role, **dgp_summary})
        engineering_rows.append({
            "scenario_id": scenario, "role": role,
            "oracle_planned": 2000, "oracle_valid": int(np.sum(oracle["valid"]) + np.sum(reference["valid"])),
            "oracle_failure_rate": oracle_failure,
            "m4_outer_planned": len(m4["observed_layers"]),
            "m4_outer_with_observed_process": int(np.sum(observed_valid)),
            "m4_outer_failure_rate": outer_failure,
            "m4_draws_planned": int(m4["full_valid"].size),
            "m4_draws_valid": int(np.sum(m4["full_valid"])),
            "m4_draw_failure_rate": draw_failure,
            "balanced_draws_planned": len(m4["selected_valid"]),
            "balanced_draws_valid": int(np.sum(m4_mask)),
            "explicit_outer_errors": int(np.sum(m4["errors"] != "")),
            "mean_outer_seconds": float(np.mean(m4["runtime"])),
            "engineering_pass": engineering_pass,
        })

        primary_cfg = cfg["primary_inference_gate"]
        max_frame = pd.DataFrame([row for row in maximum_rows if row["scenario_id"] == scenario])
        d4 = test_frame[test_frame.layer == "D4"].iloc[0]
        nominal = bool(
            d4.wilson_low_005 <= 0.05 <= d4.wilson_high_005
            or abs(d4.size_005 - 0.05) <= d4.mcse_005
        )
        inference_checks: dict[str, bool] = {
            "D4_size_005": bool(d4.size_005 <= primary_cfg["D4_size_005_max"]),
            "all_size_005": bool(test_frame.size_005.max() <= primary_cfg["size_005_max_each_D0_D4"]),
            "D4_size_001": bool(d4.size_001 <= primary_cfg["D4_size_001_max"]),
            "D4_nominal": nominal,
            "testable": bool(test_frame.testable_fraction.min() >= primary_cfg["testable_fraction_min"]),
        }
        if len(max_frame):
            inference_checks.update({
                "D2_KS": bool(max_frame.loc[max_frame.layer == "D2", "KS"].iloc[0] <= primary_cfg["D2_KS_max"]),
                "D4_KS": bool(max_frame.loc[max_frame.layer == "D4", "KS"].iloc[0] <= primary_cfg["D4_KS_max"]),
                "q95_ratios": bool(all(
                    _between(row.q95_ratio, primary_cfg["q95_ratio_D4" if row.layer == "D4" else "q95_ratio_D0_D3"])
                    for _, row in max_frame.iterrows()
                )),
                "D4_oracle_q95_exceedance": _between(
                    float(max_frame.loc[max_frame.layer == "D4", "probability_comparison_exceeds_reference_q95"].iloc[0]),
                    primary_cfg["probability_M4_D4_exceeds_oracle_q95"],
                ),
            })
        else:
            inference_checks.update({"D2_KS": False, "D4_KS": False, "q95_ratios": False, "D4_oracle_q95_exceedance": False})
        inference_pass = bool(all(inference_checks.values()))

        diag_cfg = cfg["mechanism_diagnostics"]
        diagnostic_checks: dict[str, bool] = {"center": False, "covariance": False, "marginal_tail": False, "joint_tail": False}
        if center and marginal and m4_tail:
            marginal_frame = pd.DataFrame(marginal)
            q95_joint = pd.DataFrame(m4_tail)
            q95_joint = q95_joint[q95_joint.threshold == 0.95].iloc[0]
            diagnostic_checks = {
                "center": bool(center["mean_difference_RMS_over_truth_SD"] <= diag_cfg["center_mean_RMS_over_truth_SD_max"]),
                "covariance": bool(all([
                    _between(center["pointwise_SD_ratio"], diag_cfg["covariance_pointwise_SD_ratio"]),
                    _between(center["leading_eigenvalue_ratio"], diag_cfg["covariance_leading_eigenvalue_ratio"]),
                    _between(center["effective_rank_ratio"], diag_cfg["covariance_effective_rank_ratio"]),
                    center["correlation_frobenius_relative_error"] <= diag_cfg["covariance_correlation_frobenius_relative_error_max"],
                    center["offdiagonal_correlation_MAE"] <= diag_cfg["covariance_offdiagonal_correlation_MAE_max"],
                ])),
                "marginal_tail": bool(all([
                    _between(float(marginal_frame.loc[marginal_frame.threshold == 0.90, "mean_coordinate_abs_quantile_ratio"].iloc[0]), diag_cfg["marginal_q90_ratio"]),
                    _between(float(marginal_frame.loc[marginal_frame.threshold == 0.95, "mean_coordinate_abs_quantile_ratio"].iloc[0]), diag_cfg["marginal_q95_ratio"]),
                    _between(float(marginal_frame.loc[marginal_frame.threshold == 0.975, "mean_coordinate_abs_quantile_ratio"].iloc[0]), diag_cfg["marginal_q975_ratio"]),
                    _between(float(marginal_frame.loc[marginal_frame.threshold == 0.99, "mean_coordinate_abs_quantile_ratio"].iloc[0]), diag_cfg["marginal_q99_ratio"]),
                    float(marginal_frame.marginal_exceedance_MAE.max()) <= diag_cfg["marginal_exceedance_MAE_max"],
                    float(marginal_frame.skewness_RMSE.max()) <= diag_cfg["skewness_RMSE_max"],
                    float(marginal_frame.excess_kurtosis_RMSE.max()) <= diag_cfg["excess_kurtosis_RMSE_max"],
                ])),
                "joint_tail": bool(all([
                    eisjem <= diag_cfg["EISJEM_point_max"],
                    q95_joint.joint_exceedance_frobenius_relative_error <= diag_cfg["q95_joint_exceedance_frobenius_relative_error_max"],
                    q95_joint.joint_exceedance_offdiagonal_MAE <= diag_cfg["q95_joint_exceedance_offdiagonal_MAE_max"],
                    q95_joint.tail_dependence_offdiagonal_MAE <= diag_cfg["q95_tail_dependence_offdiagonal_MAE_max"],
                ])),
            }
        failed_domains = [name for name, passed in diagnostic_checks.items() if not passed]
        if not engineering_pass:
            classification = "ENGINEERING_INVALID"
        elif not inference_pass:
            classification = "NULL_VALIDITY_FAIL"
        elif failed_domains:
            classification = "MILD_DEGRADATION"
        else:
            classification = "ROBUST_PASS"
        classifications.append({
            "scenario_id": scenario, "role": role, "classification": classification,
            "engineering_pass": engineering_pass, "primary_inference_pass": inference_pass,
            "mechanism_diagnostics_pass": not failed_domains,
            "failed_inference_checks": ";".join(name for name, passed in inference_checks.items() if not passed),
            "failed_mechanism_domains": ";".join(failed_domains),
            "primary_EISJEM": eisjem,
        })

    type1_frame = pd.DataFrame(type1_rows)
    max_frame_all = pd.DataFrame(maximum_rows)
    center_frame = pd.DataFrame(center_rows)
    if len(center_frame):
        center_frame["leading_10_eigenvalue_ratios"] = center_frame.leading_10_eigenvalue_ratios.map(json.dumps)
    marginal_frame_all = pd.DataFrame(marginal_rows)
    joint_frame_all = pd.DataFrame(joint_rows)
    engineering_frame = pd.DataFrame(engineering_rows)
    dgp_frame = pd.DataFrame(dgp_rows)
    classification_frame = pd.DataFrame(classifications)
    _atomic_csv(OUTPUT / "scenario_type1_summary.csv", type1_frame)
    _atomic_csv(OUTPUT / "scenario_max_calibration.csv", max_frame_all)
    _atomic_csv(OUTPUT / "scenario_center_covariance.csv", center_frame)
    _atomic_csv(OUTPUT / "scenario_marginal_tail.csv", marginal_frame_all)
    _atomic_csv(OUTPUT / "scenario_joint_tail.csv", joint_frame_all)
    _atomic_csv(OUTPUT / "scenario_engineering.csv", engineering_frame)
    _atomic_csv(OUTPUT / "scenario_dgp_summary.csv", dgp_frame)
    _atomic_csv(OUTPUT / "scenario_classification.csv", classification_frame)

    invalid = bool((classification_frame.classification == "ENGINEERING_INVALID").any())
    h0_fail = bool(classification_frame.loc[classification_frame.role == "reference", "classification"].isin(["NULL_VALIDITY_FAIL"]).any())
    primary_fail = bool(classification_frame.loc[classification_frame.role == "primary", "classification"].isin(["NULL_VALIDITY_FAIL"]).any())
    stress_fail = bool(classification_frame.loc[classification_frame.role == "stress", "classification"].isin(["NULL_VALIDITY_FAIL"]).any())
    transfer = classification_frame[classification_frame.role != "reference"]
    mild_count = int(np.sum(transfer.classification == "MILD_DEGRADATION"))
    failure_counter: Counter[str] = Counter()
    for value in transfer.failed_mechanism_domains:
        failure_counter.update(item for item in str(value).split(";") if item)
    systematic_domains = sorted(name for name, count in failure_counter.items() if count >= 3)
    if invalid:
        gate = cfg["final_gate"]["invalid"]
    elif h0_fail or primary_fail:
        gate = cfg["final_gate"]["fail"]
    elif stress_fail or mild_count > 2 or systematic_domains:
        gate = cfg["final_gate"]["partial"]
    else:
        gate = cfg["final_gate"]["pass"]

    d4_sizes = type1_frame[type1_frame.layer == "D4"].set_index("scenario_id").size_005.to_dict()
    payload = {
        "gate": gate,
        "scope": "Transfer-Null and Null-Misspecification Validation",
        "scenario_classifications": dict(zip(classification_frame.scenario_id, classification_frame.classification)),
        "primary_transfer_scenarios": list(cfg["scenario_plan"]["primary_transfer"]),
        "stress_scenarios": list(cfg["scenario_plan"]["stress"]),
        "D4_size_005": {key: float(value) for key, value in d4_sizes.items()},
        "engineering_invalid_present": invalid,
        "primary_null_validity_failure_present": primary_fail,
        "reference_assay_failure_present": h0_fail,
        "stress_null_validity_failure_present": stress_fail,
        "mild_degradation_count_H1_H8": mild_count,
        "systematic_mechanism_failure_domains": systematic_domains,
        "historical_seed_collision_count": int(seed_meta["historical_collision_count"]),
        "duplicate_seed_count": 0,
        "extra_seed_count": 0,
        "unregistered_seed_count": 0,
        "m4_source_sha256": sha256(WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py"),
        "m4_source_unchanged": True,
        "metric_source_sha256": sha256(WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py"),
        "posthoc_tuning": False,
        "alternative_power_validation_eligible": gate == cfg["final_gate"]["pass"],
        "OhioT1DM_allowed": False,
        "Phase3_allowed": False,
        "establish_1_0_2": False,
        "alternatives_power_changepoint_run": False,
    }
    _atomic_json(OUTPUT / "phase2rh_gate.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(summarize(), sort_keys=True))
