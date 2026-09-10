"""Frozen true/fitted component hybrid diagnostics for Phase 2R-E."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.stats import kurtosis, skew

from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2.types import SyntheticDataset
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_phase2rd.engine import (
    FittedGenerativeNull,
    fit_generative_null,
    prepare_bootstrap_analysis,
    simulate_design,
    simulate_rewards,
)
from src.rs_cusum_phase2rd.protocol import load_config as load_phase2rd_config

from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds, stable_seed


def true_model(fitted: FittedGenerativeNull, cfg: dict[str, Any]) -> FittedGenerativeNull:
    values = cfg["hybrid"]["truth_parameters"]
    return replace(
        fitted,
        initial_x0_beta=np.asarray(values["initial_x0_beta"], float),
        initial_x0_sd=float(values["initial_x0_sd"]),
        initial_other_mean=float(values["initial_other_mean"]),
        initial_other_sd=float(values["initial_other_sd"]),
        action_beta=np.asarray(values["action_beta"], float),
        action_random_effect_sd=float(values["action_random_effect_sd"]),
        transition_x0_beta=np.asarray(values["transition_x0_beta"], float),
        transition_x0_sd=float(values["transition_x0_sd"]),
        transition_other_beta=np.asarray(values["transition_other_beta"], float),
        transition_other_sd=float(values["transition_other_sd"]),
        reward_beta=np.asarray(values["reward_beta"], float),
        reward_sd=float(values["reward_sd"]),
        diagnostics={"source": "exact_N0_ground_truth"},
    )


def model_for_arm(
    fitted: FittedGenerativeNull, truth: FittedGenerativeNull, arm: str
) -> FittedGenerativeNull:
    initial = {
        "initial_x0_beta": truth.initial_x0_beta,
        "initial_x0_sd": truth.initial_x0_sd,
        "initial_other_mean": truth.initial_other_mean,
        "initial_other_sd": truth.initial_other_sd,
    }
    policy = {"action_beta": truth.action_beta, "action_random_effect_sd": truth.action_random_effect_sd}
    transition = {
        "transition_x0_beta": truth.transition_x0_beta,
        "transition_x0_sd": truth.transition_x0_sd,
        "transition_other_beta": truth.transition_other_beta,
        "transition_other_sd": truth.transition_other_sd,
    }
    reward = {"reward_beta": truth.reward_beta, "reward_sd": truth.reward_sd}
    if arm == "fitted_all":
        return fitted
    if arm == "true_all":
        return truth
    if arm == "true_initial":
        return replace(fitted, **initial)
    if arm == "true_policy":
        return replace(fitted, **policy)
    if arm == "true_random_effect":
        return replace(fitted, action_random_effect_sd=truth.action_random_effect_sd)
    if arm == "true_transition":
        return replace(fitted, **transition)
    if arm == "true_reward":
        return replace(fitted, **reward)
    if arm == "true_design":
        return replace(fitted, **initial, **policy, **transition)
    if arm == "true_innovation_scales":
        return replace(
            fitted,
            initial_x0_sd=truth.initial_x0_sd,
            initial_other_sd=truth.initial_other_sd,
            transition_x0_sd=truth.transition_x0_sd,
            transition_other_sd=truth.transition_other_sd,
            reward_sd=truth.reward_sd,
        )
    if arm == "true_structural_coefficients":
        return replace(
            fitted,
            initial_x0_beta=truth.initial_x0_beta,
            initial_other_mean=truth.initial_other_mean,
            action_beta=truth.action_beta,
            transition_x0_beta=truth.transition_x0_beta,
            transition_other_beta=truth.transition_other_beta,
            reward_beta=truth.reward_beta,
        )
    raise ValueError(arm)


def simulate_hybrid_dataset(
    model: FittedGenerativeNull, common_seed: int, draw: int, rd_cfg: dict[str, Any]
) -> SyntheticDataset:
    design_seed = stable_seed(common_seed, "hybrid_design", draw)
    reward_seed = stable_seed(common_seed, "hybrid_reward", draw)
    states, actions, _ = simulate_design(model, design_seed, rd_cfg)
    rewards, _ = simulate_rewards(model, states, actions, reward_seed)
    dataset = SyntheticDataset(
        scenario="N0_complete_balanced_null", family="N0_complete_balanced", is_null=True,
        effect_level="none", effect_size=0.0,
        seed=int(stable_seed(common_seed, "hybrid_dataset", draw)),
        states=states, actions=actions, rewards=rewards,
        observed_mask=np.ones_like(actions, dtype=bool), patient_ids=model.patient_ids,
        phenotypes=model.phenotypes.copy(), analysis_start=model.analysis_start,
        analysis_end=model.horizon, true_change_point=None,
    )
    dataset.validate()
    return dataset


def _grid(rd_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(WORKSPACE / rd_cfg["frozen_observed_method"]["evaluation_grid_file"], allow_pickle=False) as data:
        return np.asarray(data["states"], float), np.asarray(data["actions"], int)


def _one_process(
    dataset: SyntheticDataset, rd_cfg: dict[str, Any], parent: dict[str, Any],
    states: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    prepared = prepare_bootstrap_analysis(dataset, rd_cfg, parent, states, actions)
    _, _, joint = observed_arrays(prepared)
    if joint.shape != (7, 64) or np.any(~np.isfinite(joint)):
        raise FloatingPointError("invalid hybrid full process")
    return joint.reshape(-1)


def _parameter_rows(index: int, fitted: FittedGenerativeNull, truth: FittedGenerativeNull) -> list[dict[str, Any]]:
    rows = []
    fields = (
        "initial_x0_beta", "initial_x0_sd", "initial_other_mean", "initial_other_sd",
        "action_beta", "action_random_effect_sd", "transition_x0_beta", "transition_x0_sd",
        "transition_other_beta", "transition_other_sd", "reward_beta", "reward_sd",
    )
    for field in fields:
        estimate = np.atleast_1d(getattr(fitted, field)); target = np.atleast_1d(getattr(truth, field))
        for component, (value, true_value) in enumerate(zip(estimate, target)):
            rows.append({
                "outer_replicate": index, "diagnostic_type": "parameter", "component": field,
                "subcomponent": component, "estimate": float(value), "truth": float(true_value),
                "error": float(value - true_value),
            })
    return rows


def _lag1_mean(values: np.ndarray) -> float:
    values = np.asarray(values, float).reshape(-1, values.shape[-1])
    correlations = []
    for row in values:
        if np.std(row[:-1]) > 0 and np.std(row[1:]) > 0:
            correlations.append(np.corrcoef(row[:-1], row[1:])[0, 1])
    return float(np.mean(correlations)) if correlations else np.nan


def _residual_row(index: int, component: str, values: np.ndarray, scale: float) -> dict[str, Any]:
    flat = np.asarray(values, float).ravel(); standardized = flat / scale
    if values.ndim == 2:
        lag = _lag1_mean(values)
    elif values.ndim == 3:
        lag = _lag1_mean(np.transpose(values, (0, 2, 1)))
    else:
        lag = np.nan
    extreme = np.abs(values) > 2 * scale
    adjacent = extreme[:, :-1] & extreme[:, 1:] if values.ndim == 2 else np.transpose(extreme, (0, 2, 1))[:, :, :-1] & np.transpose(extreme, (0, 2, 1))[:, :, 1:]
    marginal = float(np.mean(extreme))
    return {
        "outer_replicate": index, "diagnostic_type": "innovation", "component": component,
        "subcomponent": 0, "estimate": float(np.std(flat, ddof=1)), "truth": float(scale),
        "error": float(np.std(flat, ddof=1) - scale), "mean": float(np.mean(flat)),
        "skewness": float(skew(standardized, bias=False)),
        "excess_kurtosis": float(kurtosis(standardized, fisher=True, bias=False)),
        "lag1_correlation": lag, "extreme_fraction_abs_gt_2sd": marginal,
        "adjacent_extreme_probability": float(np.mean(adjacent)),
        "adjacent_extreme_ratio_to_independence": float(np.mean(adjacent) / max(marginal**2, 1e-15)),
    }


def _innovation_rows(index: int, dataset: SyntheticDataset, truth: FittedGenerativeNull) -> list[dict[str, Any]]:
    state = dataset.states; action = dataset.actions; current = state[:, :-1]; following = state[:, 1:]
    initial_x0 = state[:, 0, 0] - (truth.initial_x0_beta[0] + truth.initial_x0_beta[1] * state[:, 0, 1])
    initial_other = state[:, 0, 2:] - truth.initial_other_mean
    tx0 = following[:, :, 0] - (
        truth.transition_x0_beta[0] + truth.transition_x0_beta[1] * current[:, :, 0]
        + truth.transition_x0_beta[2] * current[:, :, 1] + truth.transition_x0_beta[3] * action
    )
    tother = following[:, :, 2:] - (
        truth.transition_other_beta[0] + truth.transition_other_beta[1] * current[:, :, 2:]
        + truth.transition_other_beta[2] * current[:, :, [0]]
    )
    reward = dataset.rewards - (
        truth.reward_beta[0] + truth.reward_beta[1] * current[:, :, 0]
        + truth.reward_beta[2] * current[:, :, 1] + truth.reward_beta[3] * current[:, :, 2]
        + truth.reward_beta[4] * action
    )
    rows = [
        _residual_row(index, "initial_x0", initial_x0[:, None], truth.initial_x0_sd),
        _residual_row(index, "initial_other", initial_other[:, None, :], truth.initial_other_sd),
        _residual_row(index, "transition_x0", tx0, truth.transition_x0_sd),
        _residual_row(index, "transition_other", tother, truth.transition_other_sd),
        _residual_row(index, "reward", reward, truth.reward_sd),
    ]
    corr = np.corrcoef(tother.reshape(-1, tother.shape[-1]), rowvar=False); mask = ~np.eye(corr.shape[0], dtype=bool)
    joint_channels = np.column_stack([
        tx0.reshape(-1) / truth.transition_x0_sd,
        np.mean(tother, axis=2).reshape(-1) / (truth.transition_other_sd / np.sqrt(tother.shape[2])),
        reward.reshape(-1) / truth.reward_sd,
    ])
    joint_corr = np.corrcoef(joint_channels, rowvar=False); joint_mask = ~np.eye(joint_corr.shape[0], dtype=bool)
    for row in rows:
        if row["component"] == "transition_other":
            row["mean_abs_cross_coordinate_correlation"] = float(np.mean(np.abs(corr[mask])))
            row["max_abs_cross_coordinate_correlation"] = float(np.max(np.abs(corr[mask])))
        if row["component"] == "transition_x0":
            row["mean_abs_cross_coordinate_correlation"] = float(np.mean(np.abs(joint_corr[joint_mask])))
            row["max_abs_cross_coordinate_correlation"] = float(np.max(np.abs(joint_corr[joint_mask])))
    return rows


def _worker(task: tuple[int, int, int]) -> dict[str, Any]:
    index, outer_seed, common_seed = task
    cfg = load_config(); rd_cfg = load_phase2rd_config("D2"); parent = load_phase2_config(); states, actions = _grid(rd_cfg)
    outer = generate_dataset("N0_complete_balanced_null", int(outer_seed), parent, "none")
    fitted = fit_generative_null(outer, rd_cfg); truth = true_model(fitted, cfg)
    arms = list(cfg["hybrid"]["arms"]); draws = int(cfg["hybrid"]["draws_per_outer_per_arm"])
    arrays = {arm: np.full((draws, 448), np.nan, np.float32) for arm in arms}; failures = {arm: {} for arm in arms}
    started = time.perf_counter()
    for arm in arms:
        model = model_for_arm(fitted, truth, arm)
        for draw in range(draws):
            try:
                dataset = simulate_hybrid_dataset(model, common_seed, draw, rd_cfg)
                arrays[arm][draw] = _one_process(dataset, rd_cfg, parent, states, actions)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures[arm][str(draw)] = f"{type(error).__name__}: {error}"
    diagnostics = _parameter_rows(index, fitted, truth) + _innovation_rows(index, outer, truth)
    return {"index": index, "arrays": arrays, "failures": failures, "diagnostics": diagnostics, "seconds": time.perf_counter() - started}


def _write_csv(path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def run(workers: int) -> dict[str, Any]:
    cfg = load_config(); OUTPUT.mkdir(parents=True, exist_ok=True)
    outer = load_seeds("hybrid_outer_dataset"); common = load_seeds("hybrid_common_draw")
    tasks = [(index, int(outer[index]), int(common[index])) for index in range(len(outer))]
    arms = list(cfg["hybrid"]["arms"]); draws = int(cfg["hybrid"]["draws_per_outer_per_arm"]); r = len(tasks)
    output = {arm: np.full((r, draws, 448), np.nan, np.float32) for arm in arms}
    failure_rows = []; diagnostic_rows = []; runtime_rows = []; started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_worker, task) for task in tasks}; complete = 0
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result(); index = int(result["index"]); complete += 1
                for arm in arms:
                    output[arm][index] = result["arrays"][arm]
                    for draw, message in result["failures"][arm].items():
                        failure_rows.append({"outer_replicate": index, "arm": arm, "draw": draw, "message": message})
                diagnostic_rows.extend(result["diagnostics"]); runtime_rows.append({"outer_replicate": index, "seconds": result["seconds"]})
                print(f"Phase2R-E hybrid: {complete}/{r}, {time.perf_counter()-started:.1f}s", flush=True)
    np.savez_compressed(OUTPUT / "hybrid_process_draws.npz", **output)
    if failure_rows:
        _write_csv(OUTPUT / "hybrid_failures.csv", failure_rows)
    _write_csv(OUTPUT / "hybrid_component_fit_raw.csv", diagnostic_rows)
    _write_csv(OUTPUT / "hybrid_runtime.csv", runtime_rows)
    payload = {
        "outer_replicates": r, "draws_per_arm": r * draws, "arms": arms,
        "failed_draws": len(failure_rows), "common_random_numbers": True,
    }
    (OUTPUT / "hybrid_run_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=1); args = parser.parse_args()
    print(json.dumps(run(args.workers), sort_keys=True))
