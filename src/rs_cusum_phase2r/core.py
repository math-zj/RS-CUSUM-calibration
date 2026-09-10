"""Low-level N0 forensic fits, influence objects, and bootstrap processes."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from src.rs_cusum.patient_balance import (
    action_feature,
    fit_patient_balanced_fqi,
    fit_pooled_fqi,
    q_values,
)
from src.rs_cusum.types import FQIResult, RiskSetPanel, TransitionRecord
from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.types import SyntheticDataset


@dataclass(frozen=True)
class SideFit:
    records: tuple[TransitionRecord, ...]
    model: FQIResult
    design: np.ndarray
    next_greedy_design: np.ndarray
    rewards: np.ndarray
    residual: np.ndarray
    weights: np.ndarray
    patient_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateFit:
    candidate: int
    left: SideFit
    right: SideFit
    beta_difference: np.ndarray
    cluster_ids: tuple[str, ...]
    coefficient_contributions: np.ndarray
    cluster_unit: str


def make_n0_dataset(
    parent_cfg: dict,
    seed: int,
    patients: int = 12,
    horizon: int = 240,
    analysis_start: int = 48,
) -> SyntheticDataset:
    cfg = copy.deepcopy(parent_cfg)
    cfg["simulation"]["patients"] = int(patients)
    cfg["simulation"]["initial_record_history_transitions"] = int(analysis_start)
    cfg["null_scenarios"]["N0_complete_balanced_null"]["horizon"] = int(horizon)
    dataset = generate_dataset("N0_complete_balanced_null", int(seed), cfg, "none")
    if not np.all(dataset.observed_mask):
        raise RuntimeError("forensic N0 must be complete")
    return dataset


def n0_panel(dataset: SyntheticDataset) -> RiskSetPanel:
    return build_panel(dataset, "original_complete")


def fit_candidate(
    panel: RiskSetPanel,
    candidate: int,
    analysis_start: int,
    analysis_end: int,
    gamma: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
    estimator: str = "patient_balanced",
    cluster_unit: str = "patient",
) -> CandidateFit:
    left_records = tuple(
        record for record in panel.records if analysis_start <= record.elapsed_index < candidate
    )
    right_records = tuple(
        record for record in panel.records if candidate <= record.elapsed_index < analysis_end
    )
    if not left_records or not right_records:
        raise ValueError("both candidate sides require transitions")
    left = fit_side(left_records, gamma, alpha, max_iterations, tolerance, estimator)
    right = fit_side(right_records, gamma, alpha, max_iterations, tolerance, estimator)
    if not left.model.converged or not right.model.converged:
        raise FloatingPointError("left or right FQI did not converge")
    left_ids, left_values = side_contributions(left, cluster_unit)
    right_ids, right_values = side_contributions(right, cluster_unit)
    cluster_ids = tuple(sorted(set(left_ids) | set(right_ids)))
    locations = {value: index for index, value in enumerate(cluster_ids)}
    contributions = np.zeros((len(cluster_ids), len(left.model.beta)), dtype=float)
    for cluster_id, vector in zip(left_ids, left_values):
        contributions[locations[cluster_id]] += vector
    for cluster_id, vector in zip(right_ids, right_values):
        contributions[locations[cluster_id]] -= vector
    return CandidateFit(
        candidate=int(candidate),
        left=left,
        right=right,
        beta_difference=left.model.beta - right.model.beta,
        cluster_ids=cluster_ids,
        coefficient_contributions=contributions,
        cluster_unit=cluster_unit,
    )


def fit_side(
    records: Sequence[TransitionRecord],
    gamma: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
    estimator: str,
) -> SideFit:
    if estimator == "patient_balanced":
        model = fit_patient_balanced_fqi(records, gamma, alpha, max_iterations, tolerance)
    elif estimator == "pooled":
        model = fit_pooled_fqi(records, gamma, alpha, max_iterations, tolerance)
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    states = np.vstack([record.state for record in records])
    next_states = np.vstack([record.next_state for record in records])
    actions = np.asarray([record.action for record in records], dtype=int)
    rewards = np.asarray([record.reward_binary for record in records], dtype=float)
    design = action_feature(states, actions)
    next_q = q_values(next_states, model.beta)
    greedy = np.argmax(next_q, axis=1)
    next_design = action_feature(next_states, greedy)
    residual = rewards + gamma * np.max(next_q, axis=1) - design @ model.beta
    return SideFit(
        records=tuple(records),
        model=model,
        design=design,
        next_greedy_design=next_design,
        rewards=rewards,
        residual=residual,
        weights=np.asarray(model.transition_weights, dtype=float),
        patient_ids=tuple(sorted({record.patient_id for record in records})),
    )


def side_contributions(side: SideFit, cluster_unit: str) -> tuple[tuple[str, ...], np.ndarray]:
    if cluster_unit == "patient":
        ids = side.patient_ids
        values = np.vstack(
            [side.model.patient_influences[patient_id] / len(ids) for patient_id in ids]
        )
        return ids, values
    if cluster_unit != "transition":
        raise ValueError(f"unknown cluster unit {cluster_unit!r}")
    ids = tuple(f"{record.patient_id}@{record.elapsed_index}" for record in side.records)
    if len(set(ids)) != len(ids):
        raise ValueError("transition IDs are not unique")
    raw = side.weights[:, None] * side.design * side.residual[:, None]
    centered = raw - np.mean(raw, axis=0, keepdims=True)
    values = np.linalg.solve(side.model.influence_matrix, centered.T).T
    return ids, values


def evaluate_candidate(
    fit: CandidateFit, evaluation_design: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contrast = evaluation_design @ fit.beta_difference
    contributions = fit.coefficient_contributions @ evaluation_design.T
    clusters = len(fit.cluster_ids)
    if clusters <= 1:
        variance = np.full(len(contrast), np.nan)
    else:
        variance = clusters / (clusters - 1.0) * np.sum(contributions**2, axis=0)
    return contrast, contributions, variance


def multiplier_draws(
    distribution: str, draws: int, clusters: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if distribution == "gaussian":
        return rng.standard_normal((draws, clusters))
    if distribution == "rademacher":
        return rng.choice(np.asarray([-1.0, 1.0]), size=(draws, clusters))
    if distribution == "webb_six_point":
        values = np.asarray(
            [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]
        )
        return rng.choice(values, size=(draws, clusters))
    raise ValueError(f"unknown multiplier distribution {distribution!r}")


def finite_p(observed: float, bootstrap: np.ndarray) -> float:
    bootstrap = np.asarray(bootstrap, dtype=float)
    return float((1 + np.sum(bootstrap >= observed)) / (len(bootstrap) + 1))


def decomposition_statistics(
    fits: dict[int, CandidateFit],
    midpoint: int,
    evaluation_design: np.ndarray,
    analysis_start: int,
    analysis_end: int,
    draws: int,
    seed: int,
    distribution: str = "gaussian",
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if midpoint not in fits:
        raise ValueError("midpoint candidate is missing")
    global_ids = tuple(sorted({cid for fit in fits.values() for cid in fit.cluster_ids}))
    location = {cid: index for index, cid in enumerate(global_ids)}
    xi = multiplier_draws(distribution, draws, len(global_ids), seed)
    observed = {key: -np.inf for key in ("D0", "D1", "D2", "D3", "D4")}
    bootstrap = {key: np.full(draws, -np.inf) for key in observed}
    for candidate, fit in sorted(fits.items()):
        contrast, contributions, variance = evaluate_candidate(fit, evaluation_design)
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise FloatingPointError("nonpositive forensic variance")
        columns = [location[cid] for cid in fit.cluster_ids]
        coefficient_draws = xi[:, columns] @ fit.coefficient_contributions
        process = coefficient_draws @ evaluation_design.T
        standardized = np.abs(contrast) / np.sqrt(variance)
        standardized_star = np.abs(process) / np.sqrt(variance)[None, :]
        left = candidate - analysis_start
        right = analysis_end - candidate
        boundary = float(np.sqrt(left * right / (left + right)))
        if candidate == midpoint:
            observed["D0"] = float(abs(contrast[0]))
            observed["D1"] = float(standardized[0])
            observed["D2"] = float(np.max(standardized))
            bootstrap["D0"] = np.abs(process[:, 0])
            bootstrap["D1"] = standardized_star[:, 0]
            bootstrap["D2"] = np.max(standardized_star, axis=1)
        observed["D3"] = max(observed["D3"], float(boundary * standardized[0]))
        observed["D4"] = max(observed["D4"], float(boundary * np.max(standardized)))
        bootstrap["D3"] = np.maximum(
            bootstrap["D3"], boundary * standardized_star[:, 0]
        )
        bootstrap["D4"] = np.maximum(
            bootstrap["D4"], boundary * np.max(standardized_star, axis=1)
        )
    if any(not np.isfinite(value) for value in observed.values()):
        raise FloatingPointError("decomposition statistic is non-finite")
    return observed, bootstrap


def estimating_equation(side: SideFit, beta: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    next_states = np.vstack([record.next_state for record in side.records])
    next_max = np.max(q_values(next_states, beta), axis=1)
    residual = side.rewards + gamma * next_max - side.design @ beta
    return side.design.T @ (side.weights * residual) - alpha * beta


def numerical_negative_jacobian(
    side: SideFit, beta: np.ndarray, gamma: float, alpha: float, step: float
) -> np.ndarray:
    dimension = len(beta)
    jacobian = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        shift = np.zeros(dimension)
        shift[column] = step
        plus = estimating_equation(side, beta + shift, gamma, alpha)
        minus = estimating_equation(side, beta - shift, gamma, alpha)
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return -jacobian


def linearized_error(
    side: SideFit, beta0: np.ndarray, w0: np.ndarray, gamma: float, alpha: float
) -> np.ndarray:
    next_states = np.vstack([record.next_state for record in side.records])
    next_max = np.max(q_values(next_states, beta0), axis=1)
    residual = side.rewards + gamma * next_max - side.design @ beta0
    patient_vector = []
    record_ids = np.asarray([record.patient_id for record in side.records], dtype=object)
    for patient_id in side.patient_ids:
        selected = record_ids == patient_id
        patient_vector.append(np.mean(side.design[selected] * residual[selected, None], axis=0))
    psi = np.mean(np.vstack(patient_vector), axis=0) - alpha * beta0
    return np.linalg.solve(w0, psi)


def greedy_diagnostics(
    side: SideFit,
    epsilons: Iterable[float],
    gap_thresholds: Iterable[float],
    directions: np.ndarray,
) -> tuple[dict[float, float], dict[float, tuple[float, float]]]:
    next_states = np.vstack([record.next_state for record in side.records])
    base_q = q_values(next_states, side.model.beta)
    base_action = np.argmax(base_q, axis=1)
    gap = np.abs(base_q[:, 1] - base_q[:, 0])
    gap_fraction = {float(value): float(np.mean(gap <= value)) for value in gap_thresholds}
    switches: dict[float, tuple[float, float]] = {}
    for epsilon in epsilons:
        values = []
        for direction in directions:
            for sign in (-1.0, 1.0):
                perturbed = side.model.beta + sign * float(epsilon) * direction
                values.append(float(np.mean(np.argmax(q_values(next_states, perturbed), axis=1) != base_action)))
        switches[float(epsilon)] = (float(np.mean(values)), float(np.max(values)))
    return gap_fraction, switches


def reference_model(
    dataset: SyntheticDataset,
    gamma: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
) -> SideFit:
    panel = n0_panel(dataset)
    records = tuple(
        record
        for record in panel.records
        if dataset.analysis_start <= record.elapsed_index < dataset.analysis_end
    )
    return fit_side(records, gamma, alpha, max_iterations, tolerance, "pooled")


def jackknife_pseudo_contributions(
    panel: RiskSetPanel,
    candidate: int,
    analysis_start: int,
    analysis_end: int,
    gamma: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
    evaluation_design: np.ndarray,
) -> tuple[np.ndarray, float]:
    patient_ids = panel.patient_ids
    leave_one = []
    for omitted in patient_ids:
        records = tuple(record for record in panel.records if record.patient_id != omitted)
        subpanel = RiskSetPanel(records, panel.feature_names, panel.clock_type, panel.interval_min, panel.edge_gap_min)
        fit = fit_candidate(
            subpanel,
            candidate,
            analysis_start,
            analysis_end,
            gamma,
            alpha,
            max_iterations,
            tolerance,
            "patient_balanced",
            "patient",
        )
        leave_one.append(float(evaluation_design[0] @ fit.beta_difference))
    values = np.asarray(leave_one)
    mean = float(np.mean(values))
    clusters = len(values)
    pseudo = -(clusters - 1.0) / clusters * (values - mean)
    variance = clusters / (clusters - 1.0) * float(np.sum(pseudo**2))
    return pseudo, variance


def replace_rewards(records: Sequence[TransitionRecord], rewards: np.ndarray) -> tuple[TransitionRecord, ...]:
    if len(records) != len(rewards):
        raise ValueError("replacement reward length mismatch")
    return tuple(
        replace(record, reward_binary=float(reward), reward_weighted=float(reward))
        for record, reward in zip(records, rewards)
    )


def patient_centered_residual(side: SideFit) -> np.ndarray:
    values = np.asarray(side.residual, dtype=float).copy()
    ids = np.asarray([record.patient_id for record in side.records], dtype=object)
    for patient_id in side.patient_ids:
        selected = ids == patient_id
        values[selected] -= np.mean(values[selected])
    return values


def summary_condition(side: SideFit, alpha: float) -> dict[str, float]:
    gram = side.design.T @ (side.weights[:, None] * side.design)
    hessian = gram + alpha * np.eye(gram.shape[0])
    w = side.model.influence_matrix
    return {
        "ridge_condition": float(np.linalg.cond(hessian)),
        "W_condition": float(np.linalg.cond(w)),
        "W_min_singular": float(np.min(np.linalg.svd(w, compute_uv=False))),
        "W_max_singular": float(np.max(np.linalg.svd(w, compute_uv=False))),
        "W_minus_hessian_fro": float(np.linalg.norm(w - hessian)),
        "beta_norm": float(np.linalg.norm(side.model.beta)),
        "q_abs_max": float(
            np.max(
                np.abs(
                    q_values(
                        np.vstack([record.state for record in side.records]), side.model.beta
                    )
                )
            )
        ),
    }
