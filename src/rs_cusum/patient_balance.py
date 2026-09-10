"""Patient-balanced linear fitted Q iteration and cluster influence scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

from .types import FQIResult, TransitionRecord


def action_feature(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Action-specific linear state feature map with an intercept per action."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int)
    if states.ndim != 2 or actions.shape != (states.shape[0],):
        raise ValueError("states/actions have incompatible shapes")
    if not set(np.unique(actions)).issubset({0, 1}):
        raise ValueError("actions must be binary")
    base = np.column_stack([np.ones(states.shape[0]), states])
    design = np.zeros((states.shape[0], 2 * base.shape[1]), dtype=float)
    for action in (0, 1):
        selected = actions == action
        start = action * base.shape[1]
        design[selected, start : start + base.shape[1]] = base[selected]
    return design


def q_values(states: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return Q(s,0), Q(s,1) for every state."""
    states = np.asarray(states, dtype=float)
    beta = np.asarray(beta, dtype=float)
    expected = 2 * (states.shape[1] + 1)
    if beta.shape != (expected,):
        raise ValueError(f"beta must have shape ({expected},)")
    q = np.empty((states.shape[0], 2), dtype=float)
    for action in (0, 1):
        q[:, action] = action_feature(states, np.full(states.shape[0], action)) @ beta
    return q


def fit_patient_balanced_fqi(
    records: Sequence[TransitionRecord],
    gamma: float,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
    reward_field: str = "reward_binary",
) -> FQIResult:
    """Fit FQI with equal total objective weight for each original patient."""
    if not records:
        raise ValueError("FQI requires at least one transition")
    patient_ids = np.asarray([record.patient_id for record in records], dtype=object)
    counts = Counter(patient_ids.tolist())
    n_patients = len(counts)
    transition_weights = np.asarray(
        [1.0 / (n_patients * counts[patient_id]) for patient_id in patient_ids], dtype=float
    )
    patient_weights = {patient_id: 1.0 / n_patients for patient_id in sorted(counts)}
    return _fit_weighted_fqi(
        records,
        transition_weights,
        patient_weights,
        gamma,
        ridge_alpha,
        max_iterations,
        tolerance,
        reward_field,
        estimator="patient_balanced",
    )


def fit_pooled_fqi(
    records: Sequence[TransitionRecord],
    gamma: float,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
    reward_field: str = "reward_binary",
) -> FQIResult:
    """Normalized pooled benchmark used only for degeneration diagnostics."""
    if not records:
        raise ValueError("FQI requires at least one transition")
    n_transitions = len(records)
    counts = Counter(record.patient_id for record in records)
    transition_weights = np.full(n_transitions, 1.0 / n_transitions)
    patient_weights = {
        patient_id: count / n_transitions for patient_id, count in sorted(counts.items())
    }
    return _fit_weighted_fqi(
        records,
        transition_weights,
        patient_weights,
        gamma,
        ridge_alpha,
        max_iterations,
        tolerance,
        reward_field,
        estimator="normalized_pooled",
    )


def _fit_weighted_fqi(
    records: Sequence[TransitionRecord],
    transition_weights: np.ndarray,
    patient_weights: dict[str, float],
    gamma: float,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
    reward_field: str,
    estimator: str,
) -> FQIResult:
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must lie in [0,1)")
    if ridge_alpha < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid FQI numerical configuration")
    if reward_field not in {"reward_binary", "reward_weighted"}:
        raise ValueError("reward_field must be reward_binary or reward_weighted")

    states = np.vstack([np.asarray(record.state, dtype=float) for record in records])
    next_states = np.vstack([np.asarray(record.next_state, dtype=float) for record in records])
    actions = np.asarray([record.action for record in records], dtype=int)
    rewards = np.asarray([getattr(record, reward_field) for record in records], dtype=float)
    patient_ids = np.asarray([record.patient_id for record in records], dtype=object)
    design = action_feature(states, actions)
    if transition_weights.shape != (len(records),):
        raise ValueError("transition weight shape is inconsistent")
    if not np.isclose(float(transition_weights.sum()), 1.0, atol=1e-12):
        raise ValueError("normalized FQI transition weights must sum to one")
    if np.any(transition_weights <= 0):
        raise ValueError("transition weights must be strictly positive")

    weighted_gram = design.T @ (transition_weights[:, None] * design)
    hessian = weighted_gram + ridge_alpha * np.eye(design.shape[1])
    beta = np.zeros(design.shape[1], dtype=float)
    coefficient_delta = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        targets = rewards + gamma * np.max(q_values(next_states, beta), axis=1)
        rhs = design.T @ (transition_weights * targets)
        beta_new = np.linalg.solve(hessian, rhs)
        coefficient_delta = float(np.linalg.norm(beta_new - beta))
        beta = beta_new
        if coefficient_delta <= tolerance:
            converged = True
            break

    fitted_q = q_values(states, beta)
    next_q = q_values(next_states, beta)
    next_greedy_actions = np.argmax(next_q, axis=1)
    next_greedy_design = action_feature(next_states, next_greedy_actions)
    influence_matrix = (
        design.T
        @ (transition_weights[:, None] * (design - gamma * next_greedy_design))
        + ridge_alpha * np.eye(design.shape[1])
    )
    td_residual = rewards + gamma * np.max(next_q, axis=1) - design @ beta
    raw_scores: dict[str, np.ndarray] = {}
    for patient_id in sorted(set(patient_ids.tolist())):
        selected = patient_ids == patient_id
        raw_scores[patient_id] = np.mean(design[selected] * td_residual[selected, None], axis=0)
    mean_score = np.mean(np.vstack(list(raw_scores.values())), axis=0)
    centered_scores = {patient_id: score - mean_score for patient_id, score in raw_scores.items()}
    influences = {
        patient_id: np.linalg.solve(influence_matrix, score)
        for patient_id, score in centered_scores.items()
    }

    action_counts = {action: int(np.sum(actions == action)) for action in (0, 1)}
    per_patient_action_counts = {
        patient_id: {
            action: int(np.sum((patient_ids == patient_id) & (actions == action)))
            for action in (0, 1)
        }
        for patient_id in sorted(set(patient_ids.tolist()))
    }
    action1_outside_fraction = _outside_range_fraction(states[actions == 1], states[actions == 0])
    action0_outside_fraction = _outside_range_fraction(states[actions == 0], states[actions == 1])
    diagnostics = {
        "estimator": estimator,
        "n_transitions": len(records),
        "n_patients": len(raw_scores),
        "action_counts": action_counts,
        "per_patient_action_counts": per_patient_action_counts,
        "per_patient_total_estimation_weights": dict(patient_weights),
        "patients_missing_action0": [pid for pid, value in per_patient_action_counts.items() if value[0] == 0],
        "patients_missing_action1": [pid for pid, value in per_patient_action_counts.items() if value[1] == 0],
        "design_rank": int(np.linalg.matrix_rank(design)),
        "action0_design_rank": int(np.linalg.matrix_rank(design[actions == 0])) if action_counts[0] else 0,
        "action1_design_rank": int(np.linalg.matrix_rank(design[actions == 1])) if action_counts[1] else 0,
        "weighted_gram_condition": _safe_condition(weighted_gram),
        "ridge_hessian_condition": _safe_condition(hessian),
        "influence_matrix_condition": _safe_condition(influence_matrix),
        "next_greedy_action_counts": {
            action: int(np.sum(next_greedy_actions == action)) for action in (0, 1)
        },
        "beta_l2_norm": float(np.linalg.norm(beta)),
        "td_residual_mean": float(np.mean(td_residual)),
        "td_residual_sd": float(np.std(td_residual)),
        "q0_summary": _summary(fitted_q[:, 0]),
        "q1_summary": _summary(fitted_q[:, 1]),
        "q_gap_summary": _summary(fitted_q[:, 1] - fitted_q[:, 0]),
        "extreme_absolute_q_count": int(np.sum(np.abs(fitted_q) > 1.0e6)),
        "action1_outside_action0_feature_range_fraction": action1_outside_fraction,
        "action0_outside_action1_feature_range_fraction": action0_outside_fraction,
        "extrapolation_flag": bool(action1_outside_fraction > 0 or action0_outside_fraction > 0),
    }
    return FQIResult(
        beta=beta,
        converged=converged,
        iterations=iteration,
        coefficient_delta=coefficient_delta,
        patient_weights=patient_weights,
        transition_weights=transition_weights,
        patient_scores=raw_scores,
        centered_patient_scores=centered_scores,
        patient_influences=influences,
        influence_matrix=influence_matrix,
        diagnostics=diagnostics,
    )


def _safe_condition(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
    }


def _outside_range_fraction(target: np.ndarray, reference: np.ndarray) -> float:
    if target.shape[0] == 0 or reference.shape[0] == 0:
        return 1.0
    lower = np.min(reference, axis=0)
    upper = np.max(reference, axis=0)
    outside = np.any((target < lower) | (target > upper), axis=1)
    return float(np.mean(outside))
