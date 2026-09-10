"""RS-CUSUM candidate contrasts for deterministic Phase-1 toy validation."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .patient_balance import action_feature, fit_patient_balanced_fqi
from .types import CandidateStatistic, CandidateSupport, RiskSetPanel, TransitionRecord


BOUNDARY_SCALINGS = {
    "both_side_patients",
    "harmonic_active_patients",
    "count_effective_risk",
}


def boundary_factor(
    support: CandidateSupport,
    analysis_start: int,
    analysis_end: int,
    total_analysis_patients: int,
    method: str,
) -> float:
    """Outcome-blind dynamic-risk boundary factor frozen in the method spec."""
    if method not in BOUNDARY_SCALINGS:
        raise ValueError(f"unknown boundary scaling {method!r}")
    if total_analysis_patients <= 0:
        raise ValueError("total_analysis_patients must be positive")
    left_length = support.candidate - analysis_start
    right_length = analysis_end - support.candidate
    if left_length <= 0 or right_length <= 0:
        raise ValueError("candidate must be strictly inside the analysis interval")
    nominal = np.sqrt(left_length * right_length / (left_length + right_length))
    if method == "both_side_patients":
        effective = len(support.patients_both_sides)
    elif method == "harmonic_active_patients":
        effective = _harmonic(len(support.left.active_patients), len(support.right.active_patients))
    else:
        effective = _harmonic(support.left.total_count_ess, support.right.total_count_ess)
    return float(nominal * np.sqrt(effective / total_analysis_patients))


def compute_candidate_statistic(
    panel: RiskSetPanel,
    support: CandidateSupport,
    analysis_start: int,
    analysis_end: int,
    evaluation_states: np.ndarray,
    evaluation_actions: np.ndarray,
    gamma: float,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
    boundary_scaling: str = "harmonic_active_patients",
    reward_field: str = "reward_binary",
) -> CandidateStatistic:
    """Fit left/right patient-balanced FQI and compute one fixed candidate."""
    panel.validate()
    if not support.admissible:
        raise ValueError("statistic may only be computed for an outcome-blind admissible candidate")
    if not (analysis_start < support.candidate < analysis_end):
        raise ValueError("support candidate is outside the analysis interval")
    evaluation_states = np.asarray(evaluation_states, dtype=float)
    evaluation_actions = np.asarray(evaluation_actions, dtype=int)
    if evaluation_states.ndim != 2 or evaluation_states.shape[1] != panel.state_dimension:
        raise ValueError("evaluation state dimension does not match the panel")
    if evaluation_actions.shape != (evaluation_states.shape[0],):
        raise ValueError("evaluation action shape is inconsistent")

    left_records = tuple(
        record for record in panel.records if analysis_start <= record.elapsed_index < support.candidate
    )
    right_records = tuple(
        record for record in panel.records if support.candidate <= record.elapsed_index < analysis_end
    )
    if not left_records or not right_records:
        raise ValueError("both sides require transitions")
    left_model = fit_patient_balanced_fqi(
        left_records, gamma, ridge_alpha, max_iterations, tolerance, reward_field
    )
    right_model = fit_patient_balanced_fqi(
        right_records, gamma, ridge_alpha, max_iterations, tolerance, reward_field
    )

    evaluation_design = action_feature(evaluation_states, evaluation_actions)
    contrast = evaluation_design @ (left_model.beta - right_model.beta)
    left_ids = set(left_model.patient_influences)
    right_ids = set(right_model.patient_influences)
    patient_ids = tuple(sorted(left_ids | right_ids))
    patient_contributions = np.zeros((len(patient_ids), len(evaluation_states)), dtype=float)
    for patient_index, patient_id in enumerate(patient_ids):
        coefficient_contribution = np.zeros_like(left_model.beta)
        if patient_id in left_ids:
            coefficient_contribution += left_model.patient_influences[patient_id] / len(left_ids)
        if patient_id in right_ids:
            coefficient_contribution -= right_model.patient_influences[patient_id] / len(right_ids)
        patient_contributions[patient_index] = evaluation_design @ coefficient_contribution

    cluster_count = len(patient_ids)
    if cluster_count <= 1:
        cluster_variance = np.full(len(evaluation_states), np.nan)
    else:
        cluster_variance = (
            cluster_count / (cluster_count - 1.0)
        ) * np.sum(patient_contributions * patient_contributions, axis=0)
        # Algebraically zero centered-cluster influence can leave only roundoff.
        # Clamp machine-precision scale to exact zero before studentization.
        contribution_scale = max(1.0, float(np.max(np.abs(patient_contributions))) ** 2)
        numerical_zero = np.finfo(float).eps * contribution_scale
        cluster_variance[cluster_variance <= numerical_zero] = 0.0
    normalized_values = np.full(len(evaluation_states), np.nan)
    positive_variance = np.isfinite(cluster_variance) & (cluster_variance > 0)

    analysis_patients = {
        record.patient_id
        for record in panel.records
        if analysis_start <= record.elapsed_index < analysis_end
    }
    scale = boundary_factor(
        support, analysis_start, analysis_end, len(analysis_patients), boundary_scaling
    )
    normalized_values[positive_variance] = (
        scale * np.abs(contrast[positive_variance]) / np.sqrt(cluster_variance[positive_variance])
    )
    normalized_max = (
        float(np.max(normalized_values[positive_variance])) if np.any(positive_variance) else float("nan")
    )
    unnormalized_max = float(scale * np.max(np.abs(contrast)))
    l1_integral = float(scale * np.mean(np.abs(contrast)))
    diagnostics = {
        "left_patients": sorted(left_ids),
        "right_patients": sorted(right_ids),
        "patients_both_sides": sorted(left_ids & right_ids),
        "patient_union": list(patient_ids),
        "left_transition_count": len(left_records),
        "right_transition_count": len(right_records),
        "left_action_counts": left_model.diagnostics["action_counts"],
        "right_action_counts": right_model.diagnostics["action_counts"],
        "left_converged": left_model.converged,
        "right_converged": right_model.converged,
        "left_iterations": left_model.iterations,
        "right_iterations": right_model.iterations,
        "left_design_rank": left_model.diagnostics["design_rank"],
        "right_design_rank": right_model.diagnostics["design_rank"],
        "left_ridge_condition": left_model.diagnostics["ridge_hessian_condition"],
        "right_ridge_condition": right_model.diagnostics["ridge_hessian_condition"],
        "left_influence_matrix_condition": left_model.diagnostics["influence_matrix_condition"],
        "right_influence_matrix_condition": right_model.diagnostics["influence_matrix_condition"],
        "zero_variance_evaluation_points": int(np.sum(~positive_variance)),
        "patient_contribution_column_sums_max_abs": float(
            np.max(np.abs(np.sum(patient_contributions, axis=0)))
        ),
    }
    return CandidateStatistic(
        candidate=support.candidate,
        boundary_scaling=boundary_scaling,
        boundary_factor=scale,
        patient_ids=patient_ids,
        evaluation_actions=evaluation_actions,
        evaluation_states=evaluation_states,
        contrast=contrast,
        patient_contributions=patient_contributions,
        cluster_variance=cluster_variance,
        normalized_values=normalized_values,
        normalized_max=normalized_max,
        unnormalized_max=unnormalized_max,
        l1_integral=l1_integral,
        left_model=left_model,
        right_model=right_model,
        diagnostics=diagnostics,
    )


def maximum_over_fixed_candidates(statistics: Sequence[CandidateStatistic]) -> dict[str, float | str]:
    """Aggregate a pre-screened fixed candidate set without estimating a real-data CP."""
    if not statistics:
        return {
            "status": "NOT_TESTABLE_UNDER_SUPPORT_RULE",
            "normalized_max": float("nan"),
            "unnormalized_max": float("nan"),
            "l1_integral_max": float("nan"),
        }
    normalized = np.asarray([item.normalized_max for item in statistics], dtype=float)
    return {
        "status": "STATISTIC_AVAILABLE",
        "normalized_max": float(np.nanmax(normalized)) if np.any(np.isfinite(normalized)) else float("nan"),
        "unnormalized_max": float(max(item.unnormalized_max for item in statistics)),
        "l1_integral_max": float(max(item.l1_integral for item in statistics)),
    }


def _harmonic(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return float(2.0 * left * right / (left + right))
