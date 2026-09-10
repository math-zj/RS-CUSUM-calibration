"""Candidate fitting, cluster multiplier calibration, and explicit statuses."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.rs_cusum.patient_balance import (
    action_feature,
    fit_patient_balanced_fqi,
    fit_pooled_fqi,
    q_values,
)
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum.support import screen_candidates
from src.rs_cusum.types import (
    CandidateSupport,
    FQIResult,
    RiskSetPanel,
    SupportRule,
    SupportScreenResult,
    TransitionRecord,
)

from .types import CandidateCore, MethodResult


VALID_STATUSES = {
    "TESTABLE",
    "NOT_TESTABLE_SUPPORT",
    "NUMERICAL_FAILURE",
    "BOOTSTRAP_FAILURE",
    "ZERO_VARIANCE",
    "INVALID_SCENARIO",
}


@dataclass(frozen=True)
class CoreCollection:
    cores: dict[int, CandidateCore]
    errors: dict[int, str]
    estimator: str
    cluster_unit: str


def no_support_rule() -> SupportRule:
    return SupportRule(
        name="no_support_screen",
        absolute_count_floor=0,
        parameter_multiplier=0.0,
        minimum_active_patients_each_side=1,
        minimum_patients_both_sides=0,
        minimum_unique_patients_each_side_action=0,
        maximum_action1_patient_share_each_side=1.0,
        maximum_total_patient_share_each_side=1.0,
    )


def build_support_screen(
    panel: RiskSetPanel,
    candidates: Iterable[int],
    analysis_start: int,
    analysis_end: int,
    rule: SupportRule,
) -> SupportScreenResult:
    return screen_candidates(
        panel.support_view(), candidates, analysis_start, analysis_end, rule
    )


def fit_candidate_cores(
    panel: RiskSetPanel,
    candidates: Iterable[int],
    analysis_start: int,
    analysis_end: int,
    estimator: str,
    cluster_unit: str,
    gamma: float,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
) -> CoreCollection:
    """Fit each raw candidate once; failures are recorded by candidate."""
    if estimator not in {"patient_balanced", "pooled"}:
        raise ValueError(f"unknown estimator {estimator!r}")
    if cluster_unit not in {"patient", "transition"}:
        raise ValueError(f"unknown cluster unit {cluster_unit!r}")
    cores: dict[int, CandidateCore] = {}
    errors: dict[int, str] = {}
    for candidate in sorted(set(int(value) for value in candidates)):
        left = tuple(
            record
            for record in panel.records
            if analysis_start <= record.elapsed_index < candidate
        )
        right = tuple(
            record
            for record in panel.records
            if candidate <= record.elapsed_index < analysis_end
        )
        if not left or not right:
            errors[candidate] = "empty left or right transition set"
            continue
        try:
            core = _fit_one_candidate(
                candidate,
                left,
                right,
                estimator,
                cluster_unit,
                gamma,
                ridge_alpha,
                max_iterations,
                tolerance,
            )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            errors[candidate] = f"{type(error).__name__}: {error}"
            continue
        cores[candidate] = core
    return CoreCollection(cores, errors, estimator, cluster_unit)


def run_calibrated_method(
    collection: CoreCollection,
    support_screen: SupportScreenResult,
    evaluation_states: np.ndarray,
    evaluation_actions: np.ndarray,
    analysis_start: int,
    analysis_end: int,
    total_analysis_patients: int,
    scaling: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> MethodResult:
    """Compute normalized-max statistic and finite-bootstrap-corrected p-value."""
    admissible_support = {
        item.candidate: item for item in support_screen.candidates if item.admissible
    }
    admissible = tuple(sorted(admissible_support))
    support_diag = _support_diagnostics(support_screen)
    if not admissible:
        return MethodResult(
            status="NOT_TESTABLE_SUPPORT",
            observed_statistic=None,
            p_value=None,
            estimated_change_point=None,
            admissible_candidates=(),
            candidate_observed_statistics={},
            bootstrap_statistics=None,
            diagnostics={"support": support_diag, "core_errors": dict(collection.errors)},
        )
    failed = {candidate: collection.errors[candidate] for candidate in admissible if candidate in collection.errors}
    missing = [candidate for candidate in admissible if candidate not in collection.cores]
    if failed or missing:
        return MethodResult(
            status="NUMERICAL_FAILURE",
            observed_statistic=None,
            p_value=None,
            estimated_change_point=None,
            admissible_candidates=admissible,
            candidate_observed_statistics={},
            bootstrap_statistics=None,
            diagnostics={
                "support": support_diag,
                "candidate_failures": failed,
                "missing_candidate_cores": missing,
            },
        )

    evaluation_design = action_feature(evaluation_states, evaluation_actions)
    evaluated: dict[int, dict] = {}
    zero_variance_candidates: list[int] = []
    for candidate in admissible:
        core = collection.cores[candidate]
        evaluated_candidate = _evaluate_candidate(
            core,
            admissible_support[candidate],
            evaluation_design,
            analysis_start,
            analysis_end,
            total_analysis_patients,
            scaling,
        )
        if evaluated_candidate is None:
            zero_variance_candidates.append(candidate)
        else:
            evaluated[candidate] = evaluated_candidate
    if zero_variance_candidates:
        return MethodResult(
            status="ZERO_VARIANCE",
            observed_statistic=None,
            p_value=None,
            estimated_change_point=None,
            admissible_candidates=admissible,
            candidate_observed_statistics={},
            bootstrap_statistics=None,
            diagnostics={
                "support": support_diag,
                "zero_variance_candidates": zero_variance_candidates,
            },
        )

    candidate_statistics = {
        candidate: float(value["observed"]) for candidate, value in evaluated.items()
    }
    observed = max(candidate_statistics.values())
    estimated_cp = min(
        candidate for candidate, value in candidate_statistics.items() if value == observed
    )
    try:
        bootstrap, bootstrap_diag = _bootstrap_maximum(
            collection, evaluated, evaluation_design, bootstrap_draws, bootstrap_seed
        )
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        return MethodResult(
            status="BOOTSTRAP_FAILURE",
            observed_statistic=observed,
            p_value=None,
            estimated_change_point=estimated_cp,
            admissible_candidates=admissible,
            candidate_observed_statistics=candidate_statistics,
            bootstrap_statistics=None,
            diagnostics={
                "support": support_diag,
                "bootstrap_error": f"{type(error).__name__}: {error}",
            },
        )
    p_value = float((1 + np.sum(bootstrap >= observed)) / (bootstrap_draws + 1))
    diagnostics = {
        "support": support_diag,
        "bootstrap": bootstrap_diag,
        "candidate": {
            str(candidate): {
                "boundary_factor": value["boundary_factor"],
                "cluster_count": value["cluster_count"],
                "variance_min": value["variance_min"],
                "variance_max": value["variance_max"],
                "core": collection.cores[candidate].diagnostics,
            }
            for candidate, value in evaluated.items()
        },
    }
    return MethodResult(
        status="TESTABLE",
        observed_statistic=observed,
        p_value=p_value,
        estimated_change_point=estimated_cp,
        admissible_candidates=admissible,
        candidate_observed_statistics=candidate_statistics,
        bootstrap_statistics=bootstrap,
        diagnostics=diagnostics,
    )


def _fit_one_candidate(
    candidate: int,
    left: Sequence[TransitionRecord],
    right: Sequence[TransitionRecord],
    estimator: str,
    cluster_unit: str,
    gamma: float,
    ridge_alpha: float,
    max_iterations: int,
    tolerance: float,
) -> CandidateCore:
    fit = fit_patient_balanced_fqi if estimator == "patient_balanced" else fit_pooled_fqi
    left_model = fit(left, gamma, ridge_alpha, max_iterations, tolerance)
    right_model = fit(right, gamma, ridge_alpha, max_iterations, tolerance)
    if not left_model.converged or not right_model.converged:
        raise FloatingPointError(
            f"FQI did not converge: left={left_model.converged}, right={right_model.converged}"
        )
    left_ids, left_contributions, left_diag = _side_coefficient_contributions(
        left, left_model, gamma, ridge_alpha, cluster_unit
    )
    right_ids, right_contributions, right_diag = _side_coefficient_contributions(
        right, right_model, gamma, ridge_alpha, cluster_unit
    )
    all_ids = tuple(sorted(set(left_ids) | set(right_ids)))
    location = {cluster_id: index for index, cluster_id in enumerate(all_ids)}
    contributions = np.zeros((len(all_ids), len(left_model.beta)), dtype=float)
    for cluster_id, vector in zip(left_ids, left_contributions):
        contributions[location[cluster_id]] += vector
    for cluster_id, vector in zip(right_ids, right_contributions):
        contributions[location[cluster_id]] -= vector
    centered_error = float(np.max(np.abs(np.sum(contributions, axis=0))))
    return CandidateCore(
        candidate=candidate,
        beta_difference=left_model.beta - right_model.beta,
        cluster_ids=all_ids,
        coefficient_contributions=contributions,
        diagnostics={
            "left_transition_count": len(left),
            "right_transition_count": len(right),
            "left_patient_count": len({record.patient_id for record in left}),
            "right_patient_count": len({record.patient_id for record in right}),
            "left_action1_count": int(sum(record.action == 1 for record in left)),
            "right_action1_count": int(sum(record.action == 1 for record in right)),
            "left_fqi_iterations": left_model.iterations,
            "right_fqi_iterations": right_model.iterations,
            "left_design_rank": left_model.diagnostics["design_rank"],
            "right_design_rank": right_model.diagnostics["design_rank"],
            "left_influence_condition": left_diag["influence_condition"],
            "right_influence_condition": right_diag["influence_condition"],
            "cluster_unit": cluster_unit,
            "cluster_count": len(all_ids),
            "centered_contribution_sum_max_abs": centered_error,
        },
    )


def _side_coefficient_contributions(
    records: Sequence[TransitionRecord],
    model: FQIResult,
    gamma: float,
    ridge_alpha: float,
    cluster_unit: str,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, float]]:
    states = np.vstack([record.state for record in records])
    next_states = np.vstack([record.next_state for record in records])
    actions = np.asarray([record.action for record in records], dtype=int)
    rewards = np.asarray([record.reward_binary for record in records], dtype=float)
    design = action_feature(states, actions)
    next_q = q_values(next_states, model.beta)
    greedy = np.argmax(next_q, axis=1)
    next_design = action_feature(next_states, greedy)
    weights = np.asarray(model.transition_weights, dtype=float)
    influence_matrix = (
        design.T @ (weights[:, None] * (design - gamma * next_design))
        + ridge_alpha * np.eye(design.shape[1])
    )
    residual = rewards + gamma * np.max(next_q, axis=1) - design @ model.beta
    row_scores = weights[:, None] * design * residual[:, None]
    if cluster_unit == "patient":
        cluster_ids = tuple(sorted({record.patient_id for record in records}))
        raw = np.vstack(
            [
                np.sum(
                    row_scores[
                        np.asarray([record.patient_id == patient_id for record in records])
                    ],
                    axis=0,
                )
                for patient_id in cluster_ids
            ]
        )
    else:
        cluster_ids = tuple(
            f"{record.patient_id}@{record.elapsed_index}" for record in records
        )
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("transition cluster IDs are not unique")
        raw = row_scores
    centered = raw - np.mean(raw, axis=0, keepdims=True)
    contributions = np.linalg.solve(influence_matrix, centered.T).T
    return cluster_ids, contributions, {
        "influence_condition": float(np.linalg.cond(influence_matrix)),
        "raw_score_sum_norm": float(np.linalg.norm(np.sum(raw, axis=0))),
        "centered_score_sum_max_abs": float(np.max(np.abs(np.sum(centered, axis=0)))),
    }


def _evaluate_candidate(
    core: CandidateCore,
    support: CandidateSupport,
    evaluation_design: np.ndarray,
    analysis_start: int,
    analysis_end: int,
    total_analysis_patients: int,
    scaling: str,
) -> dict | None:
    contrast = evaluation_design @ core.beta_difference
    evaluation_contributions = core.coefficient_contributions @ evaluation_design.T
    clusters = len(core.cluster_ids)
    if clusters <= 1:
        return None
    variance = clusters / (clusters - 1.0) * np.sum(
        evaluation_contributions * evaluation_contributions, axis=0
    )
    numerical_zero = np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(evaluation_contributions))) ** 2
    )
    variance[variance <= numerical_zero] = 0.0
    valid = np.isfinite(variance) & (variance > 0)
    if not np.any(valid):
        return None
    factor = boundary_factor(
        support, analysis_start, analysis_end, total_analysis_patients, scaling
    )
    observed = factor * np.max(np.abs(contrast[valid]) / np.sqrt(variance[valid]))
    return {
        "observed": float(observed),
        "boundary_factor": factor,
        "variance": variance,
        "valid": valid,
        "cluster_count": clusters,
        "variance_min": float(np.min(variance[valid])),
        "variance_max": float(np.max(variance[valid])),
    }


def _bootstrap_maximum(
    collection: CoreCollection,
    evaluated: dict[int, dict],
    evaluation_design: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    global_ids = tuple(
        sorted(
            {
                cluster_id
                for candidate in evaluated
                for cluster_id in collection.cores[candidate].cluster_ids
            }
        )
    )
    if not global_ids:
        raise ValueError("bootstrap cluster set is empty")
    global_location = {cluster_id: index for index, cluster_id in enumerate(global_ids)}
    multipliers = np.random.default_rng(seed).standard_normal((draws, len(global_ids)))
    maximum = np.full(draws, -np.inf)
    for candidate, values in evaluated.items():
        core = collection.cores[candidate]
        columns = [global_location[cluster_id] for cluster_id in core.cluster_ids]
        coefficient_draw = multipliers[:, columns] @ core.coefficient_contributions
        process = coefficient_draw @ evaluation_design.T
        valid = values["valid"]
        candidate_stat = values["boundary_factor"] * np.max(
            np.abs(process[:, valid]) / np.sqrt(values["variance"][valid]), axis=1
        )
        maximum = np.maximum(maximum, candidate_stat)
    if not np.all(np.isfinite(maximum)):
        raise FloatingPointError("bootstrap maximum contains non-finite values")
    return maximum, {
        "draws": draws,
        "seed": seed,
        "global_cluster_count": len(global_ids),
        "cluster_unit": collection.cluster_unit,
        "minimum": float(np.min(maximum)),
        "median": float(np.median(maximum)),
        "maximum": float(np.max(maximum)),
    }


def _support_diagnostics(screen: SupportScreenResult) -> dict:
    reason_counts: Counter[str] = Counter()
    for candidate in screen.candidates:
        reason_counts.update(candidate.failure_reasons)
    if screen.candidates:
        max_total = max(
            max(item.left.maximum_total_patient_share, item.right.maximum_total_patient_share)
            for item in screen.candidates
        )
        max_action1 = max(
            max(
                item.left.maximum_action_patient_share[1],
                item.right.maximum_action_patient_share[1],
            )
            for item in screen.candidates
        )
        mean_active = float(
            np.mean(
                [
                    (len(item.left.active_patients) + len(item.right.active_patients)) / 2
                    for item in screen.candidates
                ]
            )
        )
        minimum_action1 = min(
            min(item.left.action_counts[1], item.right.action_counts[1])
            for item in screen.candidates
        )
    else:
        max_total = max_action1 = mean_active = 0.0
        minimum_action1 = 0
    return {
        "rule": screen.rule_name,
        "raw_candidate_count": len(screen.candidates),
        "admissible_candidate_count": len(screen.admissible_candidates),
        "admissible_candidates": list(screen.admissible_candidates),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "maximum_total_patient_share": max_total,
        "maximum_action1_patient_share": max_action1,
        "mean_active_patients": mean_active,
        "minimum_side_action1_count": minimum_action1,
    }
