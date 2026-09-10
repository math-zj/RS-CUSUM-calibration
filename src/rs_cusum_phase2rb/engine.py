"""M0/M1/M2 joint-process inference arms with full per-draw FQI refitting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.rs_cusum.patient_balance import action_feature, q_values
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum.support import load_support_rule, screen_candidates
from src.rs_cusum.types import CandidateSupport, RiskSetPanel, TransitionRecord
from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.grids import candidate_grid
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2r.core import (
    CandidateFit,
    evaluate_candidate,
    finite_p,
    fit_candidate,
    fit_side,
    multiplier_draws,
    patient_centered_residual,
)
from src.rs_cusum_phase2.protocol import WORKSPACE


@dataclass(frozen=True)
class PreparedAnalysis:
    dataset: Any
    panel: RiskSetPanel
    candidates: tuple[int, ...]
    support: dict[int, CandidateSupport]
    fits: dict[int, CandidateFit]
    evaluation_design: np.ndarray
    midpoint: int
    global_patient_ids: tuple[str, ...]
    boundaries: dict[int, float]


@dataclass(frozen=True)
class ArmResult:
    arm: str
    distribution: str
    observed: dict[str, float]
    bootstrap: dict[str, np.ndarray]
    p_values: dict[str, float]
    valid_draws: int
    failed_draws: int
    midpoint_process: np.ndarray
    joint_process: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CommonNull:
    records: tuple[TransitionRecord, ...]
    common_side: Any
    base_reward: np.ndarray
    centered_residual: np.ndarray
    record_patient_columns: np.ndarray
    patient_ids: tuple[str, ...]
    ridge_offset: np.ndarray
    ridge_score_error: float
    zero_refit_error: float


@dataclass(frozen=True)
class VectorizedSideResult:
    beta: np.ndarray
    patient_ids: tuple[str, ...]
    contributions: np.ndarray
    W: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    coefficient_delta: np.ndarray


def prepare_analysis(
    scenario: str,
    seed: int,
    cfg: dict[str, Any],
    parent_cfg: dict[str, Any],
    evaluation_states: np.ndarray,
    evaluation_actions: np.ndarray,
) -> PreparedAnalysis:
    dataset = generate_dataset(scenario, int(seed), parent_cfg, "none")
    panel = build_panel(
        dataset, "rs_reentry", int(parent_cfg["simulation"]["post_gap_burnin_transitions"])
    )
    candidates = candidate_grid(
        dataset.analysis_start, dataset.analysis_end,
        cfg["frozen_observed_method"]["candidate_fractions"],
    )
    rule = load_support_rule(WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml", "primary")
    screen = screen_candidates(
        panel.support_view(), candidates, dataset.analysis_start, dataset.analysis_end, rule
    )
    support = {item.candidate: item for item in screen.candidates if item.admissible}
    gamma = float(cfg["frozen_observed_method"]["gamma"])
    alpha = float(cfg["frozen_observed_method"]["ridge_alpha"])
    max_iterations = int(cfg["frozen_observed_method"]["fqi_max_iterations"])
    tolerance = float(cfg["frozen_observed_method"]["fqi_tolerance"])
    fits = {
        candidate: fit_candidate(
            panel, candidate, dataset.analysis_start, dataset.analysis_end,
            gamma, alpha, max_iterations, tolerance, "patient_balanced", "patient",
        )
        for candidate in sorted(support)
    }
    midpoint = dataset.analysis_start + int(
        np.floor(0.5*(dataset.analysis_end-dataset.analysis_start)+0.5)
    )
    global_patient_ids = tuple(sorted(panel.patient_ids))
    boundaries = {
        candidate: boundary_factor(
            item, dataset.analysis_start, dataset.analysis_end,
            len(global_patient_ids), "harmonic_active_patients",
        )
        for candidate, item in support.items()
    }
    return PreparedAnalysis(
        dataset=dataset, panel=panel, candidates=candidates, support=support, fits=fits,
        evaluation_design=action_feature(evaluation_states, evaluation_actions),
        midpoint=midpoint, global_patient_ids=global_patient_ids, boundaries=boundaries,
    )


def observed_process(prepared: PreparedAnalysis) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    candidates = sorted(prepared.fits)
    C, Z = len(candidates), prepared.evaluation_design.shape[0]
    process = np.full((C, Z), np.nan)
    raw = np.full((C, Z), np.nan)
    for c_index, candidate in enumerate(candidates):
        contrast, _, variance = evaluate_candidate(
            prepared.fits[candidate], prepared.evaluation_design
        )
        raw[c_index] = contrast
        process[c_index] = prepared.boundaries[candidate]*contrast/np.sqrt(variance)
    layers = {
        "D3": float(np.max(np.abs(process[:, 0]))),
        "D4": float(np.max(np.abs(process))),
    }
    if prepared.midpoint in candidates:
        fit = prepared.fits[prepared.midpoint]
        contrast, _, variance = evaluate_candidate(fit, prepared.evaluation_design)
        layers.update({
            "D0": float(abs(contrast[0])),
            "D1": float(abs(contrast[0])/np.sqrt(variance[0])),
            "D2": float(np.max(np.abs(contrast)/np.sqrt(variance))),
        })
    return layers, process, raw


def _align_candidate_contributions(
    prepared: PreparedAnalysis, fit: CandidateFit
) -> np.ndarray:
    aligned = np.zeros(
        (len(prepared.global_patient_ids), fit.coefficient_contributions.shape[1]), dtype=float
    )
    locations = {pid: index for index, pid in enumerate(prepared.global_patient_ids)}
    for row, patient_id in enumerate(fit.cluster_ids):
        aligned[locations[patient_id]] = fit.coefficient_contributions[row]
    return aligned


def run_influence_arm(
    prepared: PreparedAnalysis,
    arm: str,
    draws: int,
    seed: int,
    distribution: str = "gaussian",
    global_scale: float = 1.0,
) -> ArmResult:
    if arm not in {"M0", "M1", "M0_global_scale"}:
        raise ValueError(arm)
    observed, _, _ = observed_process(prepared)
    if not observed:
        raise ValueError("no observed admissible statistic")
    candidates = sorted(prepared.fits); Z = prepared.evaluation_design.shape[0]
    xi = multiplier_draws(distribution, draws, len(prepared.global_patient_ids), seed)
    joint = np.full((draws, len(candidates), Z), np.nan)
    raw_mid = None
    for c_index, candidate in enumerate(candidates):
        fit = prepared.fits[candidate]
        coefficient = _align_candidate_contributions(prepared, fit)
        delta = coefficient @ prepared.evaluation_design.T
        raw_star = xi @ delta
        if arm == "M1":
            weighted = xi[:, :, None]*delta[None, :, :]
            active = np.asarray(
                [prepared.global_patient_ids.index(pid) for pid in fit.cluster_ids], dtype=int
            )
            active_weighted = weighted[:, active, :]
            centered = active_weighted-np.mean(active_weighted, axis=1, keepdims=True)
            G = len(active)
            variance_star = G/(G-1.0)*np.sum(centered**2, axis=1)
            normalized = raw_star/np.sqrt(np.maximum(variance_star, 1e-15))
        else:
            _, _, variance = evaluate_candidate(fit, prepared.evaluation_design)
            normalized = raw_star/np.sqrt(variance)[None, :]
        if arm == "M0_global_scale":
            raw_star = raw_star*global_scale
            normalized = normalized*global_scale
        joint[:, c_index, :] = prepared.boundaries[candidate]*normalized
        if candidate == prepared.midpoint:
            raw_mid = raw_star
            normalized_mid = normalized
    layer_draws = {
        "D3": np.max(np.abs(joint[:, :, 0]), axis=1),
        "D4": np.max(np.abs(joint), axis=(1, 2)),
    }
    if raw_mid is not None:
        layer_draws.update({
            "D0": np.abs(raw_mid[:, 0]),
            "D1": np.abs(normalized_mid[:, 0]),
            "D2": np.max(np.abs(normalized_mid), axis=1),
        })
    p_values = {layer: finite_p(observed[layer], values) for layer, values in layer_draws.items()}
    return ArmResult(
        arm=arm, distribution=distribution, observed=observed, bootstrap=layer_draws,
        p_values=p_values, valid_draws=draws, failed_draws=0,
        midpoint_process=(normalized_mid if raw_mid is not None else np.empty((draws, 0))),
        joint_process=joint.reshape(draws, -1),
        diagnostics={
            "refit_count": 0, "support_candidates_before": tuple(sorted(prepared.support)),
            "support_candidates_after": tuple(sorted(prepared.support)),
            "same_multiplier_across_candidates": True, "m1_refit": False,
        },
    )


def build_common_null(prepared: PreparedAnalysis, cfg: dict[str, Any]) -> CommonNull:
    records = tuple(
        sorted(prepared.panel.records, key=lambda r: (r.patient_id, r.elapsed_index))
    )
    gamma = float(cfg["frozen_observed_method"]["gamma"])
    alpha = float(cfg["frozen_observed_method"]["ridge_alpha"])
    max_iterations = int(cfg["frozen_observed_method"]["fqi_max_iterations"])
    tolerance = float(cfg["frozen_observed_method"]["fqi_tolerance"])
    common = fit_side(records, gamma, alpha, max_iterations, tolerance, "patient_balanced")
    gram = common.design.T @ (common.weights[:, None]*common.design)
    ridge_offset = np.linalg.solve(gram, alpha*common.model.beta)
    next_states = np.vstack([record.next_state for record in records])
    base_reward = (
        common.design @ common.model.beta
        - gamma*np.max(q_values(next_states, common.model.beta), axis=1)
        + common.design @ ridge_offset
    )
    centered = patient_centered_residual(common)
    patient_ids = tuple(sorted({record.patient_id for record in records}))
    locations = {pid: index for index, pid in enumerate(patient_ids)}
    record_columns = np.asarray([locations[record.patient_id] for record in records], dtype=int)
    ridge_score = common.design.T @ (common.weights*(
        base_reward+gamma*np.max(q_values(next_states, common.model.beta), axis=1)
        - common.design@common.model.beta
    ))-alpha*common.model.beta
    # Zero perturbation is audited by a genuine full-window frozen FQI refit.
    from src.rs_cusum_phase2r.core import replace_rewards
    zero = fit_side(
        replace_rewards(records, base_reward), gamma, alpha, max_iterations, tolerance,
        "patient_balanced",
    )
    return CommonNull(
        records=records, common_side=common, base_reward=base_reward,
        centered_residual=centered, record_patient_columns=record_columns,
        patient_ids=patient_ids, ridge_offset=ridge_offset,
        ridge_score_error=float(np.linalg.norm(ridge_score)),
        zero_refit_error=float(np.linalg.norm(zero.model.beta-common.model.beta)),
    )


def pseudo_rewards(common: CommonNull, multipliers: np.ndarray) -> np.ndarray:
    multipliers = np.asarray(multipliers, dtype=float)
    if multipliers.ndim != 2 or multipliers.shape[1] != len(common.patient_ids):
        raise ValueError("multiplier shape does not match common-null patients")
    return (
        common.base_reward[None, :]
        + multipliers[:, common.record_patient_columns]*common.centered_residual[None, :]
    )


def _vectorized_side_refit(
    records: Sequence[TransitionRecord],
    rewards: np.ndarray,
    cfg: dict[str, Any],
) -> VectorizedSideResult:
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2 or rewards.shape[1] != len(records):
        raise ValueError("draw reward matrix shape mismatch")
    gamma = float(cfg["frozen_observed_method"]["gamma"])
    alpha = float(cfg["frozen_observed_method"]["ridge_alpha"])
    max_iterations = int(cfg["frozen_observed_method"]["fqi_max_iterations"])
    tolerance = float(cfg["frozen_observed_method"]["fqi_tolerance"])
    states = np.vstack([record.state for record in records])
    next_states = np.vstack([record.next_state for record in records])
    actions = np.asarray([record.action for record in records], dtype=int)
    X = action_feature(states, actions)
    Xn0 = action_feature(next_states, np.zeros(len(records), dtype=int))
    Xn1 = action_feature(next_states, np.ones(len(records), dtype=int))
    record_pid = np.asarray([record.patient_id for record in records], dtype=object)
    patient_ids = tuple(sorted(set(record_pid.tolist())))
    counts = Counter(record_pid.tolist()); G = len(patient_ids)
    weights = np.asarray([1.0/(G*counts[pid]) for pid in record_pid], dtype=float)
    H = X.T @ (weights[:, None]*X)+alpha*np.eye(X.shape[1])
    draws = rewards.shape[0]; beta = np.zeros((draws, X.shape[1]))
    converged = np.zeros(draws, dtype=bool); iterations = np.zeros(draws, dtype=int)
    coefficient_delta = np.full(draws, np.inf)
    for iteration in range(1, max_iterations+1):
        next_max = np.maximum(beta@Xn0.T, beta@Xn1.T)
        rhs = ((rewards+gamma*next_max)*weights[None, :])@X
        beta_new = np.linalg.solve(H, rhs.T).T
        coefficient_delta = np.linalg.norm(beta_new-beta, axis=1)
        newly = (~converged)&(coefficient_delta<=tolerance)
        iterations[newly] = iteration; converged |= newly; beta = beta_new
        if np.all(converged): break
    iterations[~converged] = max_iterations
    q0 = beta@Xn0.T; q1 = beta@Xn1.T
    greedy = q1>q0  # exact ties retain action 0, matching np.argmax
    next_max = np.maximum(q0, q1)
    residual = rewards+gamma*next_max-beta@X.T
    W = np.empty((draws, X.shape[1], X.shape[1]))
    base = X.T @ (weights[:, None]*X)+alpha*np.eye(X.shape[1])
    for draw in range(draws):
        chosen = np.where(greedy[draw, :, None], Xn1, Xn0)
        W[draw] = base-gamma*(X.T @ (weights[:, None]*chosen))
    raw_scores = np.empty((draws, G, X.shape[1]))
    for patient_index, patient_id in enumerate(patient_ids):
        selected = record_pid==patient_id
        raw_scores[:, patient_index, :] = residual[:, selected]@X[selected]/np.sum(selected)
    centered_scores = raw_scores-np.mean(raw_scores, axis=1, keepdims=True)
    eta = np.linalg.solve(W, np.swapaxes(centered_scores, 1, 2)).swapaxes(1, 2)
    contributions = eta/G
    return VectorizedSideResult(
        beta=beta, patient_ids=patient_ids, contributions=contributions, W=W,
        converged=converged, iterations=iterations, coefficient_delta=coefficient_delta,
    )


def run_full_refit_arm(
    prepared: PreparedAnalysis,
    cfg: dict[str, Any],
    draws: int,
    seed: int,
    distribution: str = "gaussian",
) -> ArmResult:
    observed, _, _ = observed_process(prepared)
    if not observed:
        raise ValueError("no observed admissible statistic")
    common = build_common_null(prepared, cfg)
    xi = multiplier_draws(distribution, draws, len(common.patient_ids), seed)
    reward_matrix = pseudo_rewards(common, xi)
    record_keys = [(record.patient_id, record.elapsed_index) for record in common.records]
    candidates = sorted(prepared.fits); Z = prepared.evaluation_design.shape[0]
    joint = np.full((draws, len(candidates), Z), np.nan)
    raw_mid = np.full((draws, Z), np.nan); normalized_mid = np.full((draws, Z), np.nan)
    valid = np.ones(draws, dtype=bool); all_iterations = []
    w_checksum = np.zeros(draws); left_beta_var = []; right_beta_var = []
    for c_index, candidate in enumerate(candidates):
        left_indices = np.asarray([
            index for index, (_, elapsed) in enumerate(record_keys)
            if prepared.dataset.analysis_start <= elapsed < candidate
        ], dtype=int)
        right_indices = np.asarray([
            index for index, (_, elapsed) in enumerate(record_keys)
            if candidate <= elapsed < prepared.dataset.analysis_end
        ], dtype=int)
        left_records = tuple(common.records[index] for index in left_indices)
        right_records = tuple(common.records[index] for index in right_indices)
        left = _vectorized_side_refit(left_records, reward_matrix[:, left_indices], cfg)
        right = _vectorized_side_refit(right_records, reward_matrix[:, right_indices], cfg)
        valid &= left.converged & right.converged
        all_iterations.extend([left.iterations, right.iterations])
        w_checksum += np.linalg.norm(left.W, axis=(1, 2))+np.linalg.norm(right.W, axis=(1, 2))
        left_beta_var.append(float(np.mean(np.var(left.beta, axis=0, ddof=1))))
        right_beta_var.append(float(np.mean(np.var(right.beta, axis=0, ddof=1))))
        beta_diff = left.beta-right.beta
        global_locations = {pid: index for index, pid in enumerate(prepared.global_patient_ids)}
        coefficient = np.zeros((draws, len(prepared.global_patient_ids), beta_diff.shape[1]))
        for index, pid in enumerate(left.patient_ids): coefficient[:, global_locations[pid]] += left.contributions[:, index]
        for index, pid in enumerate(right.patient_ids): coefficient[:, global_locations[pid]] -= right.contributions[:, index]
        active_ids = tuple(sorted(set(left.patient_ids)|set(right.patient_ids)))
        active = np.asarray([global_locations[pid] for pid in active_ids], dtype=int)
        delta = np.einsum("bgp,zp->bgz", coefficient, prepared.evaluation_design)
        G = len(active)
        variance = G/(G-1.0)*np.sum(delta[:, active, :]**2, axis=1)
        contrast = beta_diff@prepared.evaluation_design.T
        normalized = contrast/np.sqrt(np.maximum(variance, 1e-15))
        joint[:, c_index, :] = prepared.boundaries[candidate]*normalized
        if candidate==prepared.midpoint:
            raw_mid = contrast; normalized_mid = normalized
    joint[~valid] = np.nan; raw_mid[~valid] = np.nan; normalized_mid[~valid] = np.nan
    d3 = np.full(draws, np.nan); d4 = np.full(draws, np.nan)
    if np.any(valid):
        d3[valid] = np.max(np.abs(joint[valid, :, 0]), axis=1)
        d4[valid] = np.max(np.abs(joint[valid]), axis=(1, 2))
    layer_draws = {"D3": d3, "D4": d4}
    if prepared.midpoint in candidates:
        d2 = np.full(draws, np.nan)
        if np.any(valid):
            d2[valid] = np.max(np.abs(normalized_mid[valid]), axis=1)
        layer_draws.update({
            "D0": np.abs(raw_mid[:, 0]), "D1": np.abs(normalized_mid[:, 0]),
            "D2": d2,
        })
    valid_count = int(np.sum(valid)); failed_count = draws-valid_count
    p_values = {}
    for layer, values in layer_draws.items():
        finite = values[np.isfinite(values)]
        p_values[layer] = finite_p(observed[layer], finite) if len(finite) else float("nan")
    return ArmResult(
        arm="M2", distribution=distribution, observed=observed, bootstrap=layer_draws,
        p_values=p_values, valid_draws=valid_count, failed_draws=failed_count,
        midpoint_process=(normalized_mid if prepared.midpoint in candidates else np.empty((draws, 0))),
        joint_process=joint.reshape(draws, -1),
        diagnostics={
            "refit_count": draws*2*len(candidates),
            "W_recomputed_count": draws*2*len(candidates),
            "V_recomputed_count": draws*len(candidates),
            "support_candidates_before": tuple(sorted(prepared.support)),
            "support_candidates_after": tuple(sorted(prepared.support)),
            "same_multiplier_across_candidates": True,
            "pseudo_uses_observed_left_right_difference": False,
            "common_source_identical_left_right": True,
            "ridge_score_error": common.ridge_score_error,
            "zero_refit_error": common.zero_refit_error,
            "W_checksum_sd": float(np.std(w_checksum, ddof=1)),
            "mean_iterations": float(np.mean(np.concatenate(all_iterations))),
            "left_right_beta_variance_ratio": float(np.mean(left_beta_var)/np.mean(right_beta_var)),
        },
    )
