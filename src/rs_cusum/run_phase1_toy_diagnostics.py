"""Run only the frozen deterministic Phase-1 degeneration diagnostic.

No OhioT1DM records, p-values, or change-point estimates are accessed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cluster_bootstrap import bootstrap_fixed_candidate_null
from .diagnostics import (
    candidate_statistic_diagnostics,
    risk_panel_diagnostics,
    support_screen_diagnostics,
    write_json,
)
from .patient_balance import action_feature, fit_patient_balanced_fqi, fit_pooled_fqi
from .risk_set import rectangular_to_panel
from .statistic import boundary_factor, compute_candidate_statistic
from .support import screen_candidates
from .types import SupportRule


def run(output_path: str | Path) -> dict:
    seed = 20260828
    rng = np.random.default_rng(seed)
    n_patients, n_steps, state_dimension = 6, 40, 3
    states = rng.normal(size=(n_patients, n_steps + 1, state_dimension))
    actions = np.asarray(
        [[(patient + time) % 2 for time in range(n_steps)] for patient in range(n_patients)]
    )
    rewards = (
        0.4 * states[:, :-1, 0]
        - 0.25 * actions
        + 0.2 * states[:, 1:, 1]
    )
    panel = rectangular_to_panel(states, actions, rewards)
    rule = SupportRule(
        "degeneration_toy", 1, 0.0, n_patients, n_patients, n_patients, 0.20, 0.20
    )
    screen = screen_candidates(panel.support_view(), [n_steps // 2], 0, n_steps, rule)
    support = screen.candidates[0]
    if not support.admissible:
        raise AssertionError(f"balanced degeneration toy unexpectedly rejected: {support.failure_reasons}")

    evaluation_base = np.asarray(
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, -0.5, 0.25], [0.2, 0.4, -0.3]]
    )
    evaluation_states = np.vstack([evaluation_base, evaluation_base])
    evaluation_actions = np.repeat([0, 1], len(evaluation_base))
    gamma = 0.9825931938526898
    alpha = 1.0
    max_iterations = 800
    tolerance = 1.0e-10
    statistics = {
        scaling: compute_candidate_statistic(
            panel,
            support,
            0,
            n_steps,
            evaluation_states,
            evaluation_actions,
            gamma,
            alpha,
            max_iterations,
            tolerance,
            scaling,
        )
        for scaling in (
            "both_side_patients",
            "harmonic_active_patients",
            "count_effective_risk",
        )
    }
    primary = statistics["harmonic_active_patients"]
    left_records = tuple(record for record in panel.records if record.elapsed_index < n_steps // 2)
    right_records = tuple(record for record in panel.records if record.elapsed_index >= n_steps // 2)
    balanced_left = fit_patient_balanced_fqi(
        left_records, gamma, alpha, max_iterations, tolerance
    )
    balanced_right = fit_patient_balanced_fqi(
        right_records, gamma, alpha, max_iterations, tolerance
    )
    pooled_left = fit_pooled_fqi(left_records, gamma, alpha, max_iterations, tolerance)
    pooled_right = fit_pooled_fqi(right_records, gamma, alpha, max_iterations, tolerance)
    design = action_feature(evaluation_states, evaluation_actions)
    pooled_contrast = design @ (pooled_left.beta - pooled_right.beta)
    original_time_factor = float(np.sqrt(20 * 20 / 40))
    pooled_statistic = float(original_time_factor * np.max(np.abs(pooled_contrast)))
    scale_errors = {
        name: abs(item.boundary_factor - original_time_factor)
        for name, item in statistics.items()
    }
    beta_error = max(
        float(np.max(np.abs(balanced_left.beta - pooled_left.beta))),
        float(np.max(np.abs(balanced_right.beta - pooled_right.beta))),
    )
    statistic_error = abs(primary.unnormalized_max - pooled_statistic)
    bootstrap = bootstrap_fixed_candidate_null([primary], n_draws=8, seed=seed)
    bootstrap_repeat = bootstrap_fixed_candidate_null([primary], n_draws=8, seed=seed)
    checks = {
        "risk_panel_equals_rectangular_transition_count": len(panel.records) == n_patients * n_steps,
        "risk_panel_has_original_patient_count": len(panel.patient_ids) == n_patients,
        "patient_balanced_equals_normalized_pooled": beta_error <= 1.0e-10,
        "all_scalings_equal_original_time_factor": max(scale_errors.values()) <= 1.0e-12,
        "rs_unnormalized_equals_rectangular_benchmark": statistic_error <= 1.0e-10,
        "patient_contributions_center": float(
            np.max(np.abs(np.sum(primary.patient_contributions, axis=0)))
        ) <= 1.0e-10,
        "cluster_multiplier_seed_reproducible": bool(
            np.array_equal(bootstrap.multipliers, bootstrap_repeat.multipliers)
            and np.array_equal(bootstrap.unnormalized_max, bootstrap_repeat.unnormalized_max)
        ),
    }
    payload = {
        "phase": "Phase 1 deterministic toy only",
        "formal_analysis": False,
        "p_value_computed": False,
        "change_point_estimated": False,
        "seed": seed,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "dimensions": {
            "patients": n_patients,
            "transitions_each": n_steps,
            "state_dimension": state_dimension,
            "evaluation_points": len(evaluation_states),
        },
        "ridge_objective_note": (
            "The benchmark is normalized pooled FQI with mean loss + alpha||beta||^2. "
            "Legacy sklearn Ridge on unnormalized SSE requires alpha rescaling and is not "
            "the same estimator merely because its API alpha is also written as 1."
        ),
        "maximum_beta_absolute_error": beta_error,
        "original_time_factor": original_time_factor,
        "scaling_absolute_errors": scale_errors,
        "normalized_pooled_unnormalized_statistic": pooled_statistic,
        "rs_unnormalized_statistic": primary.unnormalized_max,
        "statistic_absolute_error": statistic_error,
        "bootstrap_smoke_draws": 8,
        "bootstrap_null_summary_no_pvalue": {
            "unnormalized_min": float(np.min(bootstrap.unnormalized_max)),
            "unnormalized_max": float(np.max(bootstrap.unnormalized_max)),
            "normalized_min": float(np.min(bootstrap.normalized_max)),
            "normalized_max": float(np.max(bootstrap.normalized_max)),
        },
        "risk_panel_diagnostics": risk_panel_diagnostics(panel),
        "support_diagnostics": support_screen_diagnostics(screen),
        "candidate_diagnostics": candidate_statistic_diagnostics(primary),
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    path = Path("results_rs_cusum/phase1/diagnostic_toy_results.json")
    payload = run(path)
    print(f"toy degeneration: {payload['status']}")
    print(f"maximum beta error: {payload['maximum_beta_absolute_error']:.3e}")
    print(f"statistic error: {payload['statistic_absolute_error']:.3e}")
    print(f"output: {path}")


if __name__ == "__main__":
    main()
