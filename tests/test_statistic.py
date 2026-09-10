from __future__ import annotations

import unittest

import numpy as np

from src.rs_cusum.patient_balance import action_feature, fit_pooled_fqi
from src.rs_cusum.risk_set import rectangular_to_panel
from src.rs_cusum.statistic import (
    boundary_factor,
    compute_candidate_statistic,
    maximum_over_fixed_candidates,
)
from src.rs_cusum.support import screen_candidates
from src.rs_cusum.types import SupportRule


def balanced_fixture(identical_patients: bool = False):
    rng = np.random.default_rng(77)
    n, t_steps, p = 6, 40, 2
    states = rng.normal(size=(n, t_steps + 1, p))
    if identical_patients:
        states[:] = states[0]
    actions = np.asarray([[(patient + time) % 2 for time in range(t_steps)] for patient in range(n)])
    rewards = 0.3 * states[:, :-1, 0] - 0.2 * actions + 0.1 * states[:, 1:, 1]
    if identical_patients:
        actions[:] = actions[0]
        rewards[:] = rewards[0]
    panel = rectangular_to_panel(states, actions, rewards)
    rule = SupportRule("toy", 1, 0.0, 6, 6, 6, 0.20, 0.20)
    support = screen_candidates(panel.support_view(), [20], 0, 40, rule).candidates[0]
    return panel, support


class StatisticTests(unittest.TestCase):
    def test_rectangular_degeneration_matches_pooled_benchmark(self) -> None:
        panel, support = balanced_fixture()
        evaluation_states = np.asarray([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5], [0.2, 0.4]])
        evaluation_actions = np.asarray([0, 0, 1, 1])
        statistic = compute_candidate_statistic(
            panel, support, 0, 40, evaluation_states, evaluation_actions,
            gamma=0.5, ridge_alpha=0.1, max_iterations=500, tolerance=1e-10
        )
        left = [record for record in panel.records if record.elapsed_index < 20]
        right = [record for record in panel.records if record.elapsed_index >= 20]
        pooled_left = fit_pooled_fqi(left, 0.5, 0.1, 500, 1e-10)
        pooled_right = fit_pooled_fqi(right, 0.5, 0.1, 500, 1e-10)
        design = action_feature(evaluation_states, evaluation_actions)
        benchmark_contrast = design @ (pooled_left.beta - pooled_right.beta)
        original_factor = np.sqrt(20 * 20 / 40)
        benchmark = original_factor * np.max(np.abs(benchmark_contrast))
        np.testing.assert_allclose(statistic.contrast, benchmark_contrast, atol=1e-10, rtol=1e-10)
        self.assertAlmostEqual(statistic.unnormalized_max, benchmark, places=10)

    def test_all_scalings_equal_original_factor_on_balanced_rectangle(self) -> None:
        _, support = balanced_fixture()
        expected = np.sqrt(20 * 20 / 40)
        for method in ("both_side_patients", "harmonic_active_patients", "count_effective_risk"):
            self.assertAlmostEqual(boundary_factor(support, 0, 40, 6, method), expected, places=12)

    def test_scalings_remain_nonnegative_and_bounded(self) -> None:
        panel, _ = balanced_fixture()
        mask_records = tuple(
            record for record in panel.records
            if record.patient_id == "P000" or record.elapsed_index < 30
        )
        panel = type(panel)(mask_records, panel.feature_names, panel.clock_type)
        rule = SupportRule("permissive", 1, 0.0, 1, 1, 1, 1.0, 1.0)
        support = screen_candidates(panel.support_view(), [20], 0, 40, rule).candidates[0]
        nominal = np.sqrt(20 * 20 / 40)
        factors = [boundary_factor(support, 0, 40, 6, method) for method in (
            "both_side_patients", "harmonic_active_patients", "count_effective_risk"
        )]
        self.assertTrue(all(0.0 <= value <= nominal for value in factors))

    def test_zero_cluster_variance_is_not_divided(self) -> None:
        panel, support = balanced_fixture(identical_patients=True)
        evaluation_states = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        statistic = compute_candidate_statistic(
            panel, support, 0, 40, evaluation_states, np.asarray([0, 1]),
            gamma=0.5, ridge_alpha=0.1, max_iterations=500, tolerance=1e-10
        )
        self.assertTrue(np.isnan(statistic.normalized_max))
        self.assertTrue(np.all(np.isnan(statistic.normalized_values)))

    def test_empty_fixed_candidate_set_is_not_testable(self) -> None:
        result = maximum_over_fixed_candidates([])
        self.assertEqual(result["status"], "NOT_TESTABLE_UNDER_SUPPORT_RULE")
        self.assertTrue(np.isnan(result["normalized_max"]))


if __name__ == "__main__":
    unittest.main()
