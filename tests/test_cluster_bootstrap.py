from __future__ import annotations

import unittest

import numpy as np

from src.rs_cusum.cluster_bootstrap import (
    bootstrap_fixed_candidate_null,
    draw_patient_multipliers,
)
from src.rs_cusum.types import CandidateStatistic, FQIResult


def dummy_fqi() -> FQIResult:
    return FQIResult(
        beta=np.zeros(2), converged=True, iterations=1, coefficient_delta=0.0,
        patient_weights={}, transition_weights=np.ones(1), patient_scores={},
        centered_patient_scores={}, patient_influences={}, influence_matrix=np.eye(2), diagnostics={}
    )


def fake_statistic(candidate: int, scale: float, contributions: np.ndarray) -> CandidateStatistic:
    contributions = np.asarray(contributions, dtype=float)
    variance = (contributions.shape[0] / (contributions.shape[0] - 1)) * np.sum(
        contributions * contributions, axis=0
    )
    model = dummy_fqi()
    return CandidateStatistic(
        candidate=candidate,
        boundary_scaling="toy",
        boundary_factor=scale,
        patient_ids=tuple(f"P{index}" for index in range(contributions.shape[0])),
        evaluation_actions=np.arange(contributions.shape[1]) % 2,
        evaluation_states=np.zeros((contributions.shape[1], 1)),
        contrast=np.zeros(contributions.shape[1]),
        patient_contributions=contributions,
        cluster_variance=variance,
        normalized_values=np.zeros(contributions.shape[1]),
        normalized_max=0.0,
        unnormalized_max=0.0,
        l1_integral=0.0,
        left_model=model,
        right_model=model,
    )


class ClusterBootstrapTests(unittest.TestCase):
    def test_multiplier_is_one_column_per_original_patient(self) -> None:
        patient_ids, multipliers = draw_patient_multipliers(["P0", "P1", "P2"], 5, 123)
        self.assertEqual(patient_ids, ("P0", "P1", "P2"))
        self.assertEqual(multipliers.shape, (5, 3))
        with self.assertRaisesRegex(ValueError, "unique"):
            draw_patient_multipliers(["P0", "P0"], 5, 123)

    def test_seed_is_deterministic(self) -> None:
        statistic = fake_statistic(10, 2.0, np.asarray([[1.0, -1.0], [-1.0, 1.0]]))
        first = bootstrap_fixed_candidate_null([statistic], 7, 999)
        second = bootstrap_fixed_candidate_null([statistic], 7, 999)
        np.testing.assert_array_equal(first.multipliers, second.multipliers)
        np.testing.assert_array_equal(first.unnormalized_max, second.unnormalized_max)

    def test_fixed_candidate_max_matches_manual_calculation(self) -> None:
        first = fake_statistic(10, 1.0, np.asarray([[1.0], [-1.0]]))
        second = fake_statistic(20, 0.5, np.asarray([[2.0], [-2.0]]))
        result = bootstrap_fixed_candidate_null([first, second], 4, 42)
        contrast = result.multipliers[:, 0] - result.multipliers[:, 1]
        expected_first = np.abs(contrast)
        expected_second = 0.5 * np.abs(2.0 * contrast)
        np.testing.assert_allclose(result.unnormalized_max, np.maximum(expected_first, expected_second))
        self.assertFalse(hasattr(result, "p_value"))


if __name__ == "__main__":
    unittest.main()
