from __future__ import annotations

import unittest

import numpy as np

from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.protocol import load_phase2_config


class Phase2GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_phase2_config()

    def test_null_has_no_declared_change_and_is_deterministic(self) -> None:
        first = generate_dataset("N0_complete_balanced_null", 123, self.cfg)
        second = generate_dataset("N0_complete_balanced_null", 123, self.cfg)
        self.assertIsNone(first.true_change_point)
        self.assertTrue(first.is_null)
        np.testing.assert_array_equal(first.states, second.states)
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(first.rewards, second.rewards)

    def test_alternative_change_is_fixed_at_analysis_midpoint(self) -> None:
        dataset = generate_dataset("A0_complete_balanced_change", 456, self.cfg, "moderate")
        expected = dataset.analysis_start + round(
            0.5 * (dataset.analysis_end - dataset.analysis_start)
        )
        self.assertEqual(dataset.true_change_point, expected)
        self.assertFalse(dataset.is_null)

    def test_composition_shift_changes_mask_not_environment_time_definition(self) -> None:
        dataset = generate_dataset("N7_risk_composition_shift_null", 789, self.cfg)
        midpoint = dataset.analysis_start + round(
            0.5 * (dataset.analysis_end - dataset.analysis_start)
        )
        self.assertTrue(np.all(~dataset.observed_mask[dataset.phenotypes > 0, midpoint:]))
        self.assertIsNone(dataset.true_change_point)

    def test_informative_missing_mask_has_gaps_without_future_access_marker(self) -> None:
        dataset = generate_dataset("N8_informative_missing_strong_null", 321, self.cfg)
        self.assertGreater(np.sum(~dataset.observed_mask), 0)
        self.assertIsNone(dataset.true_change_point)


if __name__ == "__main__":
    unittest.main()
