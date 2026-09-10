"""Frozen engineering tests for Phase 2R-E diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2rd.engine import fit_generative_null, simulate_design, simulate_rewards
from src.rs_cusum_phase2rd.protocol import load_config as load_phase2rd_config
from src.rs_cusum_phase2re.forensics import _tail_matrices, coordinate_metadata, load_existing
from src.rs_cusum_phase2re.hybrid import model_for_arm, simulate_hybrid_dataset, true_model
from src.rs_cusum_phase2re.protocol import load_raw_config


class Phase2RETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_raw_config(); cls.rd = load_phase2rd_config("D2"); cls.parent = load_phase2_config()
        cls.dataset = generate_dataset("N0_complete_balanced_null", 83105731, cls.parent, "none")
        cls.fitted = fit_generative_null(cls.dataset, cls.rd); cls.truth = true_model(cls.fitted, cls.cfg)

    def test_01_parent_failure_is_immutable(self):
        self.assertEqual(self.cfg["protocol"]["parent_gate"], "M4_MECHANISM_JOINT_TAIL_FAIL")
        self.assertFalse(self.cfg["protocol"]["m4_modification"])

    def test_02_coordinate_metadata_is_candidate_major_7_by_64(self):
        metadata = coordinate_metadata(self.cfg)
        self.assertEqual(metadata.shape[0], 448)
        self.assertEqual(metadata.iloc[64].candidate_index, 1)
        self.assertEqual(metadata.iloc[64].evaluation_index, 0)

    def test_03_grid_is_32_states_duplicated_across_actions(self):
        metadata = coordinate_metadata(self.cfg).iloc[:64]
        self.assertEqual(metadata.base_state_index.nunique(), 32)
        self.assertTrue(np.array_equal(metadata.iloc[:32].base_state_index, metadata.iloc[32:].base_state_index))
        self.assertTrue(np.all(metadata.iloc[:32].value_action == 0)); self.assertTrue(np.all(metadata.iloc[32:].value_action == 1))

    def test_04_existing_draw_shapes_are_exact(self):
        truth, m4, clustered = load_existing(self.cfg)
        self.assertEqual(truth.shape, (120, 448)); self.assertEqual(m4.shape, (12000, 448)); self.assertEqual(clustered.shape, (120, 100, 448))

    def test_05_original_p95_gap_is_exactly_reproduced(self):
        truth, m4, _ = load_existing(self.cfg); threshold = np.quantile(np.abs(truth), 0.95, axis=0)
        _, _, tc = _tail_matrices(np.abs(truth) > threshold); _, _, bc = _tail_matrices(np.abs(m4) > threshold); mask = ~np.eye(448, dtype=bool)
        gap = abs(np.quantile(bc[mask], 0.95) - np.quantile(tc[mask], 0.95))
        self.assertAlmostEqual(gap, 0.14181287586688995, places=7)

    def test_06_true_model_matches_frozen_n0_parameters(self):
        np.testing.assert_array_equal(self.truth.action_beta, [0.0, 0.35, 0.20])
        np.testing.assert_array_equal(self.truth.transition_x0_beta, [0.0, 0.72, 0.12, 0.18])
        self.assertEqual(self.truth.action_random_effect_sd, 0.15); self.assertEqual(self.truth.reward_sd, 0.60)

    def test_07_single_component_replacements_are_exact(self):
        policy = model_for_arm(self.fitted, self.truth, "true_policy")
        np.testing.assert_array_equal(policy.action_beta, self.truth.action_beta)
        np.testing.assert_array_equal(policy.transition_x0_beta, self.fitted.transition_x0_beta)
        reward = model_for_arm(self.fitted, self.truth, "true_reward")
        np.testing.assert_array_equal(reward.reward_beta, self.truth.reward_beta)
        np.testing.assert_array_equal(reward.action_beta, self.fitted.action_beta)

    def test_08_common_random_numbers_hold_design_fixed_for_reward_replacement(self):
        fitted = model_for_arm(self.fitted, self.truth, "fitted_all"); reward = model_for_arm(self.fitted, self.truth, "true_reward")
        first = simulate_design(fitted, 9101, self.rd); second = simulate_design(reward, 9101, self.rd)
        np.testing.assert_array_equal(first[0], second[0]); np.testing.assert_array_equal(first[1], second[1])
        first_reward = simulate_rewards(fitted, first[0], first[1], 9201)[0]; second_reward = simulate_rewards(reward, second[0], second[1], 9201)[0]
        self.assertFalse(np.array_equal(first_reward, second_reward))

    def test_09_true_design_and_true_all_share_design(self):
        design = model_for_arm(self.fitted, self.truth, "true_design")
        first = simulate_design(design, 10101, self.rd); second = simulate_design(self.truth, 10101, self.rd)
        np.testing.assert_array_equal(first[0], second[0]); np.testing.assert_array_equal(first[1], second[1])

    def test_10_hybrid_dataset_preserves_bellman_shapes(self):
        dataset = simulate_hybrid_dataset(self.truth, 11101, 0, self.rd)
        self.assertEqual(dataset.states.shape[1], dataset.actions.shape[1] + 1)
        self.assertEqual(dataset.actions.shape, dataset.rewards.shape)

    def test_11_all_hybrid_arms_are_declared_diagnostic_only(self):
        self.assertEqual(len(self.cfg["hybrid"]["arms"]), 10)
        self.assertIn("true_all", self.cfg["hybrid"]["arms"]); self.assertIn("fitted_all", self.cfg["hybrid"]["arms"])
        self.assertIn("no arm is an inference candidate", self.cfg["hybrid"]["purpose"])

    def test_12_every_prohibited_stage_remains_closed(self):
        protocol = self.cfg["protocol"]
        for key in ("d3_allowed", "ohiot1dm_allowed", "phase3_allowed", "transfer_null_allowed", "alternatives_allowed", "power_allowed", "changepoint_allowed", "establish_1_0_2"):
            self.assertFalse(protocol[key])


if __name__ == "__main__":
    unittest.main()
