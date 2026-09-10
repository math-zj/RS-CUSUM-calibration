"""Engineering and regression tests for frozen Phase 2R-D M4."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

import numpy as np

from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2rb.engine import prepare_analysis
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_phase2rd.engine import (
    fit_generative_null,
    prepare_bootstrap_analysis,
    run_m4,
    simulate_dataset,
    simulate_design,
    simulate_rewards,
)
from src.rs_cusum_phase2rd.protocol import WORKSPACE, load_raw_config, stable_seed


class Phase2RDM4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_raw_config()
        cls.parent = load_phase2_config()
        with np.load(
            WORKSPACE / cls.cfg["frozen_observed_method"]["evaluation_grid_file"],
            allow_pickle=False,
        ) as data:
            cls.evaluation_states = np.asarray(data["states"], float)
            cls.evaluation_actions = np.asarray(data["actions"], int)
        # Deliberately not a registered Monte Carlo seed: this is an engineering fixture.
        cls.dataset = generate_dataset(
            "N0_complete_balanced_null", 73519842, cls.parent, "none"
        )
        cls.prepared = prepare_analysis(
            "N0_complete_balanced_null",
            73519842,
            cls.cfg,
            cls.parent,
            cls.evaluation_states,
            cls.evaluation_actions,
        )
        cls.model = fit_generative_null(cls.dataset, cls.cfg)
        cls.simulated, cls.simulated_diagnostics = simulate_dataset(
            cls.model, 24680247, 0, cls.cfg
        )
        cls.bootstrap_prepared = prepare_bootstrap_analysis(
            cls.simulated,
            cls.cfg,
            cls.parent,
            cls.evaluation_states,
            cls.evaluation_actions,
        )
        cls.m4 = run_m4(
            cls.prepared,
            cls.cfg,
            cls.parent,
            cls.evaluation_states,
            cls.evaluation_actions,
            2,
            86420976,
        )

    def test_01_fitted_model_is_stationary_and_candidate_independent(self):
        self.assertFalse(self.model.diagnostics["candidate_specific_parameters"])
        self.assertEqual(self.model.horizon, self.dataset.actions.shape[1])
        self.assertEqual(self.model.analysis_start, self.dataset.analysis_start)

    def test_02_simulated_transition_shapes_are_exact(self):
        n, t = self.simulated.actions.shape
        self.assertEqual(
            self.simulated.states.shape, (n, t + 1, self.model.state_dimension)
        )
        self.assertEqual(self.simulated.rewards.shape, (n, t))
        self.assertEqual(self.simulated.observed_mask.shape, (n, t))

    def test_03_phenotype_is_fixed_and_time_invariant(self):
        np.testing.assert_array_equal(self.simulated.phenotypes, self.dataset.phenotypes)
        np.testing.assert_array_equal(
            self.simulated.states[:, :, 1],
            np.repeat(self.dataset.phenotypes[:, None], self.simulated.states.shape[1], 1),
        )

    def test_04_design_seed_is_exactly_deterministic(self):
        first = simulate_design(self.model, 10101, self.cfg)
        second = simulate_design(self.model, 10101, self.cfg)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_05_different_design_seed_changes_states_and_actions(self):
        first = simulate_design(self.model, 10101, self.cfg)
        second = simulate_design(self.model, 20202, self.cfg)
        self.assertFalse(np.array_equal(first[0], second[0]))
        self.assertFalse(np.array_equal(first[1], second[1]))

    def test_06_reward_seed_is_exactly_deterministic(self):
        first = simulate_rewards(
            self.model, self.simulated.states, self.simulated.actions, 30303
        )[0]
        second = simulate_rewards(
            self.model, self.simulated.states, self.simulated.actions, 30303
        )[0]
        np.testing.assert_array_equal(first, second)

    def test_07_reward_seed_changes_reward_without_changing_design(self):
        states = self.simulated.states.copy()
        actions = self.simulated.actions.copy()
        first = simulate_rewards(self.model, states, actions, 30303)[0]
        second = simulate_rewards(self.model, states, actions, 40404)[0]
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(states, self.simulated.states)
        np.testing.assert_array_equal(actions, self.simulated.actions)

    def test_08_design_and_reward_subseeds_are_disjoint(self):
        design = stable_seed(24680247, "M4", "design", 0)
        reward = stable_seed(24680247, "M4", "reward", 0)
        self.assertNotEqual(design, reward)
        self.assertEqual(self.simulated_diagnostics["design_seed"], design)
        self.assertEqual(self.simulated_diagnostics["reward_seed"], reward)

    def test_09_actions_and_probabilities_are_legal(self):
        self.assertTrue(set(np.unique(self.simulated.actions)).issubset({0, 1}))
        lower, upper = self.cfg["generative_model"]["action_probability_clip"]
        self.assertGreaterEqual(self.simulated_diagnostics["action_probability_min"], lower)
        self.assertLessEqual(self.simulated_diagnostics["action_probability_max"], upper)

    def test_10_transition_alignment_survives_panel_construction(self):
        panel = build_panel(
            self.simulated,
            "rs_reentry",
            int(self.parent["simulation"]["post_gap_burnin_transitions"]),
        )
        locations = {pid: index for index, pid in enumerate(self.simulated.patient_ids)}
        for record in panel.records[:50]:
            patient = locations[record.patient_id]
            time = record.elapsed_index
            np.testing.assert_array_equal(record.state, self.simulated.states[patient, time])
            self.assertEqual(record.action, self.simulated.actions[patient, time])
            self.assertEqual(record.reward_binary, self.simulated.rewards[patient, time])
            np.testing.assert_array_equal(
                record.next_state, self.simulated.states[patient, time + 1]
            )

    def test_11_simulated_dataset_is_complete_stationary_n0(self):
        self.simulated.validate()
        self.assertEqual(self.simulated.scenario, "N0_complete_balanced_null")
        self.assertTrue(self.simulated.is_null)
        self.assertTrue(np.all(self.simulated.observed_mask))

    def test_12_every_draw_rebuilds_all_seven_candidates(self):
        expected = tuple(self.cfg["frozen_observed_method"]["base_candidates"])
        self.assertEqual(tuple(sorted(self.bootstrap_prepared.fits)), expected)
        self.assertEqual(self.m4.diagnostics["all_seven_candidates_fraction"], 1.0)

    def test_13_every_valid_draw_refits_both_sides_of_every_candidate(self):
        valid = int(np.sum(self.m4.valid))
        self.assertEqual(self.m4.diagnostics["refit_count"], valid * 2 * 7)
        self.assertEqual(self.m4.diagnostics["W_recomputed_count"], valid * 2 * 7)
        self.assertEqual(self.m4.diagnostics["V_recomputed_count"], valid * 7)

    def test_14_m4_process_shapes_and_values_are_finite(self):
        self.assertEqual(self.m4.raw.shape, (2, 7, 64))
        self.assertEqual(self.m4.normalized.shape, (2, 7, 64))
        self.assertEqual(self.m4.joint.shape, (2, 7, 64))
        self.assertTrue(np.all(self.m4.valid))
        self.assertTrue(np.all(np.isfinite(self.m4.joint)))

    def test_15_m4_reproducibility_is_exact(self):
        repeated = run_m4(
            self.prepared,
            self.cfg,
            self.parent,
            self.evaluation_states,
            self.evaluation_actions,
            2,
            86420976,
        )
        np.testing.assert_array_equal(self.m4.raw, repeated.raw)
        np.testing.assert_array_equal(self.m4.normalized, repeated.normalized)
        np.testing.assert_array_equal(self.m4.joint, repeated.joint)

    def test_16_m4_changes_both_design_and_reward(self):
        self.assertEqual(self.m4.diagnostics["design_changed_fraction"], 1.0)
        self.assertEqual(self.m4.diagnostics["reward_changed_fraction"], 1.0)
        self.assertGreater(self.m4.diagnostics["design_state_mean_sd"], 0.0)

    def test_17_m4_uses_neither_pseudo_reward_nor_patient_weight(self):
        self.assertFalse(self.m4.diagnostics["pseudo_reward_used"])
        self.assertFalse(self.m4.diagnostics["patient_weight_used"])
        self.assertTrue(self.m4.diagnostics["support_rescreened"])

    def test_18_observed_analysis_is_not_mutated(self):
        before = observed_arrays(self.prepared)
        run_m4(
            self.prepared,
            self.cfg,
            self.parent,
            self.evaluation_states,
            self.evaluation_actions,
            1,
            97531865,
        )
        after = observed_arrays(self.prepared)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_19_incomplete_or_non_n0_input_hard_fails(self):
        mask = self.dataset.observed_mask.copy()
        mask[0, 60] = False
        with self.assertRaises(ValueError):
            fit_generative_null(replace(self.dataset, observed_mask=mask), self.cfg)
        with self.assertRaises(ValueError):
            fit_generative_null(replace(self.dataset, scenario="N1_monotone_dropout_null"), self.cfg)

    def test_20_no_enabled_forbidden_stage_or_fallback(self):
        protocol = self.cfg["protocol"]
        for key in (
            "phase3_allowed",
            "ohiot1dm_allowed",
            "transfer_null_allowed",
            "alternatives_allowed",
            "power_allowed",
            "changepoint_allowed",
            "posthoc_patch_allowed",
        ):
            self.assertFalse(protocol[key])
        self.assertTrue(self.cfg["m4"]["primary_only"])
        self.assertTrue(self.cfg["stop_rules"]["no_new_variant_after_failure"])


if __name__ == "__main__":
    unittest.main()
