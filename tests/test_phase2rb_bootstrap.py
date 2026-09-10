import copy
import inspect
import unittest

import numpy as np

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2r.core import decomposition_statistics, fit_side, replace_rewards
from src.rs_cusum_phase2rb.engine import (
    _vectorized_side_refit,
    build_common_null,
    observed_process,
    prepare_analysis,
    pseudo_rewards,
    run_full_refit_arm,
    run_influence_arm,
)
from src.rs_cusum_phase2rb.protocol import WORKSPACE, load_raw_config


class Phase2RBBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_raw_config()
        cls.parent = load_phase2_config()
        with np.load(
            WORKSPACE / cls.cfg["frozen_observed_method"]["evaluation_grid_file"],
            allow_pickle=False,
        ) as data:
            states = np.asarray(data["states"], dtype=float)
            actions = np.asarray(data["actions"], dtype=int)
        cls.prepared = prepare_analysis(
            "N0_complete_balanced_null", 918273, cls.cfg, cls.parent, states, actions
        )
        cls.common = build_common_null(cls.prepared, cls.cfg)
        cls.m0 = run_influence_arm(cls.prepared, "M0", 5, 111, "gaussian")
        cls.m1 = run_influence_arm(cls.prepared, "M1", 5, 111, "gaussian")
        cls.m2 = run_full_refit_arm(cls.prepared, cls.cfg, 3, 111, "gaussian")

    def test_01_zero_perturbation_recovers_common_null(self):
        self.assertLess(self.common.zero_refit_error, 2e-5)
        self.assertLess(self.common.ridge_score_error, 1e-10)

    def test_02_patient_multiplier_shared_across_all_times(self):
        xi = np.arange(1, len(self.common.patient_ids)+1, dtype=float)[None, :]
        pseudo = pseudo_rewards(self.common, xi)[0]
        perturbation = pseudo-self.common.base_reward
        for patient_index in range(len(self.common.patient_ids)):
            selected = self.common.record_patient_columns==patient_index
            usable = selected & (np.abs(self.common.centered_residual)>1e-10)
            np.testing.assert_allclose(
                perturbation[usable]/self.common.centered_residual[usable], xi[0, patient_index]
            )

    def test_03_patient_multiplier_shared_across_candidates(self):
        self.assertTrue(self.m2.diagnostics["same_multiplier_across_candidates"])
        self.assertEqual(
            self.m2.diagnostics["support_candidates_before"],
            self.m2.diagnostics["support_candidates_after"],
        )

    def test_04_pseudo_reward_api_has_no_left_right_contrast(self):
        self.assertEqual(list(inspect.signature(pseudo_rewards).parameters), ["common", "multipliers"])
        self.assertFalse(self.m2.diagnostics["pseudo_uses_observed_left_right_difference"])

    def test_05_common_source_is_identical_for_left_and_right(self):
        self.assertTrue(self.m2.diagnostics["common_source_identical_left_right"])

    def test_06_every_draw_refits_both_sides_for_every_candidate(self):
        self.assertEqual(self.m2.diagnostics["refit_count"], 3*2*len(self.prepared.fits))

    def test_07_every_draw_recomputes_W(self):
        self.assertEqual(self.m2.diagnostics["W_recomputed_count"], 3*2*len(self.prepared.fits))
        self.assertGreater(self.m2.diagnostics["W_checksum_sd"], 0)

    def test_08_every_draw_recomputes_V(self):
        self.assertEqual(self.m2.diagnostics["V_recomputed_count"], 3*len(self.prepared.fits))

    def test_09_m1_never_refits_fqi(self):
        self.assertEqual(self.m1.diagnostics["refit_count"], 0)
        self.assertFalse(self.m1.diagnostics["m1_refit"])

    def test_10_support_is_frozen_before_bootstrap(self):
        expected = tuple(sorted(self.prepared.support))
        self.assertEqual(self.m0.diagnostics["support_candidates_before"], expected)
        self.assertEqual(self.m1.diagnostics["support_candidates_before"], expected)

    def test_11_bootstrap_cannot_create_or_remove_candidates(self):
        self.assertEqual(
            self.m1.diagnostics["support_candidates_before"],
            self.m1.diagnostics["support_candidates_after"],
        )
        self.assertEqual(self.m1.joint_process.shape[1], len(self.prepared.fits)*64)

    def test_12_same_seed_exactly_reproduces_full_refit(self):
        repeat = run_full_refit_arm(self.prepared, self.cfg, 3, 111, "gaussian")
        for layer in self.m2.bootstrap:
            np.testing.assert_array_equal(self.m2.bootstrap[layer], repeat.bootstrap[layer])

    def test_13_invalid_multiplier_is_exception_not_p1(self):
        with self.assertRaises(ValueError):
            run_influence_arm(self.prepared, "M1", 3, 99, "not-a-distribution")

    def test_14_failed_fqi_draw_is_explicit(self):
        broken = copy.deepcopy(self.cfg)
        broken["frozen_observed_method"]["fqi_max_iterations"] = 1
        result = run_full_refit_arm(self.prepared, broken, 2, 222, "gaussian")
        self.assertEqual(result.failed_draws, 2)
        self.assertTrue(np.isnan(result.p_values["D4"]))

    def test_15_observed_statistic_is_unchanged_across_arms(self):
        expected, _, _ = observed_process(self.prepared)
        self.assertEqual(self.m0.observed, expected)
        self.assertEqual(self.m1.observed, expected)
        self.assertEqual(self.m2.observed, expected)

    def test_16_vectorized_refit_matches_scalar_frozen_fqi(self):
        xi = np.zeros((1, len(self.common.patient_ids)))
        rewards = pseudo_rewards(self.common, xi)
        candidate = self.prepared.midpoint
        indices = np.asarray([
            index for index, record in enumerate(self.common.records)
            if self.prepared.dataset.analysis_start <= record.elapsed_index < candidate
        ])
        records = tuple(self.common.records[index] for index in indices)
        vectorized = _vectorized_side_refit(records, rewards[:, indices], self.cfg)
        scalar = fit_side(
            replace_rewards(records, rewards[0, indices]),
            self.cfg["frozen_observed_method"]["gamma"],
            self.cfg["frozen_observed_method"]["ridge_alpha"],
            self.cfg["frozen_observed_method"]["fqi_max_iterations"],
            self.cfg["frozen_observed_method"]["fqi_tolerance"],
            "patient_balanced",
        )
        np.testing.assert_allclose(vectorized.beta[0], scalar.model.beta, rtol=0, atol=2e-12)

    def test_17_observed_statistic_matches_phase2r_frozen_calculation(self):
        current, _, _ = observed_process(self.prepared)
        frozen, _ = decomposition_statistics(
            self.prepared.fits,
            self.prepared.midpoint,
            self.prepared.evaluation_design,
            self.prepared.dataset.analysis_start,
            self.prepared.dataset.analysis_end,
            3,
            333,
            "gaussian",
        )
        self.assertEqual(set(current), set(frozen))
        np.testing.assert_allclose(
            [current[key] for key in sorted(current)],
            [frozen[key] for key in sorted(frozen)],
            rtol=0,
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()
