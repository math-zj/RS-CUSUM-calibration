import unittest

import numpy as np

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2r.core import (
    decomposition_statistics,
    evaluate_candidate,
    fit_candidate,
    make_n0_dataset,
    n0_panel,
    numerical_negative_jacobian,
)


class Phase2RCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = load_phase2_config()
        cls.dataset = make_n0_dataset(cls.parent, 24680, 12, 120, 48)
        cls.panel = n0_panel(cls.dataset)
        cls.fit = fit_candidate(
            cls.panel, 84, 48, 120, 0.9825931938526898, 1.0, 800, 1e-5,
            "patient_balanced", "patient",
        )
        cls.eval_design = action_feature(
            np.zeros((2, 21)), np.asarray([0, 1], dtype=int)
        )

    def test_n0_is_complete_and_rectangular(self):
        self.assertTrue(np.all(self.dataset.observed_mask))
        self.assertEqual(len(self.panel.records), 12*(120-48))
        self.assertEqual(self.fit.cluster_ids, tuple(f"P{i:02d}" for i in range(12)))

    def test_cluster_contributions_are_centered(self):
        self.assertLess(np.max(np.abs(self.fit.coefficient_contributions.sum(axis=0))), 1e-12)

    def test_variance_matches_hc1_formula(self):
        _, contribution, variance = evaluate_candidate(self.fit, self.eval_design)
        expected = 12/11*np.sum(contribution**2, axis=0)
        np.testing.assert_allclose(variance, expected, rtol=0, atol=1e-15)

    def test_d0_d1_have_identical_fixed_variance_p_value(self):
        observed, draws = decomposition_statistics(
            {84: self.fit}, 84, self.eval_design, 48, 120, 199, 991, "gaussian"
        )
        rank0 = np.sum(draws["D0"] >= observed["D0"])
        rank1 = np.sum(draws["D1"] >= observed["D1"])
        self.assertEqual(rank0, rank1)

    def test_analytic_W_matches_numeric_jacobian(self):
        numeric = numerical_negative_jacobian(
            self.fit.left, self.fit.left.model.beta, 0.9825931938526898, 1.0, 1e-6
        )
        np.testing.assert_allclose(
            self.fit.left.model.influence_matrix, numeric, rtol=2e-7, atol=2e-8
        )


if __name__ == "__main__":
    unittest.main()

