from __future__ import annotations

import unittest

import numpy as np

from src.rs_cusum.patient_balance import (
    action_feature,
    fit_patient_balanced_fqi,
    fit_pooled_fqi,
    q_values,
)
from src.rs_cusum.simulation.generator import SimulationDesign, generate_dataset


class PatientBalanceTests(unittest.TestCase):
    def setUp(self) -> None:
        design = SimulationDesign(
            "patient_balance_fixture", n_patients=4, n_transitions=30, state_dimension=3,
            seed=1122, missing_pattern="complete"
        )
        self.records = generate_dataset(design).to_risk_panel().records

    def fit_balanced(self, records):
        return fit_patient_balanced_fqi(records, 0.5, 0.1, 500, 1e-10)

    def test_each_patient_has_equal_total_objective_weight(self) -> None:
        result = self.fit_balanced(self.records)
        ids = sorted({record.patient_id for record in self.records})
        for patient_id in ids:
            selected = [index for index, record in enumerate(self.records) if record.patient_id == patient_id]
            self.assertAlmostEqual(float(np.sum(result.transition_weights[selected])), 1 / len(ids), places=12)

    def test_duplicate_one_patients_rows_does_not_change_balanced_fit(self) -> None:
        base = self.fit_balanced(self.records)
        patient_zero = tuple(record for record in self.records if record.patient_id == "P000")
        duplicated_records = tuple(self.records) + patient_zero
        duplicated = self.fit_balanced(duplicated_records)
        pooled_base = fit_pooled_fqi(self.records, 0.5, 0.1, 500, 1e-10)
        pooled_duplicate = fit_pooled_fqi(duplicated_records, 0.5, 0.1, 500, 1e-10)
        np.testing.assert_allclose(base.beta, duplicated.beta, atol=1e-10, rtol=1e-10)
        self.assertGreater(float(np.linalg.norm(pooled_base.beta - pooled_duplicate.beta)), 1e-5)

    def test_equal_counts_degenerate_to_normalized_pooled_fit(self) -> None:
        balanced = self.fit_balanced(self.records)
        pooled = fit_pooled_fqi(self.records, 0.5, 0.1, 500, 1e-10)
        np.testing.assert_allclose(balanced.beta, pooled.beta, atol=1e-10, rtol=1e-10)

    def test_patient_scores_are_centered(self) -> None:
        result = self.fit_balanced(self.records)
        score_sum = np.sum(np.vstack(list(result.centered_patient_scores.values())), axis=0)
        np.testing.assert_allclose(score_sum, 0.0, atol=1e-12)
        self.assertTrue(result.converged)

    def test_influence_matrix_is_bellman_fixed_point_jacobian(self) -> None:
        result = self.fit_balanced(self.records)
        states = np.vstack([record.state for record in self.records])
        next_states = np.vstack([record.next_state for record in self.records])
        actions = np.asarray([record.action for record in self.records])
        design = action_feature(states, actions)
        greedy = np.argmax(q_values(next_states, result.beta), axis=1)
        next_design = action_feature(next_states, greedy)
        expected = (
            design.T @ (result.transition_weights[:, None] * (design - 0.5 * next_design))
            + 0.1 * np.eye(design.shape[1])
        )
        np.testing.assert_allclose(result.influence_matrix, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
