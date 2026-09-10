from __future__ import annotations

import unittest

import numpy as np

from src.rs_cusum.simulation.generator import SimulationDesign, generate_dataset
from src.rs_cusum.simulation.run_full import require_phase2_full_authorization
from src.rs_cusum.simulation.run_pilot import require_phase2_pilot_authorization
from src.rs_cusum.simulation.scenarios import phase1_scenarios


class SimulationGeneratorTests(unittest.TestCase):
    def test_shapes_and_seed_are_deterministic(self) -> None:
        design = SimulationDesign("deterministic", n_patients=5, n_transitions=30, state_dimension=3, seed=10)
        first = generate_dataset(design)
        second = generate_dataset(design)
        self.assertEqual(first.states.shape, (5, 31, 3))
        self.assertEqual(first.actions.shape, (5, 30))
        np.testing.assert_array_equal(first.states, second.states)
        np.testing.assert_array_equal(first.actions, second.actions)

    def test_intermittent_reentry_never_creates_new_patient_ids(self) -> None:
        design = SimulationDesign(
            "reentry", n_patients=12, n_transitions=80, state_dimension=2,
            seed=20, missing_pattern="intermittent_reentry"
        )
        dataset = generate_dataset(design)
        panel = dataset.to_risk_panel()
        segment_pairs = {(record.patient_id, record.segment_id) for record in panel.records}
        self.assertEqual(len(panel.patient_ids), 12)
        self.assertGreater(len(segment_pairs), 12)

    def test_monotone_dropout_has_no_reentry(self) -> None:
        design = SimulationDesign(
            "dropout", n_patients=8, n_transitions=50, state_dimension=2,
            seed=30, missing_pattern="monotone_dropout"
        )
        mask = generate_dataset(design).observed_mask
        for row in mask:
            zero_seen = False
            for value in row:
                if not value:
                    zero_seen = True
                if zero_seen:
                    self.assertFalse(value)

    def test_one_change_index_is_declared(self) -> None:
        design = SimulationDesign(
            "change", n_patients=4, n_transitions=40, state_dimension=2,
            seed=40, alternative="one_change", change_index=17
        )
        self.assertEqual(generate_dataset(design).true_change_index, 17)

    def test_rare_action_scenario_has_lower_action1_rate(self) -> None:
        base = SimulationDesign("base", n_patients=12, n_transitions=200, seed=50)
        rare = SimulationDesign("rare", n_patients=12, n_transitions=200, seed=50, rare_action1=True)
        self.assertLess(np.mean(generate_dataset(rare).actions), np.mean(generate_dataset(base).actions))

    def test_phase1_catalog_and_execution_locks(self) -> None:
        expected = {
            "complete_balanced_null", "complete_balanced_one_change", "monotone_dropout",
            "intermittent_missing_with_reentry", "unequal_length", "single_patient_dominance",
            "rare_action1", "dropout_plus_rare_action1", "intermittent_plus_dominance",
        }
        self.assertEqual(set(phase1_scenarios()), expected)
        with self.assertRaisesRegex(RuntimeError, "locked"):
            require_phase2_pilot_authorization("configs/rs_cusum_simulation.yaml")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            require_phase2_full_authorization("configs/rs_cusum_simulation.yaml")


if __name__ == "__main__":
    unittest.main()
