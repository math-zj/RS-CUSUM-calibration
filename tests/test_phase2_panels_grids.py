from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.grids import candidate_grid, evaluation_grid
from src.rs_cusum_phase2.panels import build_panel, rs_eligible_mask, strict_eligible_mask
from src.rs_cusum_phase2.protocol import load_phase2_config


class Phase2PanelGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_phase2_config()

    def test_reentry_requires_24_preceding_observed_edges(self) -> None:
        observed = np.ones((1, 100), dtype=bool)
        observed[0, 60:66] = False
        eligible = rs_eligible_mask(observed, 48, 24)
        self.assertTrue(eligible[0, 59])
        self.assertFalse(np.any(eligible[0, 66:90]))
        self.assertTrue(eligible[0, 90])

    def test_strict_never_reenters_after_first_gap(self) -> None:
        observed = np.ones((1, 100), dtype=bool)
        observed[0, 60:66] = False
        strict = strict_eligible_mask(observed, 48)
        self.assertTrue(strict[0, 59])
        self.assertFalse(np.any(strict[0, 60:]))

    def test_candidate_grid_is_fixed_20_to_80_percent(self) -> None:
        self.assertEqual(candidate_grid(48, 240, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), (86, 106, 125, 144, 163, 182, 202))

    def test_evaluation_grid_is_reward_blind_and_patient_balanced(self) -> None:
        dataset = generate_dataset("N0_complete_balanced_null", 777, self.cfg)
        panel = build_panel(dataset, "rs_reentry")
        states1, actions1, sources1 = evaluation_grid(panel, 48, 240, 32, 9173)
        changed_records = tuple(replace(record, reward_binary=99999.0) for record in panel.records)
        changed_panel = replace(panel, records=changed_records)
        states2, actions2, sources2 = evaluation_grid(changed_panel, 48, 240, 32, 9173)
        np.testing.assert_array_equal(states1, states2)
        np.testing.assert_array_equal(actions1, actions2)
        self.assertEqual(sources1, sources2)
        source_patients = [patient for patient, _ in sources1[:32]]
        counts = {patient: source_patients.count(patient) for patient in set(source_patients)}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
