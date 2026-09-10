from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.rs_cusum.risk_set import rectangular_to_panel
from src.rs_cusum.support import screen_candidates
from src.rs_cusum.types import SupportPanel, SupportRule, SupportTransition


def balanced_support_panel(n_patients: int = 4, transitions: int = 20) -> SupportPanel:
    records = tuple(
        SupportTransition(f"P{patient}", time, (patient + time) % 2)
        for patient in range(n_patients)
        for time in range(transitions)
    )
    return SupportPanel(records, state_dimension=1, clock_type="training")


def equality_rule() -> SupportRule:
    return SupportRule("equality", 20, 0.0, 4, 4, 4, 0.25, 0.25)


class SupportTests(unittest.TestCase):
    def test_threshold_equality_is_admissible(self) -> None:
        screen = screen_candidates(balanced_support_panel(), [10], 0, 20, equality_rule())
        candidate = screen.candidates[0]
        self.assertTrue(candidate.admissible)
        self.assertEqual(candidate.left.action_counts, {0: 20, 1: 20})
        self.assertEqual(candidate.required_count_each_side_action, 20)
        self.assertEqual(candidate.left.maximum_action_patient_share[1], 0.25)

    def test_support_is_reward_blind_by_construction(self) -> None:
        states = np.zeros((4, 21, 1))
        actions = np.asarray([[(patient + time) % 2 for time in range(20)] for patient in range(4)])
        rewards = np.arange(80, dtype=float).reshape(4, 20)
        panel = rectangular_to_panel(states, actions, rewards)
        changed = tuple(replace(record, reward_binary=9999.0) for record in panel.records)
        changed_panel = replace(panel, records=changed)
        original_screen = screen_candidates(panel.support_view(), [10], 0, 20, equality_rule())
        changed_screen = screen_candidates(changed_panel.support_view(), [10], 0, 20, equality_rule())
        self.assertEqual(original_screen, changed_screen)
        with self.assertRaisesRegex(TypeError, "SupportPanel only"):
            screen_candidates(panel, [10], 0, 20, equality_rule())  # type: ignore[arg-type]

    def test_rare_action_fails_despite_large_total_count(self) -> None:
        records = tuple(
            SupportTransition(f"P{patient}", time, int(patient == 0 and time in (1, 11)))
            for patient in range(8)
            for time in range(20)
        )
        panel = SupportPanel(records, 1, "training")
        rule = SupportRule("rare", 2, 0.0, 4, 4, 2, 0.80, 0.40)
        candidate = screen_candidates(panel, [10], 0, 20, rule).candidates[0]
        self.assertFalse(candidate.admissible)
        self.assertIn("left:action1_unique_patients", candidate.failure_reasons)
        self.assertIn("right:action1_unique_patients", candidate.failure_reasons)

    def test_patient_dominance_is_rejected(self) -> None:
        records = []
        for time in range(20):
            records.append(SupportTransition("dominant", time, time % 2))
        for patient in range(1, 5):
            records.extend(
                SupportTransition(f"P{patient}", time, time % 2)
                for time in (0, 10)
            )
        panel = SupportPanel(tuple(records), 1, "training")
        rule = SupportRule("dominance", 1, 0.0, 2, 2, 2, 0.60, 0.60)
        candidate = screen_candidates(panel, [10], 0, 20, rule).candidates[0]
        self.assertFalse(candidate.admissible)
        self.assertTrue(any("dominance" in reason for reason in candidate.failure_reasons))

    def test_empty_admissible_set_has_explicit_status(self) -> None:
        impossible = SupportRule("impossible", 1000, 0.0, 10, 10, 10, 0.1, 0.1)
        screen = screen_candidates(balanced_support_panel(), [5, 10, 15], 0, 20, impossible)
        self.assertEqual(screen.status, "NOT_TESTABLE_UNDER_SUPPORT_RULE")
        self.assertEqual(screen.admissible_candidates, ())


if __name__ == "__main__":
    unittest.main()
