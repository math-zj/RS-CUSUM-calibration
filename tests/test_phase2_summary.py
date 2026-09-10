from __future__ import annotations

import unittest

from src.rs_cusum_phase2.summarize import _calibration_summary


def row(status: str, reject: bool | None) -> dict:
    return {
        "scenario": "N",
        "family": "N",
        "effect_level": "none",
        "effect_size": 0.0,
        "variant_id": "primary",
        "method": "RS-full",
        "threshold": "primary",
        "scaling": "harmonic_active_patients",
        "evaluation_grid": "primary",
        "bootstrap_draws": 499,
        "status": status,
        "number_admissible_candidates": 3 if status == "TESTABLE" else 0,
        "mean_active_patients": 10,
        "minimum_side_action1_count": 70,
        "maximum_patient_transition_share": 0.2,
        "maximum_patient_action1_share": 0.3,
        "support_failure_reasons": "{}",
        "reject_001": False if reject is not None else None,
        "reject_005": reject,
        "reject_010": reject,
    }


class Phase2SummaryTests(unittest.TestCase):
    def test_all_and_testable_denominators_are_both_reported(self) -> None:
        summary = _calibration_summary(
            [row("TESTABLE", True), row("TESTABLE", False), row("NOT_TESTABLE_SUPPORT", None)]
        )[0]
        self.assertEqual(summary["total_replicates"], 3)
        self.assertEqual(summary["testable_replicates"], 2)
        self.assertAlmostEqual(summary["rejection_rate_all_005"], 1 / 3)
        self.assertAlmostEqual(summary["rejection_rate_testable_005"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
