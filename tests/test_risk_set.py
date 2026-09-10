from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from src.rs_cusum.risk_set import build_risk_set_panel, rounded_elapsed_index
from src.rs_cusum.types import RiskSetPanel, TransitionRecord


class FakeSegment:
    def __init__(self, patient_id: str, segment_id: int, start: datetime, transitions: int):
        self.patient_id = patient_id
        self.split = "training"
        self.segment_id = segment_id
        self.tau = np.asarray(
            [start + timedelta(minutes=5 * index) for index in range(transitions + 1)],
            dtype="datetime64[ns]",
        )
        self.states = np.column_stack(
            [np.arange(transitions + 1, dtype=float), np.ones(transitions + 1)]
        )
        self.actions = np.asarray([index % 2 for index in range(transitions)], dtype=int)
        self.rewards_binary = np.asarray([index % 3 == 0 for index in range(transitions)], dtype=float)
        self.rewards_weighted = self.rewards_binary * 2.0 - 1.0

    def validate(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("bad fixture")


def config() -> dict:
    return {
        "risk_panel": {
            "interval_min": 5,
            "max_abs_lattice_residual_min": 0.5,
            "transition_edge_gap_min": [4.5, 5.5],
        },
        "reentry": {
            "initial_record_history_min": 240,
            "post_gap_required_valid_5min_edges": 24,
            "main_rule": "allow_after_continuous_120min_cgm_burnin",
        },
    }


class RiskSetTests(unittest.TestCase):
    def test_elapsed_rounding_half_up_and_residual(self) -> None:
        start = datetime(2020, 1, 1)
        index, residual = rounded_elapsed_index(start + timedelta(minutes=12.5), start, 5)
        self.assertEqual(index, 3)
        self.assertAlmostEqual(residual, -2.5)

    def test_reentry_keeps_original_patient_and_uncompressed_time(self) -> None:
        start = datetime(2020, 1, 1)
        segments = [
            FakeSegment("540", 0, start, 60),
            FakeSegment("540", 1, start + timedelta(minutes=500), 30),
        ]
        panel = build_risk_set_panel(
            segments, {"540": start}, ("x", "constant"), "training", config()
        )
        first_segment = [record for record in panel.records if record.segment_id == 0]
        second_segment = [record for record in panel.records if record.segment_id == 1]
        self.assertEqual(len(first_segment), 12)
        self.assertEqual(len(second_segment), 6)
        self.assertEqual(second_segment[0].elapsed_index, 124)
        self.assertEqual(panel.patient_ids, ("540",))
        self.assertEqual({record.patient_id for record in second_segment}, {"540"})

    def test_sparse_reentry_is_explicit_sensitivity(self) -> None:
        start = datetime(2020, 1, 1)
        segments = [
            FakeSegment("540", 0, start, 60),
            FakeSegment("540", 1, start + timedelta(minutes=500), 30),
        ]
        panel = build_risk_set_panel(
            segments,
            {"540": start},
            ("x", "constant"),
            "training",
            config(),
            reentry_rule="allow_sparse_real_history_after_gap",
        )
        second_segment = [record for record in panel.records if record.segment_id == 1]
        self.assertEqual(len(second_segment), 30)
        self.assertEqual(second_segment[0].elapsed_index, 100)

    def test_duplicate_elapsed_index_is_hard_error(self) -> None:
        start = datetime(2020, 1, 1)
        segments = [
            FakeSegment("540", 0, start, 60),
            FakeSegment("540", 1, start + timedelta(minutes=240), 30),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate patient/elapsed"):
            build_risk_set_panel(
                segments,
                {"540": start},
                ("x", "constant"),
                "training",
                config(),
                reentry_rule="allow_sparse_real_history_after_gap",
            )

    def test_lattice_residual_is_hard_error(self) -> None:
        start = datetime(2020, 1, 1)
        shifted = FakeSegment("540", 0, start + timedelta(minutes=241), 2)
        with self.assertRaisesRegex(ValueError, "lattice residual"):
            build_risk_set_panel([shifted], {"540": start}, ("x", "constant"), "training", config())

    def test_clock_mixing_is_rejected(self) -> None:
        start = datetime(2020, 1, 1)
        base = TransitionRecord(
            "540", "training", "training", 0, start, start + timedelta(minutes=5),
            np.zeros(1), np.zeros(1), 0, 1.0, 1.0, 0, "fixture"
        )
        mixed = replace(base, patient_id="544", clock_type="testing_reset")
        panel = RiskSetPanel((base, mixed), ("x",), "training")
        with self.assertRaisesRegex(ValueError, "clock types"):
            panel.validate()


if __name__ == "__main__":
    unittest.main()
