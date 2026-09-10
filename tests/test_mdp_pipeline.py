from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parse_xml import parse_xml_file
from mdp_pipeline import (
    _basal_rate_at, _bolus_delivered_between, _state_raw, _unique_glucose,
    build_patient_mdp, load_protocol,
)


class TestFrozenMDPPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_protocol(ROOT / "configs" / "mdp_5min_protocol.yaml")
        cls.p540 = parse_xml_file(ROOT / "OhioT1DM" / "train" / "540-ws-training.xml")

    def test_every_saved_segment_has_exact_bellman_shapes(self):
        segments, _, _ = build_patient_mdp(self.p540, self.cfg, "training", 5)
        self.assertGreater(len(segments), 1)  # long gaps must split the patient
        for seg in segments:
            seg.validate()
            self.assertEqual(seg.states.shape[0], len(seg.actions) + 1)
            self.assertEqual(len(seg.rewards_binary), len(seg.actions))

    def test_reward_is_next_state_glucose_utility(self):
        segments, _, _ = build_patient_mdp(self.p540, self.cfg, "training", 5)
        seg = max(segments, key=lambda x: len(x.actions))
        md = seg.transition_metadata.reset_index(drop=True)
        self.assertTrue(np.allclose(md["reward_cgm_value"], seg.states_raw[1:, 0]))
        expected = ((seg.states_raw[1:, 0] >= 70) & (seg.states_raw[1:, 0] <= 180)).astype(float)
        self.assertTrue(np.array_equal(expected, seg.rewards_binary))

    def test_future_glucose_cannot_change_prior_state(self):
        patient = deepcopy(self.p540)
        glucose = _unique_glucose(patient)
        idx = 200
        times = [g["timestamp"] for g in glucose]
        before = _state_raw(patient, glucose, times, idx, self.cfg)
        glucose[idx + 50]["glucose"] += 5000.0
        after = _state_raw(patient, glucose, times, idx, self.cfg)
        self.assertTrue(np.array_equal(before, after))

    def test_long_cgm_gap_is_not_a_saved_transition(self):
        p591 = parse_xml_file(ROOT / "OhioT1DM" / "train" / "591-ws-training.xml")
        segments, candidates, _ = build_patient_mdp(p591, self.cfg, "training", 5)
        bad = candidates[candidates["duration_min"] > 5.5]
        self.assertGreater(len(bad), 0)
        self.assertFalse(bad["valid"].any())
        saved_pairs = set()
        for seg in segments:
            saved_pairs.update(zip(seg.transition_metadata["tau_t"], seg.transition_metadata["tau_next"]))
        for row in bad.itertuples(index=False):
            self.assertNotIn((row.tau_t, row.tau_next), saved_pairs)

    def test_extended_bolus_total_is_allocated_once(self):
        extended = next(b for b in self.p540["bolus"] if b["ts_end"] > b["ts_begin"])
        delivered = _bolus_delivered_between([extended], extended["ts_begin"], extended["ts_end"])
        self.assertAlmostEqual(delivered, extended["dose"], places=10)
        midpoint = extended["ts_begin"] + (extended["ts_end"] - extended["ts_begin"]) / 2
        first = _bolus_delivered_between([extended], extended["ts_begin"], midpoint)
        second = _bolus_delivered_between([extended], midpoint, extended["ts_end"])
        self.assertAlmostEqual(first + second, extended["dose"], places=10)

    def test_temp_basal_overrides_scheduled_basal(self):
        temp = self.p540["temp_basal"][0]
        midpoint = temp["ts_begin"] + (temp["ts_end"] - temp["ts_begin"]) / 2
        self.assertAlmostEqual(_basal_rate_at(self.p540, midpoint), temp["rate"])

    def test_exact_meal_bolus_tie_is_flagged(self):
        p544 = parse_xml_file(ROOT / "OhioT1DM" / "train" / "544-ws-training.xml")
        _, candidates, _ = build_patient_mdp(p544, self.cfg, "training", 5)
        tied = candidates[candidates["meal_bolus_same_timestamp_in_cell"]]
        self.assertGreaterEqual(len(tied), 1)
        target = datetime(2027, 6, 13, 10, 45)
        self.assertTrue(any(target.isoformat(sep=" ") in x for x in tied["bolus_timestamps"]))


if __name__ == "__main__":
    unittest.main()
