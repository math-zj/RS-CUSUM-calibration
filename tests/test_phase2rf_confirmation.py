"""Engineering tests for frozen Phase 2R-F confirmation measurement."""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.rs_cusum_phase2rd.protocol import stable_seed as m4_stable_seed
from src.rs_cusum_phase2rf.metrics import primary_contrast, tail_matrices, threshold_tail_metrics
from src.rs_cusum_phase2rf.protocol import OUTPUT, WORKSPACE, load_raw_config, load_seeds, sha256
from src.rs_cusum_phase2rf.runner import _grid, _oracle_chunk
from src.rs_cusum_phase2rf.summarize import _cluster_balanced_indices


class Phase2RFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_raw_config()

    def test_01_parent_gates_are_immutable(self):
        self.assertEqual(self.cfg["protocol"]["phase2rd_gate_immutable"], "M4_MECHANISM_JOINT_TAIL_FAIL")
        self.assertEqual(self.cfg["protocol"]["phase2re_gate_immutable"], "TAIL_METRIC_UNSTABLE")

    def test_02_every_prohibited_stage_is_closed(self):
        protocol = self.cfg["protocol"]
        for key in (
            "m4_modification", "posthoc_calibration", "d3_allowed", "ohiot1dm_allowed",
            "phase3_allowed", "transfer_null_allowed", "alternatives_allowed", "power_allowed",
            "changepoint_allowed", "establish_1_0_2",
        ):
            self.assertFalse(protocol[key])

    def test_03_m4_source_matches_frozen_phase2rd_hash(self):
        path = WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py"
        self.assertEqual(sha256(path), self.cfg["frozen_method"]["m4_engine_sha256"])

    def test_04_seed_registry_is_unique_and_disjoint_by_declaration(self):
        meta = json.loads((OUTPUT / "seed_registry.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["all_values_unique"])
        self.assertTrue(meta["disjoint_from_all_prior_registries"])
        with np.load(OUTPUT / "seed_registry.npz", allow_pickle=False) as data:
            values = np.concatenate([np.asarray(data[key]).ravel() for key in data.files])
        self.assertEqual(len(values), len(np.unique(values)))

    def test_05_each_m4_process_subseed_is_registered_exactly(self):
        bootstrap = load_seeds("m4_outer_bootstrap")
        design = load_seeds("m4_design_process")
        reward = load_seeds("m4_reward_process")
        dataset = load_seeds("m4_dataset_process")
        for outer, draw in ((0, 0), (99, 8), (299, 198)):
            self.assertEqual(int(design[outer, draw]), m4_stable_seed(int(bootstrap[outer]), "M4", "design", draw))
            self.assertEqual(int(reward[outer, draw]), m4_stable_seed(int(bootstrap[outer]), "M4", "reward", draw))
            self.assertEqual(int(dataset[outer, draw]), m4_stable_seed(int(bootstrap[outer]), "M4", "dataset", draw))

    def test_06_primary_plan_has_two_independent_equal_oracle_banks(self):
        primary = load_seeds("oracle_primary_process")
        reference = load_seeds("oracle_reference_process")
        self.assertEqual(primary.shape, (2500,))
        self.assertEqual(reference.shape, (2500,))
        self.assertFalse(np.intersect1d(primary, reference).size)

    def test_07_tail_conditional_direction_is_p_j_given_p_i(self):
        events = np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]], bool)
        marginal, joint, conditional = tail_matrices(events)
        self.assertAlmostEqual(marginal[0], 0.5)
        self.assertAlmostEqual(joint[0, 1], 0.25)
        self.assertAlmostEqual(conditional[0, 1], 0.5)

    def test_08_primary_contrast_is_zero_when_m4_equals_oracle_reference(self):
        rng = np.random.default_rng(87123)
        first = rng.normal(size=(300, 8)).astype(np.float32)
        second = rng.normal(size=(300, 8)).astype(np.float32)
        result = primary_contrast(first, second, second, (0.90, 0.95))
        self.assertAlmostEqual(result["primary_EISJEM"], 0.0, places=12)

    def test_09_primary_metric_has_no_conditional_denominator(self):
        rng = np.random.default_rng(97123)
        first = rng.normal(size=(120, 5)).astype(np.float32)
        second = rng.normal(size=(120, 5)).astype(np.float32)
        rows = threshold_tail_metrics(first, second, (0.99,))
        self.assertTrue(np.isfinite(rows[0]["standardized_joint_exceedance_MAE"]))
        self.assertGreaterEqual(rows[0]["standardized_joint_exceedance_MAE"], 0.0)

    def test_10_cluster_balanced_selection_covers_clusters_before_reuse(self):
        cluster = np.repeat(np.arange(300), [9] * 100 + [8] * 200)
        selected = _cluster_balanced_indices(cluster, 250, 19381)
        self.assertEqual(len(selected), 250)
        self.assertEqual(len(np.unique(cluster[selected])), 250)

    def test_11_full_cluster_balanced_selection_is_exactly_2500(self):
        cluster = np.repeat(np.arange(300), [9] * 100 + [8] * 200)
        selected = _cluster_balanced_indices(cluster, 2500, 29381)
        self.assertEqual(len(selected), 2500)
        self.assertEqual(len(np.unique(selected)), 2500)
        self.assertEqual(len(np.unique(cluster[selected])), 300)

    def test_12_oracle_engine_returns_full_finite_process_and_layers(self):
        states, actions = _grid(self.cfg)
        result = _oracle_chunk(
            {
                "cfg": self.cfg, "bank": "engineering", "indices": np.asarray([0]),
                "seeds": np.asarray([76192381], np.uint64),
                "evaluation_states": states, "evaluation_actions": actions,
            }
        )
        self.assertEqual(result["joint"].shape, (1, 448))
        self.assertEqual(result["layers"].shape, (1, 5))
        self.assertTrue(result["valid"][0])
        self.assertTrue(np.all(np.isfinite(result["joint"])))

    def test_13_planning_precedes_formal_draws_and_uses_phase2re_only(self):
        plan = (OUTPUT / "sample_size_precision_plan.csv").read_text(encoding="utf-8")
        self.assertIn("Phase2R-E saved true_all/fitted_all only", plan)
        self.assertIn("2500", plan)

    def test_14_gate_labels_are_exhaustive_and_no_retroactive_pass_exists(self):
        labels = set(self.cfg["stop_rules"].values())
        for label in (
            "M4_JOINT_TAIL_CONFIRMED", "M4_STABLE_JOINT_TAIL_FAIL",
            "JOINT_TAIL_VALIDATION_UNSTABLE", "JOINT_TAIL_VALIDATION_INCONCLUSIVE",
        ):
            self.assertIn(label, labels)
        self.assertNotIn("M4_MECHANISM_PASS_TO_LOCKED_HOLDOUT", labels)


if __name__ == "__main__":
    unittest.main()
