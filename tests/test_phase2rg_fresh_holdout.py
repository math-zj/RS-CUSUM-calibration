"""Pre-run engineering tests for the frozen Phase 2R-G holdout."""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.rs_cusum_phase2rd.protocol import stable_seed as m4_stable_seed
from src.rs_cusum_phase2rf.metrics import primary_contrast, threshold_tail_metrics
from src.rs_cusum_phase2rg.protocol import OUTPUT, WORKSPACE, load_raw_config, load_seeds, sha256
from src.rs_cusum_phase2rg.runner import _grid
from src.rs_cusum_phase2rg.summarize import _primary_draw_count


class Phase2RGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_raw_config()

    def test_01_historical_gates_are_immutable(self):
        protocol = self.cfg["protocol"]
        self.assertEqual(protocol["phase2rd_gate_immutable"], "M4_MECHANISM_JOINT_TAIL_FAIL")
        self.assertEqual(protocol["phase2re_gate_immutable"], "TAIL_METRIC_UNSTABLE")
        self.assertEqual(protocol["phase2rf_gate_immutable"], "M4_JOINT_TAIL_CONFIRMED")

    def test_02_every_prohibited_operation_is_closed(self):
        protocol = self.cfg["protocol"]
        for key in (
            "m4_modification", "posthoc_calibration", "mechanism_exploration",
            "ohiot1dm_allowed", "phase3_allowed", "transfer_null_allowed",
            "alternatives_allowed", "power_allowed", "changepoint_allowed", "establish_1_0_2",
        ):
            self.assertFalse(protocol[key])
        self.assertTrue(protocol["one_time_holdout"])

    def test_03_m4_source_exactly_matches_phase2rf_freeze(self):
        path = WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py"
        self.assertEqual(sha256(path), self.cfg["frozen_method"]["m4_engine_sha256"])

    def test_04_metric_source_exactly_matches_phase2rf_freeze(self):
        path = WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py"
        self.assertEqual(sha256(path), self.cfg["frozen_method"]["phase2rf_metric_sha256"])

    def test_05_seed_registry_unique_and_fresh(self):
        meta = json.loads((OUTPUT / "seed_registry.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["all_values_unique"])
        self.assertTrue(meta["disjoint_from_all_prior_registries"])
        self.assertEqual(meta["historical_collision_count"], 0)
        with np.load(OUTPUT / "seed_registry.npz", allow_pickle=False) as data:
            flat = np.concatenate([np.asarray(data[key]).ravel() for key in data.files])
        self.assertEqual(len(flat), len(np.unique(flat)))

    def test_06_fresh_bank_sizes_are_frozen(self):
        self.assertEqual(load_seeds("oracle_primary_process").shape, (2500,))
        self.assertEqual(load_seeds("oracle_reference_process").shape, (2500,))
        self.assertEqual(load_seeds("m4_outer_dataset").shape, (300,))
        self.assertEqual(load_seeds("m4_outer_bootstrap").shape, (300,))
        self.assertEqual(load_seeds("m4_design_process").shape, (300, 199))
        self.assertEqual(load_seeds("m4_reward_process").shape, (300, 199))
        self.assertEqual(load_seeds("m4_dataset_process").shape, (300, 199))

    def test_07_every_m4_derived_subseed_is_registered(self):
        bootstrap = load_seeds("m4_outer_bootstrap")
        for label in ("design", "reward", "dataset"):
            registered = load_seeds(f"m4_{label}_process")
            for outer, draw in ((0, 0), (99, 8), (100, 7), (299, 198)):
                expected = m4_stable_seed(int(bootstrap[outer]), "M4", label, draw)
                self.assertEqual(int(registered[outer, draw]), expected)

    def test_08_oracle_banks_are_disjoint(self):
        primary = load_seeds("oracle_primary_process")
        reference = load_seeds("oracle_reference_process")
        self.assertFalse(np.intersect1d(primary, reference).size)

    def test_09_primary_m4_selection_is_exact_and_cluster_balanced(self):
        counts = np.asarray([_primary_draw_count(outer) for outer in range(300)])
        self.assertEqual(int(np.sum(counts)), 2500)
        self.assertEqual(set(counts.tolist()), {8, 9})
        self.assertEqual(int(np.sum(counts == 9)), 100)

    def test_10_primary_metric_is_the_phase2rf_definition(self):
        rng = np.random.default_rng(781923)
        oracle = rng.normal(size=(300, 8)).astype(np.float32)
        reference = rng.normal(size=(300, 8)).astype(np.float32)
        result = primary_contrast(oracle, reference, reference, (0.90, 0.95, 0.975, 0.99))
        self.assertAlmostEqual(result["primary_EISJEM"], 0.0, places=12)
        rows = threshold_tail_metrics(oracle, reference, (0.99,))
        self.assertTrue(np.isfinite(rows[0]["standardized_joint_exceedance_MAE"]))

    def test_11_grid_dimension_is_unchanged(self):
        states, actions = _grid(self.cfg)
        self.assertEqual(states.shape[0], 64)
        self.assertEqual(actions.shape, (64,))

    def test_12_gates_are_predeclared_and_exhaustive(self):
        labels = {self.cfg["final_gate"][key] for key in ("pass", "fail", "invalid")}
        self.assertEqual(labels, {
            "M4_FRESH_GLOBAL_NULL_PASS", "M4_FRESH_GLOBAL_NULL_FAIL", "M4_FRESH_GLOBAL_NULL_INVALID"
        })
        self.assertTrue(self.cfg["final_gate"]["no_rerun_after_scientific_fail"])

    def test_13_sample_size_plan_uses_phase2rf_only(self):
        text = (OUTPUT / "sample_size_precision_plan.csv").read_text(encoding="utf-8")
        self.assertIn("Phase 2R-F frozen output", text)
        self.assertNotIn("Phase 2R-G result", text)
        self.assertEqual(self.cfg["sample_plan"]["m4_inner_draws_per_outer"], 199)

    def test_14_phase2rg_formal_holdout_not_started_before_tests(self):
        self.assertFalse((OUTPUT / "phase2rg_gate.json").exists())
        checkpoints = OUTPUT / "checkpoints"
        self.assertFalse(checkpoints.exists() and any(checkpoints.rglob("*.npz")))


if __name__ == "__main__":
    unittest.main()
