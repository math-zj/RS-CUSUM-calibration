from __future__ import annotations

import unittest

from src.rs_cusum.support import load_support_rule
from src.rs_cusum_phase2.engine import build_support_screen, fit_candidate_cores, run_calibrated_method
from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.grids import candidate_grid, evaluation_grid
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.protocol import WORKSPACE, load_phase2_config


class Phase2EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_phase2_config()

    def test_finite_bootstrap_correction_and_seed_determinism(self) -> None:
        dataset = generate_dataset("N0_complete_balanced_null", 2024, self.cfg)
        panel = build_panel(dataset, "rs_reentry")
        candidates = (144,)
        rule = load_support_rule(WORKSPACE / "configs/rs_cusum_sensitivity.yaml", "primary")
        screen = build_support_screen(panel, candidates, 48, 240, rule)
        sim = self.cfg["simulation"]
        cores = fit_candidate_cores(
            panel, candidates, 48, 240, "patient_balanced", "patient",
            sim["gamma"], sim["ridge_alpha"], sim["fqi_max_iterations"], sim["fqi_tolerance"]
        )
        states, actions, _ = evaluation_grid(panel, 48, 240, 8, 9173)
        first = run_calibrated_method(
            cores, screen, states, actions, 48, 240, 12,
            "harmonic_active_patients", 19, 55
        )
        second = run_calibrated_method(
            cores, screen, states, actions, 48, 240, 12,
            "harmonic_active_patients", 19, 55
        )
        self.assertEqual(first.status, "TESTABLE")
        self.assertGreaterEqual(first.p_value, 1 / 20)
        self.assertLessEqual(first.p_value, 1.0)
        self.assertEqual(first.p_value, second.p_value)
        self.assertEqual(first.estimated_change_point, 144)
        self.assertEqual(first.diagnostics["bootstrap"]["cluster_unit"], "patient")


if __name__ == "__main__":
    unittest.main()
