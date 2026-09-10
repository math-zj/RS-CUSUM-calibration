"""Observed-process preparation for complete alternative trajectories."""
from __future__ import annotations
from typing import Any
import numpy as np
from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum.support import load_support_rule, screen_candidates
from src.rs_cusum_phase2.grids import candidate_grid
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.protocol import WORKSPACE
from src.rs_cusum_phase2r.core import fit_candidate
from src.rs_cusum_phase2rb.engine import PreparedAnalysis

def prepare_alternative_analysis(dataset: Any, rd_cfg: dict[str, Any], parent: dict[str, Any], states: np.ndarray, actions: np.ndarray) -> PreparedAnalysis:
    dataset.validate()
    if not np.all(dataset.observed_mask):
        raise ValueError("representative alternatives require complete observation")
    panel = build_panel(dataset, "rs_reentry", int(parent["simulation"]["post_gap_burnin_transitions"]))
    candidates = candidate_grid(dataset.analysis_start, dataset.analysis_end, rd_cfg["frozen_observed_method"]["candidate_fractions"])
    support_rule = load_support_rule(WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml", "primary")
    screen = screen_candidates(panel.support_view(), candidates, dataset.analysis_start, dataset.analysis_end, support_rule)
    support = {x.candidate: x for x in screen.candidates if x.admissible}
    required = tuple(rd_cfg["frozen_observed_method"]["base_candidates"])
    if tuple(sorted(support)) != required:
        raise ValueError(f"alternative observed support lost frozen candidates: {sorted(support)}")
    frozen = rd_cfg["frozen_observed_method"]
    fits = {c: fit_candidate(panel, c, dataset.analysis_start, dataset.analysis_end, float(frozen["gamma"]), float(frozen["ridge_alpha"]), int(frozen["fqi_max_iterations"]), float(frozen["fqi_tolerance"]), "patient_balanced", "patient") for c in required}
    midpoint = dataset.analysis_start + int(np.floor(.5 * (dataset.analysis_end-dataset.analysis_start) + .5))
    ids = tuple(sorted(panel.patient_ids))
    boundaries = {c: boundary_factor(support[c], dataset.analysis_start, dataset.analysis_end, len(ids), "harmonic_active_patients") for c in required}
    return PreparedAnalysis(dataset, panel, tuple(candidates), support, fits, action_feature(states, actions), midpoint, ids, boundaries)
