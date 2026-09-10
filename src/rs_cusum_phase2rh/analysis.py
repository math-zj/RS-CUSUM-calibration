"""Observed-process construction for an externally generated Phase 2R-H dataset."""

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


def prepare_transfer_analysis(
    dataset: Any,
    rd_cfg: dict[str, Any],
    parent_cfg: dict[str, Any],
    evaluation_states: np.ndarray,
    evaluation_actions: np.ndarray,
) -> PreparedAnalysis:
    dataset.validate()
    if not dataset.is_null or dataset.true_change_point is not None:
        raise ValueError("Phase 2R-H accepts stationary null datasets only")
    if not np.all(dataset.observed_mask):
        raise ValueError("Phase 2R-H frozen scenarios must be complete")
    panel = build_panel(
        dataset, "rs_reentry", int(parent_cfg["simulation"]["post_gap_burnin_transitions"])
    )
    candidates = candidate_grid(
        dataset.analysis_start,
        dataset.analysis_end,
        rd_cfg["frozen_observed_method"]["candidate_fractions"],
    )
    rule = load_support_rule(WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml", "primary")
    screen = screen_candidates(
        panel.support_view(), candidates, dataset.analysis_start, dataset.analysis_end, rule
    )
    support = {item.candidate: item for item in screen.candidates if item.admissible}
    required = tuple(rd_cfg["frozen_observed_method"]["base_candidates"])
    if tuple(sorted(support)) != required:
        raise ValueError(f"Phase 2R-H observed support lost frozen candidates: {sorted(support)}")
    frozen = rd_cfg["frozen_observed_method"]
    fits = {
        candidate: fit_candidate(
            panel,
            candidate,
            dataset.analysis_start,
            dataset.analysis_end,
            float(frozen["gamma"]),
            float(frozen["ridge_alpha"]),
            int(frozen["fqi_max_iterations"]),
            float(frozen["fqi_tolerance"]),
            "patient_balanced",
            "patient",
        )
        for candidate in required
    }
    midpoint = dataset.analysis_start + int(
        np.floor(0.5 * (dataset.analysis_end - dataset.analysis_start) + 0.5)
    )
    global_ids = tuple(sorted(panel.patient_ids))
    boundaries = {
        candidate: boundary_factor(
            support[candidate],
            dataset.analysis_start,
            dataset.analysis_end,
            len(global_ids),
            "harmonic_active_patients",
        )
        for candidate in required
    }
    return PreparedAnalysis(
        dataset=dataset,
        panel=panel,
        candidates=tuple(candidates),
        support=support,
        fits=fits,
        evaluation_design=action_feature(evaluation_states, evaluation_actions),
        midpoint=midpoint,
        global_patient_ids=global_ids,
        boundaries=boundaries,
    )
