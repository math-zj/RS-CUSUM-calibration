"""Apply frozen history/re-entry interpretations to synthetic observation masks."""

from __future__ import annotations

import numpy as np

from src.rs_cusum.risk_set import rectangular_to_panel
from src.rs_cusum.types import RiskSetPanel

from .types import SyntheticDataset


def build_panel(dataset: SyntheticDataset, mode: str, post_gap_burnin: int = 24) -> RiskSetPanel:
    dataset.validate()
    if mode == "rs_reentry":
        eligible = rs_eligible_mask(
            dataset.observed_mask, dataset.analysis_start, post_gap_burnin
        )
    elif mode == "strict_first_gap":
        eligible = strict_eligible_mask(dataset.observed_mask, dataset.analysis_start)
    elif mode == "original_complete":
        if not np.all(dataset.observed_mask):
            raise ValueError("original_complete benchmark requires a rectangular observed mask")
        eligible = np.zeros_like(dataset.observed_mask, dtype=bool)
        eligible[:, dataset.analysis_start : dataset.analysis_end] = True
    else:
        raise ValueError(f"unknown panel mode {mode!r}")
    panel = rectangular_to_panel(
        dataset.states,
        dataset.actions,
        dataset.rewards,
        dataset.rewards,
        eligible,
        dataset.patient_ids,
    )
    return panel.subset(dataset.analysis_start, dataset.analysis_end)


def rs_eligible_mask(observed: np.ndarray, analysis_start: int, post_gap_burnin: int) -> np.ndarray:
    observed = np.asarray(observed, dtype=bool)
    eligible = np.zeros_like(observed)
    for patient in range(observed.shape[0]):
        segment_index = -1
        in_segment = False
        local_index = -1
        for time_index, is_observed in enumerate(observed[patient]):
            if not is_observed:
                in_segment = False
                local_index = -1
                continue
            if not in_segment:
                segment_index += 1
                in_segment = True
                local_index = 0
            else:
                local_index += 1
            if time_index < analysis_start:
                continue
            if segment_index == 0 or local_index >= post_gap_burnin:
                eligible[patient, time_index] = True
    return eligible


def strict_eligible_mask(observed: np.ndarray, analysis_start: int) -> np.ndarray:
    observed = np.asarray(observed, dtype=bool)
    eligible = np.zeros_like(observed)
    for patient in range(observed.shape[0]):
        invalid_seen = False
        for time_index, is_observed in enumerate(observed[patient]):
            if not is_observed:
                invalid_seen = True
            if time_index >= analysis_start and is_observed and not invalid_seen:
                eligible[patient, time_index] = True
    return eligible
