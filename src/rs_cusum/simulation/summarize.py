"""Outcome-neutral summaries of generated simulation designs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .generator import GeneratedDataset


def summarize_dataset(dataset: GeneratedDataset) -> dict[str, Any]:
    dataset.validate()
    observed_actions = dataset.actions[dataset.observed_mask]
    patient_transition_counts = np.sum(dataset.observed_mask, axis=1)
    patient_action1_counts = np.sum((dataset.actions == 1) & dataset.observed_mask, axis=1)
    return {
        "scenario": dataset.design.name,
        "seed": dataset.design.seed,
        "n_patients": dataset.design.n_patients,
        "n_transitions_nominal": dataset.design.n_transitions,
        "observed_transition_count": int(np.sum(dataset.observed_mask)),
        "observed_action1_count": int(np.sum(observed_actions == 1)),
        "observed_action1_rate": float(np.mean(observed_actions == 1)),
        "patient_transition_counts": patient_transition_counts.tolist(),
        "patient_action1_counts": patient_action1_counts.tolist(),
        "true_change_index": dataset.true_change_index,
    }
