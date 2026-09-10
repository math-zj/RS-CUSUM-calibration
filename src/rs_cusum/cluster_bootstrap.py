"""Patient-cluster multiplier null draws for fixed toy candidate sets.

This module deliberately contains no p-value function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import CandidateStatistic


@dataclass(frozen=True)
class BootstrapNullResult:
    patient_ids: tuple[str, ...]
    multipliers: np.ndarray
    normalized_max: np.ndarray
    unnormalized_max: np.ndarray
    l1_integral_max: np.ndarray
    candidate_count: int
    seed: int


def draw_patient_multipliers(
    patient_ids: Sequence[str], n_draws: int, seed: int
) -> tuple[tuple[str, ...], np.ndarray]:
    """Draw one reusable N(0,1) multiplier per original patient and draw."""
    ids = tuple(sorted(set(str(patient_id) for patient_id in patient_ids)))
    if len(ids) != len(patient_ids):
        raise ValueError("patient_ids must be unique")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    return ids, rng.standard_normal((n_draws, len(ids)))


def bootstrap_fixed_candidate_null(
    statistics: Sequence[CandidateStatistic], n_draws: int, seed: int
) -> BootstrapNullResult:
    """Generate cluster multiplier null maxima over an already fixed set."""
    if not statistics:
        raise ValueError("bootstrap null is undefined for an empty candidate set")
    all_ids = sorted({patient_id for item in statistics for patient_id in item.patient_ids})
    patient_ids, multipliers = draw_patient_multipliers(all_ids, n_draws, seed)
    id_to_column = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    normalized_by_candidate: list[np.ndarray] = []
    unnormalized_by_candidate: list[np.ndarray] = []
    integral_by_candidate: list[np.ndarray] = []
    for item in statistics:
        aligned = np.zeros((len(patient_ids), item.patient_contributions.shape[1]), dtype=float)
        for local_index, patient_id in enumerate(item.patient_ids):
            aligned[id_to_column[patient_id]] = item.patient_contributions[local_index]
        process = multipliers @ aligned
        absolute_process = np.abs(process)
        unnormalized_by_candidate.append(item.boundary_factor * np.max(absolute_process, axis=1))
        integral_by_candidate.append(item.boundary_factor * np.mean(absolute_process, axis=1))
        valid = np.isfinite(item.cluster_variance) & (item.cluster_variance > 0)
        if np.any(valid):
            normalized_by_candidate.append(
                item.boundary_factor
                * np.max(absolute_process[:, valid] / np.sqrt(item.cluster_variance[valid]), axis=1)
            )
        else:
            normalized_by_candidate.append(np.full(n_draws, np.nan))
    normalized_stack = np.column_stack(normalized_by_candidate)
    normalized_max = np.full(n_draws, np.nan)
    for draw in range(n_draws):
        finite = np.isfinite(normalized_stack[draw])
        if np.any(finite):
            normalized_max[draw] = np.max(normalized_stack[draw, finite])
    return BootstrapNullResult(
        patient_ids=patient_ids,
        multipliers=multipliers,
        normalized_max=normalized_max,
        unnormalized_max=np.max(np.column_stack(unnormalized_by_candidate), axis=1),
        l1_integral_max=np.max(np.column_stack(integral_by_candidate), axis=1),
        candidate_count=len(statistics),
        seed=seed,
    )
