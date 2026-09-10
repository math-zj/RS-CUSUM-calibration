"""Frozen outcome-blind state-reference and candidate-grid construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Iterable

import numpy as np

from src.rs_cusum.types import RiskSetPanel


def candidate_grid(analysis_start: int, analysis_end: int, fractions: Iterable[float]) -> tuple[int, ...]:
    if analysis_end <= analysis_start:
        raise ValueError("analysis window must be positive")
    length = analysis_end - analysis_start
    candidates = {
        analysis_start + int(np.floor(float(fraction) * length + 0.5))
        for fraction in fractions
    }
    result = tuple(sorted(value for value in candidates if analysis_start < value < analysis_end))
    if not result:
        raise ValueError("candidate grid is empty")
    return result


def evaluation_grid(
    panel: RiskSetPanel,
    analysis_start: int,
    analysis_end: int,
    reference_states: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, int], ...]]:
    """Patient-balanced deterministic state-only reference selection."""
    if reference_states <= 0:
        raise ValueError("reference_states must be positive")
    grouped: dict[str, list] = defaultdict(list)
    for record in panel.records:
        if analysis_start <= record.elapsed_index < analysis_end:
            grouped[record.patient_id].append(record)
    if not grouped:
        raise ValueError("no states are available for the evaluation grid")
    queues: dict[str, deque] = {}
    for patient_id in sorted(grouped):
        records = sorted(grouped[patient_id], key=lambda item: item.elapsed_index)
        patient_seed = _stable_patient_seed(seed, patient_id)
        order = np.random.default_rng(patient_seed).permutation(len(records))
        queues[patient_id] = deque(records[index] for index in order)
    selected = []
    while len(selected) < reference_states and any(queues.values()):
        for patient_id in sorted(queues):
            if queues[patient_id] and len(selected) < reference_states:
                selected.append(queues[patient_id].popleft())
    base_states = np.vstack([record.state for record in selected])
    states = np.vstack([base_states, base_states])
    actions = np.concatenate(
        [np.zeros(len(base_states), dtype=int), np.ones(len(base_states), dtype=int)]
    )
    sources = tuple(
        [(record.patient_id, record.elapsed_index) for record in selected]
        + [(record.patient_id, record.elapsed_index) for record in selected]
    )
    return states, actions, sources


def _stable_patient_seed(seed: int, patient_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)
