"""Typed Phase-2 simulation and replicate records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    scenario: str
    family: str
    is_null: bool
    effect_level: str
    effect_size: float
    seed: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    observed_mask: np.ndarray
    patient_ids: tuple[str, ...]
    phenotypes: np.ndarray
    analysis_start: int
    analysis_end: int
    true_change_point: int | None

    def validate(self) -> None:
        n_patients, t_steps = self.actions.shape
        if self.states.ndim != 3 or self.states.shape[:2] != (n_patients, t_steps + 1):
            raise ValueError("states must have shape N x (T+1) x p")
        if self.rewards.shape != (n_patients, t_steps):
            raise ValueError("rewards must have shape N x T")
        if self.observed_mask.shape != (n_patients, t_steps):
            raise ValueError("observed_mask must have shape N x T")
        if len(self.patient_ids) != n_patients or len(set(self.patient_ids)) != n_patients:
            raise ValueError("patient IDs must be unique and have length N")
        if self.phenotypes.shape != (n_patients,):
            raise ValueError("phenotype shape is inconsistent")
        if not set(np.unique(self.actions)).issubset({0, 1}):
            raise ValueError("actions must be binary")
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.rewards)):
            raise ValueError("states/rewards must be finite")
        if not (0 <= self.analysis_start < self.analysis_end <= t_steps):
            raise ValueError("analysis window is invalid")
        if self.is_null and self.true_change_point is not None:
            raise ValueError("null scenario may not have a true change point")
        if not self.is_null and self.true_change_point is None:
            raise ValueError("alternative scenario requires a true change point")


@dataclass(frozen=True)
class CandidateCore:
    candidate: int
    beta_difference: np.ndarray
    cluster_ids: tuple[str, ...]
    coefficient_contributions: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MethodResult:
    status: str
    observed_statistic: float | None
    p_value: float | None
    estimated_change_point: int | None
    admissible_candidates: tuple[int, ...]
    candidate_observed_statistics: dict[int, float]
    bootstrap_statistics: np.ndarray | None
    diagnostics: dict[str, Any]
