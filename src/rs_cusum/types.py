"""Typed records shared by the RS-CUSUM-RL Phase-1 modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TransitionRecord:
    """One valid one-step transition on an uncompressed elapsed-time clock."""

    patient_id: str
    split: str
    clock_type: str
    elapsed_index: int
    tau_t: datetime
    tau_next: datetime
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward_binary: float
    reward_weighted: float
    segment_id: int
    valid_reason: str
    lattice_residual_min: float = 0.0
    next_lattice_residual_min: float = 0.0

    def validate(self, state_dimension: int, edge_gap_min: tuple[float, float]) -> None:
        if not self.patient_id:
            raise ValueError("patient_id must be non-empty")
        if self.elapsed_index < 0:
            raise ValueError("elapsed_index must be nonnegative")
        if self.action not in (0, 1):
            raise ValueError("action must be binary")
        state = np.asarray(self.state, dtype=float)
        next_state = np.asarray(self.next_state, dtype=float)
        if state.shape != (state_dimension,) or next_state.shape != (state_dimension,):
            raise ValueError(
                f"state shapes must both be ({state_dimension},), got {state.shape} and {next_state.shape}"
            )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(next_state)):
            raise ValueError("states must be finite")
        if not np.isfinite(self.reward_binary) or not np.isfinite(self.reward_weighted):
            raise ValueError("rewards must be finite")
        gap = (self.tau_next - self.tau_t).total_seconds() / 60.0
        if not (edge_gap_min[0] <= gap <= edge_gap_min[1]):
            raise ValueError(f"transition edge {gap:.6f} min is outside {edge_gap_min}")


@dataclass(frozen=True)
class SupportTransition:
    """Outcome-free projection used by the admissibility screen."""

    patient_id: str
    elapsed_index: int
    action: int

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise ValueError("patient_id must be non-empty")
        if self.elapsed_index < 0:
            raise ValueError("elapsed_index must be nonnegative")
        if self.action not in (0, 1):
            raise ValueError("action must be binary")


@dataclass(frozen=True)
class SupportPanel:
    """The only input type accepted by outcome-blind support screening."""

    records: tuple[SupportTransition, ...]
    state_dimension: int
    clock_type: str

    def __post_init__(self) -> None:
        if self.state_dimension <= 0:
            raise ValueError("state_dimension must be positive")
        seen: set[tuple[str, int]] = set()
        for record in self.records:
            key = (record.patient_id, record.elapsed_index)
            if key in seen:
                raise ValueError(f"duplicate support transition {key}")
            seen.add(key)


@dataclass(frozen=True)
class RiskSetPanel:
    """Collection of valid transitions, clustered by original patient."""

    records: tuple[TransitionRecord, ...]
    feature_names: tuple[str, ...]
    clock_type: str
    interval_min: int = 5
    edge_gap_min: tuple[float, float] = (4.5, 5.5)

    @property
    def state_dimension(self) -> int:
        return len(self.feature_names)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.patient_id for record in self.records}))

    def validate(self) -> None:
        if self.interval_min <= 0:
            raise ValueError("interval_min must be positive")
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        seen: set[tuple[str, int]] = set()
        for record in self.records:
            if record.clock_type != self.clock_type:
                raise ValueError("clock types may not be mixed in one RiskSetPanel")
            record.validate(self.state_dimension, self.edge_gap_min)
            key = (record.patient_id, record.elapsed_index)
            if key in seen:
                raise ValueError(f"duplicate patient/elapsed transition {key}")
            seen.add(key)

    def support_view(self) -> SupportPanel:
        """Drop every outcome/state field before support logic is called."""
        records = tuple(
            SupportTransition(record.patient_id, record.elapsed_index, record.action)
            for record in self.records
        )
        return SupportPanel(records, self.state_dimension, self.clock_type)

    def subset(self, start: int, end: int) -> "RiskSetPanel":
        records = tuple(record for record in self.records if start <= record.elapsed_index < end)
        panel = RiskSetPanel(records, self.feature_names, self.clock_type, self.interval_min, self.edge_gap_min)
        panel.validate()
        return panel


@dataclass(frozen=True)
class SupportRule:
    name: str
    absolute_count_floor: int
    parameter_multiplier: float
    minimum_active_patients_each_side: int
    minimum_patients_both_sides: int
    minimum_unique_patients_each_side_action: int
    maximum_action1_patient_share_each_side: float
    maximum_total_patient_share_each_side: float

    def required_count(self, per_action_dimension: int) -> int:
        return max(self.absolute_count_floor, int(np.ceil(self.parameter_multiplier * per_action_dimension)))


@dataclass(frozen=True)
class SideSupport:
    side: str
    transition_count: int
    action_counts: Mapping[int, int]
    active_patients: tuple[str, ...]
    unique_action_patients: Mapping[int, tuple[str, ...]]
    patient_total_counts: Mapping[str, int]
    patient_action_counts: Mapping[int, Mapping[str, int]]
    patient_total_shares: Mapping[str, float]
    patient_action_shares: Mapping[int, Mapping[str, float]]
    maximum_total_patient_share: float
    maximum_action_patient_share: Mapping[int, float]
    total_count_ess: float
    action_count_ess: Mapping[int, float]


@dataclass(frozen=True)
class CandidateSupport:
    candidate: int
    left: SideSupport
    right: SideSupport
    patients_both_sides: tuple[str, ...]
    required_count_each_side_action: int
    admissible: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SupportScreenResult:
    rule_name: str
    analysis_start: int
    analysis_end: int
    candidates: tuple[CandidateSupport, ...]

    @property
    def admissible_candidates(self) -> tuple[int, ...]:
        return tuple(candidate.candidate for candidate in self.candidates if candidate.admissible)

    @property
    def status(self) -> str:
        return "ADMISSIBLE_CANDIDATES_AVAILABLE" if self.admissible_candidates else "NOT_TESTABLE_UNDER_SUPPORT_RULE"


@dataclass(frozen=True)
class FQIResult:
    beta: np.ndarray
    converged: bool
    iterations: int
    coefficient_delta: float
    patient_weights: Mapping[str, float]
    transition_weights: np.ndarray
    patient_scores: Mapping[str, np.ndarray]
    centered_patient_scores: Mapping[str, np.ndarray]
    patient_influences: Mapping[str, np.ndarray]
    influence_matrix: np.ndarray
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateStatistic:
    candidate: int
    boundary_scaling: str
    boundary_factor: float
    patient_ids: tuple[str, ...]
    evaluation_actions: np.ndarray
    evaluation_states: np.ndarray
    contrast: np.ndarray
    patient_contributions: np.ndarray
    cluster_variance: np.ndarray
    normalized_values: np.ndarray
    normalized_max: float
    unnormalized_max: float
    l1_integral: float
    left_model: FQIResult
    right_model: FQIResult
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def as_array(values: Sequence[float]) -> np.ndarray:
    """Small explicit conversion helper used by diagnostics and tests."""
    return np.asarray(values, dtype=float)
