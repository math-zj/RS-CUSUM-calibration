"""Build uncompressed elapsed-time risk panels from valid MDP segments."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .types import RiskSetPanel, TransitionRecord


ALLOWED_CLOCKS = {"training", "testing_reset", "testing_continuous"}


def load_risk_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["protocol"]["status"] != "phase1_method_frozen":
        raise ValueError("RS-CUSUM protocol is not phase1_method_frozen")
    return cfg


def rounded_elapsed_index(tau: datetime, recording_start: datetime, interval_min: int) -> tuple[int, float]:
    """Nearest lattice index with nonnegative half ties rounded upward."""
    elapsed_min = (tau - recording_start).total_seconds() / 60.0
    if elapsed_min < 0:
        raise ValueError("tau precedes recording_start")
    raw = elapsed_min / interval_min
    index = int(np.floor(raw + 0.5))
    lattice_time = recording_start + timedelta(minutes=index * interval_min)
    residual_min = (tau - lattice_time).total_seconds() / 60.0
    return index, float(residual_min)


def build_risk_set_panel(
    segments: Iterable[Any],
    recording_starts: Mapping[str, datetime],
    feature_names: Iterable[str],
    clock_type: str,
    config: Mapping[str, Any],
    reentry_rule: str | None = None,
) -> RiskSetPanel:
    """Convert valid one-step segments to one patient-clustered risk panel.

    The input objects are expected to follow ``mdp_pipeline.MDPSegment``.  They
    are intentionally duck-typed so deterministic synthetic fixtures can use
    the same builder without importing the XML pipeline.
    """
    if clock_type not in ALLOWED_CLOCKS:
        raise ValueError(f"unknown clock_type {clock_type!r}")
    interval_min = int(config["risk_panel"]["interval_min"])
    residual_limit = float(config["risk_panel"]["max_abs_lattice_residual_min"])
    edge_limits = tuple(float(value) for value in config["risk_panel"]["transition_edge_gap_min"])
    initial_history_min = int(config["reentry"]["initial_record_history_min"])
    post_gap_edges = int(config["reentry"]["post_gap_required_valid_5min_edges"])
    chosen_reentry = reentry_rule or str(config["reentry"]["main_rule"])
    valid_reentry = {
        "allow_after_continuous_120min_cgm_burnin",
        "allow_sparse_real_history_after_gap",
    }
    if chosen_reentry not in valid_reentry:
        raise ValueError(f"unknown re-entry rule {chosen_reentry!r}")

    by_patient: dict[str, list[Any]] = defaultdict(list)
    for segment in segments:
        segment.validate()
        by_patient[str(segment.patient_id)].append(segment)

    records: list[TransitionRecord] = []
    for patient_id, patient_segments in sorted(by_patient.items()):
        if patient_id not in recording_starts:
            raise KeyError(f"missing recording_start for patient {patient_id}")
        start = recording_starts[patient_id]
        patient_segments.sort(key=lambda segment: segment.tau[0])
        for segment_rank, segment in enumerate(patient_segments):
            for local_t in range(len(segment.actions)):
                tau_t = _as_datetime(segment.tau[local_t])
                tau_next = _as_datetime(segment.tau[local_t + 1])
                record_age_min = (tau_t - start).total_seconds() / 60.0
                if record_age_min < initial_history_min:
                    continue
                if (
                    chosen_reentry == "allow_after_continuous_120min_cgm_burnin"
                    and segment_rank > 0
                    and local_t < post_gap_edges
                ):
                    continue

                elapsed_index, residual = rounded_elapsed_index(tau_t, start, interval_min)
                next_index, next_residual = rounded_elapsed_index(tau_next, start, interval_min)
                if abs(residual) > residual_limit or abs(next_residual) > residual_limit:
                    raise ValueError(
                        f"lattice residual exceeds {residual_limit} min for patient {patient_id}: "
                        f"{residual:.6f}, {next_residual:.6f}"
                    )
                if next_index != elapsed_index + 1:
                    raise ValueError(
                        f"valid local edge does not advance one elapsed cell for patient {patient_id}: "
                        f"{elapsed_index}->{next_index}"
                    )
                reason = (
                    "eligible_initial_record_history"
                    if segment_rank == 0
                    else (
                        "eligible_post_gap_continuous_120min"
                        if chosen_reentry == "allow_after_continuous_120min_cgm_burnin"
                        else "eligible_post_gap_sparse_real_history"
                    )
                )
                records.append(
                    TransitionRecord(
                        patient_id=patient_id,
                        split=str(segment.split),
                        clock_type=clock_type,
                        elapsed_index=elapsed_index,
                        tau_t=tau_t,
                        tau_next=tau_next,
                        state=np.asarray(segment.states[local_t], dtype=float).copy(),
                        next_state=np.asarray(segment.states[local_t + 1], dtype=float).copy(),
                        action=int(segment.actions[local_t]),
                        reward_binary=float(segment.rewards_binary[local_t]),
                        reward_weighted=float(segment.rewards_weighted[local_t]),
                        segment_id=int(segment.segment_id),
                        valid_reason=reason,
                        lattice_residual_min=residual,
                        next_lattice_residual_min=next_residual,
                    )
                )

    records.sort(key=lambda record: (record.patient_id, record.elapsed_index))
    panel = RiskSetPanel(tuple(records), tuple(feature_names), clock_type, interval_min, edge_limits)
    panel.validate()
    return panel


def rectangular_to_panel(
    states: np.ndarray,
    actions: np.ndarray,
    rewards_binary: np.ndarray,
    rewards_weighted: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
    patient_ids: Iterable[str] | None = None,
    clock_type: str = "training",
    start_time: datetime = datetime(2020, 1, 1),
) -> RiskSetPanel:
    """Create a risk panel from rectangular synthetic arrays without compression."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=int)
    rewards_binary = np.asarray(rewards_binary, dtype=float)
    if states.ndim != 3 or actions.ndim != 2:
        raise ValueError("states must be N x (T+1) x p and actions N x T")
    n_patients, t_plus_one, dimension = states.shape
    n_steps = t_plus_one - 1
    if actions.shape != (n_patients, n_steps) or rewards_binary.shape != actions.shape:
        raise ValueError("rectangular array shapes are inconsistent")
    if rewards_weighted is None:
        rewards_weighted = rewards_binary.copy()
    rewards_weighted = np.asarray(rewards_weighted, dtype=float)
    if rewards_weighted.shape != actions.shape:
        raise ValueError("weighted reward shape is inconsistent")
    if observed_mask is None:
        observed_mask = np.ones_like(actions, dtype=bool)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if observed_mask.shape != actions.shape:
        raise ValueError("observed mask shape is inconsistent")
    ids = tuple(patient_ids or (f"P{i:03d}" for i in range(n_patients)))
    if len(ids) != n_patients or len(set(ids)) != n_patients:
        raise ValueError("patient_ids must be unique and have length N")

    records: list[TransitionRecord] = []
    for patient_index, patient_id in enumerate(ids):
        segment_id = -1
        previously_observed = False
        for elapsed in range(n_steps):
            observed = bool(observed_mask[patient_index, elapsed])
            if observed and not previously_observed:
                segment_id += 1
            if observed:
                tau_t = start_time + timedelta(minutes=5 * elapsed)
                records.append(
                    TransitionRecord(
                        patient_id=patient_id,
                        split="synthetic",
                        clock_type=clock_type,
                        elapsed_index=elapsed,
                        tau_t=tau_t,
                        tau_next=tau_t + timedelta(minutes=5),
                        state=states[patient_index, elapsed].copy(),
                        next_state=states[patient_index, elapsed + 1].copy(),
                        action=int(actions[patient_index, elapsed]),
                        reward_binary=float(rewards_binary[patient_index, elapsed]),
                        reward_weighted=float(rewards_weighted[patient_index, elapsed]),
                        segment_id=segment_id,
                        valid_reason="synthetic_observed_transition",
                    )
                )
            previously_observed = observed
    feature_names = tuple(f"state_{index}" for index in range(dimension))
    panel = RiskSetPanel(tuple(records), feature_names, clock_type)
    panel.validate()
    return panel


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").astype(datetime)
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    raise TypeError(f"cannot convert {type(value)!r} to datetime")
