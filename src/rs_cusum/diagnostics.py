"""Structured Phase-1 diagnostics with explicit failure fields."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .types import CandidateStatistic, RiskSetPanel, SupportScreenResult


def risk_panel_diagnostics(panel: RiskSetPanel) -> dict[str, Any]:
    panel.validate()
    patient_counts = Counter(record.patient_id for record in panel.records)
    patient_action1 = Counter(record.patient_id for record in panel.records if record.action == 1)
    patient_segments: dict[str, set[int]] = defaultdict(set)
    patient_elapsed: dict[str, list[int]] = defaultdict(list)
    residuals: list[float] = []
    next_residuals: list[float] = []
    for record in panel.records:
        patient_segments[record.patient_id].add(record.segment_id)
        patient_elapsed[record.patient_id].append(record.elapsed_index)
        residuals.append(record.lattice_residual_min)
        next_residuals.append(record.next_lattice_residual_min)
    reentry_count = sum(max(0, len(segments) - 1) for segments in patient_segments.values())
    elapsed_holes = {
        patient_id: int(
            sum(max(0, right - left - 1) for left, right in zip(sorted(values)[:-1], sorted(values)[1:]))
        )
        for patient_id, values in patient_elapsed.items()
    }
    return {
        "clock_type": panel.clock_type,
        "interval_min": panel.interval_min,
        "state_dimension": panel.state_dimension,
        "transition_count": len(panel.records),
        "patient_count": len(patient_counts),
        "action_counts": {
            "0": int(sum(record.action == 0 for record in panel.records)),
            "1": int(sum(record.action == 1 for record in panel.records)),
        },
        "patient_transition_counts": dict(sorted(patient_counts.items())),
        "patient_action1_counts": dict(sorted(patient_action1.items())),
        "patient_segment_counts": {
            patient_id: len(segments) for patient_id, segments in sorted(patient_segments.items())
        },
        "reentry_count": reentry_count,
        "elapsed_holes_by_patient": dict(sorted(elapsed_holes.items())),
        "maximum_abs_lattice_residual_min": float(
            max((abs(value) for value in residuals + next_residuals), default=0.0)
        ),
        "duplicate_patient_elapsed_indices": 0,
        "clock_mixing_detected": False,
    }


def support_screen_diagnostics(screen: SupportScreenResult) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for item in screen.candidates:
        candidates.append(
            {
                "candidate": item.candidate,
                "admissible": item.admissible,
                "failure_reasons": list(item.failure_reasons),
                "exact_rejection_reason": "NONE" if item.admissible else ";".join(item.failure_reasons),
                "required_count_each_side_action": item.required_count_each_side_action,
                "n_left": item.left.transition_count,
                "n_right": item.right.transition_count,
                "patients_left": list(item.left.active_patients),
                "patients_right": list(item.right.active_patients),
                "patients_both": list(item.patients_both_sides),
                "A0_left": item.left.action_counts[0],
                "A1_left": item.left.action_counts[1],
                "A0_right": item.right.action_counts[0],
                "A1_right": item.right.action_counts[1],
                "unique_patients_by_side_action": {
                    "left_A0": list(item.left.unique_action_patients[0]),
                    "left_A1": list(item.left.unique_action_patients[1]),
                    "right_A0": list(item.right.unique_action_patients[0]),
                    "right_A1": list(item.right.unique_action_patients[1]),
                },
                "max_patient_transition_share": {
                    "left": item.left.maximum_total_patient_share,
                    "right": item.right.maximum_total_patient_share,
                },
                "max_patient_A1_share": {
                    "left": item.left.maximum_action_patient_share[1],
                    "right": item.right.maximum_action_patient_share[1],
                },
                "patients_both_sides": list(item.patients_both_sides),
                "left": _side_to_dict(item.left),
                "right": _side_to_dict(item.right),
            }
        )
    return {
        "rule_name": screen.rule_name,
        "analysis_start": screen.analysis_start,
        "analysis_end": screen.analysis_end,
        "status": screen.status,
        "admissible_candidates": list(screen.admissible_candidates),
        "candidate_count": len(screen.candidates),
        "candidates": candidates,
    }


def candidate_statistic_diagnostics(statistic: CandidateStatistic) -> dict[str, Any]:
    return {
        "candidate": statistic.candidate,
        "boundary_scaling": statistic.boundary_scaling,
        "boundary_factor": statistic.boundary_factor,
        "patient_ids": list(statistic.patient_ids),
        "evaluation_point_count": len(statistic.contrast),
        "normalized_max": statistic.normalized_max,
        "unnormalized_max": statistic.unnormalized_max,
        "l1_integral": statistic.l1_integral,
        "contrast_min": float(np.min(statistic.contrast)),
        "contrast_max": float(np.max(statistic.contrast)),
        "cluster_variance_min": float(np.nanmin(statistic.cluster_variance)),
        "cluster_variance_max": float(np.nanmax(statistic.cluster_variance)),
        "left_fqi": dict(statistic.left_model.diagnostics),
        "right_fqi": dict(statistic.right_model.diagnostics),
        "candidate_diagnostics": dict(statistic.diagnostics),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _side_to_dict(side: Any) -> dict[str, Any]:
    return {
        "side": side.side,
        "transition_count": side.transition_count,
        "action_counts": dict(side.action_counts),
        "active_patients": list(side.active_patients),
        "unique_action_patients": {
            str(action): list(values) for action, values in side.unique_action_patients.items()
        },
        "patient_total_counts": dict(side.patient_total_counts),
        "patient_action_counts": {
            str(action): dict(values) for action, values in side.patient_action_counts.items()
        },
        "patient_total_shares": dict(side.patient_total_shares),
        "patient_action_shares": {
            str(action): dict(values) for action, values in side.patient_action_shares.items()
        },
        "maximum_total_patient_share": side.maximum_total_patient_share,
        "maximum_action_patient_share": dict(side.maximum_action_patient_share),
        "total_count_ess": side.total_count_ess,
        "action_count_ess": dict(side.action_count_ess),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value
