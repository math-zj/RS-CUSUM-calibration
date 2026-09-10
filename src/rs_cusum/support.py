"""Outcome-blind candidate support diagnostics and filtering."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .types import (
    CandidateSupport,
    SideSupport,
    SupportPanel,
    SupportRule,
    SupportScreenResult,
    SupportTransition,
)


def load_support_rule(path: str | Path, name: str) -> SupportRule:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["protocol"]["status"] != "phase1_method_frozen":
        raise ValueError("support protocol is not phase1_method_frozen")
    raw = cfg["thresholds"][name]
    count = raw["transition_count_each_side_action"]
    return SupportRule(
        name=name,
        absolute_count_floor=int(count["absolute_floor"]),
        parameter_multiplier=float(count["multiplier_per_action_parameter_dimension"]),
        minimum_active_patients_each_side=int(raw["minimum_active_patients_each_side"]),
        minimum_patients_both_sides=int(raw["minimum_patients_observed_on_both_sides"]),
        minimum_unique_patients_each_side_action=int(raw["minimum_unique_patients_each_side_action"]),
        maximum_action1_patient_share_each_side=float(raw["maximum_single_patient_action1_share_each_side"]),
        maximum_total_patient_share_each_side=float(raw["maximum_single_patient_total_transition_share_each_side"]),
    )


def screen_candidates(
    panel: SupportPanel,
    candidates: Iterable[int],
    analysis_start: int,
    analysis_end: int,
    rule: SupportRule,
) -> SupportScreenResult:
    """Screen candidates using only patient/time/action support information."""
    if not isinstance(panel, SupportPanel):
        raise TypeError("screen_candidates accepts SupportPanel only; outcomes/states must be projected away")
    if analysis_end <= analysis_start:
        raise ValueError("analysis_end must exceed analysis_start")
    required_count = rule.required_count(panel.state_dimension + 1)
    results: list[CandidateSupport] = []
    for candidate in sorted(set(int(value) for value in candidates)):
        if not (analysis_start < candidate < analysis_end):
            raise ValueError(f"candidate {candidate} is outside the open analysis interval")
        left_records = tuple(
            record for record in panel.records if analysis_start <= record.elapsed_index < candidate
        )
        right_records = tuple(
            record for record in panel.records if candidate <= record.elapsed_index < analysis_end
        )
        left = _side_support("left", left_records)
        right = _side_support("right", right_records)
        both = tuple(sorted(set(left.active_patients) & set(right.active_patients)))
        reasons = _failure_reasons(left, right, both, required_count, rule)
        results.append(
            CandidateSupport(
                candidate=candidate,
                left=left,
                right=right,
                patients_both_sides=both,
                required_count_each_side_action=required_count,
                admissible=not reasons,
                failure_reasons=tuple(reasons),
            )
        )
    return SupportScreenResult(rule.name, analysis_start, analysis_end, tuple(results))


def _side_support(side: str, records: tuple[SupportTransition, ...]) -> SideSupport:
    patient_total = Counter(record.patient_id for record in records)
    patient_action: dict[int, Counter[str]] = {
        0: Counter(record.patient_id for record in records if record.action == 0),
        1: Counter(record.patient_id for record in records if record.action == 1),
    }
    action_counts = {action: int(sum(patient_action[action].values())) for action in (0, 1)}
    total = len(records)
    total_shares = _shares(patient_total, total)
    action_shares = {action: _shares(patient_action[action], action_counts[action]) for action in (0, 1)}
    return SideSupport(
        side=side,
        transition_count=total,
        action_counts=action_counts,
        active_patients=tuple(sorted(patient_total)),
        unique_action_patients={action: tuple(sorted(patient_action[action])) for action in (0, 1)},
        patient_total_counts=dict(sorted(patient_total.items())),
        patient_action_counts={action: dict(sorted(patient_action[action].items())) for action in (0, 1)},
        patient_total_shares=total_shares,
        patient_action_shares=action_shares,
        maximum_total_patient_share=max(total_shares.values(), default=0.0),
        maximum_action_patient_share={
            action: max(action_shares[action].values(), default=0.0) for action in (0, 1)
        },
        total_count_ess=_count_ess(total_shares),
        action_count_ess={action: _count_ess(action_shares[action]) for action in (0, 1)},
    )


def _shares(counts: Mapping[str, int], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {}
    return {patient: count / denominator for patient, count in sorted(counts.items())}


def _count_ess(shares: Mapping[str, float]) -> float:
    denominator = sum(value * value for value in shares.values())
    return 1.0 / denominator if denominator > 0 else 0.0


def _failure_reasons(
    left: SideSupport,
    right: SideSupport,
    both: tuple[str, ...],
    required_count: int,
    rule: SupportRule,
) -> list[str]:
    reasons: list[str] = []
    for side in (left, right):
        if len(side.active_patients) < rule.minimum_active_patients_each_side:
            reasons.append(f"{side.side}:active_patients")
        for action in (0, 1):
            if side.action_counts[action] < required_count:
                reasons.append(f"{side.side}:action{action}_count")
            if len(side.unique_action_patients[action]) < rule.minimum_unique_patients_each_side_action:
                reasons.append(f"{side.side}:action{action}_unique_patients")
        if side.maximum_action_patient_share[1] > rule.maximum_action1_patient_share_each_side:
            reasons.append(f"{side.side}:action1_patient_dominance")
        if side.maximum_total_patient_share > rule.maximum_total_patient_share_each_side:
            reasons.append(f"{side.side}:total_patient_dominance")
    if len(both) < rule.minimum_patients_both_sides:
        reasons.append("patients_observed_on_both_sides")
    return reasons
