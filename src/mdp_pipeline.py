"""Causal-time construction of the frozen OhioT1DM analysis MDP.

This module deliberately contains no CUSUM, bootstrap, change-point, or p-value
code.  Its only responsibilities are XML-to-transition construction, strict
shape/timestamp validation, and forensic metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

try:
    from .parse_xml import parse_xml_file
except ImportError:  # pragma: no cover - script execution from src/
    from parse_xml import parse_xml_file


@dataclass
class MDPSegment:
    """One uninterrupted trajectory; every array obeys the Bellman shape."""

    patient_id: str
    split: str
    interval_min: int
    segment_id: int
    states: np.ndarray
    states_raw: np.ndarray
    actions: np.ndarray
    rewards_binary: np.ndarray
    rewards_weighted: np.ndarray
    tau: np.ndarray
    transition_metadata: pd.DataFrame

    def validate(self) -> None:
        t = len(self.actions)
        if self.states.shape[0] != t + 1:
            raise AssertionError(f"States must be T+1, got {self.states.shape[0]} for T={t}")
        if self.states_raw.shape != self.states.shape:
            raise AssertionError("raw/scaled state shapes differ")
        if len(self.rewards_binary) != t or len(self.rewards_weighted) != t:
            raise AssertionError("Rewards must have shape T")
        if len(self.tau) != t + 1:
            raise AssertionError("Boundary timestamps must have shape T+1")
        if len(self.transition_metadata) != t:
            raise AssertionError("Transition metadata must have T rows")
        if not np.all(np.isfinite(self.states)):
            raise AssertionError("Non-finite scaled state")
        if not np.all(np.isfinite(self.states_raw)):
            raise AssertionError("Non-finite raw state")
        if not np.all(np.diff(self.tau.astype("datetime64[ns]")).astype("timedelta64[ns]") > np.timedelta64(0, "ns")):
            raise AssertionError("Boundary timestamps are not strictly increasing")
        if not set(np.unique(self.actions)).issubset({0, 1}):
            raise AssertionError("Actions are not binary")
        for row in self.transition_metadata.itertuples(index=False):
            if not (row.tau_t < row.tau_next):
                raise AssertionError("Non-positive transition duration")
            for ts in _split_timestamps(row.bolus_timestamps):
                if not (row.tau_t <= ts < row.tau_next):
                    raise AssertionError(f"Bolus {ts} is outside [{row.tau_t}, {row.tau_next})")
            if row.reward_cgm_timestamp != row.tau_next:
                raise AssertionError("Reward CGM timestamp is not the next-state boundary")


def load_protocol(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg["protocol"]["status"] != "frozen_pre_cusum":
        raise ValueError("The MDP protocol is not frozen_pre_cusum")
    return cfg


def reward_binary_tir(glucose: float, cfg: dict[str, Any]) -> float:
    rcfg = cfg["reward"]["candidate_1"]
    return float(rcfg["tir_low_mg_dl"] <= glucose <= rcfg["tir_high_mg_dl"])


def reward_clinically_weighted(glucose: float, cfg: dict[str, Any]) -> float:
    rcfg = cfg["reward"]["candidate_2"]
    th = rcfg["thresholds_mg_dl"]
    u = rcfg["utilities"]
    if glucose < th["severe_hypo"]:
        return float(u["severe_hypo"])
    if glucose < th["tir_low"]:
        return float(u["hypo"])
    if glucose <= th["tir_high"]:
        return float(u["in_range"])
    if glucose <= th["severe_hyper"]:
        return float(u["hyper"])
    return float(u["severe_hyper"])


def _unique_glucose(patient_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Sort glucose and resolve exact duplicate timestamps without interpolation."""
    by_ts: dict[datetime, float] = {}
    for event in patient_data["glucose"]:
        by_ts[event["timestamp"]] = float(event["glucose"])
    return [{"timestamp": ts, "glucose": by_ts[ts]} for ts in sorted(by_ts)]


def _events_between(events: Iterable[dict], key: str, start: datetime, end: datetime) -> list[dict]:
    return [e for e in events if start <= e[key] < end]


def _linear_slope(glucose: list[dict[str, Any]], glucose_times: list[datetime], tau: datetime, minutes: int) -> float:
    start = tau - timedelta(minutes=minutes)
    left = bisect_left(glucose_times, start)
    right = bisect_left(glucose_times, tau) + 1
    vals = [(g["timestamp"], g["glucose"]) for g in glucose[left:right]]
    if len(vals) < 2:
        return 0.0
    x = np.array([(ts - vals[0][0]).total_seconds() / 60.0 for ts, _ in vals])
    y = np.array([v for _, v in vals], dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denom) if denom > 0 else 0.0


def _glucose_values(glucose: list[dict[str, Any]], glucose_times: list[datetime], tau: datetime, minutes: int) -> np.ndarray:
    start = tau - timedelta(minutes=minutes)
    left = bisect_left(glucose_times, start)
    right = bisect_left(glucose_times, tau) + 1
    return np.asarray([g["glucose"] for g in glucose[left:right]], dtype=float)


def _bolus_delivered_between(bolus: list[dict[str, Any]], start: datetime, end: datetime) -> float:
    """Insulin physically delivered in [start, end); extended delivery is uniform."""
    total = 0.0
    for b in bolus:
        begin, finish, dose = b["ts_begin"], b["ts_end"], float(b["dose"])
        if finish <= begin:
            if start <= begin < end:
                total += dose
            continue
        left, right = max(start, begin), min(end, finish)
        if right > left:
            total += dose * (right - left).total_seconds() / (finish - begin).total_seconds()
    return float(total)


def _iob_linear(bolus: list[dict[str, Any]], tau: datetime, horizon_min: float) -> float:
    """Linear remaining-fraction IOB from insulin delivered strictly before tau."""
    horizon_s = horizon_min * 60.0
    total = 0.0
    for b in bolus:
        begin, finish, dose = b["ts_begin"], b["ts_end"], float(b["dose"])
        if begin >= tau:
            continue
        if finish <= begin:
            age_s = (tau - begin).total_seconds()
            if 0 < age_s < horizon_s:
                total += dose * (1.0 - age_s / horizon_s)
            continue

        # Integrate uniform delivery times u with weight 1-(tau-u)/H.
        left = max(begin, tau - timedelta(seconds=horizon_s))
        right = min(finish, tau)
        if right <= left:
            continue
        duration_s = (finish - begin).total_seconds()
        a = (left - tau).total_seconds()
        z = (right - tau).total_seconds()
        weighted_seconds = (z - a) + (z * z - a * a) / (2.0 * horizon_s)
        total += (dose / duration_s) * weighted_seconds
    return float(max(total, 0.0))


def _scheduled_basal_at(basal: list[dict[str, Any]], tau: datetime) -> float:
    prior = [e for e in basal if e["timestamp"] <= tau]
    if not prior:
        return 0.0
    return float(max(prior, key=lambda e: e["timestamp"])["rate"])


def _basal_rate_at(patient_data: dict[str, Any], tau: datetime) -> float:
    active_temp = [e for e in patient_data.get("temp_basal", []) if e["ts_begin"] <= tau < e["ts_end"]]
    if active_temp:
        return float(max(active_temp, key=lambda e: e["ts_begin"])["rate"])
    return _scheduled_basal_at(patient_data["basal"], tau)


def _basal_mean(patient_data: dict[str, Any], tau: datetime, minutes: int) -> float:
    # Left-endpoint five-minute quadrature is causal and adequate for pump rates.
    points = [tau - timedelta(minutes=m) for m in range(minutes, 0, -5)]
    return float(np.mean([_basal_rate_at(patient_data, p) for p in points])) if points else _basal_rate_at(patient_data, tau)


def _time_since(events: list[dict[str, Any]], key: str, tau: datetime, cap_min: float) -> float:
    prior = [e[key] for e in events if e[key] < tau]
    if not prior:
        return float(cap_min)
    return float(min((tau - max(prior)).total_seconds() / 60.0, cap_min))


def _meal_sum(meals: list[dict[str, Any]], tau: datetime, minutes: int) -> float:
    start = tau - timedelta(minutes=minutes)
    return float(sum(float(e["carbs"]) for e in meals if start < e["timestamp"] <= tau))


def _state_raw(
    patient_data: dict[str, Any],
    glucose: list[dict[str, Any]],
    glucose_times: list[datetime],
    index: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    tau = glucose[index]["timestamp"]
    current = float(glucose[index]["glucose"])
    previous = glucose[index - 1] if index > 0 else glucose[index]
    prev_gap = (tau - previous["timestamp"]).total_seconds() / 60.0 if index > 0 else cfg["time"]["interval_min"]
    delta = current - float(previous["glucose"]) if index > 0 and prev_gap <= cfg["time"]["max_allowed_cgm_gap_min"] else 0.0
    g30 = _glucose_values(glucose, glucose_times, tau, 30)
    g120 = _glucose_values(glucose, glucose_times, tau, 120)
    expected_30 = 7  # inclusive readings at -30,...,0 on a nominal 5-minute clock
    bcfg = cfg["insulin"]
    bolus = patient_data["bolus"]
    meals = patient_data["meal"]
    hour = tau.hour + tau.minute / 60.0 + tau.second / 3600.0

    return np.asarray([
        current,
        delta,
        _linear_slope(glucose, glucose_times, tau, 15),
        _linear_slope(glucose, glucose_times, tau, 30),
        float(np.mean(g30)) if len(g30) else current,
        float(np.std(g30)) if len(g30) else 0.0,
        float(np.mean(g120)) if len(g120) else current,
        float(np.std(g120)) if len(g120) else 0.0,
        _basal_rate_at(patient_data, tau),
        _basal_mean(patient_data, tau, 30),
        _bolus_delivered_between(bolus, tau - timedelta(minutes=30), tau),
        _bolus_delivered_between(bolus, tau - timedelta(minutes=120), tau),
        _iob_linear(bolus, tau, float(bcfg["iob_horizon_min"])),
        _time_since(bolus, "ts_begin", tau, 240),
        _meal_sum(meals, tau, 30),
        _meal_sum(meals, tau, 120),
        _time_since(meals, "timestamp", tau + timedelta(microseconds=1), 240),
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        min(float(prev_gap), 240.0),
        max(0.0, 1.0 - min(len(g30), expected_30) / expected_30),
    ], dtype=float)


def _scale_state(raw: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    names = cfg["state"]["feature_names"]
    scaling = cfg["state"]["fixed_scaling"]
    if len(raw) != len(names):
        raise AssertionError("Feature count does not match the frozen protocol")
    return np.asarray([(raw[i] - float(scaling[n]["center"])) / float(scaling[n]["scale"]) for i, n in enumerate(names)])


def _bolus_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bolus_count": len(events),
        "bolus_timestamps": "|".join(e["ts_begin"].isoformat(sep=" ") for e in events),
        "bolus_end_timestamps": "|".join(e["ts_end"].isoformat(sep=" ") for e in events),
        "bolus_types": "|".join(str(e["type"]) for e in events),
        "bolus_doses": "|".join(f"{float(e['dose']):.6g}" for e in events),
        "bolus_total_programmed_dose": float(sum(float(e["dose"]) for e in events)),
    }


def _transition_row(
    patient_data: dict[str, Any], glucose: list[dict[str, Any]], i: int, j: int,
    raw0: np.ndarray, raw1: np.ndarray, cfg: dict[str, Any], valid: bool,
    reason: str, interval_min: int, segment_id: int,
) -> dict[str, Any]:
    t0, t1 = glucose[i]["timestamp"], glucose[j]["timestamp"]
    bolus_events = sorted(_events_between(patient_data["bolus"], "ts_begin", t0, t1), key=lambda e: e["ts_begin"])
    meal_ties = [m for m in patient_data["meal"] if m["timestamp"] == t0]
    bolus_ties = [b for b in bolus_events if b["ts_begin"] == t0]
    same_timestamp_meal_bolus = any(
        m["timestamp"] == b["ts_begin"]
        for m in patient_data["meal"] if t0 <= m["timestamp"] < t1
        for b in bolus_events
    )
    names = cfg["state"]["feature_names"]
    state = dict(zip(names, raw0))
    next_state = dict(zip(names, raw1))
    row = {
        "patient_id": str(patient_data["patient_id"]),
        "interval_min": interval_min,
        "segment_id": segment_id,
        "tau_t": t0,
        "tau_next": t1,
        "duration_min": (t1 - t0).total_seconds() / 60.0,
        "state_last_cgm_timestamp": t0,
        "state_last_cgm_value": float(glucose[i]["glucose"]),
        "state_glucose_history": (
            f"G={state['glucose_current']:.3f};d5={state['glucose_delta_5m']:.3f};"
            f"s15={state['glucose_slope_15m']:.3f};mean30={state['glucose_mean_30m']:.3f};"
            f"mean120={state['glucose_mean_120m']:.3f}"
        ),
        "state_iob": float(state["iob_linear_4h"]),
        "state_basal_current": float(state["basal_current"]),
        "state_basal_mean_30m": float(state["basal_mean_30m"]),
        "temp_basal_active_at_tau": bool(any(e["ts_begin"] <= t0 < e["ts_end"] for e in patient_data.get("temp_basal", []))),
        "state_meal_history": (
            f"carbs30={state['meal_carbs_30m']:.3f};carbs120={state['meal_carbs_120m']:.3f};"
            f"since={state['time_since_meal_min']:.3f}min"
        ),
        "action": int(bool(bolus_events)),
        "reward_cgm_timestamp": t1,
        "reward_cgm_value": float(glucose[j]["glucose"]),
        "reward_binary_tir": reward_binary_tir(float(glucose[j]["glucose"]), cfg),
        "reward_clinically_weighted": reward_clinically_weighted(float(glucose[j]["glucose"]), cfg),
        "next_state_glucose": float(next_state["glucose_current"]),
        "next_state_iob": float(next_state["iob_linear_4h"]),
        "next_state_basal_current": float(next_state["basal_current"]),
        "next_state_meal_carbs_30m": float(next_state["meal_carbs_30m"]),
        "next_state_summary": (
            f"G={next_state['glucose_current']:.3f};IOB={next_state['iob_linear_4h']:.3f};"
            f"basal={next_state['basal_current']:.3f};meal30={next_state['meal_carbs_30m']:.3f}"
        ),
        "valid": bool(valid),
        "invalid_reason": reason,
        "exact_meal_bolus_tie_at_tau": bool(meal_ties and bolus_ties),
        "meal_bolus_same_timestamp_in_cell": bool(same_timestamp_meal_bolus),
    }
    row.update(_bolus_metadata(bolus_events))
    for name, value in state.items():
        row[f"state_raw__{name}"] = float(value)
    return row


def _make_segment(
    patient_data: dict[str, Any], split: str, interval_min: int, segment_id: int,
    boundary_indices: list[int], glucose: list[dict[str, Any]], raw_cache: dict[int, np.ndarray],
    cfg: dict[str, Any], rows: list[dict[str, Any]],
) -> MDPSegment:
    raw = np.vstack([raw_cache[i] for i in boundary_indices])
    scaled = np.vstack([_scale_state(v, cfg) for v in raw])
    if len(rows) != len(boundary_indices) - 1:
        raise AssertionError("Segment rows and boundary indices are not one-step aligned")
    transition_rows = list(rows)
    actions = [r["action"] for r in transition_rows]
    rb = [r["reward_binary_tir"] for r in transition_rows]
    rw = [r["reward_clinically_weighted"] for r in transition_rows]
    seg = MDPSegment(
        patient_id=str(patient_data["patient_id"]), split=split, interval_min=interval_min,
        segment_id=segment_id, states=scaled, states_raw=raw,
        actions=np.asarray(actions, dtype=np.int8), rewards_binary=np.asarray(rb, dtype=float),
        rewards_weighted=np.asarray(rw, dtype=float),
        tau=np.asarray([glucose[i]["timestamp"] for i in boundary_indices], dtype="datetime64[ns]"),
        transition_metadata=pd.DataFrame(transition_rows),
    )
    seg.validate()
    return seg


def build_patient_mdp(
    patient_data: dict[str, Any], cfg: dict[str, Any], split: str,
    interval_min: int = 5,
) -> tuple[list[MDPSegment], pd.DataFrame, dict[str, Any]]:
    """Build valid uninterrupted segments plus a candidate-transition audit."""
    glucose = _unique_glucose(patient_data)
    if len(glucose) < 2:
        return [], pd.DataFrame(), {"error": "fewer than two glucose events"}
    glucose_times = [g["timestamp"] for g in glucose]
    raw_cache = {i: _state_raw(patient_data, glucose, glucose_times, i, cfg) for i in range(len(glucose))}
    five_lo, five_hi = map(float, cfg["time"]["valid_transition_gap_min"])
    edge_valid = []
    for i in range(len(glucose) - 1):
        gap = (glucose[i + 1]["timestamp"] - glucose[i]["timestamp"]).total_seconds() / 60.0
        edge_valid.append(five_lo <= gap <= five_hi)

    rows: list[dict[str, Any]] = []
    segments: list[MDPSegment] = []
    invalid_gap_minutes = []

    if interval_min == 5:
        segment_id = 0
        current_boundaries: list[int] = []
        current_rows: list[dict[str, Any]] = []
        for i, ok in enumerate(edge_valid):
            gap = (glucose[i + 1]["timestamp"] - glucose[i]["timestamp"]).total_seconds() / 60.0
            reason = "" if ok else f"cgm_gap_{gap:.3f}min_outside_[{five_lo},{five_hi}]"
            row = _transition_row(patient_data, glucose, i, i + 1, raw_cache[i], raw_cache[i + 1], cfg, ok, reason, interval_min, segment_id if ok else -1)
            rows.append(row)
            if ok:
                if not current_boundaries:
                    current_boundaries = [i, i + 1]
                else:
                    current_boundaries.append(i + 1)
                current_rows.append(row)
            else:
                invalid_gap_minutes.append(gap)
                if current_boundaries:
                    segments.append(_make_segment(patient_data, split, interval_min, segment_id, current_boundaries, glucose, raw_cache, cfg, current_rows))
                    segment_id += 1
                    current_boundaries, current_rows = [], []
        if current_boundaries:
            segments.append(_make_segment(patient_data, split, interval_min, segment_id, current_boundaries, glucose, raw_cache, cfg, current_rows))
    elif interval_min == 15:
        lo15, hi15 = map(float, cfg["time"]["sensitivity_valid_total_gap_min"])
        # Split at every invalid native edge, then take non-overlapping groups of
        # three valid five-minute transitions within each uninterrupted run.
        run_start = 0
        segment_id = 0
        valid_runs = []
        for i, ok in enumerate(edge_valid):
            if not ok:
                if i >= run_start:
                    valid_runs.append((run_start, i))
                invalid_gap_minutes.append((glucose[i + 1]["timestamp"] - glucose[i]["timestamp"]).total_seconds() / 60.0)
                run_start = i + 1
        if len(glucose) - 1 >= run_start:
            valid_runs.append((run_start, len(glucose) - 1))

        for run_start, run_end in valid_runs:
            boundaries = list(range(run_start, run_end + 1, 3))
            current_boundaries: list[int] = []
            local_rows: list[dict[str, Any]] = []
            for i, j in zip(boundaries[:-1], boundaries[1:]):
                gap = (glucose[j]["timestamp"] - glucose[i]["timestamp"]).total_seconds() / 60.0
                ok = lo15 <= gap <= hi15 and all(edge_valid[i:j])
                reason = "" if ok else f"15min_gap_{gap:.3f}min_or_native_edge_invalid"
                row = _transition_row(patient_data, glucose, i, j, raw_cache[i], raw_cache[j], cfg, ok, reason, interval_min, segment_id if ok else -1)
                rows.append(row)
                if ok:
                    if not current_boundaries:
                        current_boundaries = [i, j]
                    else:
                        current_boundaries.append(j)
                    local_rows.append(row)
                else:
                    invalid_gap_minutes.append(gap)
                    if current_boundaries:
                        segments.append(_make_segment(patient_data, split, interval_min, segment_id, current_boundaries, glucose, raw_cache, cfg, local_rows))
                        segment_id += 1
                        current_boundaries, local_rows = [], []
            if current_boundaries:
                segments.append(_make_segment(patient_data, split, interval_min, segment_id, current_boundaries, glucose, raw_cache, cfg, local_rows))
                segment_id += 1
    else:
        raise ValueError("Only the frozen 5-minute protocol and 15-minute sensitivity are supported")

    for seg in segments:
        seg.validate()
    transitions = pd.DataFrame(rows)
    meta = {
        "patient_id": str(patient_data["patient_id"]),
        "split": split,
        "interval_min": interval_min,
        "n_glucose": len(glucose),
        "n_candidate_transitions": len(transitions),
        "n_valid_transitions": int(transitions["valid"].sum()) if len(transitions) else 0,
        "n_invalid_transitions": int((~transitions["valid"]).sum()) if len(transitions) else 0,
        "n_invalid_native_5min_edges": int(sum(not x for x in edge_valid)),
        "n_native_edges_not_used_by_interval": int(sum(edge_valid) - (int(transitions["valid"].sum()) if interval_min == 5 and len(transitions) else 3 * int(transitions["valid"].sum()) if len(transitions) else 0)),
        "n_segments": len(segments),
        "max_invalid_gap_min": float(max(invalid_gap_minutes)) if invalid_gap_minutes else 0.0,
        "n_action_positive": int(transitions.loc[transitions["valid"], "action"].sum()) if len(transitions) else 0,
    }
    return segments, transitions, meta


def build_from_xml(xml_path: str | Path, cfg: dict[str, Any], split: str, interval_min: int = 5):
    return build_patient_mdp(parse_xml_file(xml_path), cfg, split=split, interval_min=interval_min)


def save_patient_segments(segments: list[MDPSegment], output_path: str | Path, cfg: dict[str, Any]) -> None:
    """Save ragged, gap-separated trajectories without ever stitching gaps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=np.asarray([s.states for s in segments], dtype=object),
        states_raw=np.asarray([s.states_raw for s in segments], dtype=object),
        actions=np.asarray([s.actions for s in segments], dtype=object),
        rewards_binary=np.asarray([s.rewards_binary for s in segments], dtype=object),
        rewards_weighted=np.asarray([s.rewards_weighted for s in segments], dtype=object),
        tau=np.asarray([s.tau for s in segments], dtype=object),
        feature_names=np.asarray(cfg["state"]["feature_names"], dtype=object),
        segment_ids=np.asarray([s.segment_id for s in segments], dtype=int),
        protocol_version=cfg["protocol"]["version"],
    )


def flatten_valid_segments(segments: list[MDPSegment], cfg: dict[str, Any]) -> pd.DataFrame:
    """Return one row per valid transition for descriptive/QC models only."""
    names = cfg["state"]["feature_names"]
    out = []
    for seg in segments:
        md = seg.transition_metadata.reset_index(drop=True)
        for t in range(len(seg.actions)):
            row = {
                "patient_id": seg.patient_id,
                "split": seg.split,
                "interval_min": seg.interval_min,
                "segment_id": seg.segment_id,
                "transition_index": t,
                "tau_t": pd.Timestamp(seg.tau[t]),
                "tau_next": pd.Timestamp(seg.tau[t + 1]),
                "action": int(seg.actions[t]),
                "reward_binary_tir": float(seg.rewards_binary[t]),
                "reward_clinically_weighted": float(seg.rewards_weighted[t]),
            }
            for k, name in enumerate(names):
                row[f"state__{name}"] = float(seg.states[t, k])
                row[f"state_raw__{name}"] = float(seg.states_raw[t, k])
                row[f"next_state__{name}"] = float(seg.states[t + 1, k])
            for col in ["bolus_count", "bolus_timestamps", "bolus_types", "bolus_doses", "bolus_total_programmed_dose", "exact_meal_bolus_tie_at_tau", "meal_bolus_same_timestamp_in_cell"]:
                row[col] = md.loc[t, col]
            out.append(row)
    return pd.DataFrame(out)


def _split_timestamps(value: Any) -> list[datetime]:
    if value in (None, "") or (isinstance(value, float) and np.isnan(value)):
        return []
    return [datetime.fromisoformat(x) for x in str(value).split("|") if x]
