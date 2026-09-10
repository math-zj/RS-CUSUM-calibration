"""Forensic CGM-gap and fixed-lattice QC; deliberately contains no CUSUM.

Outputs are descriptive only.  The script never imports or calls the project's
CUSUM, bootstrap, p-value, or change-point modules.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from .mdp_pipeline import _unique_glucose, load_protocol
    from .parse_xml import parse_xml_file
except ImportError:  # pragma: no cover
    from mdp_pipeline import _unique_glucose, load_protocol
    from parse_xml import parse_xml_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results_mdp_gate" / "gap_audit"
QC_CONFIG = PROJECT_ROOT / "configs" / "gap_handling_qc.yaml"
MDP_CONFIG = PROJECT_ROOT / "configs" / "mdp_5min_protocol.yaml"


def _load_qc() -> dict[str, Any]:
    with QC_CONFIG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg["audit"]["status"] != "pre_cusum_qc_only":
        raise ValueError("gap audit config is not pre_cusum_qc_only")
    return cfg


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive edge-index runs for True values."""
    if len(mask) == 0:
        return []
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return list(zip(starts.tolist(), ends.tolist()))


def _gap_bin(x: float, cfg: dict[str, Any]) -> str:
    for b in cfg["gap_bins_min"]:
        lo, hi = b.get("lower"), b.get("upper")
        lo_ok = True if lo is None else (x >= lo if b.get("lower_inclusive", False) else x > lo)
        hi_ok = True if hi is None else (x <= hi if b.get("upper_inclusive", False) else x < hi)
        if lo_ok and hi_ok:
            return str(b["label"])
    raise AssertionError(f"gap {x} did not match a configured bin")


def _gap_class(x: float, threshold: float) -> str:
    if x < 4.5:
        return "sub_nominal_clock_jitter_or_clock_correction"
    if x <= 5.5:
        return "nominal_5min"
    if x <= threshold:
        return "prolonged_timestamp_jitter_no_full_missing_cycle_implied"
    missing = max(int(round(x / 5.0)) - 1, 1)
    if x <= 12.5:
        return "approximately_one_missing_cgm_sample"
    if x <= 17.5:
        return "approximately_two_missing_cgm_samples"
    return f"genuine_long_gap_approximately_{missing}_missing_cycles"


def _action_assignment(patient: dict[str, Any], times: list[pd.Timestamp], valid: np.ndarray) -> dict[str, Any]:
    py_times = [x.to_pydatetime() for x in times]
    actions = np.zeros(len(valid), dtype=np.int8)
    captured: set[int] = set()
    in_record: set[int] = set()
    for event_id, bolus in enumerate(sorted(patient["bolus"], key=lambda x: x["ts_begin"])):
        ts = bolus["ts_begin"]
        if not (py_times[0] <= ts < py_times[-1]):
            continue
        in_record.add(event_id)
        i = bisect_right(py_times, ts) - 1
        if 0 <= i < len(valid) and ts < py_times[i + 1] and bool(valid[i]):
            actions[i] = 1
            captured.add(event_id)
    return {
        "action_positive_cells": int(actions.sum()),
        "raw_bolus_episodes": len(patient["bolus"]),
        "bolus_in_record": len(in_record),
        "bolus_captured": len(captured),
        "bolus_coverage_pct": 100.0 * len(captured) / len(in_record) if in_record else np.nan,
        "bolus_coverage_all_raw_pct": 100.0 * len(captured) / len(patient["bolus"]) if patient["bolus"] else np.nan,
        "captured_ids": captured,
    }


def _quantiles(values: pd.Series, prefix: str = "") -> dict[str, float]:
    v = values.astype(float)
    if len(v) == 0:
        return {f"{prefix}{x}": np.nan for x in ("median", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}median": float(v.median()),
        f"{prefix}p90": float(v.quantile(.90)),
        f"{prefix}p95": float(v.quantile(.95)),
        f"{prefix}p99": float(v.quantile(.99)),
        f"{prefix}max": float(v.max()),
    }


def _actual_rule_rows(split: str, pid: str, patient: dict[str, Any], gaps: np.ndarray,
                      times: list[pd.Timestamp], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rule in cfg["adjacency_rules_min"]:
        lo, hi = float(rule["lower"]), float(rule["upper"])
        valid = (gaps >= lo) & (gaps <= hi)
        runs = _runs(valid)
        action = _action_assignment(patient, times, valid)
        err = np.abs(gaps[valid] - 5.0)
        signed = gaps[valid] - 5.0
        rows.append({
            "split": split,
            "patient_id": pid,
            "rule": rule["name"],
            "candidate_edges": len(gaps),
            "valid_transitions": int(valid.sum()),
            "valid_pct": 100.0 * float(valid.mean()) if len(valid) else np.nan,
            "transition_segments": len(runs),
            "longest_segment_transitions": max((b - a + 1 for a, b in runs), default=0),
            "action_positive_cells": action["action_positive_cells"],
            "raw_bolus_episodes": action["raw_bolus_episodes"],
            "bolus_in_record": action["bolus_in_record"],
            "bolus_captured": action["bolus_captured"],
            "bolus_coverage_pct": action["bolus_coverage_pct"],
            "bolus_coverage_all_raw_pct": action["bolus_coverage_all_raw_pct"],
            "timing_error_signed_median_min": float(np.median(signed)) if len(signed) else np.nan,
            "timing_error_abs_median_min": float(np.median(err)) if len(err) else np.nan,
            "timing_error_abs_p90_min": float(np.quantile(err, .90)) if len(err) else np.nan,
            "timing_error_abs_p95_min": float(np.quantile(err, .95)) if len(err) else np.nan,
            "timing_error_abs_p99_min": float(np.quantile(err, .99)) if len(err) else np.nan,
            "timing_error_abs_max_min": float(np.max(err)) if len(err) else np.nan,
        })
    return rows


def _fixed_lattice(patient: dict[str, Any], split: str, pid: str, glucose: list[dict[str, Any]],
                   cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], set[int]]:
    """Map CGM causally to a first-observation-anchored 5-minute lattice."""
    lc = cfg["fixed_lattice_candidate"]
    step = timedelta(minutes=float(lc["interval_min"]))
    tol = timedelta(minutes=float(lc["past_tolerance_min"]))
    genuine = float(lc["genuine_gap_threshold_min"])
    obs_ts = [g["timestamp"] for g in glucose]
    tau = []
    x = obs_ts[0]
    while x <= obs_ts[-1]:
        tau.append(x)
        x += step

    mapped: list[int | None] = []
    j = -1
    for boundary in tau:
        while j + 1 < len(obs_ts) and obs_ts[j + 1] <= boundary:
            j += 1
        if j >= 0 and boundary - obs_ts[j] <= tol:
            mapped.append(j)
        else:
            mapped.append(None)

    bolus_sorted = sorted(patient["bolus"], key=lambda e: e["ts_begin"])
    rows: list[dict[str, Any]] = []
    captured_ids: set[int] = set()
    for k in range(len(tau) - 1):
        i, j = mapped[k], mapped[k + 1]
        reasons: list[str] = []
        if i is None:
            reasons.append("no_causal_state_cgm_within_1min")
        if j is None:
            reasons.append("no_causal_next_cgm_within_1min")
        if i is not None and j is not None:
            if j == i:
                reasons.append("same_observation_reused")
            elif obs_ts[j] <= obs_ts[i]:
                reasons.append("nonincreasing_mapped_observation")
            elif (obs_ts[j] - obs_ts[i]).total_seconds() / 60.0 > genuine:
                reasons.append("crosses_genuine_cgm_gap")
        cell_bolus = [(eid, b) for eid, b in enumerate(bolus_sorted) if tau[k] <= b["ts_begin"] < tau[k + 1]]
        if j is not None and any(b["ts_begin"] >= obs_ts[j] for _, b in cell_bolus):
            reasons.append("reward_cgm_not_after_all_cell_bolus_starts")
        valid = len(reasons) == 0
        if valid:
            captured_ids.update(eid for eid, _ in cell_bolus)
        rows.append({
            "split": split,
            "patient_id": pid,
            "lattice_index": k,
            "tau_t": tau[k],
            "tau_next": tau[k + 1],
            "state_cgm_timestamp": obs_ts[i] if i is not None else pd.NaT,
            "next_cgm_timestamp": obs_ts[j] if j is not None else pd.NaT,
            "state_mapping_age_min": (tau[k] - obs_ts[i]).total_seconds() / 60.0 if i is not None else np.nan,
            "next_mapping_age_min": (tau[k + 1] - obs_ts[j]).total_seconds() / 60.0 if j is not None else np.nan,
            "mapped_observation_step_min": (obs_ts[j] - obs_ts[i]).total_seconds() / 60.0 if i is not None and j is not None else np.nan,
            "action": int(bool(cell_bolus)),
            "bolus_count": len(cell_bolus),
            "valid": valid,
            "invalid_reason": "|".join(reasons),
        })
    table = pd.DataFrame(rows)
    valid = table["valid"].to_numpy(dtype=bool)
    runs = _runs(valid)
    in_record = {eid for eid, b in enumerate(bolus_sorted) if tau[0] <= b["ts_begin"] < tau[-1]}
    valid_t = table.loc[table["valid"]]
    summary = {
        "split": split,
        "patient_id": pid,
        "scheme": "fixed_5min_lattice_causal_latest_past_1min",
        "candidate_edges": len(table),
        "valid_transitions": int(table["valid"].sum()),
        "valid_pct": 100.0 * float(table["valid"].mean()) if len(table) else np.nan,
        "transition_segments": len(runs),
        "longest_segment_transitions": max((b - a + 1 for a, b in runs), default=0),
        "action_positive_cells": int(valid_t["action"].sum()),
        "raw_bolus_episodes": len(bolus_sorted),
        "bolus_in_record": len(in_record),
        "bolus_captured": len(captured_ids),
        "bolus_coverage_pct": 100.0 * len(captured_ids) / len(in_record) if in_record else np.nan,
        "bolus_coverage_all_raw_pct": 100.0 * len(captured_ids) / len(bolus_sorted) if bolus_sorted else np.nan,
        "state_mapping_age_median_min": float(valid_t["state_mapping_age_min"].median()) if len(valid_t) else np.nan,
        "state_mapping_age_p95_min": float(valid_t["state_mapping_age_min"].quantile(.95)) if len(valid_t) else np.nan,
        "next_mapping_age_median_min": float(valid_t["next_mapping_age_min"].median()) if len(valid_t) else np.nan,
        "next_mapping_age_p95_min": float(valid_t["next_mapping_age_min"].quantile(.95)) if len(valid_t) else np.nan,
        "mapped_step_abs_error_median_min": float((valid_t["mapped_observation_step_min"] - 5).abs().median()) if len(valid_t) else np.nan,
        "mapped_step_abs_error_p95_min": float((valid_t["mapped_observation_step_min"] - 5).abs().quantile(.95)) if len(valid_t) else np.nan,
        "future_state_mapping_count": int(((table["state_mapping_age_min"] < 0) | (table["next_mapping_age_min"] < 0)).sum()),
        "reward_precedes_action_invalid": int(table["invalid_reason"].str.contains("reward_cgm_not_after", regex=False).sum()),
    }
    return table, summary, captured_ids


def _strategy_rows(split: str, patients: list[dict[str, Any]], actual_details: dict[str, dict[str, Any]],
                   lattice_tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for strategy in ("longest_current_segment", "first_current_segment", "lattice_all_valid_mask", "actual_all_valid_mask"):
        total_t = total_a = segment_count = patient_count = 0
        for p in patients:
            pid = str(p["patient_id"])
            if strategy in ("longest_current_segment", "first_current_segment"):
                d = actual_details[pid]
                runs = d["runs"]
                if not runs:
                    continue
                run = max(runs, key=lambda ab: ab[1] - ab[0] + 1) if strategy.startswith("longest") else runs[0]
                a, b = run
                t = b - a + 1
                actions = d["actions"]
                a1 = int(actions[a:b + 1].sum())
                segs = 1
            elif strategy == "lattice_all_valid_mask":
                tab = lattice_tables[pid]
                t = int(tab["valid"].sum())
                a1 = int(tab.loc[tab["valid"], "action"].sum())
                segs = len(_runs(tab["valid"].to_numpy(bool)))
            else:
                d = actual_details[pid]
                t = int(d["valid"].sum())
                a1 = int(d["actions"].sum())
                segs = len(d["runs"])
            if t:
                patient_count += 1
            total_t += t
            total_a += a1
            segment_count += segs
        out.append({
            "split": split,
            "strategy_data_view": strategy,
            "patients_retained": patient_count,
            "valid_transitions": total_t,
            "action_positive": total_a,
            "action_positive_pct": 100.0 * total_a / total_t if total_t else np.nan,
            "within_patient_segments_represented": segment_count,
        })
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qc = _load_qc()
    load_protocol(MDP_CONFIG)  # confirms the referenced MDP protocol remains frozen
    threshold = float(qc["classification"]["genuine_missing_gap_threshold_min"])

    gap_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    lattice_summary: list[dict[str, Any]] = []
    lattice_all: list[pd.DataFrame] = []
    strategy_all: list[dict[str, Any]] = []
    strategy_patient_rows: list[dict[str, Any]] = []
    direct_suffix_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    assignment_overlap_rows: list[dict[str, Any]] = []

    for split, directory, pattern in (
        ("training", PROJECT_ROOT / "OhioT1DM" / "train", "*-ws-training.xml"),
        ("testing", PROJECT_ROOT / "OhioT1DM" / "test", "*-ws-testing.xml"),
    ):
        patients: list[dict[str, Any]] = []
        actual_details: dict[str, dict[str, Any]] = {}
        lattice_tables: dict[str, pd.DataFrame] = {}
        for path in sorted(directory.glob(pattern)):
            patient = parse_xml_file(path)
            patients.append(patient)
            pid = str(patient["patient_id"])
            glucose = _unique_glucose(patient)
            times = [pd.Timestamp(g["timestamp"]) for g in glucose]
            gaps = np.asarray([(times[i + 1] - times[i]).total_seconds() / 60.0 for i in range(len(times) - 1)])
            for i, gap in enumerate(gaps):
                gap_rows.append({
                    "split": split,
                    "patient_id": pid,
                    "edge_index": i,
                    "timestamp_t": times[i],
                    "timestamp_next": times[i + 1],
                    "gap_min": gap,
                    "gap_bin": _gap_bin(float(gap), qc),
                    "gap_class": _gap_class(float(gap), threshold),
                    "current_4p5_5p5_valid": bool(4.5 <= gap <= 5.5),
                    "genuine_missing_gap": bool(gap > threshold),
                })
            rule_rows.extend(_actual_rule_rows(split, pid, patient, gaps, times, qc))
            valid = (gaps >= 4.5) & (gaps <= 5.5)
            # Edge action vector for exact first/longest segment strategy counts.
            py_times = [t.to_pydatetime() for t in times]
            edge_actions = np.zeros(len(valid), dtype=np.int8)
            for b in patient["bolus"]:
                if py_times[0] <= b["ts_begin"] < py_times[-1]:
                    i = bisect_right(py_times, b["ts_begin"]) - 1
                    if 0 <= i < len(valid) and valid[i] and b["ts_begin"] < py_times[i + 1]:
                        edge_actions[i] = 1
            actual_assignment = _action_assignment(patient, times, valid)
            actual_details[pid] = {
                "valid": valid, "runs": _runs(valid), "actions": edge_actions,
                "captured_ids": actual_assignment["captured_ids"], "times": times,
            }

            lattice_table, lattice_row, lattice_ids = _fixed_lattice(patient, split, pid, glucose, qc)
            lattice_tables[pid] = lattice_table
            lattice_all.append(lattice_table)
            lattice_summary.append(lattice_row)
            actual_ids = actual_assignment["captured_ids"]
            all_ids = set(range(len(patient["bolus"])))
            assignment_overlap_rows.append({
                "split": split,
                "patient_id": pid,
                "raw_bolus_episodes": len(all_ids),
                "captured_actual_timestamp_scheme": len(actual_ids),
                "captured_fixed_lattice": len(lattice_ids),
                "captured_both": len(actual_ids & lattice_ids),
                "actual_only": len(actual_ids - lattice_ids),
                "lattice_only": len(lattice_ids - actual_ids),
                "captured_neither": len(all_ids - (actual_ids | lattice_ids)),
            })
            record_rows.append({
                "split": split,
                "patient_id": pid,
                "first_cgm": times[0],
                "last_cgm": times[-1],
                "record_duration_days": (times[-1] - times[0]).total_seconds() / 86400.0,
                "unique_cgm": len(times),
                "raw_bolus_episodes": len(patient["bolus"]),
            })
        strategy_all.extend(_strategy_rows(split, patients, actual_details, lattice_tables))
        # Patient-level retention and timing for the two one-episode strategies.
        first_sequences: dict[str, np.ndarray] = {}
        for patient in patients:
            pid = str(patient["patient_id"])
            d = actual_details[pid]
            for label, run in (
                ("first_current_segment", d["runs"][0]),
                ("longest_current_segment", max(d["runs"], key=lambda ab: ab[1] - ab[0] + 1)),
            ):
                a, b = run
                seq = d["actions"][a:b + 1]
                strategy_patient_rows.append({
                    "split": split, "patient_id": pid, "strategy": label,
                    "start_timestamp": d["times"][a], "end_timestamp": d["times"][b + 1],
                    "start_elapsed_from_record_h": (d["times"][a] - d["times"][0]).total_seconds() / 3600.0,
                    "transitions": len(seq), "action_positive": int(seq.sum()),
                    "action_positive_pct": 100.0 * float(seq.mean()) if len(seq) else np.nan,
                })
                if label == "first_current_segment":
                    first_sequences[pid] = seq
        # These are diagnostics for the legacy ratio list only.  They are not a
        # frozen new CUSUM plan and no statistic or candidate change point is computed.
        t_max = max(map(len, first_sequences.values()))
        for ratio in (.2, .3, .4, .5, .6, .7, .8, .9):
            kappa = max(1, int(round(ratio * t_max)))
            start = t_max - kappa
            mid = start + kappa // 2
            total_t = total_a = left_a = right_a = active = 0
            for seq in first_sequences.values():
                if len(seq) <= start:
                    continue
                active += 1
                tail = seq[start:min(len(seq), t_max)]
                total_t += len(tail)
                total_a += int(tail.sum())
                left_a += int(seq[start:min(len(seq), mid)].sum()) if len(seq) > start else 0
                right_a += int(seq[mid:min(len(seq), t_max)].sum()) if len(seq) > mid else 0
            direct_suffix_rows.append({
                "split": split, "data_strategy": "first_current_segment_varying_termination",
                "legacy_kappa_ratio_not_frozen": ratio, "T_max": t_max, "suffix_start": start,
                "suffix_nominal_length": kappa, "subjects_observed_after_suffix_start": active,
                "observed_transitions": total_t, "action_positive": total_a,
                "midpoint_action_positive_left": left_a, "midpoint_action_positive_right": right_a,
            })

    gaps = pd.DataFrame(gap_rows)
    rules = pd.DataFrame(rule_rows)
    lattice = pd.concat(lattice_all, ignore_index=True)
    lattice_s = pd.DataFrame(lattice_summary)
    records = pd.DataFrame(record_rows)

    bin_order = [b["label"] for b in qc["gap_bins_min"]]
    dist_rows = []
    for keys, group in gaps.groupby(["split", "patient_id"], sort=True):
        split, pid = keys
        counts = group["gap_bin"].value_counts()
        for label in bin_order:
            n = int(counts.get(label, 0))
            dist_rows.append({"scope": "patient", "split": split, "patient_id": pid,
                              "gap_bin": label, "count": n, "percentage": 100.0 * n / len(group)})
    for split, group in gaps.groupby("split", sort=True):
        counts = group["gap_bin"].value_counts()
        for label in bin_order:
            n = int(counts.get(label, 0))
            dist_rows.append({"scope": "split", "split": split, "patient_id": "ALL",
                              "gap_bin": label, "count": n, "percentage": 100.0 * n / len(group)})
    distribution = pd.DataFrame(dist_rows)

    summary_rows = []
    for (split, pid), group in gaps.groupby(["split", "patient_id"], sort=True):
        row = {"scope": "patient", "split": split, "patient_id": pid, "n_adjacent_gaps": len(group)}
        row.update(_quantiles(group["gap_min"]))
        inv = group.loc[~group["current_4p5_5p5_valid"], "gap_min"]
        row["n_current_breaks"] = len(inv)
        row.update(_quantiles(inv, "break_"))
        summary_rows.append(row)
    for split, group in gaps.groupby("split", sort=True):
        row = {"scope": "split", "split": split, "patient_id": "ALL", "n_adjacent_gaps": len(group)}
        row.update(_quantiles(group["gap_min"]))
        inv = group.loc[~group["current_4p5_5p5_valid"], "gap_min"]
        row["n_current_breaks"] = len(inv)
        row.update(_quantiles(inv, "break_"))
        summary_rows.append(row)
    gap_summary = pd.DataFrame(summary_rows)

    breaks = gaps.loc[~gaps["current_4p5_5p5_valid"]].copy()
    breaks["break_number_within_patient"] = breaks.groupby(["split", "patient_id"]).cumcount() + 1

    # Aggregate rule and lattice summaries while retaining patient-level files.
    rule_split = rules.groupby(["split", "rule"], as_index=False).agg(
        patients=("patient_id", "nunique"), candidate_edges=("candidate_edges", "sum"),
        valid_transitions=("valid_transitions", "sum"), transition_segments=("transition_segments", "sum"),
        longest_segment_transitions=("longest_segment_transitions", "max"),
        action_positive_cells=("action_positive_cells", "sum"), raw_bolus_episodes=("raw_bolus_episodes", "sum"),
        bolus_in_record=("bolus_in_record", "sum"),
        bolus_captured=("bolus_captured", "sum"))
    rule_split["valid_pct"] = 100.0 * rule_split["valid_transitions"] / rule_split["candidate_edges"]
    rule_split["bolus_coverage_pct"] = 100.0 * rule_split["bolus_captured"] / rule_split["bolus_in_record"]
    rule_split["bolus_coverage_all_raw_pct"] = 100.0 * rule_split["bolus_captured"] / rule_split["raw_bolus_episodes"]
    for (split, rule), group in rules.groupby(["split", "rule"]):
        idx = (rule_split["split"] == split) & (rule_split["rule"] == rule)
        # Recompute edge-weighted timing errors directly from the raw gap table.
        rc = next(x for x in qc["adjacency_rules_min"] if x["name"] == rule)
        g = gaps.loc[gaps["split"] == split, "gap_min"].to_numpy(float)
        e = np.abs(g[(g >= float(rc["lower"])) & (g <= float(rc["upper"]))] - 5.0)
        signed = g[(g >= float(rc["lower"])) & (g <= float(rc["upper"]))] - 5.0
        rule_split.loc[idx, "timing_error_signed_median_min"] = np.median(signed) if len(signed) else np.nan
        rule_split.loc[idx, "timing_error_abs_median_min"] = np.median(e) if len(e) else np.nan
        rule_split.loc[idx, "timing_error_abs_p95_min"] = np.quantile(e, .95) if len(e) else np.nan
        rule_split.loc[idx, "timing_error_abs_p99_min"] = np.quantile(e, .99) if len(e) else np.nan
        rule_split.loc[idx, "timing_error_abs_max_min"] = np.max(e) if len(e) else np.nan

    lattice_split = lattice_s.groupby("split", as_index=False).agg(
        patients=("patient_id", "nunique"), candidate_edges=("candidate_edges", "sum"),
        valid_transitions=("valid_transitions", "sum"), transition_segments=("transition_segments", "sum"),
        longest_segment_transitions=("longest_segment_transitions", "max"),
        action_positive_cells=("action_positive_cells", "sum"), raw_bolus_episodes=("raw_bolus_episodes", "sum"),
        bolus_in_record=("bolus_in_record", "sum"),
        bolus_captured=("bolus_captured", "sum"), future_state_mapping_count=("future_state_mapping_count", "sum"),
        reward_precedes_action_invalid=("reward_precedes_action_invalid", "sum"))
    lattice_split["valid_pct"] = 100.0 * lattice_split["valid_transitions"] / lattice_split["candidate_edges"]
    lattice_split["bolus_coverage_pct"] = 100.0 * lattice_split["bolus_captured"] / lattice_split["bolus_in_record"]
    lattice_split["bolus_coverage_all_raw_pct"] = 100.0 * lattice_split["bolus_captured"] / lattice_split["raw_bolus_episodes"]
    for split, group in lattice.groupby("split"):
        v = group[group["valid"]]
        idx = lattice_split["split"] == split
        lattice_split.loc[idx, "state_mapping_age_median_min"] = v["state_mapping_age_min"].median()
        lattice_split.loc[idx, "state_mapping_age_p95_min"] = v["state_mapping_age_min"].quantile(.95)
        lattice_split.loc[idx, "next_mapping_age_median_min"] = v["next_mapping_age_min"].median()
        lattice_split.loc[idx, "next_mapping_age_p95_min"] = v["next_mapping_age_min"].quantile(.95)
        abs_e = (v["mapped_observation_step_min"] - 5.0).abs()
        lattice_split.loc[idx, "mapped_step_abs_error_median_min"] = abs_e.median()
        lattice_split.loc[idx, "mapped_step_abs_error_p95_min"] = abs_e.quantile(.95)

    # Invalid-lattice reason counts help distinguish causal-mapping loss from true gaps.
    reason_rows = []
    for split, group in lattice.loc[~lattice["valid"]].groupby("split"):
        for reason in sorted({r for x in group["invalid_reason"] for r in str(x).split("|") if r}):
            reason_rows.append({"split": split, "invalid_reason": reason,
                                "count": int(group["invalid_reason"].str.contains(reason, regex=False).sum())})

    gaps.to_csv(OUT / "all_adjacent_cgm_gaps.csv", index=False)
    distribution.to_csv(OUT / "gap_bin_distribution.csv", index=False)
    gap_summary.to_csv(OUT / "gap_quantiles.csv", index=False)
    breaks.to_csv(OUT / "current_rule_segment_breaks.csv", index=False)
    rules.to_csv(OUT / "adjacency_rule_patient_comparison.csv", index=False)
    rule_split.to_csv(OUT / "adjacency_rule_split_comparison.csv", index=False)
    lattice.to_csv(OUT / "fixed_lattice_transition_audit.csv", index=False)
    lattice_s.to_csv(OUT / "fixed_lattice_patient_summary.csv", index=False)
    lattice_split.to_csv(OUT / "fixed_lattice_split_summary.csv", index=False)
    pd.DataFrame(reason_rows).to_csv(OUT / "fixed_lattice_invalid_reasons.csv", index=False)
    pd.DataFrame(strategy_all).to_csv(OUT / "strategy_data_retention.csv", index=False)
    pd.DataFrame(strategy_patient_rows).to_csv(OUT / "strategy_patient_retention.csv", index=False)
    pd.DataFrame(direct_suffix_rows).to_csv(OUT / "direct_varying_termination_suffix_support.csv", index=False)
    pd.DataFrame(assignment_overlap_rows).to_csv(OUT / "action_assignment_scheme_overlap.csv", index=False)
    records.to_csv(OUT / "patient_record_time_axes.csv", index=False)

    manifest = {
        "audit_config": str(QC_CONFIG.relative_to(PROJECT_ROOT)),
        "mdp_protocol_read_only": str(MDP_CONFIG.relative_to(PROJECT_ROOT)),
        "formal_cusum_called": False,
        "bootstrap_called": False,
        "p_values_computed": False,
        "changepoints_computed": False,
        "gap_edges": int(len(gaps)),
        "current_rule_breaks": int(len(breaks)),
        "outputs": sorted(p.name for p in OUT.glob("*.csv")),
    }
    import json
    with (OUT / "gap_audit_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
