"""Common-landmark/monotone-censoring feasibility; no inference is performed."""

from __future__ import annotations

from bisect import bisect_left
from datetime import timedelta
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml

try:
    from .mdp_pipeline import _unique_glucose
    from .parse_xml import parse_xml_file
except ImportError:  # pragma: no cover
    from mdp_pipeline import _unique_glucose
    from parse_xml import parse_xml_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "common_landmark_feasibility.yaml"
OUT = ROOT / "results_mdp_gate" / "landmark_feasibility"


def load_cfg() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg["study"]["status"] != "pre_cusum_time_action_qc_only":
        raise ValueError("common landmark study is not QC-only")
    return cfg


def load_split(split: str) -> dict[str, dict[str, Any]]:
    if split == "training":
        directory, pattern = ROOT / "OhioT1DM" / "train", "*-ws-training.xml"
    elif split == "testing":
        directory, pattern = ROOT / "OhioT1DM" / "test", "*-ws-testing.xml"
    else:
        raise ValueError(split)
    return {str(p["patient_id"]): p for p in (parse_xml_file(f) for f in sorted(directory.glob(pattern)))}


def merge_patient(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    out = {"patient_id": str(train["patient_id"])}
    for key in ("glucose", "bolus", "meal", "basal", "temp_basal", "context_events"):
        out[key] = list(train.get(key, [])) + list(test.get(key, []))
    return out


def action_vector(patient: dict[str, Any], times: list[pd.Timestamp], start_i: int, t_count: int) -> np.ndarray:
    actions = np.zeros(t_count, dtype=np.int8)
    if t_count == 0:
        return actions
    starts = [x.to_pydatetime() for x in times[start_i:start_i + t_count + 1]]
    for event in patient["bolus"]:
        ts = event["ts_begin"]
        if not (starts[0] <= ts < starts[-1]):
            continue
        j = bisect_left(starts, ts)
        edge = j - 1 if j == len(starts) or starts[j] != ts else j
        if 0 <= edge < t_count and starts[edge] <= ts < starts[edge + 1]:
            actions[edge] = 1
    return actions


def build_monotone_trajectory(
    patient: dict[str, Any], landmark_h: float, cfg: dict[str, Any],
    clock_start: pd.Timestamp | None = None,
    allowed_entry_start: pd.Timestamp | None = None,
    allowed_entry_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Return one trajectory starting near L and ending at its first illegal edge."""
    glucose = _unique_glucose(patient)
    times = [pd.Timestamp(g["timestamp"]) for g in glucose]
    pid = str(patient["patient_id"])
    if clock_start is None:
        clock_start = times[0]
    target = clock_start + pd.Timedelta(hours=float(landmark_h))
    search_from = max(target, allowed_entry_start) if allowed_entry_start is not None else target
    idx = bisect_left(times, search_from)
    base = {
        "patient_id": pid, "landmark_h": float(landmark_h), "target_timestamp": target,
        "clock_start": clock_start, "entry_timestamp": pd.NaT, "entry_delay_min": np.nan,
        "state_eligible": False, "trajectory_eligible": False, "T": 0, "A1": 0,
        "A1_per_observed_day": np.nan, "censor_reason": "", "censor_gap_min": np.nan,
        "end_timestamp": pd.NaT, "start_index": -1, "actions": np.zeros(0, dtype=np.int8),
    }
    if idx >= len(times):
        base["ineligible_reason"] = "no_cgm_at_or_after_landmark"
        return base
    entry = times[idx]
    if allowed_entry_end is not None and entry > allowed_entry_end:
        base["ineligible_reason"] = "landmark_outside_allowed_testing_window"
        return base
    delay = (entry - target).total_seconds() / 60.0
    if delay < 0 or delay >= float(cfg["entry"]["max_entry_delay_min_exclusive"]):
        base["ineligible_reason"] = "no_boundary_within_entry_tolerance"
        base["entry_timestamp"], base["entry_delay_min"] = entry, delay
        return base
    base["entry_timestamp"], base["entry_delay_min"] = entry, delay
    record_age = (entry - clock_start).total_seconds() / 60.0
    if record_age < float(cfg["entry"]["record_history_burnin_min"]):
        base["ineligible_reason"] = "insufficient_240min_record_burnin"
        return base
    n_burn = int(cfg["entry"]["cgm_history_required_preceding_edges"])
    if idx < n_burn:
        base["ineligible_reason"] = "insufficient_120min_cgm_burnin"
        return base
    lo_b, hi_b = map(float, cfg["entry"]["cgm_burnin_legal_gap_min"])
    burn_gaps = np.asarray([(times[j + 1] - times[j]).total_seconds() / 60.0 for j in range(idx - n_burn, idx)])
    if not np.all((burn_gaps >= lo_b) & (burn_gaps <= hi_b)):
        base["ineligible_reason"] = "noncontinuous_120min_cgm_burnin"
        return base
    base["state_eligible"] = True

    lo, hi = map(float, cfg["followup"]["legal_adjacent_gap_min"])
    genuine = float(cfg["followup"]["genuine_gap_threshold_min"])
    t_count = 0
    censor_reason, censor_gap = "record_termination", np.nan
    for j in range(idx, len(times) - 1):
        if allowed_entry_end is not None and times[j + 1] > allowed_entry_end:
            censor_reason = "allowed_testing_window_end"
            break
        gap = (times[j + 1] - times[j]).total_seconds() / 60.0
        if lo <= gap <= hi:
            t_count += 1
            continue
        censor_gap = gap
        censor_reason = "genuine_cgm_gap" if gap > genuine else "non_genuine_illegal_timing_edge"
        break
    actions = action_vector(patient, times, idx, t_count)
    end = times[idx + t_count]
    days = (end - entry).total_seconds() / 86400.0
    base.update({
        "trajectory_eligible": bool(t_count >= int(cfg["followup"]["minimum_transitions_for_N_L"])),
        "T": int(t_count), "A1": int(actions.sum()),
        "A1_per_observed_day": float(actions.sum() / days) if days > 0 else np.nan,
        "censor_reason": censor_reason, "censor_gap_min": censor_gap,
        "end_timestamp": end, "start_index": idx, "actions": actions,
        "ineligible_reason": "" if t_count else "state_available_but_zero_legal_followup",
    })
    return base


def trajectory_public_row(dataset: str, tr: dict[str, Any]) -> dict[str, Any]:
    return {"dataset_clock": dataset, **{k: v for k, v in tr.items() if k != "actions"}}


def side_support(trajectories: list[dict[str, Any]], u: int) -> dict[str, Any]:
    left_t = right_t = left_a1 = right_a1 = 0
    left_contrib = right_active = 0
    left_by_patient: list[int] = []
    right_by_patient: list[int] = []
    for tr in trajectories:
        a = tr["actions"]
        lt = min(len(a), u)
        rt = max(len(a) - u, 0)
        la = int(a[:lt].sum())
        ra = int(a[u:].sum()) if rt else 0
        if lt:
            left_contrib += 1
        if rt:
            right_active += 1
        left_t += lt
        right_t += rt
        left_a1 += la
        right_a1 += ra
        left_by_patient.append(la)
        right_by_patient.append(ra)
    return {
        "u_transition": int(u), "u_hours_nominal": u * 5.0 / 60.0,
        "left_patients_contributing": left_contrib, "patients_active_at_u": right_active,
        "left_transitions": left_t, "left_A0": left_t - left_a1, "left_A1": left_a1,
        "right_transitions": right_t, "right_A0": right_t - right_a1, "right_A1": right_a1,
        "left_max_patient_A1_share": max(left_by_patient, default=0) / left_a1 if left_a1 else np.nan,
        "right_max_patient_A1_share": max(right_by_patient, default=0) / right_a1 if right_a1 else np.nan,
        "left_dominant_patient_index": int(np.argmax(left_by_patient)) if left_a1 else -1,
        "right_dominant_patient_index": int(np.argmax(right_by_patient)) if right_a1 else -1,
    }


def summarize_landmark(dataset: str, landmark_h: float, rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [r for r in rows if r["trajectory_eligible"]]
    ts = np.asarray([r["T"] for r in eligible], dtype=int)
    total_t = int(ts.sum()) if len(ts) else 0
    total_a = int(sum(r["A1"] for r in eligible))
    summary = {
        "dataset_clock": dataset, "landmark_h": float(landmark_h),
        "state_eligible_patients": int(sum(r["state_eligible"] for r in rows)),
        "eligible_patients": len(eligible), "ineligible_patients": len(rows) - len(eligible),
        "min_T": int(ts.min()) if len(ts) else 0, "median_T": float(np.median(ts)) if len(ts) else 0.0,
        "max_T": int(ts.max()) if len(ts) else 0,
        "total_transitions": total_t, "total_A0": total_t - total_a,
        "total_action_positive": total_a,
        "action_positive_pct": 100.0 * total_a / total_t if total_t else np.nan,
        "median_A1_per_patient": float(np.median([r["A1"] for r in eligible])) if eligible else 0.0,
        "min_A1_per_patient": min((r["A1"] for r in eligible), default=0),
        "max_A1_per_patient": max((r["A1"] for r in eligible), default=0),
        "patients_with_zero_A1": int(sum(r["A1"] == 0 for r in eligible)),
        "median_A1_per_observed_day": float(np.nanmedian([r["A1_per_observed_day"] for r in eligible])) if eligible else np.nan,
        "entry_delay_median_min": float(np.median([r["entry_delay_min"] for r in eligible])) if eligible else np.nan,
        "entry_delay_max_min": max((r["entry_delay_min"] for r in eligible), default=np.nan),
        "censored_by_genuine_gap": int(sum(r["censor_reason"] == "genuine_cgm_gap" for r in eligible)),
        "censored_by_timing_edge": int(sum(r["censor_reason"] == "non_genuine_illegal_timing_edge" for r in eligible)),
    }
    fraction_rows: list[dict[str, Any]] = []
    if eligible:
        t_max = max(r["T"] for r in eligible)
        for fraction in map(float, cfg["support"]["followup_fractions"]):
            u = max(1, int(round(fraction * t_max)))
            s = side_support(eligible, u)
            fraction_rows.append({"dataset_clock": dataset, "landmark_h": landmark_h,
                                  "followup_fraction_of_Tmax": fraction, **s})
        focus = [r for r in fraction_rows if .2 <= r["followup_fraction_of_Tmax"] <= .8]
        summary["worst_side_action_20_80"] = min(min(r["left_A1"], r["right_A1"]) for r in focus)
        summary["minimum_active_patients_20_80"] = min(r["patients_active_at_u"] for r in focus)
        dominance_values = [v for r in focus for v in
                            (r["left_max_patient_A1_share"], r["right_max_patient_A1_share"])
                            if np.isfinite(v)]
        summary["maximum_patient_dominance_20_80"] = max(dominance_values) if dominance_values else np.nan
    else:
        summary.update({"worst_side_action_20_80": 0, "minimum_active_patients_20_80": 0,
                        "maximum_patient_dominance_20_80": np.nan})

    active_rows: list[dict[str, Any]] = []
    step = int(round(float(cfg["support"]["candidate_u_step_hours"]) * 12))
    max_t = max((r["T"] for r in eligible), default=0)
    for u in range(0, max_t + 1, step):
        active_rows.append({
            "dataset_clock": dataset, "landmark_h": landmark_h,
            "followup_transition": u, "followup_hours_nominal": u * 5 / 60,
            "active_patients": int(sum(r["T"] > u for r in eligible)),
        })
    if max_t and (not active_rows or active_rows[-1]["followup_transition"] != max_t):
        active_rows.append({"dataset_clock": dataset, "landmark_h": landmark_h,
                            "followup_transition": max_t, "followup_hours_nominal": max_t * 5 / 60,
                            "active_patients": 0})
    return summary, fraction_rows, active_rows


def pareto_frontier(table: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    x = table[table["eligible_patients"] > 0].reset_index(drop=True)
    keep = np.ones(len(x), dtype=bool)
    vals = x[objectives].to_numpy(float)
    for i in range(len(x)):
        dominated = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return x.loc[keep].copy()


def run_dataset(
    dataset: str, patients: dict[str, dict[str, Any]], landmarks: list[float], cfg: dict[str, Any],
    clock_starts: dict[str, pd.Timestamp] | None = None,
    allowed_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    patient_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    fractions: list[dict[str, Any]] = []
    active_curves: list[dict[str, Any]] = []
    support_map: list[dict[str, Any]] = []
    step = int(round(float(cfg["support"]["candidate_u_step_hours"]) * 12))
    low_n = int(cfg["support"]["low_active_patient_threshold"])
    dom = float(cfg["support"]["patient_action_dominance_fraction"])
    for landmark in landmarks:
        private = []
        for pid, patient in patients.items():
            window = allowed_windows.get(pid) if allowed_windows else None
            tr = build_monotone_trajectory(
                patient, landmark, cfg,
                clock_start=clock_starts.get(pid) if clock_starts else None,
                allowed_entry_start=window[0] if window else None,
                allowed_entry_end=window[1] if window else None,
            )
            private.append(tr)
            patient_rows.append(trajectory_public_row(dataset, tr))
        summary, frac, curve = summarize_landmark(dataset, landmark, private, cfg)
        summaries.append(summary); fractions.extend(frac); active_curves.extend(curve)
        eligible = [r for r in private if r["trajectory_eligible"]]
        max_t = max((r["T"] for r in eligible), default=0)
        pids = [r["patient_id"] for r in eligible]
        for u in range(step, max_t, step):
            s = side_support(eligible, u)
            left_idx, right_idx = s.pop("left_dominant_patient_index"), s.pop("right_dominant_patient_index")
            s.update({
                "dataset_clock": dataset, "landmark_h": landmark,
                "left_dominant_patient": pids[left_idx] if left_idx >= 0 else "",
                "right_dominant_patient": pids[right_idx] if right_idx >= 0 else "",
                "flag_only_1_2_active_right": s["patients_active_at_u"] <= low_n,
                "flag_left_A1_lt5": s["left_A1"] < 5, "flag_right_A1_lt5": s["right_A1"] < 5,
                "flag_left_A1_lt10": s["left_A1"] < 10, "flag_right_A1_lt10": s["right_A1"] < 10,
                "flag_left_patient_dominance_ge50pct": bool(s["left_max_patient_A1_share"] >= dom) if np.isfinite(s["left_max_patient_A1_share"]) else False,
                "flag_right_patient_dominance_ge50pct": bool(s["right_max_patient_A1_share"] >= dom) if np.isfinite(s["right_max_patient_A1_share"]) else False,
            })
            support_map.append(s)
    return patient_rows, summaries, fractions, active_curves, support_map


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    train = load_split("training")
    test = load_split("testing")

    def duration_h(patient: dict[str, Any]) -> float:
        g = _unique_glucose(patient)
        return (g[-1]["timestamp"] - g[0]["timestamp"]).total_seconds() / 3600.0

    step_h = int(cfg["landmarks"]["step_hours"])
    train_end = int(np.floor(min(map(duration_h, train.values())) / step_h) * step_h)
    test_end = int(np.floor(min(map(duration_h, test.values())) / step_h) * step_h)
    train_landmarks = list(map(float, range(0, train_end + 1, step_h)))
    test_landmarks = list(map(float, range(0, test_end + 1, step_h)))

    all_patient: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_fraction: list[dict[str, Any]] = []
    all_active: list[dict[str, Any]] = []
    all_support: list[dict[str, Any]] = []
    for dataset, patients, landmarks in (
        ("training_reset_clock", train, train_landmarks),
        ("testing_reset_clock", test, test_landmarks),
    ):
        out = run_dataset(dataset, patients, landmarks, cfg)
        for target, values in zip((all_patient, all_summary, all_fraction, all_active, all_support), out):
            target.extend(values)

    # Testing as a continuation of training: patient clock remains at training start,
    # while entry and follow-up are restricted to that patient's testing interval.
    combined: dict[str, dict[str, Any]] = {}
    clock_starts: dict[str, pd.Timestamp] = {}
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    axis_rows = []
    max_test_end_h = 0.0
    for pid in sorted(train):
        combined[pid] = merge_patient(train[pid], test[pid])
        tg, qg = _unique_glucose(train[pid]), _unique_glucose(test[pid])
        train_start, train_last = pd.Timestamp(tg[0]["timestamp"]), pd.Timestamp(tg[-1]["timestamp"])
        test_start, test_last = pd.Timestamp(qg[0]["timestamp"]), pd.Timestamp(qg[-1]["timestamp"])
        clock_starts[pid] = train_start
        windows[pid] = (test_start, test_last)
        start_h = (test_start - train_start).total_seconds() / 3600.0
        end_h = (test_last - train_start).total_seconds() / 3600.0
        max_test_end_h = max(max_test_end_h, end_h)
        axis_rows.append({
            "patient_id": pid, "training_start": train_start, "training_last_cgm": train_last,
            "testing_start": test_start, "testing_end": test_last,
            "testing_start_elapsed_from_training_h": start_h,
            "testing_end_elapsed_from_training_h": end_h,
            "train_to_test_boundary_gap_min": (test_start - train_last).total_seconds() / 60.0,
        })
    continuous_end = int(np.floor(max_test_end_h / step_h) * step_h)
    continuous_landmarks = list(map(float, range(0, continuous_end + 1, step_h)))
    out = run_dataset("testing_training_start_clock", combined, continuous_landmarks, cfg, clock_starts, windows)
    for target, values in zip((all_patient, all_summary, all_fraction, all_active, all_support), out):
        target.extend(values)

    patient_df = pd.DataFrame(all_patient)
    summary_df = pd.DataFrame(all_summary)
    fraction_df = pd.DataFrame(all_fraction)
    active_df = pd.DataFrame(all_active)
    support_df = pd.DataFrame(all_support)
    axis_df = pd.DataFrame(axis_rows)

    support_summary = support_df.groupby(["dataset_clock", "landmark_h"], as_index=False).agg(
        candidate_u_count=("u_transition", "count"),
        minimum_patients_active_at_u=("patients_active_at_u", "min"),
        candidate_u_with_only_1_2_active=("flag_only_1_2_active_right", "sum"),
        minimum_left_A1=("left_A1", "min"), minimum_right_A1=("right_A1", "min"),
        candidate_u_left_A1_lt5=("flag_left_A1_lt5", "sum"),
        candidate_u_right_A1_lt5=("flag_right_A1_lt5", "sum"),
        candidate_u_left_A1_lt10=("flag_left_A1_lt10", "sum"),
        candidate_u_right_A1_lt10=("flag_right_A1_lt10", "sum"),
        maximum_left_patient_A1_share=("left_max_patient_A1_share", "max"),
        maximum_right_patient_A1_share=("right_max_patient_A1_share", "max"),
        candidate_u_left_dominance_ge50pct=("flag_left_patient_dominance_ge50pct", "sum"),
        candidate_u_right_dominance_ge50pct=("flag_right_patient_dominance_ge50pct", "sum"),
    ) if len(support_df) else pd.DataFrame()
    reason_summary = patient_df.groupby(
        ["dataset_clock", "landmark_h", "trajectory_eligible", "ineligible_reason", "censor_reason"],
        dropna=False, as_index=False,
    ).size().rename(columns={"size": "patients"})

    natural = set(map(float, cfg["landmarks"]["natural_hours"]))
    natural_df = summary_df[summary_df["landmark_h"].isin(natural)].copy()
    objective_cols = list(cfg["pareto"]["maximize"])
    frontiers = []
    for dataset, group in summary_df.groupby("dataset_clock"):
        f = pareto_frontier(group, objective_cols)
        f["pareto_objectives"] = "|".join(objective_cols)
        frontiers.append(f)
    pareto_df = pd.concat(frontiers, ignore_index=True)

    patient_df.to_csv(OUT / "landmark_patient_trajectories.csv", index=False)
    summary_df.to_csv(OUT / "landmark_feasibility_summary.csv", index=False)
    natural_df.to_csv(OUT / "natural_landmark_summary.csv", index=False)
    fraction_df.to_csv(OUT / "followup_fraction_support.csv", index=False)
    active_df.to_csv(OUT / "active_patient_curves.csv", index=False)
    support_df.to_csv(OUT / "candidate_split_support_map.csv", index=False)
    support_summary.to_csv(OUT / "candidate_split_support_summary.csv", index=False)
    reason_summary.to_csv(OUT / "eligibility_censor_reason_summary.csv", index=False)
    pareto_df.to_csv(OUT / "landmark_pareto_frontier.csv", index=False)
    axis_df.to_csv(OUT / "testing_time_axis_structure.csv", index=False)

    max_start = float(axis_df["testing_start_elapsed_from_training_h"].max())
    min_end = float(axis_df["testing_end_elapsed_from_training_h"].min())
    manifest = {
        "protocol": str(CONFIG.relative_to(ROOT)),
        "formal_cusum_called": False, "bootstrap_called": False,
        "p_values_computed": False, "changepoints_computed": False,
        "training_landmarks_scanned": len(train_landmarks),
        "testing_reset_landmarks_scanned": len(test_landmarks),
        "continuous_testing_landmarks_scanned": len(continuous_landmarks),
        "testing_common_overlap_from_training_start_h": [max_start, min_end],
        "testing_common_overlap_duration_h": max(0.0, min_end - max_start),
        "candidate_support_rows": len(support_df),
        "candidate_support_summary_rows": len(support_summary),
        "reward_fields_read_for_selection": False,
    }
    with (OUT / "landmark_feasibility_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
