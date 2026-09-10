"""Descriptive and forensic QC for the frozen OhioT1DM analysis MDP.

No function in this module calls CUSUM, bootstrap, or change-point code.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from .mdp_pipeline import MDPSegment, _split_timestamps
except ImportError:  # pragma: no cover
    from mdp_pipeline import MDPSegment, _split_timestamps


def audit_table_around(
    transitions: pd.DataFrame, target: datetime, n_rows: int = 20,
) -> pd.DataFrame:
    """Return consecutive candidate rows around a real forensic timestamp."""
    if transitions.empty:
        return transitions.copy()
    d = transitions.sort_values("tau_t").reset_index(drop=True)
    hit = np.argmin(np.abs((pd.to_datetime(d["tau_t"]) - pd.Timestamp(target)).dt.total_seconds()))
    start = max(0, int(hit) - n_rows // 2)
    end = min(len(d), start + n_rows)
    start = max(0, end - n_rows)
    cols = [
        "patient_id", "tau_t", "state_last_cgm_timestamp", "state_last_cgm_value",
        "state_glucose_history", "state_iob", "state_basal_current", "state_basal_mean_30m",
        "temp_basal_active_at_tau", "state_meal_history", "action",
        "bolus_timestamps", "bolus_types", "bolus_doses", "reward_cgm_timestamp",
        "reward_cgm_value", "reward_binary_tir", "reward_clinically_weighted",
        "tau_next", "next_state_glucose", "next_state_iob", "next_state_basal_current",
        "next_state_meal_carbs_30m", "next_state_summary", "duration_min", "valid", "invalid_reason",
        "exact_meal_bolus_tie_at_tau", "meal_bolus_same_timestamp_in_cell",
    ]
    return d.loc[start:end - 1, cols].copy()


def _event_times_before_bolus(patient_data: dict[str, Any], start: datetime, bolus_ts: datetime) -> dict[str, int]:
    """Count information arriving after S_t but before the first cell bolus."""
    out = {
        "prior_glucose": 0,
        "prior_meal": 0,
        "prior_scheduled_basal_change": 0,
        "prior_temp_basal_begin": 0,
        "prior_temp_basal_end": 0,
        "prior_fingerstick": 0,
        "prior_hypo_event": 0,
        "prior_exercise": 0,
        "prior_stressor": 0,
        "prior_work": 0,
        "prior_illness": 0,
        "prior_sleep": 0,
        "prior_other_bolus": 0,
    }
    out["prior_glucose"] = sum(start < e["timestamp"] < bolus_ts for e in patient_data["glucose"])
    out["prior_meal"] = sum(start < e["timestamp"] < bolus_ts for e in patient_data["meal"])
    out["prior_scheduled_basal_change"] = sum(start < e["timestamp"] < bolus_ts for e in patient_data["basal"])
    out["prior_temp_basal_begin"] = sum(start < e["ts_begin"] < bolus_ts for e in patient_data.get("temp_basal", []))
    out["prior_temp_basal_end"] = sum(start < e["ts_end"] < bolus_ts for e in patient_data.get("temp_basal", []))
    for e in patient_data.get("context_events", []):
        if start < e["timestamp"] < bolus_ts:
            key = f"prior_{e['kind']}"
            if key in out:
                out[key] += 1
    out["prior_other_bolus"] = sum(start <= e["ts_begin"] < bolus_ts for e in patient_data["bolus"])
    return out


def action_timing_rows(
    patient_data: dict[str, Any], valid_transitions: pd.DataFrame, split: str, interval_min: int,
) -> pd.DataFrame:
    """One row per action-positive cell, using the first bolus as cell timing."""
    rows = []
    pos = valid_transitions[(valid_transitions["valid"]) & (valid_transitions["action"] == 1)]
    for tr in pos.itertuples(index=False):
        timestamps = sorted(_split_timestamps(tr.bolus_timestamps))
        if not timestamps:
            continue
        first = timestamps[0]
        offset = (first - tr.tau_t).total_seconds() / 60.0
        if offset < 1:
            timing_bin = "0-1"
        elif offset < 2:
            timing_bin = "1-2"
        elif offset < 3:
            timing_bin = "2-3"
        elif offset < 4:
            timing_bin = "3-4"
        elif offset < 5:
            timing_bin = "4-5"
        else:
            timing_bin = "5+"
        prior = _event_times_before_bolus(patient_data, tr.tau_t, first)
        same_time_meal = sum(e["timestamp"] == first for e in patient_data["meal"])
        decision_keys = [k for k in prior if k != "prior_other_bolus"] + ["prior_other_bolus"]
        row = {
            "split": split,
            "patient_id": str(patient_data["patient_id"]),
            "interval_min": interval_min,
            "tau_t": tr.tau_t,
            "tau_next": tr.tau_next,
            "first_bolus_timestamp": first,
            "first_bolus_offset_min": offset,
            "timing_bin": timing_bin,
            "bolus_count_in_cell": int(tr.bolus_count),
            "bolus_types": tr.bolus_types,
            "bolus_doses": tr.bolus_doses,
            **prior,
            "meal_same_timestamp_as_bolus": int(same_time_meal),
        }
        row["any_new_information_before_bolus"] = any(row[k] > 0 for k in decision_keys) or same_time_meal > 0
        rows.append(row)
    return pd.DataFrame(rows)


def action_timing_summary(timing: pd.DataFrame) -> pd.DataFrame:
    if timing.empty:
        return pd.DataFrame()
    rows = []
    order = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    for keys, group in timing.groupby(["split", "interval_min"], observed=True):
        split, interval = keys
        n = len(group)
        for b in order:
            count = int((group["timing_bin"] == b).sum())
            rows.append({"split": split, "interval_min": interval, "metric": f"timing_{b}", "count": count, "proportion": count / n})
        context_cols = [c for c in group if c.startswith("prior_")] + ["meal_same_timestamp_as_bolus"]
        for col in context_cols:
            count = int((group[col] > 0).sum())
            rows.append({"split": split, "interval_min": interval, "metric": col, "count": count, "proportion": count / n})
        count = int(group["any_new_information_before_bolus"].sum())
        rows.append({"split": split, "interval_min": interval, "metric": "any_new_information_before_bolus", "count": count, "proportion": count / n})
        multiple = int((group["bolus_count_in_cell"] > 1).sum())
        rows.append({"split": split, "interval_min": interval, "metric": "multiple_bolus_cell", "count": multiple, "proportion": multiple / n})
    return pd.DataFrame(rows)


def bolus_event_coverage(
    patient_data: dict[str, Any], transitions: pd.DataFrame, split: str, interval_min: int,
) -> pd.DataFrame:
    """Audit every raw bolus episode, including those lost at CGM gaps/edges."""
    rows = []
    valid = transitions[transitions["valid"]].sort_values("tau_t") if len(transitions) else transitions
    all_cells = transitions.sort_values("tau_t") if len(transitions) else transitions
    for event_id, b in enumerate(sorted(patient_data["bolus"], key=lambda e: e["ts_begin"])):
        ts = b["ts_begin"]
        hit = valid[(valid["tau_t"] <= ts) & (ts < valid["tau_next"])] if len(valid) else valid
        any_hit = all_cells[(all_cells["tau_t"] <= ts) & (ts < all_cells["tau_next"])] if len(all_cells) else all_cells
        captured = len(hit) > 0
        if captured:
            reason = "captured_valid_transition"
            tau_t, tau_next = hit.iloc[0][["tau_t", "tau_next"]]
        elif len(any_hit):
            reason = "inside_invalid_or_unused_cgm_interval"
            tau_t, tau_next = any_hit.iloc[0][["tau_t", "tau_next"]]
        elif len(all_cells) and ts < all_cells["tau_t"].min():
            reason = "before_first_candidate_boundary"
            tau_t = tau_next = pd.NaT
        elif len(all_cells) and ts >= all_cells["tau_next"].max():
            reason = "at_or_after_last_candidate_boundary"
            tau_t = tau_next = pd.NaT
        else:
            reason = "between_saved_macro_cells_or_cgm_gap"
            tau_t = tau_next = pd.NaT
        rows.append({
            "split": split, "patient_id": str(patient_data["patient_id"]),
            "interval_min": interval_min, "bolus_event_id": event_id,
            "ts_begin": ts, "ts_end": b["ts_end"], "type": b["type"], "dose": float(b["dose"]),
            "captured": captured, "coverage_reason": reason,
            "captured_tau_t": tau_t, "captured_tau_next": tau_next,
        })
    return pd.DataFrame(rows)


def patient_transition_summary(flat: pd.DataFrame, build_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, pid, interval), group in flat.groupby(["split", "patient_id", "interval_min"], observed=True):
        days = max(int(pd.to_datetime(group["tau_t"]).dt.date.nunique()), 1)
        meta = build_meta[(build_meta["split"] == split) & (build_meta["patient_id"].astype(str) == str(pid)) & (build_meta["interval_min"] == interval)]
        m = meta.iloc[0] if len(meta) else {}
        n1 = int(group["action"].sum())
        rows.append({
            "split": split, "patient_id": pid, "interval_min": interval,
            "total_transitions": len(group), "action_1": n1,
            "action_1_rate": n1 / len(group), "observed_days": days,
            "bolus_initiations_per_day": n1 / days,
            "binary_tir_reward_mean": group["reward_binary_tir"].mean(),
            "weighted_reward_mean": group["reward_clinically_weighted"].mean(),
            "n_segments": int(m.get("n_segments", np.nan)),
            "invalid_gap_transitions": int(m.get("n_invalid_transitions", np.nan)),
            "max_invalid_gap_min": float(m.get("max_invalid_gap_min", np.nan)),
        })
    return pd.DataFrame(rows)


def temporal_action_support(flat5: pd.DataFrame) -> pd.DataFrame:
    """Chronological within-patient quartiles; no gap is treated as a transition."""
    rows = []
    for (split, pid), group in flat5.sort_values("tau_t").groupby(["split", "patient_id"], observed=True):
        group = group.reset_index(drop=True)
        labels = pd.qcut(np.arange(len(group)), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for quartile, z in group.groupby(labels, observed=True):
            rows.append({
                "split": split, "patient_id": pid, "time_quartile": str(quartile),
                "n": len(z), "action_1": int(z["action"].sum()),
                "action_1_rate": float(z["action"].mean()),
                "start": z["tau_t"].iloc[0], "end": z["tau_next"].iloc[-1],
            })
    return pd.DataFrame(rows)


def segment_action_support(segment_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    table = segment_table[segment_table["interval_min"] == 5]
    for split, group in table.groupby("split", observed=True):
        total = group["T"].sum()
        zero_t = group.loc[group["action_1"] == 0, "T"].sum()
        rows.append({
            "split": split, "segments": len(group),
            "segments_zero_action": int((group["action_1"] == 0).sum()),
            "segments_lt5_actions": int((group["action_1"] < 5).sum()),
            "transitions_in_zero_action_segments": int(zero_t),
            "fraction_transitions_in_zero_action_segments": float(zero_t / total),
            "median_actions_per_segment": float(group["action_1"].median()),
            "p25_actions_per_segment": float(group["action_1"].quantile(.25)),
        })
    return pd.DataFrame(rows)


def state_action_distributions(flat: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for (split, pid, interval, action), group in flat.groupby(["split", "patient_id", "interval_min", "action"], observed=True):
        for name in feature_names:
            x = group[f"state_raw__{name}"].to_numpy(float)
            rows.append({
                "split": split, "patient_id": pid, "interval_min": interval, "action": int(action),
                "feature": name, "n": len(x), "mean": np.mean(x), "sd": np.std(x),
                "p05": np.quantile(x, 0.05), "median": np.median(x), "p95": np.quantile(x, 0.95),
            })
    return pd.DataFrame(rows)


def _rate_table(df: pd.DataFrame, stratum: str, labels: pd.Series) -> pd.DataFrame:
    work = df[["split", "patient_id", "interval_min", "action"]].copy()
    work["stratum_type"] = stratum
    work["stratum"] = labels.astype(str)
    return work.groupby(["split", "patient_id", "interval_min", "stratum_type", "stratum"], observed=True)["action"].agg(n="size", action_1="sum", action_1_rate="mean").reset_index()


def stratified_action_rates(flat: pd.DataFrame) -> pd.DataFrame:
    g = flat["state_raw__glucose_current"]
    glucose = pd.cut(g, [-np.inf, 54, 70, 180, 250, np.inf], right=False, labels=["<54", "54-69", "70-179", "180-249", ">=250"])
    meal = pd.cut(flat["state_raw__time_since_meal_min"], [-np.inf, 30, 120, 240, np.inf], right=False, labels=["0-29", "30-119", "120-239", "capped/no_recent"])
    iob = pd.cut(flat["state_raw__iob_linear_4h"], [-np.inf, 0.01, 1, 3, 6, np.inf], right=False, labels=["~0", "0.01-0.99", "1-2.99", "3-5.99", ">=6"])
    hours = pd.to_datetime(flat["tau_t"]).dt.hour
    tod = pd.cut(hours, [0, 6, 12, 18, 24], right=False, labels=["00-05", "06-11", "12-17", "18-23"])
    return pd.concat([
        _rate_table(flat, "glucose", glucose),
        _rate_table(flat, "meal_recency_min", meal),
        _rate_table(flat, "iob_units", iob),
        _rate_table(flat, "time_of_day", tod),
    ], ignore_index=True)


def fit_propensity_diagnostic(flat5: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Patient-group cross-fitted train propensity and held-out split predictions."""
    feature_names = cfg["state"]["feature_names"]
    cols = [f"state__{x}" for x in feature_names]
    train = flat5[flat5["split"] == "training"].copy().reset_index(drop=True)
    test = flat5[flat5["split"] == "testing"].copy().reset_index(drop=True)
    if train["action"].nunique() < 2:
        raise RuntimeError("Training data contain only one action; propensity is not identifiable")
    x, y = train[cols].to_numpy(float), train["action"].to_numpy(int)
    groups = train["patient_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(6, len(unique_groups))
    oof = np.full(len(train), np.nan)
    for fit_idx, val_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(x[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
    final = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    final.fit(x, y)
    test_p = final.predict_proba(test[cols].to_numpy(float))[:, 1] if len(test) else np.array([])

    predictions = pd.concat([
        train[["split", "patient_id", "tau_t", "action"]].assign(propensity=oof, prediction_mode="group_cross_fitted"),
        test[["split", "patient_id", "tau_t", "action"]].assign(propensity=test_p, prediction_mode="trained_on_training_split"),
    ], ignore_index=True)

    qlo, qhi = map(float, cfg["qc"]["common_support_quantiles"])
    low = float(cfg["qc"]["propensity_extreme_low"])
    high = float(cfg["qc"]["propensity_extreme_high"])
    summary_rows = []
    feature_rows = []
    for split, group in predictions.groupby("split", observed=True):
        a0 = group.loc[group["action"] == 0, "propensity"].to_numpy()
        a1 = group.loc[group["action"] == 1, "propensity"].to_numpy()
        support_lo = max(np.quantile(a0, qlo), np.quantile(a1, qlo))
        support_hi = min(np.quantile(a0, qhi), np.quantile(a1, qhi))
        summary_rows.append({
            "split": split, "n": len(group), "n_action_1": len(a1),
            "auc": roc_auc_score(group["action"], group["propensity"]),
            "brier": brier_score_loss(group["action"], group["propensity"]),
            "p_min": group["propensity"].min(), "p01": group["propensity"].quantile(0.01),
            "p05": group["propensity"].quantile(0.05), "p50": group["propensity"].median(),
            "p95": group["propensity"].quantile(0.95), "p99": group["propensity"].quantile(0.99),
            "p_max": group["propensity"].max(),
            "propensity_below_extreme_low": float((group["propensity"] < low).mean()),
            "propensity_above_extreme_high": float((group["propensity"] > high).mean()),
            "common_support_lo": support_lo, "common_support_hi": support_hi,
            "outside_common_support_all": float(((group["propensity"] < support_lo) | (group["propensity"] > support_hi)).mean()),
            "action1_outside_common_support": float(((a1 < support_lo) | (a1 > support_hi)).mean()),
            "action0_outside_common_support": float(((a0 < support_lo) | (a0 > support_hi)).mean()),
        })

        source = flat5[flat5["split"] == split].reset_index(drop=True)
        for name in feature_names:
            raw_col = f"state_raw__{name}"
            z0 = source.loc[source["action"] == 0, raw_col].to_numpy(float)
            z1 = source.loc[source["action"] == 1, raw_col].to_numpy(float)
            lo0, hi0 = np.quantile(z0, [qlo, qhi])
            lo1, hi1 = np.quantile(z1, [qlo, qhi])
            lo_i, hi_i = max(lo0, lo1), min(hi0, hi1)
            feature_rows.append({
                "split": split, "feature": name, "a0_q01": lo0, "a0_q99": hi0,
                "a1_q01": lo1, "a1_q99": hi1, "intersection_lo": lo_i, "intersection_hi": hi_i,
                "empty_robust_intersection": bool(lo_i > hi_i),
                "a1_outside_a0_robust_range": float(((z1 < lo0) | (z1 > hi0)).mean()),
            })

    # Propensity bins reveal state regions containing only one observed action.
    edges = np.unique(np.quantile(predictions["propensity"], np.linspace(0, 1, 11)))
    if len(edges) < 3:
        bins = pd.Series("all", index=predictions.index)
    else:
        bins = pd.cut(predictions["propensity"], edges, include_lowest=True, duplicates="drop")
    regions = predictions.assign(propensity_region=bins.astype(str)).groupby(
        ["split", "propensity_region"], observed=True
    )["action"].agg(n="size", action_1="sum", action_1_rate="mean").reset_index()
    regions["only_one_action_observed"] = (regions["action_1"] == 0) | (regions["action_1"] == regions["n"])

    diagnostics = {
        "model": "unweighted L2 logistic regression",
        "purpose": "overlap diagnostic only; not causal adjustment",
        "n_features": len(feature_names),
        "train_group_cross_folds": n_splits,
        "train_action_rate": float(y.mean()),
        "coefficients": dict(zip(feature_names, final.coef_[0].tolist())),
        "intercept": float(final.intercept_[0]),
    }
    return predictions, pd.DataFrame(summary_rows), regions, {"model": diagnostics, "feature_support": pd.DataFrame(feature_rows)}


def interval_comparison(patient_summary: pd.DataFrame, timing_summary: pd.DataFrame) -> pd.DataFrame:
    agg = patient_summary.groupby(["split", "interval_min"], observed=True).agg(
        patients=("patient_id", "nunique"), total_transitions=("total_transitions", "sum"),
        total_action_1=("action_1", "sum"), total_segments=("n_segments", "sum"),
        invalid_gap_transitions=("invalid_gap_transitions", "sum"),
    ).reset_index()
    agg["action_1_rate"] = agg["total_action_1"] / agg["total_transitions"]
    extras = timing_summary[timing_summary["metric"].isin(["multiple_bolus_cell", "any_new_information_before_bolus"])].pivot_table(
        index=["split", "interval_min"], columns="metric", values="proportion", aggfunc="first"
    ).reset_index()
    return agg.merge(extras, on=["split", "interval_min"], how="left")
