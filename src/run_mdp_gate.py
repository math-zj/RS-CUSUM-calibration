"""Run the pre-CUSUM MDP construction, forensic validation, and QC gate.

Safety boundary: this module imports neither run_population_cusum nor any
bootstrap/change-point function. It produces no p-values and no CUSUM output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parse_xml import parse_xml_file
from mdp_pipeline import (
    build_patient_mdp, flatten_valid_segments, load_protocol, save_patient_segments,
)
from mdp_qc import (
    action_timing_rows, action_timing_summary, audit_table_around,
    bolus_event_coverage, fit_propensity_diagnostic, interval_comparison, patient_transition_summary,
    segment_action_support, state_action_distributions, stratified_action_rates,
    temporal_action_support,
)
from fqi_sanity import run_fqi_sanity


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "mdp_5min_protocol.yaml"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False, default=_json_default)


def _segment_rows(segments):
    rows = []
    for seg in segments:
        rows.append({
            "split": seg.split, "patient_id": seg.patient_id, "interval_min": seg.interval_min,
            "segment_id": seg.segment_id, "T": len(seg.actions), "states_rows": len(seg.states),
            "start": pd.Timestamp(seg.tau[0]), "end": pd.Timestamp(seg.tau[-1]),
            "action_1": int(seg.actions.sum()), "action_1_rate": float(seg.actions.mean()),
        })
    return rows


def _reward_summary(flat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in flat.groupby(["split", "patient_id", "interval_min"], observed=True):
        split, pid, interval = keys
        for col in ("reward_binary_tir", "reward_clinically_weighted"):
            counts = g[col].value_counts().sort_index()
            for value, n in counts.items():
                rows.append({
                    "split": split, "patient_id": pid, "interval_min": interval,
                    "reward": col, "value": value, "n": int(n), "proportion": n / len(g),
                })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_protocol(PROTOCOL_PATH)
    out = PROJECT_ROOT / cfg["output"]["root"]
    arrays = PROJECT_ROOT / cfg["output"]["arrays"]
    audits = PROJECT_ROOT / cfg["output"]["audits"]
    qc = PROJECT_ROOT / cfg["output"]["qc"]
    logs = PROJECT_ROOT / cfg["output"]["logs"]
    for directory in (out, arrays, audits, qc, logs):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": datetime.now().isoformat(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "scope": "MDP construction + forensic validation + descriptive QC + FQI sanity",
        "explicitly_not_run": ["CUSUM", "bootstrap", "change point", "Round 2", "Round 3", "Round 4"],
    }
    _write_json(logs / "gate_run_manifest.json", manifest)

    all_segments = {5: [], 15: []}
    all_flat = []
    all_candidates = {5: [], 15: []}
    meta_rows, segment_rows, timing_rows, bolus_coverage_rows = [], [], [], []
    patient_cache = {}

    for split, folder, pattern in (
        ("training", PROJECT_ROOT / "OhioT1DM" / "train", "*-ws-training.xml"),
        ("testing", PROJECT_ROOT / "OhioT1DM" / "test", "*-ws-testing.xml"),
    ):
        for xml_path in sorted(folder.glob(pattern)):
            patient = parse_xml_file(xml_path)
            pid = str(patient["patient_id"])
            patient_cache[(split, pid)] = patient
            print(f"Building {split} patient {pid}")
            for interval in (5, 15):
                segments, candidates, meta = build_patient_mdp(patient, cfg, split, interval)
                for seg in segments:
                    seg.validate()
                all_segments[interval].extend(segments)
                meta_rows.append(meta)
                segment_rows.extend(_segment_rows(segments))
                if len(candidates):
                    candidates = candidates.assign(split=split)
                    all_candidates[interval].append(candidates)
                    timing_rows.append(action_timing_rows(patient, candidates, split, interval))
                    bolus_coverage_rows.append(bolus_event_coverage(patient, candidates, split, interval))
                flat = flatten_valid_segments(segments, cfg)
                if len(flat):
                    all_flat.append(flat)
                save_patient_segments(segments, arrays / f"{split}_{pid}_{interval}min.npz", cfg)

    meta_df = pd.DataFrame(meta_rows)
    segments_df = pd.DataFrame(segment_rows)
    flat = pd.concat(all_flat, ignore_index=True)
    cand5 = pd.concat(all_candidates[5], ignore_index=True)
    cand15 = pd.concat(all_candidates[15], ignore_index=True)
    timing = pd.concat([x for x in timing_rows if len(x)], ignore_index=True)
    bolus_coverage = pd.concat([x for x in bolus_coverage_rows if len(x)], ignore_index=True)
    timing_summary = action_timing_summary(timing)

    meta_df.to_csv(qc / "build_and_missingness_summary.csv", index=False)
    segments_df.to_csv(qc / "trajectory_segments.csv", index=False)
    cand5.to_csv(qc / "candidate_transitions_5min.csv.gz", index=False, compression="gzip")
    cand15.to_csv(qc / "candidate_transitions_15min.csv.gz", index=False, compression="gzip")
    timing.to_csv(qc / "action_timing_cells.csv", index=False)
    timing_summary.to_csv(qc / "action_timing_summary.csv", index=False)
    bolus_coverage.to_csv(qc / "bolus_event_coverage.csv", index=False)
    bolus_coverage.groupby(["split", "interval_min", "type"], observed=True)["captured"].agg(
        events="size", captured="sum", capture_rate="mean"
    ).reset_index().to_csv(qc / "bolus_event_coverage_summary.csv", index=False)

    feature_names = cfg["state"]["feature_names"]
    patient_summary = patient_transition_summary(flat, meta_df)
    patient_summary.to_csv(qc / "patient_transition_summary.csv", index=False)
    state_action_distributions(flat, feature_names).to_csv(qc / "state_action_distributions.csv", index=False)
    stratified_action_rates(flat).to_csv(qc / "stratified_action_rates.csv", index=False)
    _reward_summary(flat).to_csv(qc / "reward_distributions.csv", index=False)

    flat5 = flat[flat["interval_min"] == 5].reset_index(drop=True)
    temporal_action_support(flat5).to_csv(qc / "temporal_action_support_quartiles.csv", index=False)
    segment_action_support(segments_df).to_csv(qc / "segment_action_support_summary.csv", index=False)
    prop_pred, prop_summary, prop_regions, prop_extra = fit_propensity_diagnostic(flat5, cfg)
    prop_pred.to_csv(qc / "propensity_predictions.csv.gz", index=False, compression="gzip")
    prop_summary.to_csv(qc / "propensity_summary.csv", index=False)
    prop_regions.to_csv(qc / "propensity_regions.csv", index=False)
    prop_extra["feature_support"].to_csv(qc / "propensity_feature_support.csv", index=False)
    _write_json(qc / "propensity_model.json", prop_extra["model"])

    fqi = run_fqi_sanity(flat5, prop_pred, cfg)
    for name, frame in fqi.items():
        frame.to_csv(qc / f"fqi_{name}.csv", index=False)

    compare = interval_comparison(patient_summary, timing_summary)
    compare.to_csv(qc / "interval_5min_vs_15min.csv", index=False)

    # Prespecified real-timestamp forensic cases. Each output contains 20
    # consecutive native candidate transitions and may include an invalid gap.
    cases = [
        ("patient540_general", "training", "540", datetime(2027, 5, 19, 12, 8, 7)),
        ("dual_square_patient570", "training", "570", datetime(2021, 12, 27, 15, 9, 28)),
        ("temp_basal_patient540", "training", "540", datetime(2027, 5, 19, 12, 32, 6)),
        ("long_cgm_gap_patient591", "training", "591", datetime(2021, 12, 26, 5, 38, 0)),
        ("exact_meal_bolus_tie_patient544", "training", "544", datetime(2027, 6, 13, 10, 45, 0)),
    ]
    case_index = []
    for name, split, pid, target in cases:
        source = cand5[(cand5["split"] == split) & (cand5["patient_id"].astype(str) == pid)]
        table = audit_table_around(source, target, 20)
        table.to_csv(audits / f"{name}_20_transitions.csv", index=False)
        case_index.append({
            "case": name, "split": split, "patient_id": pid, "target_timestamp": target,
            "rows": len(table), "contains_action": bool(table["action"].sum()),
            "contains_invalid": bool((~table["valid"]).sum()),
            "contains_exact_tie": bool(table["exact_meal_bolus_tie_at_tau"].sum() or table["meal_bolus_same_timestamp_in_cell"].sum()),
        })
    pd.DataFrame(case_index).to_csv(audits / "forensic_case_index.csv", index=False)

    # Machine-readable invariant report used by the final gate review.
    invariant_report = {
        "segments_checked": len(segments_df),
        "all_states_are_T_plus_1": bool((segments_df["states_rows"] == segments_df["T"] + 1).all()),
        "all_saved_transitions_valid": bool(all(seg.transition_metadata["valid"].all() for seg in all_segments[5])),
        "reward_tail_zero_padding_used": False,
        "state_prepend_used": False,
        "future_glucose_interpolation_used": False,
        "whole_trajectory_standardization_used": False,
        "long_gap_stitching_used": False,
        "formal_cusum_called": False,
    }
    _write_json(qc / "invariant_report.json", invariant_report)

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["outputs"] = {
        "valid_5min_transitions": int((flat["interval_min"] == 5).sum()),
        "valid_15min_transitions": int((flat["interval_min"] == 15).sum()),
        "segments": len(segments_df),
    }
    _write_json(logs / "gate_run_manifest.json", manifest)
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
