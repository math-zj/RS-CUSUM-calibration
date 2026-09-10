"""Create denominator-explicit null, power, support, ablation, and CP summaries."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .protocol import WORKSPACE


GROUP_FIELDS = (
    "scenario",
    "family",
    "effect_level",
    "effect_size",
    "variant_id",
    "method",
    "threshold",
    "scaling",
    "evaluation_grid",
    "bootstrap_draws",
)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_absolute():
        path = WORKSPACE / path
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_phase2(rows: list[dict[str, Any]], output_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(output_root)
    if not root.is_absolute():
        root = WORKSPACE / root
    root.mkdir(parents=True, exist_ok=True)
    null_rows = [row for row in rows if _bool(row["is_null"])]
    alternative_rows = [row for row in rows if not _bool(row["is_null"])]
    null_summary = _calibration_summary(null_rows)
    power_summary = _calibration_summary(alternative_rows)
    support_summary = _support_summary(rows)
    ablation_summary = [row for row in null_summary + power_summary if row["method"] != "RS-full" or row["variant_id"] == "primary"]
    cp_accuracy = _cp_summary(alternative_rows)
    _write_csv(root / "null_summary.csv", null_summary)
    _write_csv(root / "power_summary.csv", power_summary)
    _write_csv(root / "support_summary.csv", support_summary)
    _write_csv(root / "ablation_summary.csv", ablation_summary)
    _write_csv(root / "cp_accuracy.csv", cp_accuracy)
    return {
        "null_summary": null_summary,
        "power_summary": power_summary,
        "support_summary": support_summary,
        "ablation_summary": ablation_summary,
        "cp_accuracy": cp_accuracy,
    }


def _calibration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(rows)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        testable = [row for row in group if row["status"] == "TESTABLE"]
        status_counts = Counter(row["status"] for row in group)
        base = dict(zip(GROUP_FIELDS, key))
        record: dict[str, Any] = {
            **base,
            "total_replicates": len(group),
            "testable_replicates": len(testable),
            "not_testable_replicates": status_counts["NOT_TESTABLE_SUPPORT"],
            "testable_fraction": len(testable) / len(group) if group else float("nan"),
            "status_counts": json.dumps(dict(sorted(status_counts.items())), sort_keys=True),
            "numerical_failure_rate": (
                status_counts["NUMERICAL_FAILURE"]
                + status_counts["BOOTSTRAP_FAILURE"]
                + status_counts["ZERO_VARIANCE"]
            )
            / len(group),
            "mean_admissible_candidates": _mean(group, "number_admissible_candidates"),
            "median_admissible_candidates": _median(group, "number_admissible_candidates"),
            "mean_active_patients": _mean(group, "mean_active_patients"),
            "mean_minimum_side_action1": _mean(group, "minimum_side_action1_count"),
            "mean_max_patient_transition_share": _mean(group, "maximum_patient_transition_share"),
            "mean_max_patient_action1_share": _mean(group, "maximum_patient_action1_share"),
            "support_failure_reasons": _aggregate_reason_json(group),
        }
        for alpha_label, field in (("001", "reject_001"), ("005", "reject_005"), ("010", "reject_010")):
            rejection_count = sum(_bool(row[field]) for row in testable)
            all_rate = rejection_count / len(group) if group else float("nan")
            conditional_rate = rejection_count / len(testable) if testable else float("nan")
            lower, upper = _wilson(rejection_count, len(testable))
            record[f"rejection_count_{alpha_label}"] = rejection_count
            record[f"rejection_rate_all_{alpha_label}"] = all_rate
            record[f"rejection_rate_testable_{alpha_label}"] = conditional_rate
            record[f"mcse_testable_{alpha_label}"] = (
                math.sqrt(conditional_rate * (1 - conditional_rate) / len(testable))
                if testable
                else float("nan")
            )
            record[f"wilson_low_testable_{alpha_label}"] = lower
            record[f"wilson_high_testable_{alpha_label}"] = upper
        output.append(record)
    return output


def _support_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(rows)
    output = []
    for key, group in sorted(grouped.items()):
        base = dict(zip(GROUP_FIELDS, key))
        status_counts = Counter(row["status"] for row in group)
        output.append(
            {
                **base,
                "total_replicates": len(group),
                "testable_fraction": status_counts["TESTABLE"] / len(group),
                "not_testable_fraction": status_counts["NOT_TESTABLE_SUPPORT"] / len(group),
                "mean_admissible_candidates": _mean(group, "number_admissible_candidates"),
                "median_admissible_candidates": _median(group, "number_admissible_candidates"),
                "mean_retained_transitions": _mean(group, "retained_transitions"),
                "mean_retained_action1": _mean(group, "retained_action1"),
                "mean_active_patients": _mean(group, "mean_active_patients"),
                "mean_minimum_side_action1": _mean(group, "minimum_side_action1_count"),
                "mean_max_patient_transition_share": _mean(group, "maximum_patient_transition_share"),
                "mean_max_patient_action1_share": _mean(group, "maximum_patient_action1_share"),
                "support_failure_reasons": _aggregate_reason_json(group),
            }
        )
    return output


def _cp_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(rows)
    output = []
    for key, group in sorted(grouped.items()):
        testable = [row for row in group if row["status"] == "TESTABLE"]
        detected = [row for row in testable if _bool(row["reject_005"])]
        all_errors = _float_values(testable, "cp_error")
        detected_errors = _float_values(detected, "cp_error")
        all_relative = _float_values(testable, "relative_cp_error")
        record = dict(zip(GROUP_FIELDS, key))
        record.update(
            {
                "total_replicates": len(group),
                "testable_replicates": len(testable),
                "detected_replicates_005": len(detected),
                "true_cp": _first_nonempty(group, "true_cp"),
                "median_absolute_error_all_testable": _quantile(all_errors, 0.50),
                "iqr_low_absolute_error_all_testable": _quantile(all_errors, 0.25),
                "iqr_high_absolute_error_all_testable": _quantile(all_errors, 0.75),
                "median_relative_error_all_testable": _quantile(all_relative, 0.50),
                "median_absolute_error_detection_conditioned": _quantile(detected_errors, 0.50),
                "iqr_low_absolute_error_detection_conditioned": _quantile(detected_errors, 0.25),
                "iqr_high_absolute_error_detection_conditioned": _quantile(detected_errors, 0.75),
            }
        )
        output.append(record)
    return output


def _group(rows: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    return grouped


def _aggregate_reason_json(rows: list[dict[str, Any]]) -> str:
    total: Counter[str] = Counter()
    for row in rows:
        raw = row.get("support_failure_reasons", "")
        if raw:
            total.update(json.loads(raw))
    return json.dumps(dict(sorted(total.items())), sort_keys=True)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = _float_values(rows, field)
    return float(np.mean(values)) if values else float("nan")


def _median(rows: list[dict[str, Any]], field: str) -> float:
    values = _float_values(rows, field)
    return float(np.median(values)) if values else float("nan")


def _float_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(field)
        if raw not in (None, "", "None", "nan"):
            values.append(float(raw))
    return values


def _first_nonempty(rows: list[dict[str, Any]], field: str) -> Any:
    for row in rows:
        if row.get(field) not in (None, "", "None"):
            return row[field]
    return None


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability)) if values else float("nan")


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return center - half, center + half


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
