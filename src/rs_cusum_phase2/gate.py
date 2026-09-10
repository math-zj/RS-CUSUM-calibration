"""Stage-A smoke gate and Phase-2 pilot gate checks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .engine import VALID_STATUSES
from .protocol import WORKSPACE, load_phase2_config
from .runner import run_replicate_task
from .summarize import read_rows


def evaluate_stage_a() -> dict[str, Any]:
    cfg = load_phase2_config()
    path = WORKSPACE / "results_rs_cusum" / "phase2" / "stage_a_replicate_results.csv"
    rows = read_rows(path)
    primary = [row for row in rows if row["variant_id"] == "primary"]
    failures: list[str] = []
    invalid_status = sorted({row["status"] for row in rows} - VALID_STATUSES)
    if invalid_status:
        failures.append(f"invalid statuses: {invalid_status}")
    for row in rows:
        if row["status"] == "TESTABLE":
            p_value = float(row["p_value"])
            lower = 1.0 / (int(row["bootstrap_draws"]) + 1)
            if not (lower <= p_value <= 1.0):
                failures.append(f"p-value out of range: {row['scenario']} rep {row['replicate']}")
            candidates = json.loads(row["admissible_candidates"])
            if int(row["estimated_cp"]) not in candidates:
                failures.append(f"CP outside U_adm: {row['scenario']} rep {row['replicate']}")
            numerical = json.loads(row["numerical_diagnostics"])
            for candidate in numerical.get("candidate", {}).values():
                if candidate["core"]["cluster_unit"] != (
                    "transition" if row["method"] == "original_rectangular_compatible" else "patient"
                ):
                    failures.append(f"cluster unit mismatch: {row['variant_id']}")
    numerical_statuses = {"NUMERICAL_FAILURE", "BOOTSTRAP_FAILURE", "ZERO_VARIANCE"}
    numerical_rate = sum(row["status"] in numerical_statuses for row in primary) / len(primary)
    if numerical_rate > float(cfg["gate"]["numerical_failure_max"]):
        failures.append(f"primary numerical failure rate {numerical_rate:.4f} exceeds limit")

    first = run_replicate_task(("stage_a", "N0_complete_balanced_null", "none", 0))
    second = run_replicate_task(("stage_a", "N0_complete_balanced_null", "none", 0))
    first_primary = _without_runtime(next(row for row in first if row["variant_id"] == "primary"))
    second_primary = _without_runtime(next(row for row in second if row["variant_id"] == "primary"))
    deterministic = first_primary == second_primary
    if not deterministic:
        failures.append("same-seed primary replicate is not deterministic")

    scenario_counts = Counter(row["scenario"] for row in primary)
    expected_replicates = int(cfg["stages"]["stage_a"]["replicates_each_core_scenario"])
    for scenario, count in scenario_counts.items():
        if count != expected_replicates:
            failures.append(f"{scenario} has {count} primary rows, expected {expected_replicates}")
    output = {
        "gate": "STAGE_A_PASS" if not failures else "STAGE_A_FAIL",
        "failures": failures,
        "total_result_rows": len(rows),
        "primary_rows": len(primary),
        "primary_status_counts": dict(sorted(Counter(row["status"] for row in primary).items())),
        "primary_numerical_failure_rate": numerical_rate,
        "same_seed_deterministic": deterministic,
        "scenario_primary_counts": dict(sorted(scenario_counts.items())),
        "formal_ohiot1dm_analysis": False,
    }
    target = WORKSPACE / "results_rs_cusum" / "phase2" / "stage_a_gate.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def _without_runtime(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "runtime_seconds"}


if __name__ == "__main__":
    print(json.dumps(evaluate_stage_a(), indent=2, sort_keys=True))
