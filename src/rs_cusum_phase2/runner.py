"""Parallel, checkpointed Phase-2 smoke and small-pilot runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.rs_cusum.support import load_support_rule

from .engine import (
    CoreCollection,
    build_support_screen,
    fit_candidate_cores,
    no_support_rule,
    run_calibrated_method,
)
from .generator import ALTERNATIVE_FAMILIES, NULL_SCENARIOS, generate_dataset
from .grids import candidate_grid, evaluation_grid
from .panels import build_panel
from .protocol import WORKSPACE, load_phase2_config
from .types import MethodResult, SyntheticDataset


RESULT_FIELDS = (
    "stage",
    "scenario",
    "family",
    "is_null",
    "effect_level",
    "effect_size",
    "replicate",
    "seed",
    "variant_id",
    "method",
    "threshold",
    "scaling",
    "evaluation_grid",
    "bootstrap_draws",
    "status",
    "testable",
    "true_cp",
    "number_admissible_candidates",
    "admissible_candidates",
    "observed_statistic",
    "p_value",
    "reject_001",
    "reject_005",
    "reject_010",
    "estimated_cp",
    "cp_error",
    "relative_cp_error",
    "retained_transitions",
    "retained_action1",
    "active_patients",
    "minimum_side_action1_count",
    "maximum_patient_transition_share",
    "maximum_patient_action1_share",
    "mean_active_patients",
    "support_failure_reasons",
    "numerical_diagnostics",
    "runtime_seconds",
)


def run_stage(stage: str, workers: int = 1) -> list[dict[str, Any]]:
    cfg = load_phase2_config()
    if stage not in {"stage_a", "stage_b"}:
        raise ValueError("stage must be stage_a or stage_b")
    if stage == "stage_b":
        gate_path = WORKSPACE / "results_rs_cusum" / "phase2" / "stage_a_gate.json"
        if not gate_path.is_file():
            raise RuntimeError("Stage B is blocked because Stage A gate is absent")
        with gate_path.open("r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if gate["gate"] != "STAGE_A_PASS":
            raise RuntimeError("Stage B is blocked because Stage A did not pass")
    tasks = _stage_tasks(stage, cfg)
    output = WORKSPACE / "results_rs_cusum" / "phase2" / f"{stage}_replicate_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"{stage}: {len(tasks)} datasets, workers={workers}", flush=True)
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.extend(run_replicate_task(task))
            if completed % 5 == 0 or completed == len(tasks):
                _write_rows(output, rows)
                print(
                    f"{stage}: {completed}/{len(tasks)} datasets, rows={len(rows)}, "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_replicate_task, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                if completed % 5 == 0 or completed == len(tasks):
                    rows.sort(key=_row_sort_key)
                    _write_rows(output, rows)
                    print(
                        f"{stage}: {completed}/{len(tasks)} datasets, rows={len(rows)}, "
                        f"elapsed={time.perf_counter()-started:.1f}s",
                        flush=True,
                    )
    rows.sort(key=_row_sort_key)
    _write_rows(output, rows)
    return rows


def run_replicate_task(task: tuple[str, str, str, int]) -> list[dict[str, Any]]:
    stage, scenario, effect_level, replicate = task
    cfg = load_phase2_config()
    seed = _stable_seed(
        int(cfg["simulation"]["master_seed"]), stage, scenario, effect_level, replicate
    )
    dataset = generate_dataset(scenario, seed, cfg, effect_level)
    started = time.perf_counter()
    rows = _run_dataset(stage, replicate, dataset, cfg)
    runtime = time.perf_counter() - started
    for row in rows:
        row["runtime_seconds"] = runtime
    return rows


def _run_dataset(
    stage: str, replicate: int, dataset: SyntheticDataset, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    sim = cfg["simulation"]
    candidates = candidate_grid(
        dataset.analysis_start,
        dataset.analysis_end,
        cfg["candidate_grid"]["fractions"],
    )
    rs_panel = build_panel(
        dataset,
        "rs_reentry",
        int(sim["post_gap_burnin_transitions"]),
    )
    grid_cache = {
        name: evaluation_grid(
            rs_panel,
            dataset.analysis_start,
            dataset.analysis_end,
            int(cfg["evaluation_grid"][f"{name}_reference_states"]),
            int(cfg["evaluation_grid"]["seed"]),
        )[:2]
        for name in ("small", "primary", "large")
    }
    rules = {
        name: load_support_rule(
            WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml", name
        )
        for name in ("lenient", "primary", "strict")
    }
    rs_screens = {
        name: build_support_screen(
            rs_panel, candidates, dataset.analysis_start, dataset.analysis_end, rule
        )
        for name, rule in rules.items()
    }
    rs_no_support = build_support_screen(
        rs_panel,
        candidates,
        dataset.analysis_start,
        dataset.analysis_end,
        no_support_rule(),
    )
    rs_balanced = fit_candidate_cores(
        rs_panel,
        candidates,
        dataset.analysis_start,
        dataset.analysis_end,
        "patient_balanced",
        "patient",
        float(sim["gamma"]),
        float(sim["ridge_alpha"]),
        int(sim["fqi_max_iterations"]),
        float(sim["fqi_tolerance"]),
    )
    draws = int(cfg["stages"][stage]["bootstrap_draws"])
    rows: list[dict[str, Any]] = []

    def add_variant(
        method: str,
        collection: CoreCollection,
        screen,
        panel,
        threshold: str,
        scaling: str,
        grid_name: str,
        variant_draws: int,
        variant_tag: str,
    ) -> None:
        states, actions = grid_cache[grid_name]
        result = run_calibrated_method(
            collection,
            screen,
            states,
            actions,
            dataset.analysis_start,
            dataset.analysis_end,
            len(dataset.patient_ids),
            scaling,
            variant_draws,
            _stable_seed(dataset.seed, method, threshold, scaling, grid_name, variant_draws),
        )
        rows.append(
            _result_row(
                stage,
                replicate,
                dataset,
                method,
                threshold,
                scaling,
                grid_name,
                variant_draws,
                variant_tag,
                panel,
                result,
            )
        )

    add_variant(
        "RS-full",
        rs_balanced,
        rs_screens["primary"],
        rs_panel,
        "primary",
        "harmonic_active_patients",
        "primary",
        draws,
        "primary",
    )

    if dataset.scenario in set(cfg["support_sensitivity"]["stage_b_scenarios"]):
        for threshold in ("lenient", "strict"):
            add_variant(
                "RS-full",
                rs_balanced,
                rs_screens[threshold],
                rs_panel,
                threshold,
                "harmonic_active_patients",
                "primary",
                draws,
                f"support_{threshold}",
            )
    if dataset.scenario in set(cfg["scaling_sensitivity"]["stage_b_scenarios"]):
        for scaling in ("both_side_patients", "count_effective_risk"):
            add_variant(
                "RS-full",
                rs_balanced,
                rs_screens["primary"],
                rs_panel,
                "primary",
                scaling,
                "primary",
                draws,
                f"scaling_{scaling}",
            )
    if dataset.scenario in set(cfg["grid_sensitivity"]["stage_b_scenarios"]):
        for grid_name in ("small", "large"):
            add_variant(
                "RS-full",
                rs_balanced,
                rs_screens["primary"],
                rs_panel,
                "primary",
                "harmonic_active_patients",
                grid_name,
                draws,
                f"grid_{grid_name}",
            )
    if (
        stage == "stage_b"
        and dataset.scenario in set(cfg["stages"]["bootstrap_sensitivity"]["scenarios"])
        and replicate < int(cfg["stages"]["bootstrap_sensitivity"]["replicates"])
    ):
        for sensitivity_draws in cfg["stages"]["bootstrap_sensitivity"]["draws"]:
            add_variant(
                "RS-full",
                rs_balanced,
                rs_screens["primary"],
                rs_panel,
                "primary",
                "harmonic_active_patients",
                "primary",
                int(sensitivity_draws),
                f"bootstrap_B{sensitivity_draws}",
            )

    if dataset.scenario in set(cfg["ablations"]["pooled_weight"]["scenarios"]):
        pooled = fit_candidate_cores(
            rs_panel,
            candidates,
            dataset.analysis_start,
            dataset.analysis_end,
            "pooled",
            "patient",
            float(sim["gamma"]),
            float(sim["ridge_alpha"]),
            int(sim["fqi_max_iterations"]),
            float(sim["fqi_tolerance"]),
        )
        add_variant(
            "RS-with-pooled-weight",
            pooled,
            rs_screens["primary"],
            rs_panel,
            "primary",
            "harmonic_active_patients",
            "primary",
            draws,
            "ablation_pooled_weight",
        )
    if dataset.scenario in set(cfg["ablations"]["no_support"]["scenarios"]):
        add_variant(
            "RS-no-support-screen",
            rs_balanced,
            rs_no_support,
            rs_panel,
            "none",
            "harmonic_active_patients",
            "primary",
            draws,
            "ablation_no_support",
        )
    if dataset.scenario in set(cfg["ablations"]["strict_first_gap_termination"]["scenarios"]):
        strict_panel = build_panel(dataset, "strict_first_gap")
        strict_screen = build_support_screen(
            strict_panel,
            candidates,
            dataset.analysis_start,
            dataset.analysis_end,
            rules["primary"],
        )
        strict_cores = fit_candidate_cores(
            strict_panel,
            candidates,
            dataset.analysis_start,
            dataset.analysis_end,
            "patient_balanced",
            "patient",
            float(sim["gamma"]),
            float(sim["ridge_alpha"]),
            int(sim["fqi_max_iterations"]),
            float(sim["fqi_tolerance"]),
        )
        add_variant(
            "strict_first_gap_termination",
            strict_cores,
            strict_screen,
            strict_panel,
            "primary",
            "harmonic_active_patients",
            "primary",
            draws,
            "ablation_strict_termination",
        )
    if dataset.scenario in set(cfg["ablations"]["original_rectangular"]["scenarios"]):
        original_panel = build_panel(dataset, "original_complete")
        original_screen = build_support_screen(
            original_panel,
            candidates,
            dataset.analysis_start,
            dataset.analysis_end,
            no_support_rule(),
        )
        original_cores = fit_candidate_cores(
            original_panel,
            candidates,
            dataset.analysis_start,
            dataset.analysis_end,
            "pooled",
            "transition",
            float(sim["gamma"]),
            float(sim["ridge_alpha"]),
            int(sim["fqi_max_iterations"]),
            float(sim["fqi_tolerance"]),
        )
        add_variant(
            "original_rectangular_compatible",
            original_cores,
            original_screen,
            original_panel,
            "none",
            "harmonic_active_patients",
            "primary",
            draws,
            "benchmark_original_rectangular",
        )
    return rows


def _result_row(
    stage: str,
    replicate: int,
    dataset: SyntheticDataset,
    method: str,
    threshold: str,
    scaling: str,
    grid_name: str,
    bootstrap_draws: int,
    variant_tag: str,
    panel,
    result: MethodResult,
) -> dict[str, Any]:
    support = result.diagnostics.get("support", {})
    testable = result.status == "TESTABLE"
    cp_error = (
        abs(result.estimated_change_point - dataset.true_change_point)
        if testable and dataset.true_change_point is not None
        else None
    )
    relative_cp_error = (
        cp_error / (dataset.analysis_end - dataset.analysis_start)
        if cp_error is not None
        else None
    )
    numerical = {
        key: value
        for key, value in result.diagnostics.items()
        if key != "support"
    }
    return {
        "stage": stage,
        "scenario": dataset.scenario,
        "family": dataset.family,
        "is_null": dataset.is_null,
        "effect_level": dataset.effect_level,
        "effect_size": dataset.effect_size,
        "replicate": replicate,
        "seed": dataset.seed,
        "variant_id": variant_tag,
        "method": method,
        "threshold": threshold,
        "scaling": scaling,
        "evaluation_grid": grid_name,
        "bootstrap_draws": bootstrap_draws,
        "status": result.status,
        "testable": testable,
        "true_cp": dataset.true_change_point,
        "number_admissible_candidates": len(result.admissible_candidates),
        "admissible_candidates": json.dumps(result.admissible_candidates),
        "observed_statistic": result.observed_statistic,
        "p_value": result.p_value,
        "reject_001": result.p_value <= 0.01 if testable else None,
        "reject_005": result.p_value <= 0.05 if testable else None,
        "reject_010": result.p_value <= 0.10 if testable else None,
        "estimated_cp": result.estimated_change_point,
        "cp_error": cp_error,
        "relative_cp_error": relative_cp_error,
        "retained_transitions": len(panel.records),
        "retained_action1": int(sum(record.action == 1 for record in panel.records)),
        "active_patients": len(panel.patient_ids),
        "minimum_side_action1_count": support.get("minimum_side_action1_count"),
        "maximum_patient_transition_share": support.get("maximum_total_patient_share"),
        "maximum_patient_action1_share": support.get("maximum_action1_patient_share"),
        "mean_active_patients": support.get("mean_active_patients"),
        "support_failure_reasons": json.dumps(
            support.get("failure_reason_counts", {}), sort_keys=True
        ),
        "numerical_diagnostics": json.dumps(numerical, sort_keys=True, default=_json_default),
        "runtime_seconds": None,
    }


def _stage_tasks(stage: str, cfg: dict[str, Any]) -> list[tuple[str, str, str, int]]:
    repetitions = int(cfg["stages"][stage][
        "replicates_each_core_scenario" if stage == "stage_a" else "replicates_each_scenario"
    ])
    tasks = [
        (stage, scenario, "none", replicate)
        for scenario in NULL_SCENARIOS
        for replicate in range(repetitions)
    ]
    effect_levels = ("moderate",) if stage == "stage_a" else ("weak", "moderate", "strong")
    tasks.extend(
        (stage, scenario, effect_level, replicate)
        for scenario in ALTERNATIVE_FAMILIES
        for effect_level in effect_levels
        for replicate in range(repetitions)
    )
    return tasks


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _row_sort_key(row: dict[str, Any]) -> tuple:
    return (
        row["scenario"],
        row["effect_level"],
        int(row["replicate"]),
        row["method"],
        row["variant_id"],
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage_a", "stage_b"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_stage(args.stage, args.workers)


if __name__ == "__main__":
    main()
