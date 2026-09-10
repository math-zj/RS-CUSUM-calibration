"""Precision planning using only the already-saved Phase 2R-E hybrid draws."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
INPUT = WORKSPACE / "results_rs_cusum" / "phase2re" / "hybrid_process_draws.npz"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2rf" / "sample_size_precision_plan.csv"
THRESHOLDS = (0.90, 0.95, 0.975, 0.99)
SAMPLE_SIZES = (120, 250, 500, 1000, 1500, 2000, 2500)
PLANNING_REPETITIONS = 100
PLANNING_SEED = 2026083109


def integrated_joint_mae(reference: np.ndarray, comparison: np.ndarray) -> float:
    """Equal-weighted joint-exceedance MAE, scaled by each tail probability."""
    reference = np.asarray(reference, np.float32)
    comparison = np.asarray(comparison, np.float32)
    dimension = reference.shape[1]
    off_diagonal = ~np.eye(dimension, dtype=bool)
    components: list[float] = []
    for threshold_level in THRESHOLDS:
        threshold = np.quantile(np.abs(reference), threshold_level, axis=0)
        reference_event = (np.abs(reference) > threshold).astype(np.float32)
        comparison_event = (np.abs(comparison) > threshold).astype(np.float32)
        reference_joint = reference_event.T @ reference_event / len(reference_event)
        comparison_joint = comparison_event.T @ comparison_event / len(comparison_event)
        components.append(
            float(np.mean(np.abs(comparison_joint - reference_joint)[off_diagonal]))
            / (1.0 - threshold_level)
        )
    return float(np.mean(components))


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, float)
    low, high = np.quantile(array, (0.025, 0.975))
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "mc_sd": float(np.std(array, ddof=1)),
        "q025": float(low),
        "q975": float(high),
        "interval_width": float(high - low),
        "observed_range": float(np.ptp(array)),
    }


def run() -> list[dict[str, object]]:
    """Bootstrap existing E draws; never generate a Phase 2R-F process."""
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    with np.load(INPUT, allow_pickle=False) as data:
        oracle = np.asarray(data["true_all"], np.float32).reshape(-1, 448)
        fitted = np.asarray(data["fitted_all"], np.float32).reshape(-1, 448)
    rng = np.random.default_rng(PLANNING_SEED)
    rows: list[dict[str, object]] = []
    for sample_size in SAMPLE_SIZES:
        raw_values: list[float] = []
        noise_values: list[float] = []
        excess_values: list[float] = []
        for _ in range(PLANNING_REPETITIONS):
            first = oracle[rng.choice(len(oracle), sample_size, replace=True)]
            second = oracle[rng.choice(len(oracle), sample_size, replace=True)]
            m4 = fitted[rng.choice(len(fitted), sample_size, replace=True)]
            raw = integrated_joint_mae(first, m4)
            noise = integrated_joint_mae(first, second)
            raw_values.append(raw)
            noise_values.append(noise)
            excess_values.append(raw - noise)
        for metric, values in (
            ("raw_integrated_standardized_joint_exceedance_mae", raw_values),
            ("oracle_oracle_noise_floor", noise_values),
            ("excess_over_oracle_oracle_noise", excess_values),
        ):
            rows.append(
                {
                    "source": "Phase2R-E saved true_all/fitted_all only",
                    "planning_seed": PLANNING_SEED,
                    "sample_size_per_bank": sample_size,
                    "planning_repetitions": PLANNING_REPETITIONS,
                    "metric": metric,
                    "expected_q99_exceedances_per_coordinate": sample_size * 0.01,
                    **_summarize(values),
                    "target_mc_se_max": 0.0075,
                    "target_ci_width_max": 0.03,
                    "selected_for_phase2rf": sample_size == 2500,
                    "selection_reason": (
                        "meets precision target and gives about 25 q99 exceedances per coordinate"
                        if sample_size == 2500
                        else "precision/convergence comparison only"
                    ),
                }
            )
        print(f"planning {sample_size}/{SAMPLE_SIZES[-1]}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)
    return rows


if __name__ == "__main__":
    run()
