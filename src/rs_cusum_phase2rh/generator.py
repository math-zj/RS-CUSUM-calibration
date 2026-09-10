"""Frozen complete stationary transfer-null data-generating processes for Phase 2R-H."""

from __future__ import annotations

import csv
from functools import lru_cache
from math import log, sqrt
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.rs_cusum_phase2.types import SyntheticDataset


WORKSPACE = Path(__file__).resolve().parents[2]
REGISTRY = WORKSPACE / "results_rs_cusum" / "phase2rh" / "scenario_registry.csv"
COMPONENTS = ("initial", "heterogeneity", "action", "transition", "reward")
COMPATIBILITY_LABEL = "N0_complete_balanced_null"


@lru_cache(maxsize=1)
def scenario_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["scenario_id"]: row for row in csv.DictReader(handle)}
    if len(rows) != 9:
        raise RuntimeError("Phase 2R-H must contain exactly nine frozen scenarios")
    return rows


def _number(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value in (None, "") else float(value)


def _standardized_t(rng: np.random.Generator, df: float, sd: float, size: Any) -> np.ndarray:
    return rng.standard_t(df, size=size) * (sd / sqrt(df / (df - 2.0)))


def _innovation(
    rng: np.random.Generator, law: str, sd: float, size: Any, df: float = 0.0
) -> np.ndarray:
    if law == "gaussian":
        return rng.normal(0.0, sd, size=size)
    if law.startswith("student_t_"):
        return _standardized_t(rng, df, sd, size)
    raise ValueError(f"unsupported innovation law {law!r}")


def generate_transfer_dataset(
    scenario_id: str,
    dataset_seed: int,
    component_seeds: Mapping[str, int],
    cfg: dict[str, Any],
) -> tuple[SyntheticDataset, dict[str, Any]]:
    """Generate one stationary null dataset; the actual DGP label stays external to frozen M4."""
    row = scenario_registry()[scenario_id]
    if set(component_seeds) != set(COMPONENTS):
        raise ValueError("exactly five registered DGP component seeds are required")
    dgp = cfg["dgp"]
    n = int(dgp["patients"])
    horizon = int(dgp["horizon"])
    p = int(dgp["state_dimension"])
    analysis_start = int(dgp["analysis_start"])
    negative = int(row["phenotype_negative"])
    positive = int(row["phenotype_positive"])
    if negative + positive != n:
        raise ValueError("phenotype composition does not equal frozen patient count")
    phenotypes = np.r_[-np.ones(negative), np.ones(positive)]
    patient_ids = tuple(f"P{index:02d}" for index in range(n))

    initial_rng = np.random.default_rng(int(component_seeds["initial"]))
    heterogeneity_rng = np.random.default_rng(int(component_seeds["heterogeneity"]))
    action_rng = np.random.default_rng(int(component_seeds["action"]))
    transition_rng = np.random.default_rng(int(component_seeds["transition"]))
    reward_rng = np.random.default_rng(int(component_seeds["reward"]))

    states = np.zeros((n, horizon + 1, p), dtype=float)
    actions = np.zeros((n, horizon), dtype=int)
    rewards = np.zeros((n, horizon), dtype=float)
    states[:, 0] = initial_rng.normal(0.0, float(dgp["initial_state_sd"]), (n, p))
    states[:, 0, 0] += float(dgp["phenotype_shift_x0"]) * phenotypes
    states[:, 0, 1] = phenotypes

    re_law = row["policy_random_effect_law"]
    re_sd = _number(row, "policy_random_effect_sd")
    if re_law == "gaussian":
        patient_effect = heterogeneity_rng.normal(0.0, re_sd, n)
    elif re_law.startswith("student_t_"):
        df = float(re_law.split("_")[2])
        patient_effect = _standardized_t(heterogeneity_rng, df, re_sd, n)
    else:
        raise ValueError(f"unsupported policy random-effect law {re_law!r}")

    action_target = _number(row, "action_target")
    action_x0 = _number(row, "action_x0_coefficient")
    intercept = log(action_target / (1.0 - action_target))
    transition_law = row["transition_innovation_law"]
    transition_df = _number(row, "transition_df")
    transition_sd = _number(row, "transition_sd")

    for time_index in range(horizon):
        eta = (
            intercept
            + patient_effect
            + action_x0 * states[:, time_index, 0]
            + float(dgp["phenotype_policy_coefficient"]) * states[:, time_index, 1]
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        actions[:, time_index] = action_rng.binomial(1, probability)
        states[:, time_index + 1, 0] = (
            float(dgp["state_ar"]) * states[:, time_index, 0]
            + float(dgp["state_phenotype_coefficient"]) * states[:, time_index, 1]
            + float(dgp["state_action_coefficient"]) * actions[:, time_index]
            + _innovation(transition_rng, transition_law, transition_sd, n, transition_df)
        )
        states[:, time_index + 1, 1] = phenotypes
        if p > 2:
            states[:, time_index + 1, 2:] = (
                float(dgp["other_state_ar"]) * states[:, time_index, 2:]
                + float(dgp["other_state_x0_coefficient"]) * states[:, time_index, [0]]
                + _innovation(
                    transition_rng, transition_law, transition_sd, (n, p - 2), transition_df
                )
            )

    current = states[:, :-1]
    reward_mean = (
        float(dgp["reward_x0_coefficient"]) * current[:, :, 0]
        + float(dgp["reward_phenotype_coefficient"]) * current[:, :, 1]
        + float(dgp["reward_x2_coefficient"]) * current[:, :, 2]
        + float(dgp["reward_action_coefficient"]) * actions
    )
    reward_law = row["reward_innovation_law"]
    reward_sd = _number(row, "reward_sd")
    reward_df = _number(row, "reward_df")
    if reward_law in {"gaussian", "student_t_5_standardized", "student_t_7_standardized"}:
        innovation = _innovation(reward_rng, reward_law, reward_sd, actions.shape, reward_df)
        conditional_scale = np.full(actions.shape, reward_sd)
    elif reward_law == "gaussian_state_action_scale":
        conditional_scale = reward_sd * (
            0.75 + 0.35 * np.abs(np.tanh(current[:, :, 0])) + 0.20 * actions
        )
        innovation = reward_rng.normal(0.0, 1.0, actions.shape) * conditional_scale
    elif reward_law == "stationary_gaussian_AR1":
        rho = _number(row, "reward_ar1_rho")
        innovation = np.empty(actions.shape, dtype=float)
        innovation[:, 0] = reward_rng.normal(0.0, reward_sd, n)
        driving_sd = reward_sd * sqrt(1.0 - rho * rho)
        for time_index in range(1, horizon):
            innovation[:, time_index] = (
                rho * innovation[:, time_index - 1]
                + reward_rng.normal(0.0, driving_sd, n)
            )
        conditional_scale = np.full(actions.shape, reward_sd)
    else:
        raise ValueError(f"unsupported reward innovation law {reward_law!r}")
    rewards[:] = reward_mean + innovation

    dataset = SyntheticDataset(
        scenario=COMPATIBILITY_LABEL,
        family=f"phase2rh::{scenario_id}",
        is_null=True,
        effect_level="none",
        effect_size=0.0,
        seed=int(dataset_seed),
        states=states,
        actions=actions,
        rewards=rewards,
        observed_mask=np.ones_like(actions, dtype=bool),
        patient_ids=patient_ids,
        phenotypes=phenotypes,
        analysis_start=analysis_start,
        analysis_end=horizon,
        true_change_point=None,
    )
    dataset.validate()
    diagnostics = {
        "actual_scenario_id": scenario_id,
        "compatibility_label": dataset.scenario,
        "stationary_no_changepoint": True,
        "complete_observation": True,
        "action_rate": float(np.mean(actions)),
        "action_count": int(np.sum(actions)),
        "minimum_patient_action_count": int(np.min(np.sum(actions, axis=1))),
        "maximum_patient_action_share": float(np.max(np.sum(actions, axis=1)) / max(np.sum(actions), 1)),
        "state_x0_mean": float(np.mean(current[:, :, 0])),
        "state_x0_sd": float(np.std(current[:, :, 0], ddof=1)),
        "reward_mean": float(np.mean(rewards)),
        "reward_sd": float(np.std(rewards, ddof=1)),
        "innovation_mean": float(np.mean(innovation)),
        "innovation_sd": float(np.std(innovation, ddof=1)),
        "conditional_scale_min": float(np.min(conditional_scale)),
        "conditional_scale_max": float(np.max(conditional_scale)),
        "component_seeds": {name: int(component_seeds[name]) for name in COMPONENTS},
    }
    return dataset, diagnostics
