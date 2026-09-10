"""Frozen Phase-2 synthetic Markov environments and observation processes."""

from __future__ import annotations

from math import log
from typing import Any

import numpy as np

from .types import SyntheticDataset


NULL_SCENARIOS = (
    "N0_complete_balanced_null",
    "N1_monotone_dropout_null",
    "N2_intermittent_reentry_null",
    "N3_unequal_length_null",
    "N4_rare_action_moderate_null",
    "N4_rare_action_low_null",
    "N4_rare_action_very_low_null",
    "N5_patient_dominance_null",
    "N6_missing_rare_unequal_null",
    "N7_risk_composition_shift_null",
    "N8_informative_missing_mild_null",
    "N8_informative_missing_strong_null",
)

ALTERNATIVE_FAMILIES = (
    "A0_complete_balanced_change",
    "A1_intermittent_change",
    "A2_rare_action_change",
    "A3_unequal_dominance_change",
)


def generate_dataset(
    scenario: str,
    seed: int,
    cfg: dict[str, Any],
    effect_level: str = "none",
) -> SyntheticDataset:
    sim = cfg["simulation"]
    if scenario in NULL_SCENARIOS:
        base_scenario = scenario
        is_null = True
        effect_size = 0.0
        family = scenario.split("_null")[0]
    elif scenario in ALTERNATIVE_FAMILIES:
        alternative = cfg["alternative_scenarios"][scenario]
        base_scenario = alternative["base"]
        is_null = False
        levels_key = "rare_action_effect_levels" if scenario == "A2_rare_action_change" else "common_effect_levels"
        levels = cfg["alternative_scenarios"][levels_key]
        if effect_level not in levels:
            raise ValueError(f"unknown effect level {effect_level!r}")
        effect_size = float(levels[effect_level])
        family = scenario
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    base_cfg = cfg["null_scenarios"][base_scenario]
    n_patients = int(sim["patients"])
    state_dimension = int(sim["state_dimension"])
    horizon = int(base_cfg["horizon"])
    analysis_start = int(sim["initial_record_history_transitions"])
    analysis_end = horizon
    if analysis_start >= analysis_end:
        raise ValueError("initial history exhausts the simulation horizon")
    true_change = None if is_null else _round_half_up(
        analysis_start + float(cfg["alternative_scenarios"][scenario]["true_cp_fraction"]) * (analysis_end - analysis_start)
    )

    rng = np.random.default_rng(seed)
    patient_ids = tuple(f"P{index:02d}" for index in range(n_patients))
    phenotypes = np.concatenate(
        [-np.ones(n_patients // 2), np.ones(n_patients - n_patients // 2)]
    )
    states = np.zeros((n_patients, horizon + 1, state_dimension), dtype=float)
    actions = np.zeros((n_patients, horizon), dtype=int)
    rewards = np.zeros((n_patients, horizon), dtype=float)
    states[:, 0] = rng.normal(0.0, 0.7, size=(n_patients, state_dimension))
    states[:, 0, 0] += 0.5 * phenotypes
    states[:, 0, 1] = phenotypes
    patient_policy_effect = rng.normal(0.0, 0.15, size=n_patients)
    action_target = float(base_cfg["action_target"])
    intercept = log(action_target / (1.0 - action_target))

    for time_index in range(horizon):
        policy_intercept = np.full(n_patients, intercept) + patient_policy_effect
        if base_scenario == "N5_patient_dominance_null":
            policy_intercept[0] += 2.0
            policy_intercept[1:] -= 0.35
        logits = (
            policy_intercept
            + float(sim["action_state_coefficient"]) * states[:, time_index, 0]
            + 0.20 * states[:, time_index, 1]
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        actions[:, time_index] = rng.binomial(1, probabilities)

        changed = (not is_null) and time_index >= int(true_change)
        general_change = changed and scenario != "A2_rare_action_change"
        rare_action_change = changed and scenario == "A2_rare_action_change"
        next_state = np.zeros((n_patients, state_dimension), dtype=float)
        next_state[:, 0] = (
            float(sim["state_ar"]) * states[:, time_index, 0]
            + 0.12 * states[:, time_index, 1]
            + 0.18 * actions[:, time_index]
            + (0.25 * effect_size if general_change else 0.0)
            + (0.25 * effect_size * actions[:, time_index] if rare_action_change else 0.0)
            + rng.normal(0.0, float(sim["nonphenotype_state_noise_sd"]), size=n_patients)
        )
        next_state[:, 1] = phenotypes
        if state_dimension > 2:
            next_state[:, 2:] = (
                0.55 * states[:, time_index, 2:]
                + 0.04 * states[:, time_index, [0]]
                + rng.normal(
                    0.0,
                    float(sim["nonphenotype_state_noise_sd"]),
                    size=(n_patients, state_dimension - 2),
                )
            )
        states[:, time_index + 1] = next_state
        rewards[:, time_index] = (
            0.40 * states[:, time_index, 0]
            + 0.10 * states[:, time_index, 1]
            + (0.05 * states[:, time_index, 2] if state_dimension > 2 else 0.0)
            - 0.20 * actions[:, time_index]
            + (effect_size if general_change else 0.0)
            + (effect_size * actions[:, time_index] if rare_action_change else 0.0)
            + rng.normal(0.0, float(sim["reward_noise_sd"]), size=n_patients)
        )

    observed = _observation_mask(base_scenario, states, actions, phenotypes, analysis_start, rng)
    dataset = SyntheticDataset(
        scenario=scenario,
        family=family,
        is_null=is_null,
        effect_level=effect_level,
        effect_size=effect_size,
        seed=seed,
        states=states,
        actions=actions,
        rewards=rewards,
        observed_mask=observed,
        patient_ids=patient_ids,
        phenotypes=phenotypes,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        true_change_point=true_change,
    )
    dataset.validate()
    return dataset


def _observation_mask(
    scenario: str,
    states: np.ndarray,
    actions: np.ndarray,
    phenotypes: np.ndarray,
    analysis_start: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_patients, horizon = actions.shape
    mask = np.ones((n_patients, horizon), dtype=bool)
    if scenario in {
        "N0_complete_balanced_null",
        "N4_rare_action_moderate_null",
        "N4_rare_action_low_null",
        "N4_rare_action_very_low_null",
    }:
        return mask
    if scenario == "N1_monotone_dropout_null":
        stops = rng.integers(_round_half_up(0.55 * horizon), horizon + 1, size=n_patients)
        stops[0] = horizon
        return _apply_stops(mask, stops)
    if scenario == "N2_intermittent_reentry_null":
        for patient in range(n_patients):
            _add_gap(mask[patient], rng, int(0.38 * horizon), int(0.50 * horizon), 6, 12)
            _add_gap(mask[patient], rng, int(0.68 * horizon), int(0.80 * horizon), 6, 12)
        return mask
    if scenario == "N3_unequal_length_null":
        stops = np.asarray([_round_half_up(0.40 * horizon)] * 4 + [_round_half_up(0.67 * horizon)] * 4 + [horizon] * 4)
        return _apply_stops(mask, stops)
    if scenario == "N5_patient_dominance_null":
        stops = rng.integers(_round_half_up(0.48 * horizon), _round_half_up(0.72 * horizon) + 1, size=n_patients)
        stops[0] = horizon
        return _apply_stops(mask, stops)
    if scenario == "N6_missing_rare_unequal_null":
        stops = rng.integers(_round_half_up(0.60 * horizon), horizon + 1, size=n_patients)
        stops[0] = horizon
        _apply_stops(mask, stops)
        for patient in range(n_patients):
            if stops[patient] > analysis_start + 60:
                _add_gap(mask[patient], rng, analysis_start + 20, int(stops[patient]) - 24, 6, 10)
                if patient % 2 == 0 and stops[patient] > analysis_start + 120:
                    _add_gap(mask[patient], rng, analysis_start + 80, int(stops[patient]) - 12, 5, 9)
        return mask
    if scenario == "N7_risk_composition_shift_null":
        midpoint = analysis_start + _round_half_up(0.50 * (horizon - analysis_start))
        mask[phenotypes > 0, midpoint:] = False
        return mask
    if scenario in {"N8_informative_missing_mild_null", "N8_informative_missing_strong_null"}:
        base_hazard = 0.005 if "mild" in scenario else 0.015
        for patient in range(n_patients):
            time_index = analysis_start + 12
            while time_index < horizon:
                hazard = base_hazard * np.exp(
                    0.35 * states[patient, time_index, 0] + 0.45 * actions[patient, time_index]
                )
                hazard = float(np.clip(hazard, 0.001, 0.08))
                if rng.random() < hazard:
                    length = int(rng.integers(6, 13))
                    mask[patient, time_index : min(horizon, time_index + length)] = False
                    time_index += length
                else:
                    time_index += 1
        return mask
    raise ValueError(f"observation mask is undefined for {scenario}")


def _apply_stops(mask: np.ndarray, stops: np.ndarray) -> np.ndarray:
    for patient, stop in enumerate(stops):
        mask[patient, int(stop) :] = False
    return mask


def _add_gap(
    row: np.ndarray,
    rng: np.random.Generator,
    low: int,
    high: int,
    min_length: int,
    max_length: int,
) -> None:
    if high <= low:
        return
    start = int(rng.integers(low, high + 1))
    length = int(rng.integers(min_length, max_length + 1))
    row[start : min(len(row), start + length)] = False


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))
