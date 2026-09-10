"""Deterministic synthetic MDP generator with missing-risk patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..risk_set import rectangular_to_panel
from ..types import RiskSetPanel


@dataclass(frozen=True)
class SimulationDesign:
    name: str
    n_patients: int = 12
    n_transitions: int = 240
    state_dimension: int = 4
    seed: int = 20260828
    missing_pattern: str = "complete"
    alternative: str = "null"
    change_index: int | None = None
    action_intercept: float = -1.2
    rare_action1: bool = False
    state_ar: float = 0.75
    state_noise_sd: float = 0.35
    reward_noise_sd: float = 0.5


@dataclass(frozen=True)
class GeneratedDataset:
    design: SimulationDesign
    states: np.ndarray
    actions: np.ndarray
    rewards_binary: np.ndarray
    rewards_weighted: np.ndarray
    observed_mask: np.ndarray
    patient_ids: tuple[str, ...]
    true_change_index: int | None

    def validate(self) -> None:
        n, t = self.actions.shape
        if self.states.shape != (n, t + 1, self.design.state_dimension):
            raise ValueError("synthetic state shape is inconsistent")
        if self.rewards_binary.shape != (n, t) or self.rewards_weighted.shape != (n, t):
            raise ValueError("synthetic reward shapes are inconsistent")
        if self.observed_mask.shape != (n, t):
            raise ValueError("synthetic observation-mask shape is inconsistent")
        if len(self.patient_ids) != n or len(set(self.patient_ids)) != n:
            raise ValueError("synthetic patient IDs are inconsistent")
        if not set(np.unique(self.actions)).issubset({0, 1}):
            raise ValueError("synthetic actions must be binary")

    def to_risk_panel(self) -> RiskSetPanel:
        self.validate()
        return rectangular_to_panel(
            self.states,
            self.actions,
            self.rewards_binary,
            self.rewards_weighted,
            self.observed_mask,
            self.patient_ids,
        )


def generate_dataset(design: SimulationDesign) -> GeneratedDataset:
    if design.n_patients < 2 or design.n_transitions < 4 or design.state_dimension < 1:
        raise ValueError("simulation dimensions are too small")
    if design.alternative not in {"null", "one_change"}:
        raise ValueError("alternative must be null or one_change")
    rng = np.random.default_rng(design.seed)
    n, t_steps, p = design.n_patients, design.n_transitions, design.state_dimension
    change = design.change_index
    if design.alternative == "one_change" and change is None:
        change = t_steps // 2
    if change is not None and not (1 <= change < t_steps):
        raise ValueError("change_index must be inside the trajectory")

    states = np.zeros((n, t_steps + 1, p), dtype=float)
    actions = np.zeros((n, t_steps), dtype=int)
    rewards = np.zeros((n, t_steps), dtype=float)
    states[:, 0] = rng.normal(0.0, 0.7, size=(n, p))
    patient_effect = rng.normal(0.0, 0.25, size=n)
    action_intercept = -4.0 if design.rare_action1 else design.action_intercept
    for time_index in range(t_steps):
        logit = action_intercept + 0.65 * states[:, time_index, 0] + patient_effect
        if p > 1:
            logit -= 0.25 * states[:, time_index, 1]
        action_probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        actions[:, time_index] = rng.binomial(1, action_probability)
        changed = design.alternative == "one_change" and time_index >= int(change)
        action_dynamics = 0.35 if not changed else -0.20
        drift = np.zeros((n, p))
        drift[:, 0] = action_dynamics * actions[:, time_index]
        if p > 1:
            drift[:, 1] = (0.10 if not changed else 0.45) * states[:, time_index, 0]
        states[:, time_index + 1] = (
            design.state_ar * states[:, time_index]
            + drift
            + rng.normal(0.0, design.state_noise_sd, size=(n, p))
        )
        reward_shift = 0.0 if not changed else 0.70
        rewards[:, time_index] = (
            0.35 * states[:, time_index, 0]
            - 0.20 * actions[:, time_index]
            + reward_shift
            + rng.normal(0.0, design.reward_noise_sd, size=n)
        )

    observed = _observation_mask(design, rng)
    dataset = GeneratedDataset(
        design=design,
        states=states,
        actions=actions,
        rewards_binary=rewards,
        rewards_weighted=rewards.copy(),
        observed_mask=observed,
        patient_ids=tuple(f"P{index:03d}" for index in range(n)),
        true_change_index=change if design.alternative == "one_change" else None,
    )
    dataset.validate()
    return dataset


def _observation_mask(design: SimulationDesign, rng: np.random.Generator) -> np.ndarray:
    n, t_steps = design.n_patients, design.n_transitions
    pattern = design.missing_pattern
    mask = np.ones((n, t_steps), dtype=bool)
    if pattern == "complete":
        return mask
    if pattern == "monotone_dropout":
        termination = rng.integers(max(2, t_steps // 2), t_steps + 1, size=n)
        termination[0] = t_steps
        for patient, stop in enumerate(termination):
            mask[patient, int(stop) :] = False
        return mask
    if pattern == "intermittent_reentry":
        mask = rng.random((n, t_steps)) > 0.12
        mask[:, :2] = True
        mask[:, -2:] = True
        return mask
    if pattern == "unequal_length":
        termination = np.linspace(max(2, int(0.55 * t_steps)), t_steps, n).astype(int)
        for patient, stop in enumerate(termination):
            mask[patient, stop:] = False
        return mask
    if pattern == "single_patient_dominance":
        short = max(2, int(0.30 * t_steps))
        mask[1:, short:] = False
        return mask
    if pattern == "dropout_plus_intermittent":
        termination = rng.integers(max(2, t_steps // 2), t_steps + 1, size=n)
        for patient, stop in enumerate(termination):
            mask[patient, int(stop) :] = False
        mask &= rng.random((n, t_steps)) > 0.10
        mask[:, :2] = True
        return mask
    raise ValueError(f"unknown missing pattern {pattern!r}")
