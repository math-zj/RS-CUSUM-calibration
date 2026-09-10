"""Engineering and protocol tests for frozen Phase 2R-H."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2rd.engine import run_m4
from src.rs_cusum_phase2rd.protocol import load_config as load_phase2rd_config
from src.rs_cusum_phase2rh.analysis import prepare_transfer_analysis
from src.rs_cusum_phase2rh.generator import COMPONENTS, COMPATIBILITY_LABEL, generate_transfer_dataset
from src.rs_cusum_phase2rh.protocol import (
    OUTPUT, WORKSPACE, load_config, load_seeds, sha256, stable_seed,
)
from src.rs_cusum_phase2rh.runner import _grid


def _components(prefix: str, scenario_index: int) -> dict[str, int]:
    return {name: int(load_seeds(f"{prefix}_dgp_{name}")[scenario_index]) for name in COMPONENTS}


def test_protocol_identity_and_frozen_method() -> None:
    cfg = load_config("frozen")
    assert cfg["protocol"]["phase2rd_gate_immutable"] == "M4_MECHANISM_JOINT_TAIL_FAIL"
    assert cfg["protocol"]["phase2re_gate_immutable"] == "TAIL_METRIC_UNSTABLE"
    assert cfg["protocol"]["phase2rf_gate_immutable"] == "M4_JOINT_TAIL_CONFIRMED"
    assert cfg["protocol"]["phase2rg_gate_immutable"] == "M4_FRESH_GLOBAL_NULL_PASS"
    assert sha256(WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py") == cfg["frozen_method"]["m4_engine_sha256"]
    assert sha256(WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py") == cfg["frozen_method"]["phase2rf_metric_sha256"]
    assert not any((OUTPUT / "checkpoints").rglob("*.npz")) if (OUTPUT / "checkpoints").exists() else True


def test_seed_registry_is_unique_disjoint_and_derivations_match_m4() -> None:
    meta = json.loads((OUTPUT / "seed_registry.json").read_text(encoding="utf-8"))
    assert meta["all_values_unique"]
    assert meta["disjoint_from_all_prior_registries"]
    assert meta["historical_collision_count"] == 0
    bootstrap = load_seeds("m4_outer_bootstrap")
    for label in ("design", "reward", "dataset"):
        derived = load_seeds(f"m4_{label}_process")
        for scenario_index in (0, 4, 8):
            for outer in (0, 199):
                for draw in (0, 198):
                    assert int(derived[scenario_index, outer, draw]) == stable_seed(
                        int(bootstrap[scenario_index, outer]), "M4", label, draw
                    )


@pytest.mark.parametrize("scenario_index", range(9))
def test_each_frozen_dgp_is_deterministic_complete_null_and_supported(scenario_index: int) -> None:
    cfg = load_config("frozen")
    scenarios = tuple(cfg["scenario_plan"]["order"])
    scenario = scenarios[scenario_index]
    seed = int(load_seeds("engineering_dataset")[scenario_index])
    components = _components("engineering", scenario_index)
    first, diag = generate_transfer_dataset(scenario, seed, components, cfg)
    second, second_diag = generate_transfer_dataset(scenario, seed, components, cfg)
    assert first.scenario == COMPATIBILITY_LABEL
    assert first.family == f"phase2rh::{scenario}"
    assert first.is_null and first.true_change_point is None
    assert np.all(first.observed_mask)
    assert first.states.shape == (12, 241, 21)
    assert first.actions.shape == first.rewards.shape == (12, 240)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.actions, second.actions)
    assert np.array_equal(first.rewards, second.rewards)
    assert diag == second_diag
    assert diag["actual_scenario_id"] == scenario
    assert diag["compatibility_label"] == COMPATIBILITY_LABEL
    rd_cfg = load_phase2rd_config("D2")
    parent = load_phase2_config()
    states, actions = _grid(cfg)
    prepared = prepare_transfer_analysis(first, rd_cfg, parent, states, actions)
    assert tuple(sorted(prepared.fits)) == tuple(cfg["frozen_method"]["base_candidates"])
    assert prepared.evaluation_design.shape == (64, 44)


def test_scenario_specific_semantics_are_present() -> None:
    cfg = load_config("frozen")
    scenarios = tuple(cfg["scenario_plan"]["order"])
    generated = {}
    for index, scenario in enumerate(scenarios):
        generated[scenario] = generate_transfer_dataset(
            scenario, int(load_seeds("engineering_dataset")[index]),
            _components("engineering", index), cfg,
        )[0]
    assert tuple(np.unique(generated["H0_reference_N0"].phenotypes, return_counts=True)[1]) == (6, 6)
    assert tuple(np.unique(generated["H2_imbalanced_design"].phenotypes, return_counts=True)[1]) == (8, 4)
    hetero = generated["H5_reward_heteroskedastic"]
    current = hetero.states[:, :-1]
    mean = .40*current[:, :, 0] + .10*current[:, :, 1] + .05*current[:, :, 2] - .20*hetero.actions
    residual = hetero.rewards - mean
    low = residual[np.abs(current[:, :, 0]) < .1]
    high = residual[np.abs(current[:, :, 0]) > 1.0]
    assert np.std(high) > np.std(low)
    temporal = generated["H6_reward_AR1"]
    current = temporal.states[:, :-1]
    residual = temporal.rewards - (.40*current[:, :, 0] + .10*current[:, :, 1] + .05*current[:, :, 2] - .20*temporal.actions)
    assert np.corrcoef(residual[:, :-1].ravel(), residual[:, 1:].ravel())[0, 1] > 0.15


def test_unchanged_m4_accepts_transfer_compatibility_dataset() -> None:
    cfg = load_config("frozen")
    rd_cfg = load_phase2rd_config("D2")
    parent = load_phase2_config()
    evaluation_states, evaluation_actions = _grid(cfg)
    scenario_index = 3
    scenario = cfg["scenario_plan"]["order"][scenario_index]
    dataset, _ = generate_transfer_dataset(
        scenario, int(load_seeds("engineering_dataset")[scenario_index]),
        _components("engineering", scenario_index), cfg,
    )
    prepared = prepare_transfer_analysis(dataset, rd_cfg, parent, evaluation_states, evaluation_actions)
    result = run_m4(
        prepared, rd_cfg, parent, evaluation_states, evaluation_actions, 2,
        int(load_seeds("engineering_bootstrap")[scenario_index]),
    )
    assert result.joint.shape == (2, 7, 64)
    assert np.all(result.valid)
    assert np.all(np.isfinite(result.joint))
