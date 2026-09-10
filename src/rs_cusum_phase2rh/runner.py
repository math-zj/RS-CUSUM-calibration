"""Deterministic resumable oracle and frozen-M4 execution for Phase 2R-H."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2rc.c0_engine import observed_arrays
from src.rs_cusum_phase2rd.engine import run_m4
from src.rs_cusum_phase2rd.protocol import load_config as load_phase2rd_config

from .analysis import prepare_transfer_analysis
from .generator import COMPONENTS, generate_transfer_dataset
from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds


LAYER_NAMES = ("D0", "D1", "D2", "D3", "D4")
CHECKPOINTS = OUTPUT / "checkpoints"


def _grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(WORKSPACE / cfg["frozen_method"]["evaluation_grid_file"], allow_pickle=False) as data:
        return np.asarray(data["states"], float), np.asarray(data["actions"], int)


def _layers(raw: np.ndarray, normalized: np.ndarray, joint: np.ndarray, midpoint: int) -> np.ndarray:
    return np.asarray(
        [
            abs(raw[midpoint, 0]),
            abs(normalized[midpoint, 0]),
            np.max(np.abs(normalized[midpoint])),
            np.max(np.abs(joint[:, 0])),
            np.max(np.abs(joint)),
        ],
        dtype=np.float32,
    )


def _draw_layers(raw: np.ndarray, normalized: np.ndarray, joint: np.ndarray, midpoint: int) -> np.ndarray:
    return np.column_stack(
        [
            np.abs(raw[:, midpoint, 0]),
            np.abs(normalized[:, midpoint, 0]),
            np.max(np.abs(normalized[:, midpoint]), axis=1),
            np.max(np.abs(joint[:, :, 0]), axis=1),
            np.max(np.abs(joint), axis=(1, 2)),
        ]
    ).astype(np.float32)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _oracle_path(scenario: str, bank: str, start: int, stop: int) -> Path:
    return CHECKPOINTS / scenario / f"oracle_{bank}" / f"chunk_{start:04d}_{stop:04d}.npz"


def _m4_path(scenario: str, index: int) -> Path:
    return CHECKPOINTS / scenario / "m4" / f"outer_{index:04d}.npz"


def _oracle_chunk(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["cfg"]
    rd_cfg = load_phase2rd_config("D2")
    parent = load_phase2_config()
    seeds = np.asarray(task["seeds"], np.uint64)
    indices = np.asarray(task["indices"], int)
    component = {name: np.asarray(task["component_seeds"][name], np.uint64) for name in COMPONENTS}
    joint = np.full((len(seeds), 448), np.nan, np.float32)
    layers = np.full((len(seeds), 5), np.nan, np.float32)
    valid = np.zeros(len(seeds), bool)
    errors = np.full(len(seeds), "", dtype="<U1024")
    dgp_diagnostics = np.full(len(seeds), "", dtype="<U4096")
    started = time.perf_counter()
    for local, seed in enumerate(seeds):
        try:
            components = {name: int(component[name][local]) for name in COMPONENTS}
            dataset, dgp_diag = generate_transfer_dataset(
                task["scenario"], int(seed), components, cfg
            )
            prepared = prepare_transfer_analysis(
                dataset, rd_cfg, parent, task["evaluation_states"], task["evaluation_actions"]
            )
            raw, normalized, process = observed_arrays(prepared)
            midpoint = sorted(prepared.fits).index(prepared.midpoint)
            if process.shape != (7, 64) or np.any(~np.isfinite(process)):
                raise FloatingPointError("invalid oracle full process")
            joint[local] = process.reshape(-1)
            layers[local] = _layers(raw, normalized, process, midpoint)
            valid[local] = True
            dgp_diagnostics[local] = json.dumps(dgp_diag, sort_keys=True)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            errors[local] = f"{type(error).__name__}: {error}"[:1023]
    return {
        "scenario": task["scenario"], "bank": task["bank"], "indices": indices,
        "seeds": seeds, "component_seeds": component, "joint": joint, "layers": layers,
        "valid": valid, "errors": errors, "dgp_diagnostics": dgp_diagnostics,
        "seconds": time.perf_counter() - started,
    }


def _m4_outer(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["cfg"]
    rd_cfg = load_phase2rd_config("D2")
    parent = load_phase2_config()
    scenario = task["scenario"]
    outer_index = int(task["outer_index"])
    dataset_seed = int(task["dataset_seed"])
    bootstrap_seed = int(task["bootstrap_seed"])
    draws = int(task["draws"])
    started = time.perf_counter()
    observed_joint = np.full(448, np.nan, np.float32)
    observed_layers = np.full(5, np.nan, np.float32)
    joint = np.full((draws, 448), np.nan, np.float32)
    layers = np.full((draws, 5), np.nan, np.float32)
    valid = np.zeros(draws, bool)
    error = ""
    diagnostics = ""
    dgp_diagnostics = ""
    try:
        dataset, dgp_diag = generate_transfer_dataset(
            scenario, dataset_seed, task["component_seeds"], cfg
        )
        dgp_diagnostics = json.dumps(dgp_diag, sort_keys=True)
        prepared = prepare_transfer_analysis(
            dataset, rd_cfg, parent, task["evaluation_states"], task["evaluation_actions"]
        )
        raw, normalized, observed_process = observed_arrays(prepared)
        midpoint = sorted(prepared.fits).index(prepared.midpoint)
        observed_joint[:] = observed_process.reshape(-1).astype(np.float32)
        observed_layers[:] = _layers(raw, normalized, observed_process, midpoint)
        process = run_m4(
            prepared, rd_cfg, parent, task["evaluation_states"], task["evaluation_actions"],
            draws, bootstrap_seed,
        )
        joint[:] = process.joint.reshape(draws, -1).astype(np.float32)
        layers[:] = _draw_layers(process.raw, process.normalized, process.joint, midpoint)
        valid[:] = np.asarray(process.valid, bool)
        diagnostics = json.dumps(process.diagnostics, sort_keys=True)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as caught:
        error = f"{type(caught).__name__}: {caught}"[:2047]
    return {
        "scenario": scenario, "outer_index": outer_index,
        "dataset_seed": np.uint64(dataset_seed), "bootstrap_seed": np.uint64(bootstrap_seed),
        "component_seeds": {name: np.uint64(task["component_seeds"][name]) for name in COMPONENTS},
        "observed_joint": observed_joint, "observed_layers": observed_layers,
        "joint": joint, "layers": layers, "valid": valid,
        "error": error, "diagnostics": diagnostics, "dgp_diagnostics": dgp_diagnostics,
        "seconds": time.perf_counter() - started,
    }


def _valid_oracle_checkpoint(
    path: Path, scenario: str, bank: str, indices: np.ndarray, seeds: np.ndarray,
    component_seeds: dict[str, np.ndarray],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(
                str(data["scenario"].item()) == scenario
                and str(data["bank"].item()) == bank
                and np.array_equal(data["indices"], indices)
                and np.array_equal(data["seeds"], seeds)
                and all(np.array_equal(data[f"seed_{name}"], component_seeds[name]) for name in COMPONENTS)
                and data["joint"].shape == (len(seeds), 448)
                and data["layers"].shape == (len(seeds), 5)
                and data["valid"].shape == (len(seeds),)
                and data["errors"].shape == (len(seeds),)
            )
    except (OSError, ValueError, KeyError):
        return False


def _valid_m4_checkpoint(
    path: Path, scenario: str, index: int, dataset_seed: int, bootstrap_seed: int,
    component_seeds: dict[str, int], draws: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(
                str(data["scenario"].item()) == scenario
                and int(data["outer_index"]) == index
                and int(data["dataset_seed"]) == dataset_seed
                and int(data["bootstrap_seed"]) == bootstrap_seed
                and all(int(data[f"seed_{name}"]) == component_seeds[name] for name in COMPONENTS)
                and data["joint"].shape == (draws, 448)
                and data["layers"].shape == (draws, 5)
                and data["valid"].shape == (draws,)
                and data["observed_joint"].shape == (448,)
                and data["observed_layers"].shape == (5,)
            )
    except (OSError, ValueError, KeyError):
        return False


def run_oracle(bank: str, workers: int) -> dict[str, Any]:
    if bank not in {"primary", "reference"}:
        raise ValueError(bank)
    cfg = load_config("run")
    scenarios = tuple(cfg["scenario_plan"]["order"])
    evaluation_states, evaluation_actions = _grid(cfg)
    all_seeds = load_seeds(f"oracle_{bank}_process")
    all_components = {name: load_seeds(f"oracle_{bank}_dgp_{name}") for name in COMPONENTS}
    chunk = int(cfg["sample_plan"]["oracle_chunk_size"])
    tasks: list[dict[str, Any]] = []
    resumed = 0
    for scenario_index, scenario in enumerate(scenarios):
        seeds = all_seeds[scenario_index]
        for start in range(0, len(seeds), chunk):
            stop = min(start + chunk, len(seeds))
            indices = np.arange(start, stop)
            components = {name: all_components[name][scenario_index, start:stop] for name in COMPONENTS}
            path = _oracle_path(scenario, bank, start, stop)
            if path.exists() and not _valid_oracle_checkpoint(
                path, scenario, bank, indices, seeds[start:stop], components
            ):
                raise RuntimeError(f"invalid oracle checkpoint; refusing splice/overwrite: {path}")
            if path.exists():
                resumed += stop - start
                continue
            tasks.append({
                "cfg": cfg, "scenario": scenario, "bank": bank, "indices": indices,
                "seeds": seeds[start:stop], "component_seeds": components,
                "evaluation_states": evaluation_states, "evaluation_actions": evaluation_actions,
            })
    total = int(all_seeds.size)
    started = time.perf_counter()
    complete = resumed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_oracle_chunk, task): task for task in tasks}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                start = int(result["indices"][0]); stop = int(result["indices"][-1]) + 1
                arrays: dict[str, Any] = {
                    "scenario": np.asarray(result["scenario"]), "bank": np.asarray(bank),
                    "indices": result["indices"], "seeds": result["seeds"],
                    "joint": result["joint"], "layers": result["layers"],
                    "valid": result["valid"], "errors": result["errors"],
                    "dgp_diagnostics": result["dgp_diagnostics"], "seconds": np.asarray(result["seconds"]),
                }
                arrays.update({f"seed_{name}": result["component_seeds"][name] for name in COMPONENTS})
                _atomic_npz(_oracle_path(result["scenario"], bank, start, stop), **arrays)
                complete += len(result["indices"])
                print(f"Phase2R-H oracle {bank}: {complete}/{total} [{result['scenario']}]", flush=True)
    payload = {
        "stage": f"oracle_{bank}", "scenarios": len(scenarios),
        "expected_processes": total, "completed_processes": complete,
        "resumed_processes": resumed, "new_processes": complete - resumed,
        "completed_seed_reruns": 0, "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / f"oracle_{bank}_run_status.json", payload)
    return payload


def run_m4_outer(workers: int) -> dict[str, Any]:
    cfg = load_config("run")
    scenarios = tuple(cfg["scenario_plan"]["order"])
    evaluation_states, evaluation_actions = _grid(cfg)
    dataset_seeds = load_seeds("m4_outer_dataset")
    bootstrap_seeds = load_seeds("m4_outer_bootstrap")
    components_all = {name: load_seeds(f"m4_outer_dgp_{name}") for name in COMPONENTS}
    draws = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])
    tasks: list[dict[str, Any]] = []
    resumed = 0
    for scenario_index, scenario in enumerate(scenarios):
        for index in range(dataset_seeds.shape[1]):
            components = {name: int(components_all[name][scenario_index, index]) for name in COMPONENTS}
            path = _m4_path(scenario, index)
            if path.exists() and not _valid_m4_checkpoint(
                path, scenario, index, int(dataset_seeds[scenario_index, index]),
                int(bootstrap_seeds[scenario_index, index]), components, draws,
            ):
                raise RuntimeError(f"invalid M4 checkpoint; refusing splice/overwrite: {path}")
            if path.exists():
                resumed += 1
                continue
            tasks.append({
                "cfg": cfg, "scenario": scenario, "outer_index": index,
                "dataset_seed": dataset_seeds[scenario_index, index],
                "bootstrap_seed": bootstrap_seeds[scenario_index, index],
                "component_seeds": components, "draws": draws,
                "evaluation_states": evaluation_states, "evaluation_actions": evaluation_actions,
            })
    total = int(dataset_seeds.size)
    started = time.perf_counter()
    complete = resumed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_m4_outer, task): task for task in tasks}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                arrays: dict[str, Any] = {
                    "scenario": np.asarray(result["scenario"]),
                    "outer_index": np.asarray(result["outer_index"]),
                    "dataset_seed": result["dataset_seed"], "bootstrap_seed": result["bootstrap_seed"],
                    "observed_joint": result["observed_joint"], "observed_layers": result["observed_layers"],
                    "joint": result["joint"], "layers": result["layers"], "valid": result["valid"],
                    "error": np.asarray(result["error"]), "diagnostics": np.asarray(result["diagnostics"]),
                    "dgp_diagnostics": np.asarray(result["dgp_diagnostics"]), "seconds": np.asarray(result["seconds"]),
                }
                arrays.update({f"seed_{name}": result["component_seeds"][name] for name in COMPONENTS})
                _atomic_npz(_m4_path(result["scenario"], int(result["outer_index"])), **arrays)
                complete += 1
                print(f"Phase2R-H M4: {complete}/{total} [{result['scenario']}:{result['outer_index']}]", flush=True)
    payload = {
        "stage": "m4", "scenarios": len(scenarios),
        "expected_outer_replicates": total, "completed_outer_replicates": complete,
        "resumed_outer_replicates": resumed, "new_outer_replicates": complete - resumed,
        "completed_seed_reruns": 0, "inner_draws_per_outer": draws,
        "completed_inner_draws": complete * draws, "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "m4_run_status.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["oracle-primary", "oracle-reference", "m4"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_m4_outer(args.workers) if args.stage == "m4" else run_oracle(
        args.stage.removeprefix("oracle-"), args.workers
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
