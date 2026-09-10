"""Deterministic resumable execution of the fresh locked Phase 2R-G holdout."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from src.rs_cusum_phase2rf.runner import _m4_outer, _oracle_chunk

from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds


LAYER_NAMES = ("D0", "D1", "D2", "D3", "D4")
CHECKPOINTS = OUTPUT / "checkpoints"


def _grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(WORKSPACE / cfg["frozen_method"]["evaluation_grid_file"], allow_pickle=False) as data:
        return np.asarray(data["states"], float), np.asarray(data["actions"], int)


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


def _oracle_path(bank: str, start: int, stop: int) -> Path:
    return CHECKPOINTS / f"oracle_{bank}" / f"chunk_{start:04d}_{stop:04d}.npz"


def _m4_path(index: int) -> Path:
    return CHECKPOINTS / "m4" / f"outer_{index:04d}.npz"


def _valid_oracle_checkpoint(path: Path, bank: str, indices: np.ndarray, seeds: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(
                str(data["bank"].item()) == bank
                and np.array_equal(data["indices"], indices)
                and np.array_equal(data["seeds"], seeds)
                and data["joint"].shape == (len(seeds), 448)
                and data["layers"].shape == (len(seeds), 5)
                and data["valid"].shape == (len(seeds),)
                and data["errors"].shape == (len(seeds),)
            )
    except (OSError, ValueError, KeyError):
        return False


def _valid_m4_checkpoint(path: Path, index: int, dataset_seed: int, bootstrap_seed: int, draws: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(
                int(data["outer_index"]) == index
                and int(data["dataset_seed"]) == dataset_seed
                and int(data["bootstrap_seed"]) == bootstrap_seed
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
    evaluation_states, evaluation_actions = _grid(cfg)
    seeds = load_seeds(f"oracle_{bank}_process")
    chunk = int(cfg["sample_plan"]["oracle_chunk_size"])
    tasks: list[dict[str, Any]] = []
    resumed = 0
    for start in range(0, len(seeds), chunk):
        stop = min(start + chunk, len(seeds))
        indices = np.arange(start, stop)
        path = _oracle_path(bank, start, stop)
        if path.exists() and not _valid_oracle_checkpoint(path, bank, indices, seeds[start:stop]):
            raise RuntimeError(f"invalid oracle checkpoint; refusing to splice or overwrite: {path}")
        if path.exists():
            resumed += stop - start
            continue
        tasks.append(
            {
                "cfg": cfg, "bank": bank, "indices": indices, "seeds": seeds[start:stop],
                "evaluation_states": evaluation_states, "evaluation_actions": evaluation_actions,
            }
        )
    started = time.perf_counter()
    complete = resumed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_oracle_chunk, task): task for task in tasks}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                start = int(result["indices"][0])
                stop = int(result["indices"][-1]) + 1
                _atomic_npz(
                    _oracle_path(bank, start, stop),
                    bank=np.asarray(bank), indices=result["indices"], seeds=result["seeds"],
                    joint=result["joint"], layers=result["layers"], valid=result["valid"],
                    errors=result["errors"], seconds=np.asarray(result["seconds"]),
                )
                complete += len(result["indices"])
                print(f"Phase2R-G oracle {bank}: {complete}/{len(seeds)}", flush=True)
    payload = {
        "stage": f"oracle_{bank}", "expected_processes": len(seeds),
        "completed_processes": complete, "resumed_processes": resumed,
        "new_processes": complete - resumed, "completed_seed_reruns": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / f"oracle_{bank}_run_status.json", payload)
    return payload


def run_m4_outer(workers: int) -> dict[str, Any]:
    cfg = load_config("run")
    evaluation_states, evaluation_actions = _grid(cfg)
    dataset_seeds = load_seeds("m4_outer_dataset")
    bootstrap_seeds = load_seeds("m4_outer_bootstrap")
    draws = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])
    tasks: list[dict[str, Any]] = []
    resumed = 0
    for index, (dataset_seed, bootstrap_seed) in enumerate(zip(dataset_seeds, bootstrap_seeds)):
        path = _m4_path(index)
        if path.exists() and not _valid_m4_checkpoint(path, index, int(dataset_seed), int(bootstrap_seed), draws):
            raise RuntimeError(f"invalid M4 checkpoint; refusing to splice or overwrite: {path}")
        if path.exists():
            resumed += 1
            continue
        tasks.append(
            {
                "cfg": cfg, "outer_index": index, "dataset_seed": dataset_seed,
                "bootstrap_seed": bootstrap_seed, "draws": draws,
                "evaluation_states": evaluation_states, "evaluation_actions": evaluation_actions,
            }
        )
    started = time.perf_counter()
    complete = resumed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_m4_outer, task): task for task in tasks}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                index = int(result["outer_index"])
                _atomic_npz(
                    _m4_path(index),
                    outer_index=np.asarray(index), dataset_seed=result["dataset_seed"],
                    bootstrap_seed=result["bootstrap_seed"], observed_joint=result["observed_joint"],
                    observed_layers=result["observed_layers"], joint=result["joint"],
                    layers=result["layers"], valid=result["valid"],
                    diagnostics=np.asarray(result["diagnostics"]), seconds=np.asarray(result["seconds"]),
                )
                complete += 1
                print(f"Phase2R-G M4 outer: {complete}/{len(dataset_seeds)}", flush=True)
    payload = {
        "stage": "m4", "expected_outer_replicates": len(dataset_seeds),
        "completed_outer_replicates": complete, "resumed_outer_replicates": resumed,
        "new_outer_replicates": complete - resumed, "completed_seed_reruns": 0,
        "inner_draws_per_outer": draws,
        "completed_inner_draws": complete * draws,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "m4_run_status.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["oracle-primary", "oracle-reference", "m4"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "m4":
        result = run_m4_outer(args.workers)
    else:
        result = run_oracle(args.stage.removeprefix("oracle-"), args.workers)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
