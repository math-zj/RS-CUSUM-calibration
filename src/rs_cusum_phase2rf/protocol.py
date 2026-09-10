"""Frozen configuration, seeds, and SHA-256 controls for Phase 2R-F."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = WORKSPACE / "configs" / "rs_cusum_phase2rf.yaml"
PROTOCOL_MD = WORKSPACE / "report" / "phase2rf_protocol.md"
PROTOCOL_JSON = WORKSPACE / "report" / "phase2rf_protocol.json"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2rf"
SEEDS = OUTPUT / "seed_registry.npz"
SEED_META = OUTPUT / "seed_registry.json"
PROTOCOL_HASH = OUTPUT / "hash_registry_protocol.json"
PRE_RUN_HASH = OUTPUT / "hash_registry_pre_run.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_raw_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protocol = cfg["protocol"]
    if protocol["label"] != "phase2rf-independent-stable-tail-confirmation":
        raise RuntimeError("unexpected Phase 2R-F identity")
    if protocol["phase2rd_gate_immutable"] != "M4_MECHANISM_JOINT_TAIL_FAIL":
        raise RuntimeError("Phase 2R-D gate was not preserved")
    if protocol["phase2re_gate_immutable"] != "TAIL_METRIC_UNSTABLE":
        raise RuntimeError("Phase 2R-E gate was not preserved")
    forbidden = (
        "m4_modification", "posthoc_calibration", "d3_allowed", "ohiot1dm_allowed",
        "phase3_allowed", "transfer_null_allowed", "alternatives_allowed",
        "power_allowed", "changepoint_allowed", "establish_1_0_2",
    )
    if any(bool(protocol[key]) for key in forbidden):
        raise RuntimeError("a prohibited Phase 2R-F operation is enabled")
    if cfg["primary_metric"]["thresholds"] != [0.90, 0.95, 0.975, 0.99]:
        raise RuntimeError("primary thresholds changed")
    if float(cfg["primary_metric"]["equivalence_margin"]) != 0.015:
        raise RuntimeError("primary margin changed")
    return cfg


def _prior_seed_values() -> set[int]:
    values: set[int] = set()
    for path in (WORKSPACE / "results_rs_cusum").glob("**/*seed*.npz"):
        if path.resolve() == SEEDS.resolve():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    array = np.asarray(data[key])
                    if np.issubdtype(array.dtype, np.integer):
                        values.update(map(int, array.ravel()))
        except (OSError, ValueError):
            continue
    for path in (WORKSPACE / "results_rs_cusum").glob("**/*replicate*.csv"):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("seed", "dataset_seed", "bootstrap_seed"):
                        if row.get(key) not in (None, ""):
                            values.add(int(row[key]))
        except (OSError, ValueError):
            continue
    return values


def create_seed_registry() -> dict[str, Any]:
    if SEEDS.exists() or SEED_META.exists():
        raise RuntimeError("Phase 2R-F seed registry already exists")
    cfg = load_raw_config()
    master = int(cfg["seeds"]["master_seed"])
    oracle_n = int(cfg["sample_plan"]["oracle_processes_per_bank"])
    outer_n = int(cfg["sample_plan"]["m4_outer_replicates"])
    inner_n = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])
    arrays: dict[str, np.ndarray] = {
        "planning_seed": np.asarray([cfg["sample_plan"]["planning_seed"]], dtype=np.uint64),
        "oracle_primary_process": np.asarray(
            [stable_seed(master, "oracle_primary", index) for index in range(oracle_n)], dtype=np.uint64
        ),
        "oracle_reference_process": np.asarray(
            [stable_seed(master, "oracle_reference", index) for index in range(oracle_n)], dtype=np.uint64
        ),
        "m4_outer_dataset": np.asarray(
            [stable_seed(master, "m4_outer_dataset", index) for index in range(outer_n)], dtype=np.uint64
        ),
        "m4_outer_bootstrap": np.asarray(
            [stable_seed(master, "m4_outer_bootstrap", index) for index in range(outer_n)], dtype=np.uint64
        ),
        "balanced_analysis": np.asarray(
            [stable_seed(master, "balanced_analysis", index) for index in range(int(cfg["balanced_comparison"]["repetitions"]))],
            dtype=np.uint64,
        ),
        "uncertainty_bootstrap": np.asarray(
            [stable_seed(master, "uncertainty_bootstrap", index) for index in range(int(cfg["uncertainty"]["bootstrap_replicates"]))],
            dtype=np.uint64,
        ),
    }
    bootstrap = arrays["m4_outer_bootstrap"]
    for label in ("design", "reward", "dataset"):
        arrays[f"m4_{label}_process"] = np.asarray(
            [[stable_seed(int(bootstrap[outer]), "M4", label, draw) for draw in range(inner_n)] for outer in range(outer_n)],
            dtype=np.uint64,
        )
    flat = np.concatenate([array.ravel() for array in arrays.values()])
    if len(np.unique(flat)) != len(flat):
        raise RuntimeError("within-Phase 2R-F seed collision")
    overlap = set(map(int, flat)) & _prior_seed_values()
    if overlap:
        raise RuntimeError(f"Phase 2R-F seed overlap: {sorted(overlap)[:5]}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = SEEDS.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(SEEDS)
    payload = {
        "master_seed": master,
        "algorithm": "SHA-256 first 64 bits",
        "all_values_unique": True,
        "disjoint_from_all_prior_registries": True,
        "formal_validation_draws_generated_before_registry": False,
        "planning_seed_used_only_on_saved_Phase2R_E_draws": True,
        "arrays": {
            key: {
                "shape": list(value.shape),
                "count": int(value.size),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest().upper(),
            }
            for key, value in arrays.items()
        },
    }
    SEED_META.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_seeds(name: str) -> np.ndarray:
    load_config("run")
    with np.load(SEEDS, allow_pickle=False) as data:
        return np.asarray(data[name]).copy()


def _relative(path: Path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


def _write_hash(path: Path, stage: str, paths: list[Path], extra: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"{path.name} already exists")
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in paths:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    missing = [str(item) for item in unique if not item.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "stage": stage,
        "hash_algorithm": "SHA-256",
        "files": {_relative(item): sha256(item) for item in unique},
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_hash(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for relative, expected in payload["files"].items():
        target = WORKSPACE / relative
        if not target.is_file() or sha256(target) != expected:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Phase 2R-F hash mismatch: {bad}")


def freeze_protocol() -> dict[str, Any]:
    cfg = load_raw_config()
    create_seed_registry()
    paths = [
        CONFIG, PROTOCOL_MD, PROTOCOL_JSON,
        OUTPUT / "sample_size_precision_plan.csv", SEEDS, SEED_META,
        WORKSPACE / "results_rs_cusum" / "phase2rd" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2re" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2rd" / "phase2rd_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2re" / "phase2re_gate.json",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "runner.py",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "summarize.py",
        WORKSPACE / "tests" / "test_phase2rd_m4.py",
    ]
    return _write_hash(
        PROTOCOL_HASH,
        "PHASE2RF_PROTOCOL_SEED_GATE_FREEZE",
        paths,
        {
            "frozen_before_any_formal_Phase2R_F_validation_draw": True,
            "m4_modified": False,
            "phase2rd_gate_unchanged": cfg["protocol"]["phase2rd_gate_immutable"],
            "phase2re_gate_unchanged": cfg["protocol"]["phase2re_gate_immutable"],
        },
    )


def freeze_pre_run() -> dict[str, Any]:
    verify_hash(PROTOCOL_HASH)
    sources = sorted((WORKSPACE / "src" / "rs_cusum_phase2rf").glob("*.py"))
    tests = sorted((WORKSPACE / "tests").glob("test_phase2rf*.py"))
    return _write_hash(
        PRE_RUN_HASH,
        "PHASE2RF_PRE_ENGINEERING_AND_FORMAL_RUN_FREEZE",
        [PROTOCOL_HASH, *sources, *tests],
        {"implementation_frozen_before_engineering_tests": True, "formal_draws_exist": False},
    )


def load_config(stage: str = "raw") -> dict[str, Any]:
    cfg = load_raw_config()
    if stage != "raw":
        verify_hash(PROTOCOL_HASH)
    if stage == "run":
        verify_hash(PRE_RUN_HASH)
    return cfg


if __name__ == "__main__":
    print(json.dumps(freeze_protocol(), sort_keys=True))
