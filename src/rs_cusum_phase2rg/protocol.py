"""Frozen protocol, seed registry, and hash controls for Phase 2R-G."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = WORKSPACE / "configs" / "rs_cusum_phase2rg.yaml"
PROTOCOL_MD = WORKSPACE / "report" / "phase2rg_protocol.md"
PROTOCOL_JSON = WORKSPACE / "report" / "phase2rg_protocol.json"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2rg"
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_raw_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protocol = cfg["protocol"]
    if protocol["label"] != "phase2rg-fresh-locked-n0-confirmation":
        raise RuntimeError("unexpected Phase 2R-G identity")
    expected = {
        "phase2rd_gate_immutable": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "phase2re_gate_immutable": "TAIL_METRIC_UNSTABLE",
        "phase2rf_gate_immutable": "M4_JOINT_TAIL_CONFIRMED",
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise RuntimeError(f"historical gate changed: {key}")
    forbidden = (
        "m4_modification", "posthoc_calibration", "mechanism_exploration",
        "ohiot1dm_allowed", "phase3_allowed", "transfer_null_allowed",
        "alternatives_allowed", "power_allowed", "changepoint_allowed",
        "establish_1_0_2",
    )
    if any(bool(protocol[key]) for key in forbidden):
        raise RuntimeError("a prohibited Phase 2R-G operation is enabled")
    if not protocol["one_time_holdout"]:
        raise RuntimeError("Phase 2R-G must be a one-time holdout")
    if cfg["primary_metric"]["thresholds"] != [0.90, 0.95, 0.975, 0.99]:
        raise RuntimeError("primary thresholds changed")
    if float(cfg["primary_metric"]["equivalence_margin"]) != 0.015:
        raise RuntimeError("primary equivalence margin changed")
    m4 = WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py"
    metric = WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py"
    if sha256(m4) != cfg["frozen_method"]["m4_engine_sha256"]:
        raise RuntimeError("M4 source hash differs from Phase 2R-F")
    if sha256(metric) != cfg["frozen_method"]["phase2rf_metric_sha256"]:
        raise RuntimeError("Phase 2R-F stable metric source hash changed")
    return cfg


def _collect_json_seed_values(value: Any, key_path: tuple[str, ...] = ()) -> Iterable[int]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _collect_json_seed_values(child, (*key_path, str(key).lower()))
    elif isinstance(value, list):
        for child in value:
            yield from _collect_json_seed_values(child, key_path)
    elif isinstance(value, int) and any("seed" in key for key in key_path):
        yield int(value)


def prior_seed_audit() -> tuple[set[int], dict[str, int]]:
    values: set[int] = set()
    sources: dict[str, int] = {}
    root = WORKSPACE / "results_rs_cusum"
    for path in sorted(root.glob("**/*seed*.npz")):
        if OUTPUT.resolve() in path.resolve().parents:
            continue
        found: set[int] = set()
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    array = np.asarray(data[key])
                    if np.issubdtype(array.dtype, np.integer):
                        found.update(map(int, array.ravel()))
        except (OSError, ValueError):
            continue
        values.update(found)
        sources[str(path.relative_to(WORKSPACE)).replace("\\", "/")] = len(found)
    for path in sorted(root.glob("**/*seed*.json")):
        if OUTPUT.resolve() in path.resolve().parents:
            continue
        try:
            found = set(_collect_json_seed_values(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        values.update(found)
        sources[str(path.relative_to(WORKSPACE)).replace("\\", "/")] = len(found)
    for path in sorted(root.glob("**/*replicate*.csv")):
        found: set[int] = set()
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key, item in row.items():
                        if "seed" in key.lower() and item not in (None, ""):
                            found.add(int(item))
        except (OSError, ValueError):
            continue
        if found:
            values.update(found)
            sources[str(path.relative_to(WORKSPACE)).replace("\\", "/")] = len(found)
    return values, sources


def create_seed_registry() -> dict[str, Any]:
    if SEEDS.exists() or SEED_META.exists():
        raise RuntimeError("Phase 2R-G seed registry already exists")
    cfg = load_raw_config()
    master = int(cfg["seeds"]["master_seed"])
    oracle_n = int(cfg["sample_plan"]["oracle_processes_per_bank"])
    outer_n = int(cfg["sample_plan"]["m4_outer_replicates"])
    inner_n = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])
    arrays: dict[str, np.ndarray] = {
        "master_seed": np.asarray([master], dtype=np.uint64),
        "engineering_oracle_process": np.asarray(
            [stable_seed(master, "engineering_oracle", 0)], dtype=np.uint64
        ),
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
        "uncertainty_bootstrap": np.asarray(
            [stable_seed(master, "uncertainty_bootstrap", index)
             for index in range(int(cfg["uncertainty"]["bootstrap_replicates"]))],
            dtype=np.uint64,
        ),
    }
    bootstrap = arrays["m4_outer_bootstrap"]
    for label in ("design", "reward", "dataset"):
        arrays[f"m4_{label}_process"] = np.asarray(
            [[stable_seed(int(bootstrap[outer]), "M4", label, draw)
              for draw in range(inner_n)] for outer in range(outer_n)],
            dtype=np.uint64,
        )
    flat = np.concatenate([array.ravel() for array in arrays.values()])
    unique_count = len(np.unique(flat))
    if unique_count != len(flat):
        raise RuntimeError("within-Phase 2R-G seed collision")
    prior, prior_sources = prior_seed_audit()
    overlap = set(map(int, flat)) & prior
    if overlap:
        raise RuntimeError(f"Phase 2R-G historical seed overlap: {sorted(overlap)[:10]}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = SEEDS.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(SEEDS)
    payload = {
        "master_seed": master,
        "algorithm": "SHA-256 first 64 bits",
        "all_values_unique": True,
        "within_phase_total_values": int(len(flat)),
        "within_phase_unique_values": int(unique_count),
        "historical_values_scanned": int(len(prior)),
        "historical_seed_sources": prior_sources,
        "historical_collision_count": 0,
        "historical_collisions": [],
        "disjoint_from_all_prior_registries": True,
        "formal_holdout_draws_generated_before_registry": False,
        "seed_registry_npz_sha256": sha256(SEEDS),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "count": int(value.size),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest().upper(),
            }
            for key, value in arrays.items()
        },
    }
    _atomic_json(SEED_META, payload)
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
    _atomic_json(path, payload)
    return payload


def verify_hash(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for relative, expected in payload["files"].items():
        target = WORKSPACE / relative
        if not target.is_file() or sha256(target) != expected:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Phase 2R-G hash mismatch: {bad}")


def freeze_protocol() -> dict[str, Any]:
    cfg = load_raw_config()
    create_seed_registry()
    paths = [
        CONFIG, PROTOCOL_MD, PROTOCOL_JSON, OUTPUT / "sample_size_precision_plan.csv",
        SEEDS, SEED_META, OUTPUT / "pre_freeze_engineering_log.json",
        OUTPUT / "seed_registry.pre_freeze_invalid.npz",
        OUTPUT / "seed_registry.pre_freeze_invalid.json",
        WORKSPACE / "results_rs_cusum" / "phase2rd" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2re" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2rf" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2rd" / "phase2rd_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2re" / "phase2re_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2rf" / "phase2rf_gate.json",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "runner.py",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "summarize.py",
        WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py",
        WORKSPACE / "tests" / "test_phase2rd_m4.py",
        WORKSPACE / "tests" / "test_phase2rf_confirmation.py",
    ]
    return _write_hash(
        PROTOCOL_HASH,
        "PHASE2RG_PROTOCOL_SEED_GATE_FREEZE",
        paths,
        {
            "frozen_before_any_formal_Phase2R_G_holdout_draw": True,
            "m4_modified": False,
            "metric_modified": False,
            "posthoc_tuning": False,
            "phase2rd_gate_unchanged": cfg["protocol"]["phase2rd_gate_immutable"],
            "phase2re_gate_unchanged": cfg["protocol"]["phase2re_gate_immutable"],
            "phase2rf_gate_unchanged": cfg["protocol"]["phase2rf_gate_immutable"],
        },
    )


def freeze_pre_run() -> dict[str, Any]:
    verify_hash(PROTOCOL_HASH)
    checkpoints = OUTPUT / "checkpoints"
    if checkpoints.exists() and any(checkpoints.rglob("*.npz")):
        raise RuntimeError("formal Phase 2R-G checkpoints already exist before pre-run freeze")
    sources = sorted((WORKSPACE / "src" / "rs_cusum_phase2rg").glob("*.py"))
    tests = sorted((WORKSPACE / "tests").glob("test_phase2rg*.py"))
    return _write_hash(
        PRE_RUN_HASH,
        "PHASE2RG_PRE_ENGINEERING_AND_FORMAL_RUN_FREEZE",
        [PROTOCOL_HASH, *sources, *tests],
        {
            "implementation_frozen_before_engineering_tests": True,
            "formal_draws_exist": False,
            "one_time_holdout_not_started": True,
        },
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
