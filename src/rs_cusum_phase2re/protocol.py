"""Frozen protocol, seeds, and traceability for Phase 2R-E."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = WORKSPACE / "configs" / "rs_cusum_phase2re.yaml"
PROTOCOL_MD = WORKSPACE / "report" / "phase2re_protocol.md"
PROTOCOL_JSON = WORKSPACE / "report" / "phase2re_protocol.json"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2re"
SEEDS = OUTPUT / "seed_registry.npz"
SEED_META = OUTPUT / "seed_registry.json"
PROTOCOL_HASH = OUTPUT / "hash_registry_protocol.json"
OLD_PRE_HYBRID_HASH = OUTPUT / "hash_registry_pre_hybrid.json"
PRE_HYBRID_HASH = OUTPUT / "hash_registry_pre_hybrid_v2.json"
BUG_LOG = OUTPUT / "engineering_bug_log.json"


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
    if protocol["label"] != "phase2re-joint-tail-localization":
        raise RuntimeError("unexpected Phase 2R-E identity")
    if protocol["parent_gate"] != "M4_MECHANISM_JOINT_TAIL_FAIL":
        raise RuntimeError("Phase 2R-D failure gate was changed")
    forbidden = (
        "method_optimization", "m4_modification", "new_method_allowed", "d3_allowed",
        "ohiot1dm_allowed", "phase3_allowed", "transfer_null_allowed",
        "alternatives_allowed", "power_allowed", "changepoint_allowed", "establish_1_0_2",
    )
    if any(bool(protocol[key]) for key in forbidden):
        raise RuntimeError("a prohibited Phase 2R-E operation was enabled")
    parent_gate = json.loads(
        (WORKSPACE / "results_rs_cusum" / "phase2rd" / "phase2rd_gate.json").read_text(encoding="utf-8")
    )
    if parent_gate["gate"] != "M4_MECHANISM_JOINT_TAIL_FAIL" or parent_gate["D3_run"]:
        raise RuntimeError("Phase 2R-D terminal state is inconsistent")
    return cfg


def _prior_used_seeds() -> set[int]:
    """Read prior used/development seeds, explicitly never Phase 2R-D D3 arrays."""
    prior: set[int] = set()
    root = WORKSPACE / "results_rs_cusum"
    for path in root.glob("**/*seed*.npz"):
        if path.resolve() == SEEDS.resolve():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                for name in data.files:
                    if "phase2rd" in str(path).lower() and name.lower().startswith("d3_"):
                        continue
                    array = np.asarray(data[name])
                    if np.issubdtype(array.dtype, np.integer):
                        prior.update(map(int, array.ravel()))
        except (OSError, ValueError):
            continue
    for path in root.glob("**/*replicate*.csv"):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("seed", "dataset_seed", "bootstrap_seed"):
                        if row.get(key) not in (None, ""):
                            prior.add(int(row[key]))
        except (OSError, ValueError):
            continue
    return prior


def create_seed_registry() -> dict[str, Any]:
    if SEEDS.exists() or SEED_META.exists():
        raise RuntimeError("Phase 2R-E seed registry already exists")
    cfg = load_raw_config()
    master = int(cfg["seeds"]["master_seed"])
    outer = int(cfg["hybrid"]["outer_replicates"])
    mc = int(cfg["mc_uncertainty"]["bootstrap_replicates"])
    subsample = int(cfg["mc_uncertainty"]["subsample_repetitions"])
    arrays = {
        "hybrid_outer_dataset": np.asarray(
            [stable_seed(master, "hybrid_outer_dataset", index) for index in range(outer)],
            dtype=np.uint64,
        ),
        "hybrid_common_draw": np.asarray(
            [stable_seed(master, "hybrid_common_draw", index) for index in range(outer)],
            dtype=np.uint64,
        ),
        "mc_pair_sample": np.asarray([stable_seed(master, "mc_pair_sample")], dtype=np.uint64),
        "mc_bootstrap": np.asarray(
            [stable_seed(master, "mc_bootstrap", index) for index in range(mc)],
            dtype=np.uint64,
        ),
        "subsample": np.asarray(
            [stable_seed(master, "subsample", index) for index in range(subsample)],
            dtype=np.uint64,
        ),
    }
    flat = np.concatenate([value.ravel() for value in arrays.values()])
    if len(flat) != len(np.unique(flat)):
        raise RuntimeError("within-Phase 2R-E seed collision")
    overlap = set(map(int, flat)) & _prior_used_seeds()
    if overlap:
        raise RuntimeError(f"Phase 2R-E seed overlap: {sorted(overlap)[:5]}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SEEDS, **arrays)
    payload = {
        "master_seed": master,
        "algorithm": "SHA-256 first 64 bits",
        "all_values_unique": True,
        "disjoint_from_prior_used_and_development_seeds": True,
        "phase2rd_D3_arrays_read": False,
        "diagnostic_only": True,
        "arrays": {
            key: {"count": int(value.size), "sha256": hashlib.sha256(value.tobytes()).hexdigest().upper()}
            for key, value in arrays.items()
        },
    }
    SEED_META.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _relative(path: Path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


def _write_hash(path: Path, stage: str, paths: list[Path], extra: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"{path.name} already exists")
    unique: list[Path] = []
    for item in paths:
        item = item.resolve()
        if item not in unique:
            unique.append(item)
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


def freeze_protocol() -> dict[str, Any]:
    create_seed_registry()
    parent = WORKSPACE / "results_rs_cusum" / "phase2rd"
    return _write_hash(
        PROTOCOL_HASH,
        "PHASE2RE_PROTOCOL_FREEZE",
        [
            CONFIG, PROTOCOL_MD, PROTOCOL_JSON, SEEDS, SEED_META,
            parent / "manifest.json", parent / "phase2rd_gate.json",
            parent / "d2_gate.json", parent / "d2_process_forensics.npz",
            WORKSPACE / "results_rs_cusum" / "phase2r" / "fixed_evaluation_grid.npz",
        ],
        {
            "frozen_before_new_hybrid_draw": True,
            "phase2rd_files_modified": False,
            "phase2rd_gate_preserved": "M4_MECHANISM_JOINT_TAIL_FAIL",
        },
    )


def verify_hash(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for relative, expected in payload["files"].items():
        target = WORKSPACE / relative
        if not target.is_file() or sha256(target) != expected:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Phase 2R-E hash mismatch: {bad}")


def freeze_pre_hybrid() -> dict[str, Any]:
    verify_hash(PROTOCOL_HASH)
    sources = sorted((WORKSPACE / "src" / "rs_cusum_phase2re").glob("*.py"))
    tests = sorted((WORKSPACE / "tests").glob("test_phase2re*.py"))
    return _write_hash(
        PRE_HYBRID_HASH,
        "PHASE2RE_PRE_HYBRID_FREEZE_V2",
        [PROTOCOL_HASH, OLD_PRE_HYBRID_HASH, BUG_LOG, *sources, *tests],
        {"implementation_revision": "engineering-r1", "supersedes": "hash_registry_pre_hybrid.json",
         "implementation_frozen_before_existing-draw_outputs_and_hybrid": True},
    )


def load_config(require_implementation: bool = True) -> dict[str, Any]:
    cfg = load_raw_config()
    verify_hash(PROTOCOL_HASH)
    if require_implementation:
        verify_hash(PRE_HYBRID_HASH)
    return cfg


def load_seeds(name: str) -> np.ndarray:
    load_config()
    with np.load(SEEDS, allow_pickle=False) as data:
        return np.asarray(data[name])


if __name__ == "__main__":
    print(json.dumps({"files": len(freeze_protocol()["files"])}, sort_keys=True))
