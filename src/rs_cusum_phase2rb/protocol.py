"""Freeze and verify Phase 2R-B protocol, holdout seeds, code, and parents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = WORKSPACE / "configs" / "rs_cusum_phase2rb.yaml"
PROTOCOL = WORKSPACE / "report" / "rs_cusum_phase2rb_inference_revision_protocol.md"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2rb"
HASH_FILE = OUTPUT / "pre_run_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def load_raw_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    p = cfg["protocol"]
    if p["label"] != "phase2rb-confirmatory" or p["parent_method"] != "1.0.1-phase1":
        raise RuntimeError("unexpected Phase 2R-B identity")
    if p["phase3_gate"] != "CLOSED" or p["formal_ohiot1dm_gate"] != "CLOSED":
        raise RuntimeError("Phase 3 and OhioT1DM gates must remain closed")
    if p["alternatives_allowed"] or p["power_allowed"] or p["changepoint_performance_allowed"]:
        raise RuntimeError("Phase 2R-B is null-only")
    return cfg


def load_config(require_frozen: bool = True) -> dict[str, Any]:
    cfg = load_raw_config()
    if require_frozen:
        frozen = json.loads(HASH_FILE.read_text(encoding="utf-8"))
        mismatches = []
        for relative, expected in frozen["files"].items():
            path = WORKSPACE / relative
            if not path.is_file() or sha256(path) != expected:
                mismatches.append(relative)
        if mismatches:
            raise RuntimeError(f"Phase 2R-B frozen hash mismatch: {mismatches}")
    return cfg


def create_seed_lists(cfg: dict[str, Any]) -> Path:
    master = int(cfg["fresh_seeds"]["master_seed"])
    arrays: dict[str, np.ndarray] = {}
    for stage in ("RB0", "RB1", "RB2"):
        count = int(cfg["stages"][stage]["replicates"])
        arrays[stage.lower()] = np.asarray(
            [stable_seed(master, stage, "N0_complete_balanced_null", index) for index in range(count)],
            dtype=np.uint32,
        )
    for scenario in cfg["stages"]["transfer_null"]["scenarios"]:
        count = int(cfg["stages"]["transfer_null"]["replicates"])
        arrays[f"transfer__{scenario}"] = np.asarray(
            [stable_seed(master, "transfer", scenario, index) for index in range(count)],
            dtype=np.uint32,
        )
    concatenated = np.concatenate(list(arrays.values()))
    if len(np.unique(concatenated)) != len(concatenated):
        raise RuntimeError("Phase 2R-B seed lists overlap each other")
    # Explicitly exclude every seed saved in Phase 2R-A replicate tables.
    prior = set()
    for filename in (
        "truth_replicates.csv", "decomposition_replicates.csv", "scaling_replicates.csv",
        "full_refit_replicates.csv",
    ):
        path = WORKSPACE / "results_rs_cusum" / "phase2r" / filename
        if path.is_file():
            import csv
            with path.open("r", encoding="utf-8", newline="") as handle:
                prior.update(int(row["seed"]) for row in csv.DictReader(handle))
    overlap = sorted(set(map(int, concatenated)) & prior)
    if overlap:
        raise RuntimeError(f"fresh Phase 2R-B seeds overlap Phase 2R-A: {overlap[:5]}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = WORKSPACE / cfg["fresh_seeds"]["seed_file"]
    np.savez_compressed(target, **arrays)
    return target


def load_seed_list(name: str) -> np.ndarray:
    cfg = load_config()
    with np.load(WORKSPACE / cfg["fresh_seeds"]["seed_file"], allow_pickle=False) as data:
        return np.asarray(data[name], dtype=np.uint32)


def freeze_pre_run() -> dict[str, Any]:
    if HASH_FILE.exists():
        raise RuntimeError("pre-run hashes already exist; refusing overwrite")
    cfg = load_raw_config()
    seeds = create_seed_lists(cfg)
    paths = [
        CONFIG, PROTOCOL, seeds,
        WORKSPACE / "configs" / "rs_cusum_main.yaml",
        WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml",
        WORKSPACE / "configs" / "rs_cusum_phase2_pilot.yaml",
        WORKSPACE / "results_rs_cusum" / "phase2r" / "fixed_evaluation_grid.npz",
        WORKSPACE / "results_rs_cusum" / "phase2r" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2r" / "phase2r_gate.json",
        WORKSPACE / "report" / "rs_cusum_phase2r_failure_localization_report.md",
    ]
    paths.extend(sorted((WORKSPACE / "src" / "rs_cusum_phase2rb").glob("*.py")))
    paths.extend(sorted((WORKSPACE / "tests").glob("test_phase2rb_*.py")))
    rel = lambda p: str(p.relative_to(WORKSPACE)).replace("\\", "/")
    payload = {
        "label": "phase2rb-confirmatory", "parent_method": "1.0.1-phase1",
        "frozen_before_any_phase2rb_multi_repetition_run": True,
        "rb2_seed_list_frozen_before_rb1": True, "hash_algorithm": "SHA-256",
        "files": {rel(path): sha256(path) for path in paths},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = freeze_pre_run()
    print(f"Frozen Phase 2R-B files: {len(result['files'])}")

