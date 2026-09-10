"""Freeze and verify the Phase 2R-A forensic protocol and fixed state grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2.grids import evaluation_grid
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.protocol import load_phase2_config


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG_PATH = WORKSPACE / "configs" / "rs_cusum_phase2r_forensics.yaml"
PROTOCOL_PATH = WORKSPACE / "report" / "rs_cusum_phase2r_failure_localization_protocol.md"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2r"
HASH_PATH = OUTPUT / "pre_run_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_raw_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    protocol = cfg["protocol"]
    if protocol["diagnostic_label"] != "phase2r-a-forensics":
        raise RuntimeError("unexpected Phase 2R diagnostic label")
    if protocol["parent_method"] != "1.0.1-phase1":
        raise RuntimeError("parent method must remain 1.0.1-phase1")
    if protocol["formal_ohiot1dm_gate"] != "CLOSED":
        raise RuntimeError("formal OhioT1DM gate must remain CLOSED")
    if protocol["alternatives_allowed"] or protocol["method_revision_allowed"]:
        raise RuntimeError("Phase 2R-A may not run alternatives or revise the method")
    return cfg


def load_config(require_frozen: bool = True) -> dict[str, Any]:
    cfg = load_raw_config()
    if require_frozen:
        if not HASH_PATH.is_file():
            raise RuntimeError("Phase 2R-A pre-run hash record is absent")
        frozen = json.loads(HASH_PATH.read_text(encoding="utf-8"))
        if not frozen.get("frozen_before_any_phase2r_multi_repetition_diagnostic"):
            raise RuntimeError("Phase 2R-A hash record is not marked pre-run")
        mismatches = []
        for relative, expected in frozen["files"].items():
            path = WORKSPACE / relative
            if not path.is_file() or sha256(path) != expected:
                mismatches.append(relative)
        if mismatches:
            raise RuntimeError(f"Phase 2R-A frozen hash mismatch: {mismatches}")
    return cfg


def build_fixed_grid() -> Path:
    """Create the fixed state-only evaluation grid before multi-repetition work."""
    cfg = load_raw_config()
    parent = load_phase2_config()
    seed = int(cfg["fixed_point"]["evaluation_reference_dataset_seed"])
    dataset = generate_dataset("N0_complete_balanced_null", seed, parent, "none")
    panel = build_panel(dataset, "original_complete")
    states, actions, sources = evaluation_grid(
        panel,
        dataset.analysis_start,
        dataset.analysis_end,
        int(cfg["fixed_point"]["primary_reference_states"]),
        int(cfg["fixed_point"]["evaluation_grid_seed"]),
    )
    if int(actions[0]) != int(cfg["fixed_point"]["single_point_action"]):
        raise RuntimeError("fixed single-point action is inconsistent with the grid")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = WORKSPACE / cfg["fixed_point"]["fixed_grid_file"]
    source_patient = np.asarray([value[0] for value in sources], dtype="U32")
    source_elapsed = np.asarray([value[1] for value in sources], dtype=np.int64)
    np.savez_compressed(
        target,
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=np.int64),
        source_patient=source_patient,
        source_elapsed=source_elapsed,
    )
    return target


def load_fixed_grid() -> tuple[np.ndarray, np.ndarray]:
    cfg = load_config()
    path = WORKSPACE / cfg["fixed_point"]["fixed_grid_file"]
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["states"], dtype=float), np.asarray(data["actions"], dtype=int)


def freeze_pre_run_hashes() -> dict[str, Any]:
    """Hash protocol, parents, implementation, tests, and the fixed grid exactly once."""
    if HASH_PATH.exists():
        raise RuntimeError("pre-run hash record already exists; refusing to overwrite")
    grid = build_fixed_grid()
    fixed_paths = [
        CONFIG_PATH,
        PROTOCOL_PATH,
        WORKSPACE / "configs" / "rs_cusum_main.yaml",
        WORKSPACE / "configs" / "rs_cusum_phase2_pilot.yaml",
        WORKSPACE / "report" / "rs_cusum_phase1_method_spec.md",
        WORKSPACE / "results_rs_cusum" / "phase1" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2" / "phase2_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2" / "failure_forensics.json",
        grid,
    ]
    fixed_paths.extend(sorted((WORKSPACE / "src" / "rs_cusum_phase2r").glob("*.py")))
    fixed_paths.extend(sorted((WORKSPACE / "tests").glob("test_phase2r_*.py")))
    relative = lambda path: str(path.relative_to(WORKSPACE)).replace("\\", "/")
    payload = {
        "diagnostic_label": "phase2r-a-forensics",
        "parent_method": "1.0.1-phase1",
        "formal_ohiot1dm_gate": "CLOSED",
        "frozen_before_any_phase2r_multi_repetition_diagnostic": True,
        "hash_algorithm": "SHA-256",
        "files": {relative(path): sha256(path) for path in fixed_paths},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    HASH_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


if __name__ == "__main__":
    record = freeze_pre_run_hashes()
    print(f"Frozen Phase 2R-A files: {len(record['files'])}")

