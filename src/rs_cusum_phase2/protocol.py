"""Load the pre-run protocol and verify every frozen parent hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


WORKSPACE = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_phase2_config(
    config_path: str | Path = "configs/rs_cusum_phase2_pilot.yaml",
    hash_path: str | Path = "results_rs_cusum/phase2/pre_run_config_hashes.json",
) -> dict[str, Any]:
    config_file = _resolve(config_path)
    hash_file = _resolve(hash_path)
    with config_file.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["protocol"]["status"] != "phase2_pilot_authorized":
        raise RuntimeError("Phase 2 pilot is not authorized")
    if cfg["protocol"]["formal_ohiot1dm_gate"] != "CLOSED":
        raise RuntimeError("formal OhioT1DM gate must remain CLOSED")
    with hash_file.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if not frozen["frozen_before_any_phase2_multi_repetition_simulation"]:
        raise RuntimeError("pre-run hash record is not frozen")
    mismatches: list[str] = []
    for relative_path, expected in frozen["files"].items():
        path = WORKSPACE / relative_path
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(relative_path)
    if mismatches:
        raise RuntimeError(f"frozen protocol/parent hash mismatch: {mismatches}")
    return cfg


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else WORKSPACE / path
