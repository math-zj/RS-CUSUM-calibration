"""Frozen protocol, seed registry, and hash controls for Phase 2R-H."""

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
CONFIG = WORKSPACE / "configs" / "rs_cusum_phase2rh.yaml"
PROTOCOL_MD = WORKSPACE / "report" / "phase2rh_protocol.md"
PROTOCOL_JSON = WORKSPACE / "report" / "phase2rh_protocol.json"
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2rh"
SCENARIO_CSV = OUTPUT / "scenario_registry.csv"
SCENARIO_JSON = OUTPUT / "scenario_registry.json"
PRECISION_PLAN = OUTPUT / "sample_size_precision_plan.csv"
HISTORICAL_AUDIT = OUTPUT / "historical_scenario_audit.csv"
SEEDS = OUTPUT / "seed_registry.npz"
SEED_META = OUTPUT / "seed_registry.json"
PROTOCOL_HASH = OUTPUT / "hash_registry_protocol.json"
PRE_RUN_HASH = OUTPUT / "hash_registry_pre_run.json"
TEST_HASH = OUTPUT / "hash_registry_tests.json"
DGP_COMPONENTS = ("initial", "heterogeneity", "action", "transition", "reward")


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


def scenario_ids() -> tuple[str, ...]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return tuple(cfg["scenario_plan"]["order"])


def load_raw_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protocol = cfg["protocol"]
    if protocol["label"] != "phase2rh-transfer-null":
        raise RuntimeError("unexpected Phase 2R-H identity")
    expected = {
        "phase2rd_gate_immutable": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "phase2re_gate_immutable": "TAIL_METRIC_UNSTABLE",
        "phase2rf_gate_immutable": "M4_JOINT_TAIL_CONFIRMED",
        "phase2rg_gate_immutable": "M4_FRESH_GLOBAL_NULL_PASS",
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise RuntimeError(f"historical gate changed: {key}")
    forbidden = (
        "m4_modification", "posthoc_tuning", "scenario_specific_patch",
        "alternatives_allowed", "power_allowed", "changepoint_allowed",
        "ohiot1dm_allowed", "phase3_allowed", "establish_1_0_2",
    )
    if any(bool(protocol[key]) for key in forbidden):
        raise RuntimeError("a prohibited Phase 2R-H operation is enabled")
    m4 = WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py"
    metric = WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py"
    if sha256(m4) != cfg["frozen_method"]["m4_engine_sha256"]:
        raise RuntimeError("M4 source hash changed")
    if sha256(metric) != cfg["frozen_method"]["phase2rf_metric_sha256"]:
        raise RuntimeError("Phase 2R-F metric source hash changed")
    with SCENARIO_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(row["scenario_id"] for row in rows) != tuple(cfg["scenario_plan"]["order"]):
        raise RuntimeError("scenario registry/order mismatch")
    if len(rows) != 9 or not all(row["stationary_no_changepoint"] == "true" for row in rows):
        raise RuntimeError("scenario registry is not the frozen nine-scenario null slate")
    if int(cfg["sample_plan"]["m4_inner_draws_per_outer"]) != 199:
        raise RuntimeError("frozen M4 draw count changed")
    if int(cfg["sample_plan"]["m4_outer_replicates_per_scenario"]) != 200:
        raise RuntimeError("outer count changed")
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
        if OUTPUT.resolve() in path.resolve().parents:
            continue
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


def _component_arrays(base: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_dgp_{component}": np.asarray(
            [stable_seed(int(value), "DGP", component) for value in base.ravel()], dtype=np.uint64
        ).reshape(base.shape)
        for component in DGP_COMPONENTS
    }


def create_seed_registry() -> dict[str, Any]:
    if SEEDS.exists() or SEED_META.exists():
        raise RuntimeError("Phase 2R-H seed registry already exists")
    cfg = load_raw_config()
    master = int(cfg["seeds"]["master_seed"])
    scenarios = tuple(cfg["scenario_plan"]["order"])
    s_count = len(scenarios)
    oracle_n = int(cfg["sample_plan"]["oracle_processes_per_bank_per_scenario"])
    outer_n = int(cfg["sample_plan"]["m4_outer_replicates_per_scenario"])
    inner_n = int(cfg["sample_plan"]["m4_inner_draws_per_outer"])

    def base_array(label: str, count: int) -> np.ndarray:
        return np.asarray(
            [[stable_seed(master, scenario, label, index) for index in range(count)] for scenario in scenarios],
            dtype=np.uint64,
        )

    arrays: dict[str, np.ndarray] = {"master_seed": np.asarray([master], dtype=np.uint64)}
    engineering_dataset = base_array("engineering_dataset", 1).reshape(s_count)
    engineering_bootstrap = base_array("engineering_bootstrap", 1).reshape(s_count)
    arrays["engineering_dataset"] = engineering_dataset
    arrays["engineering_bootstrap"] = engineering_bootstrap
    arrays.update(_component_arrays(engineering_dataset, "engineering"))
    for bank in ("primary", "reference"):
        base = base_array(f"oracle_{bank}_process", oracle_n)
        arrays[f"oracle_{bank}_process"] = base
        arrays.update(_component_arrays(base, f"oracle_{bank}"))
    outer = base_array("m4_outer_dataset", outer_n)
    bootstrap = base_array("m4_outer_bootstrap", outer_n)
    arrays["m4_outer_dataset"] = outer
    arrays["m4_outer_bootstrap"] = bootstrap
    arrays.update(_component_arrays(outer, "m4_outer"))
    for label in ("design", "reward", "dataset"):
        arrays[f"m4_{label}_process"] = np.asarray(
            [
                [
                    [stable_seed(int(bootstrap[s, outer_index]), "M4", label, draw)
                     for draw in range(inner_n)]
                    for outer_index in range(outer_n)
                ]
                for s in range(s_count)
            ],
            dtype=np.uint64,
        )
    flat = np.concatenate([array.ravel() for array in arrays.values()])
    unique_count = len(np.unique(flat))
    if unique_count != len(flat):
        raise RuntimeError("within-Phase 2R-H seed collision")
    prior, prior_sources = prior_seed_audit()
    overlap = set(map(int, flat)) & prior
    if overlap:
        raise RuntimeError(f"Phase 2R-H historical seed overlap: {sorted(overlap)[:10]}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = SEEDS.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(SEEDS)
    payload = {
        "master_seed": master,
        "algorithm": "SHA-256 first 64 bits",
        "scenario_order": list(scenarios),
        "all_values_unique": True,
        "within_phase_total_values": int(len(flat)),
        "within_phase_unique_values": int(unique_count),
        "historical_values_scanned": int(len(prior)),
        "historical_seed_sources": prior_sources,
        "historical_collision_count": 0,
        "disjoint_from_all_prior_registries": True,
        "formal_results_generated_before_registry": False,
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
    load_config("frozen")
    with np.load(SEEDS, allow_pickle=False) as data:
        return np.asarray(data[name]).copy()


def dgp_component_seeds(prefix: str, scenario_index: int, item_index: int | None = None) -> dict[str, int]:
    result: dict[str, int] = {}
    for component in DGP_COMPONENTS:
        array = load_seeds(f"{prefix}_dgp_{component}")
        value = array[scenario_index] if item_index is None else array[scenario_index, item_index]
        result[component] = int(value)
    return result


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
        raise RuntimeError(f"Phase 2R-H hash mismatch: {bad}")


def freeze_protocol() -> dict[str, Any]:
    cfg = load_raw_config()
    create_seed_registry()
    paths = [
        CONFIG, PROTOCOL_MD, PROTOCOL_JSON, SCENARIO_CSV, SCENARIO_JSON,
        PRECISION_PLAN, HISTORICAL_AUDIT, SEEDS, SEED_META,
        WORKSPACE / "results_rs_cusum" / "phase2rd" / "phase2rd_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2re" / "phase2re_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2rf" / "phase2rf_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2rg" / "phase2rg_gate.json",
        WORKSPACE / "results_rs_cusum" / "phase2rd" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2re" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2rf" / "manifest.json",
        WORKSPACE / "results_rs_cusum" / "phase2rg" / "manifest.json",
        WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py",
        WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py",
    ]
    return _write_hash(
        PROTOCOL_HASH,
        "PHASE2RH_PROTOCOL_SCENARIO_SEED_GATE_FREEZE",
        paths,
        {
            "frozen_before_any_formal_Phase2R_H_result": True,
            "m4_modified": False,
            "posthoc_tuning": False,
            "scenario_count": 9,
            "primary_transfer_count": 6,
            "stress_count": 2,
            "historical_gates_unchanged": True,
            "compatibility_label_is_not_actual_DGP_label": True,
            "gate": cfg["final_gate"],
        },
    )


def freeze_pre_run() -> dict[str, Any]:
    verify_hash(PROTOCOL_HASH)
    checkpoints = OUTPUT / "checkpoints"
    if checkpoints.exists() and any(checkpoints.rglob("*.npz")):
        raise RuntimeError("formal Phase 2R-H checkpoints exist before source freeze")
    sources = sorted((WORKSPACE / "src" / "rs_cusum_phase2rh").glob("*.py"))
    tests = sorted((WORKSPACE / "tests").glob("test_phase2rh*.py"))
    return _write_hash(
        PRE_RUN_HASH,
        "PHASE2RH_SOURCE_TEST_PRE_RUN_FREEZE",
        [PROTOCOL_HASH, *sources, *tests],
        {
            "implementation_frozen_before_engineering_tests": True,
            "formal_checkpoints_exist": False,
            "m4_source_sha256": sha256(WORKSPACE / "src" / "rs_cusum_phase2rd" / "engine.py"),
            "metric_source_sha256": sha256(WORKSPACE / "src" / "rs_cusum_phase2rf" / "metrics.py"),
        },
    )


def freeze_test_results() -> dict[str, Any]:
    verify_hash(PROTOCOL_HASH)
    verify_hash(PRE_RUN_HASH)
    checkpoints = OUTPUT / "checkpoints"
    if checkpoints.exists() and any(checkpoints.rglob("*.npz")):
        raise RuntimeError("formal checkpoints exist before engineering-test hash freeze")
    return _write_hash(
        TEST_HASH,
        "PHASE2RH_ENGINEERING_TEST_RESULTS_FREEZE",
        [PRE_RUN_HASH, OUTPUT / "test_results.txt"],
        {"engineering_tests_completed_before_formal_run": True},
    )


def load_config(stage: str = "raw") -> dict[str, Any]:
    cfg = load_raw_config()
    if stage in {"frozen", "run"}:
        verify_hash(PROTOCOL_HASH)
    if stage == "run":
        verify_hash(PRE_RUN_HASH)
        verify_hash(TEST_HASH)
    return cfg


if __name__ == "__main__":
    print(json.dumps(freeze_protocol(), sort_keys=True))
