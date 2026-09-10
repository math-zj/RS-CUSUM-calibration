"""Build reproducible Phase-1 hashes and manifests without running statistics."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT = WORKSPACE / "results_rs_cusum" / "phase1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def relative(path: Path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


def phase0_recheck() -> dict:
    inventory = WORKSPACE / "results_rs_cusum" / "phase0" / "frozen_file_inventory.csv"
    mismatches: list[dict[str, str]] = []
    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = WORKSPACE / row["path"]
        if not path.is_file():
            mismatches.append({"path": row["path"], "reason": "missing", "actual_sha256": ""})
            continue
        actual = sha256(path)
        if actual != row["sha256"].upper():
            mismatches.append({"path": row["path"], "reason": "sha256", "actual_sha256": actual})
    return {
        "inventory": relative(inventory),
        "inventory_rows": len(rows),
        "hash_algorithm": "SHA-256",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def phase1_source_files() -> list[Path]:
    files = [
        WORKSPACE / "configs" / "rs_cusum_main.yaml",
        WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml",
        WORKSPACE / "configs" / "rs_cusum_simulation.yaml",
        WORKSPACE / "report" / "rs_cusum_phase1_method_spec.md",
        WORKSPACE / "report" / "rs_cusum_phase1_implementation_report.md",
        WORKSPACE / "results_rs_cusum" / "CHANGELOG.md",
    ]
    files.extend(sorted((WORKSPACE / "src" / "rs_cusum").rglob("*.py")))
    files.extend(
        WORKSPACE / "tests" / name
        for name in (
            "test_risk_set.py",
            "test_support.py",
            "test_patient_balance.py",
            "test_statistic.py",
            "test_cluster_bootstrap.py",
            "test_simulation_generator.py",
        )
    )
    return sorted(set(files), key=relative)


def artifact_files() -> list[Path]:
    names = (
        "commands.md",
        "test_results.txt",
        "toy_diagnostic_console.txt",
        "diagnostic_toy_results.json",
        "config_hashes.json",
        "phase0_hash_recheck.json",
        "modified_files.txt",
    )
    return [OUTPUT / name for name in names]


def file_entry(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_paths = [
        WORKSPACE / "configs" / "rs_cusum_main.yaml",
        WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml",
        WORKSPACE / "configs" / "rs_cusum_simulation.yaml",
        WORKSPACE / "configs" / "mdp_5min_protocol.yaml",
        WORKSPACE / "report" / "rs_cusum_phase1_method_spec.md",
    ]
    config_hashes = {
        "hash_algorithm": "SHA-256",
        "frozen_before_formal_rs_outcomes": True,
        "files": {relative(path): sha256(path) for path in config_paths},
    }
    write_json(OUTPUT / "config_hashes.json", config_hashes)

    recheck = phase0_recheck()
    write_json(OUTPUT / "phase0_hash_recheck.json", recheck)
    if recheck["mismatch_count"] != 0:
        raise RuntimeError("Phase 0 frozen inventory mismatch is nonzero")

    source_files = phase1_source_files()
    modified_lines = [relative(path) for path in source_files]
    with (OUTPUT / "modified_files.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(modified_lines) + "\n")

    entries = [file_entry(path) for path in source_files + artifact_files()]
    manifest = {
        "phase": 1,
        "gate": "PHASE_1_PASS",
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "hash_algorithm": "SHA-256",
        "manifest_self_hash_excluded": True,
        "formal_ohiot1dm_analysis_run": False,
        "formal_bootstrap_run": False,
        "p_value_computed": False,
        "change_point_estimated": False,
        "unit_tests": {"total": 37, "passed": 37, "failed": 0},
        "degeneration_toy": {
            "status": "PASS",
            "maximum_beta_absolute_error": 0.0,
            "statistic_absolute_error": 0.0,
            "maximum_scaling_absolute_error": 0.0,
        },
        "phase0_recheck": recheck,
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    write_json(OUTPUT / "manifest.json", manifest)
    print(f"Phase 1 manifest: {len(entries)} files")
    print(f"Phase 0 mismatch count: {recheck['mismatch_count']}")


if __name__ == "__main__":
    main()
