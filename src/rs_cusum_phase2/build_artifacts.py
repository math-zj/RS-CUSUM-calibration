"""Build Phase-2 failure-gate artifacts and frozen-file audits."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from .protocol import WORKSPACE


OUTPUT = WORKSPACE / "results_rs_cusum" / "phase2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def rel(path: Path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


def audit_phase0() -> dict:
    inventory = WORKSPACE / "results_rs_cusum" / "phase0" / "frozen_file_inventory.csv"
    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = []
    for row in rows:
        path = WORKSPACE / row["path"]
        if not path.is_file():
            mismatches.append({"path": row["path"], "reason": "missing"})
        elif sha256(path) != row["sha256"].upper():
            mismatches.append({"path": row["path"], "reason": "hash"})
    return {
        "inventory_rows": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def audit_phase1() -> dict:
    manifest_path = WORKSPACE / "results_rs_cusum" / "phase1" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    allowed_mutable = {
        "results_rs_cusum/CHANGELOG.md": "append-only Phase 2 entry explicitly requested by the Phase 2 protocol"
    }
    expected = []
    unexpected = []
    matched = 0
    for entry in manifest["files"]:
        path = WORKSPACE / entry["path"]
        mismatch = None
        if not path.is_file():
            mismatch = "missing"
        elif sha256(path) != entry["sha256"]:
            mismatch = "hash"
        if mismatch is None:
            matched += 1
        else:
            record = {
                "path": entry["path"],
                "reason": mismatch,
                "allowed_reason": allowed_mutable.get(entry["path"]),
            }
            (expected if entry["path"] in allowed_mutable else unexpected).append(record)
    return {
        "manifest_entries": len(manifest["files"]),
        "matched_entries": matched,
        "allowed_append_only_mismatch_count": len(expected),
        "allowed_append_only_mismatches": expected,
        "unexpected_mismatch_count": len(unexpected),
        "unexpected_mismatches": unexpected,
        "frozen_method_config_source_test_result_integrity": "PASS" if not unexpected else "FAIL",
    }


def verify_pre_run_hashes() -> dict:
    path = OUTPUT / "pre_run_config_hashes.json"
    with path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    mismatches = []
    for relative, expected in frozen["files"].items():
        target = WORKSPACE / relative
        if not target.is_file() or sha256(target) != expected:
            mismatches.append(relative)
    return {
        "file_count": len(frozen["files"]),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT / "stage_a_replicate_results.csv", OUTPUT / "replicate_results.csv")
    phase0 = audit_phase0()
    phase1 = audit_phase1()
    pre_run = verify_pre_run_hashes()
    integrity = {"phase0": phase0, "phase1": phase1, "phase2_pre_run": pre_run}
    write_json(OUTPUT / "hash_integrity.json", integrity)
    if phase0["mismatch_count"] or phase1["unexpected_mismatch_count"] or pre_run["mismatch_count"]:
        raise RuntimeError("frozen-file integrity audit failed")

    run_metadata = {
        "phase": 2,
        "parent_method_version": "1.0.1-phase1",
        "stage_a": {
            "run": True,
            "datasets": 320,
            "replicate_result_rows": 1220,
            "replicates_each_scenario_or_family": 20,
            "bootstrap_draws": 199,
            "engineering_gate": "STAGE_A_PASS",
            "calibration_gate": "IMMEDIATE_FAIL",
        },
        "stage_b": {"run": False, "reason": "immediate null-calibration stop"},
        "formal_ohiot1dm_analysis": False,
        "final_gate": "PHASE_2_METHOD_REVISION_REQUIRED",
    }
    write_json(OUTPUT / "run_metadata.json", run_metadata)

    source_files = [
        WORKSPACE / "configs" / "rs_cusum_phase2_pilot.yaml",
        WORKSPACE / "report" / "rs_cusum_phase2_simulation_protocol.md",
        WORKSPACE / "report" / "rs_cusum_phase2_pilot_report.md",
        WORKSPACE / "results_rs_cusum" / "CHANGELOG.md",
    ]
    source_files.extend(sorted((WORKSPACE / "src" / "rs_cusum_phase2").glob("*.py")))
    source_files.extend(sorted((WORKSPACE / "tests").glob("test_phase2_*.py")))
    with (OUTPUT / "modified_files.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(rel(path) for path in source_files) + "\n")

    result_files = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp")
    )
    all_files = sorted(set(source_files + result_files), key=rel)
    entries = [
        {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in all_files
    ]
    manifest = {
        "phase": 2,
        "gate": "PHASE_2_METHOD_REVISION_REQUIRED",
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "hash_algorithm": "SHA-256",
        "manifest_self_hash_excluded": True,
        "parent_method_modified": False,
        "stage_a_run": True,
        "stage_b_run": False,
        "phase3_full_simulation_allowed": False,
        "formal_ohiot1dm_analysis_allowed": False,
        "unit_tests": {"total": 47, "passed": 47, "failed": 0},
        "phase0_integrity": phase0,
        "phase1_integrity": phase1,
        "phase2_pre_run_integrity": pre_run,
        "phase1_manifest_allowed_mutable_paths": {
            "results_rs_cusum/CHANGELOG.md": "append-only Phase 2 changelog entry"
        },
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    write_json(OUTPUT / "manifest.json", manifest)
    print(f"Phase 2 artifact files: {len(entries)}")
    print(f"Phase 0 mismatch: {phase0['mismatch_count']}")
    print(f"Phase 1 unexpected mismatch: {phase1['unexpected_mismatch_count']}")
    print(f"Phase 2 pre-run mismatch: {pre_run['mismatch_count']}")


if __name__ == "__main__":
    main()
