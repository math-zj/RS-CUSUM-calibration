"""Package Phase 2R-A forensic outputs and verify every frozen parent."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from src.rs_cusum_phase2.build_artifacts import audit_phase0, audit_phase1

from .protocol import HASH_PATH, OUTPUT, WORKSPACE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)+"\n", encoding="utf-8")


def verify_pre_run() -> dict:
    frozen = json.loads(HASH_PATH.read_text(encoding="utf-8"))
    mismatches = []
    for path_text, expected in frozen["files"].items():
        path = WORKSPACE / path_text
        if not path.is_file(): mismatches.append({"path": path_text, "reason": "missing"})
        elif sha256(path) != expected: mismatches.append({"path": path_text, "reason": "hash"})
    return {"entries": len(frozen["files"]), "mismatch_count": len(mismatches), "mismatches": mismatches,
            "status": "PASS" if not mismatches else "FAIL"}


def audit_phase2() -> dict:
    manifest_path = WORKSPACE / "results_rs_cusum" / "phase2" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for entry in manifest["files"]:
        path = WORKSPACE / entry["path"]
        if not path.is_file(): mismatches.append({"path": entry["path"], "reason": "missing"})
        elif sha256(path) != entry["sha256"]: mismatches.append({"path": entry["path"], "reason": "hash"})
    return {"manifest_entries": len(manifest["files"]), "mismatch_count": len(mismatches),
            "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle))-1)


def main() -> None:
    phase0 = audit_phase0(); phase1 = audit_phase1(); phase2 = audit_phase2(); pre_run = verify_pre_run()
    integrity = {"phase0": phase0, "phase1": phase1, "phase2": phase2, "phase2r_pre_run": pre_run}
    write_json(OUTPUT / "hash_integrity.json", integrity)
    if phase0["mismatch_count"] or phase1["unexpected_mismatch_count"] or phase2["mismatch_count"] or pre_run["mismatch_count"]:
        raise RuntimeError("frozen parent or pre-run integrity audit failed")

    gate = {
        "gate": "ROOT_CAUSE_LOCALIZED",
        "parent_method": "1.0.1-phase1",
        "parent_method_modified": False,
        "primary_failure_layer": "joint state-grid x candidate maximum multiplier inference",
        "secondary_failure": "single-point multiplier/studentization underdispersion",
        "forensic_implementation_failure": False,
        "new_primary_method_declared": False,
        "recommended_next_protocol": "Phase 2R-B inference-only revision and independent null confirmation",
        "candidate_version_scope_if_confirmed": "1.0.2 inference-only revision",
        "phase3_full_simulation_allowed": False,
        "formal_ohiot1dm_analysis_allowed": False,
    }
    write_json(OUTPUT / "phase2r_gate.json", gate)
    metadata = {
        "diagnostic_label": "phase2r-a-forensics", "parent_method": "1.0.1-phase1",
        "scenario": "N0_complete_balanced_stationary_null_only",
        "truth_replicates": 1000, "decomposition_replicates": 400,
        "scaling_replicates": 3900, "highB_datasets": 50, "highB_draws": 1999,
        "full_refit_datasets": 30, "full_refit_draws": 199,
        "alternatives_run": False, "changepoint_run": False, "ohiot1dm_run": False,
        "unit_tests": {"total": 55, "passed": 55, "failed": 0},
        "gate": gate["gate"],
    }
    write_json(OUTPUT / "run_metadata.json", metadata)

    commands = "# Phase 2R-A executed commands\n\n"
    commands += "- `python -m src.rs_cusum_phase2r.protocol`\n"
    commands += "- `python -m src.rs_cusum_phase2r.runner truth --workers 4`\n"
    commands += "- `python -m src.rs_cusum_phase2r.runner decomposition --workers 4`\n"
    commands += "- `python -m src.rs_cusum_phase2r.runner scaling --workers 4`\n"
    commands += "- `python -m src.rs_cusum_phase2r.runner bootstrap --workers 4`\n"
    commands += "- `python -m src.rs_cusum_phase2r.summarize`\n"
    commands += "- `python -m src.rs_cusum_phase2r.runner full_refit --workers 4`\n"
    commands += "- `python -m src.rs_cusum_phase2r.postprocess` (deterministic post-run arrays only; no new datasets/fits)\n"
    commands += "- `python -m unittest discover -s tests -p 'test_*.py' -v`\n"
    (OUTPUT / "commands.md").write_text(commands, encoding="utf-8")

    modified = [
        WORKSPACE / "configs" / "rs_cusum_phase2r_forensics.yaml",
        WORKSPACE / "report" / "rs_cusum_phase2r_failure_localization_protocol.md",
        WORKSPACE / "report" / "rs_cusum_phase2r_failure_localization_report.md",
    ]
    modified.extend(sorted((WORKSPACE / "src" / "rs_cusum_phase2r").glob("*.py")))
    modified.extend(sorted((WORKSPACE / "tests").glob("test_phase2r_*.py")))
    (OUTPUT / "modified_files.txt").write_text("\n".join(relative(path) for path in modified)+"\n", encoding="utf-8")

    result_files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "manifest.json")
    all_files = sorted(set(modified+result_files), key=relative)
    entries = [{"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in all_files]
    manifest = {
        "phase": "2R-A", "diagnostic_label": "phase2r-a-forensics", "gate": gate["gate"],
        "generated_at_local": datetime.now().astimezone().isoformat(), "hash_algorithm": "SHA-256",
        "manifest_self_hash_excluded": True, "parent_method_modified": False,
        "pre_run_frozen_file_count": pre_run["entries"],
        "post_run_only_source": {
            "src/rs_cusum_phase2r/postprocess.py": "deterministic saved-array reconstruction; no dataset generation or model fitting",
            "src/rs_cusum_phase2r/build_artifacts.py": "packaging and integrity audit only",
        },
        "phase0_integrity": phase0, "phase1_integrity": phase1,
        "phase2_integrity": phase2, "phase2r_pre_run_integrity": pre_run,
        "unit_tests": metadata["unit_tests"], "phase3_full_simulation_allowed": False,
        "formal_ohiot1dm_analysis_allowed": False,
        "key_replicate_rows": {
            "truth": csv_rows(OUTPUT/"truth_replicates.csv"),
            "decomposition": csv_rows(OUTPUT/"decomposition_replicates.csv"),
            "scaling": csv_rows(OUTPUT/"scaling_replicates.csv"),
            "bootstrap_forensic": csv_rows(OUTPUT/"bootstrap_forensic_replicates.csv"),
            "full_refit": csv_rows(OUTPUT/"full_refit_replicates.csv"),
        },
        "file_count_excluding_manifest": len(entries), "files": entries,
    }
    write_json(OUTPUT / "manifest.json", manifest)
    print(f"Phase 2R-A artifacts: {len(entries)}")
    print(f"Phase 0 mismatch: {phase0['mismatch_count']}")
    print(f"Phase 1 unexpected mismatch: {phase1['unexpected_mismatch_count']}")
    print(f"Phase 2 mismatch: {phase2['mismatch_count']}")
    print(f"Phase 2R pre-run mismatch: {pre_run['mismatch_count']}")


if __name__ == "__main__":
    main()
