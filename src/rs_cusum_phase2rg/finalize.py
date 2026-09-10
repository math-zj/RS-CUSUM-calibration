"""Write the Phase 2R-G report, final hash registry, and complete manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .protocol import (
    CONFIG, OUTPUT, PRE_RUN_HASH, PROTOCOL_HASH, PROTOCOL_JSON, PROTOCOL_MD,
    WORKSPACE, sha256, verify_hash,
)


REPORT = WORKSPACE / "report" / "rs_cusum_phase2rg_report.md"
FINAL_HASH = OUTPUT / "hash_registry_final.json"
MANIFEST = OUTPUT / "manifest.json"


def _value(frame: pd.DataFrame, column: str, **filters: object) -> float:
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    return float(selected.iloc[0][column])


def write_report() -> None:
    gate = json.loads((OUTPUT / "phase2rg_gate.json").read_text(encoding="utf-8"))
    engineering = json.loads((OUTPUT / "engineering_summary.json").read_text(encoding="utf-8"))
    center = pd.read_csv(OUTPUT / "center_covariance_summary.csv").iloc[0]
    marginal = pd.read_csv(OUTPUT / "marginal_tail_summary.csv")
    joint = pd.read_csv(OUTPUT / "joint_tail_summary.csv")
    maximum = pd.read_csv(OUTPUT / "max_distribution_summary.csv")
    type1 = pd.read_csv(OUTPUT / "type1_summary.csv")
    q95 = joint[(joint.comparison == "M4_balanced") & (joint.threshold == 0.95)].iloc[0]
    d2 = maximum[maximum.layer == "D2"].iloc[0]
    d4 = maximum[maximum.layer == "D4"].iloc[0]
    d4_type = type1[type1.layer == "D4"].iloc[0]
    q95_ratios = ", ".join(f"{row.layer}={row.q95_ratio:.6f}" for _, row in maximum.iterrows())
    sizes = ", ".join(f"{row.layer}={row.size_005:.6f}" for _, row in type1.iterrows())
    domain = json.dumps(gate["domain_checks"], sort_keys=True)
    next_step = (
        "M4 may be considered for a separately authorized, newly pre-registered transfer-null or null-misspecification validation with fresh scenarios; none is started here."
        if gate["gate"] == "M4_FRESH_GLOBAL_NULL_PASS"
        else "The frozen holdout failed; any future work must be a separately declared method-development stage and cannot replace or rerun this holdout."
    )
    text = f"""# RS-CUSUM-RL Phase 2R-G: Fresh Locked N0 Confirmation

## Scope and final gate

This was the one-time fresh locked confirmation of unchanged M4 under only `N0_complete_balanced_null`. It did not run a transfer null or establish cross-distribution robustness. Historical gates remain unchanged: Phase 2R-D=`M4_MECHANISM_JOINT_TAIL_FAIL`, Phase 2R-E=`TAIL_METRIC_UNSTABLE`, and Phase 2R-F=`M4_JOINT_TAIL_CONFIRMED`.

Final gate: `{gate['gate']}`. Domain checks: `{domain}`.

## Required answers

1. **M4 source versus Phase 2R-F.** Exact match={gate['m4_source_matches_phase2rf']}; SHA-256=`{gate['m4_source_sha256']}`.
2. **Fresh seeds.** All registered Phase 2R-G seeds were unique and disjoint from historical registries={gate['fresh_seed_integrity']}.
3. **Historical collisions.** {gate['historical_seed_collision_count']}.
4. **Post-hoc tuning.** None; `M4_modified={str(gate['M4_modified']).lower()}`, `posthoc_tuning={str(gate['posthoc_tuning']).lower()}`.
5. **Outer counts.** Planned/completed={gate['planned_outer_replicates']}/{gate['completed_outer_replicates']}.
6. **Bootstrap draws.** Planned/completed={gate['planned_outer_replicates'] * gate['planned_inner_draws_per_outer']}/{gate['completed_bootstrap_draws']} ({gate['planned_inner_draws_per_outer']} per outer).
7. **Draw failure.** M4 draw failure={gate['m4_draw_failure_rate']:.6f}; outer failure={gate['m4_outer_failure_rate']:.6f}; oracle failure={gate['oracle_failure_rate']:.6f}.
8. **Center.** Conditional-mean RMS/truth SD={center.mean_difference_RMS_over_truth_SD:.6f}.
9. **Pointwise SD ratio.** {center.pointwise_SD_ratio:.6f}.
10. **Leading eigenvalue ratio.** {center.leading_eigenvalue_ratio:.6f}.
11. **Effective-rank ratio.** {center.effective_rank_ratio:.6f}.
12. **Correlation errors.** Frobenius relative error={center.correlation_frobenius_relative_error:.6f}; off-diagonal MAE={center.offdiagonal_correlation_MAE:.6f}.
13. **Marginal q90/q95/q97.5/q99.** Mean coordinate absolute-quantile ratios={_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.90):.6f}/{_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.95):.6f}/{_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.975):.6f}/{_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.99):.6f}; maximum marginal exceedance MAE={marginal.marginal_exceedance_MAE.max():.6f}.
14. **EISJEM.** Point={gate['primary_EISJEM']:.6f}, MC SE={gate['primary_mc_standard_error']:.6f}, 95% CI=[{gate['primary_ci_low']:.6f}, {gate['primary_ci_high']:.6f}], frozen margin={gate['primary_equivalence_margin']:.6f}.
15. **Legacy p95 gap.** Point={gate['legacy_p95_gap']:.6f}, 95% CI=[{gate['legacy_p95_ci_low']:.6f}, {gate['legacy_p95_ci_high']:.6f}].
16. **Joint-tail secondary metrics at q95.** Tail-dependence MAE={q95.tail_dependence_offdiagonal_MAE:.6f}, joint-exceedance MAE={q95.joint_exceedance_offdiagonal_MAE:.6f}, joint-exceedance Frobenius discrepancy={q95.joint_exceedance_frobenius_relative_error:.6f}, pair-error p95/max={q95.pair_abs_error_p95:.6f}/{q95.pair_abs_error_max:.6f}. Threshold sensitivity at q90/q95/q97.5/q99 is in `joint_tail_summary.csv`.
17. **D2/D4 KS.** {d2.KS:.6f}/{d4.KS:.6f}.
18. **D0-D4 q95 ratios.** {q95_ratios}.
19. **D0-D4 empirical size at 0.05.** {sizes}.
20. **D4 Wilson interval.** Size={d4_type.size_005:.6f}; 95% Wilson=[{d4_type.wilson_low_005:.6f}, {d4_type.wilson_high_005:.6f}].
21. **Domain decisions.** `{domain}`.
22. **Final Phase 2R-G gate.** `{gate['gate']}`.
23. **Next-stage eligibility.** `qualifies_next_research_stage_decision={str(gate['qualifies_next_research_stage_decision']).lower()}`. This is eligibility for a separate decision only, not automatic authorization.
24. **Establish 1.0.2.** No.
25. **Allow OhioT1DM.** No.
26. **Most reasonable next experiment.** {next_step}
27. **Main files.** Protocol=`report/phase2rg_protocol.md`; gate=`results_rs_cusum/phase2rg/phase2rg_gate.json`; domain summaries, registries, run statuses, hashes, and manifest are under `results_rs_cusum/phase2rg`; this report=`report/rs_cusum_phase2rg_report.md`.

## Interpretation boundary

Even a PASS is independent computational confirmation only for unchanged M4 under the correctly specified complete balanced synthetic N0 family. It neither erases the historical Phase 2R-D/E outcomes nor proves theoretical bootstrap validity, cross-null robustness, transportability to OhioT1DM, or readiness of RS-CUSUM-RL 1.0.2. This phase ends here without starting any later experiment.
"""
    REPORT.write_text(text, encoding="utf-8")


def finalize() -> dict[str, object]:
    verify_hash(PROTOCOL_HASH)
    verify_hash(PRE_RUN_HASH)
    write_report()
    required = [
        CONFIG, PROTOCOL_MD, PROTOCOL_JSON, REPORT,
        OUTPUT / "sample_size_precision_plan.csv",
        OUTPUT / "phase2rg_gate.json", OUTPUT / "seed_registry.npz", OUTPUT / "seed_registry.json",
        PROTOCOL_HASH, PRE_RUN_HASH, OUTPUT / "test_results.txt",
        OUTPUT / "center_covariance_summary.csv", OUTPUT / "marginal_tail_summary.csv",
        OUTPUT / "joint_tail_summary.csv", OUTPUT / "max_distribution_summary.csv",
        OUTPUT / "type1_summary.csv", OUTPUT / "engineering_summary.json",
        OUTPUT / "mc_uncertainty_summary.csv", OUTPUT / "mc_uncertainty_draws.npz",
        OUTPUT / "process_banks.npz", OUTPUT / "runtime_summary.csv",
        OUTPUT / "oracle_primary_run_status.json", OUTPUT / "oracle_reference_run_status.json",
        OUTPUT / "m4_run_status.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    temporary = list(OUTPUT.rglob("*.tmp")) + list(OUTPUT.rglob("*.tmp.npz"))
    if temporary:
        raise RuntimeError(f"temporary/incomplete outputs remain: {temporary}")
    checkpoints = sorted((OUTPUT / "checkpoints").rglob("*.npz"))
    expected_checkpoints = 100 + 100 + 300 + 20
    if len(checkpoints) != expected_checkpoints:
        raise RuntimeError(f"checkpoint count {len(checkpoints)} != {expected_checkpoints}")
    source_files = sorted((WORKSPACE / "src" / "rs_cusum_phase2rg").glob("*.py"))
    test_files = sorted((WORKSPACE / "tests").glob("test_phase2rg*.py"))
    final_files = [*required, *source_files, *test_files]
    final_payload = {
        "stage": "PHASE2RG_FINAL",
        "hash_algorithm": "SHA-256",
        "files": {
            str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path)
            for path in final_files
        },
        "M4_modified": False,
        "metric_modified": False,
        "posthoc_tuning": False,
        "one_time_holdout_complete": True,
        "phase2rd_gate_unchanged": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "phase2re_gate_unchanged": "TAIL_METRIC_UNSTABLE",
        "phase2rf_gate_unchanged": "M4_JOINT_TAIL_CONFIRMED",
    }
    FINAL_HASH.write_text(json.dumps(final_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate = json.loads((OUTPUT / "phase2rg_gate.json").read_text(encoding="utf-8"))["gate"]
    manifest_files = [*final_files, FINAL_HASH, *checkpoints]
    manifest = {
        "phase": "Phase 2R-G",
        "scope": "Fresh Locked N0 Confirmation",
        "gate": gate,
        "hash_algorithm": "SHA-256",
        "files": {
            str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path)
            for path in manifest_files
        },
        "formal_outputs_complete": True,
        "checkpoint_count": len(checkpoints),
        "M4_modified": False,
        "metric_modified": False,
        "posthoc_tuning": False,
        "replacement_holdout": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
