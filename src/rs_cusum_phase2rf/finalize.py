"""Write the frozen Phase 2R-F report, final hash registry, and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .protocol import CONFIG, OUTPUT, PRE_RUN_HASH, PROTOCOL_HASH, PROTOCOL_JSON, PROTOCOL_MD, WORKSPACE, sha256, verify_hash


REPORT = WORKSPACE / "report" / "rs_cusum_phase2rf_report.md"
FINAL_HASH = OUTPUT / "hash_registry_final.json"
MANIFEST = OUTPUT / "manifest.json"


def _value(frame: pd.DataFrame, column: str, **filters: object) -> float:
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    return float(selected.iloc[0][column])


def write_report() -> None:
    gate = json.loads((OUTPUT / "phase2rf_gate.json").read_text(encoding="utf-8"))
    tail = pd.read_csv(OUTPUT / "stable_joint_tail_metrics.csv")
    center = pd.read_csv(OUTPUT / "center_covariance_summary.csv").iloc[0]
    marginal = pd.read_csv(OUTPUT / "marginal_tail_summary.csv")
    maximum = pd.read_csv(OUTPUT / "max_distribution_summary.csv")
    type1 = pd.read_csv(OUTPUT / "type1_summary.csv")
    convergence = pd.read_csv(OUTPUT / "oracle_convergence.csv")
    planning = pd.read_csv(OUTPUT / "sample_size_precision_plan.csv")
    q95 = tail[(tail.comparison == "M4_balanced") & (tail.threshold == 0.95)].iloc[0]
    d4 = maximum[maximum.layer == "D4"].iloc[0]
    d4_type = type1[type1.layer == "D4"].iloc[0]
    old_gap = 0.14181287586688995
    text = f"""# RS-CUSUM-RL Phase 2R-F: Independent Oracle and Stable Joint-Tail Confirmation

## Frozen scope and provenance

Phase 2R-F evaluated the unchanged Phase 2R-D M4 only under synthetic `N0_complete_balanced_null`. The historical gates remain immutable: Phase 2R-D=`M4_MECHANISM_JOINT_TAIL_FAIL`; Phase 2R-E=`TAIL_METRIC_UNSTABLE`. No D3, OhioT1DM, Phase 3, transfer-null, alternative, power, or changepoint analysis was run. No M4 source, fitted model, bootstrap parameter, critical value, scale, or gate was tuned after results.

Precision planning used only saved Phase 2R-E draws. Two independent 2,500-process oracle banks were frozen (5,000 total). M4 used 300 new outer N0 datasets and 199 unchanged inner draws per outer (59,700 generated), with a pre-registered cluster-balanced 2,500-process bank for fair primary comparison.

## Final gate

`{gate['gate']}`

Primary EISJEM={gate['primary_point_estimate']:.6f}, cluster-bootstrap MC SE={gate['primary_mc_standard_error']:.6f}, 95% CI=[{gate['primary_ci_low']:.6f}, {gate['primary_ci_high']:.6f}], equivalence margin=0.015, and n>=1,000 convergence median range={gate['primary_convergence_median_range_n_ge_1000']:.6f}. Domain checks: `{json.dumps(gate['domain_checks'], sort_keys=True)}`.

## Required answers

1. **Old 0.1418 metric at large oracle replication.** The q95 raw conditional-dependence p95 gap changed from {old_gap:.6f} to {gate['legacy_p95_point_estimate']:.6f}; its cluster-bootstrap 95% interval is [{gate['legacy_p95_ci_low']:.6f}, {gate['legacy_p95_ci_high']:.6f}].
2. **Convergence of the old metric.** Its n>=1,000 median range is {gate['legacy_convergence_median_range_n_ge_1000']:.6f}; see `oracle_convergence.csv` for n=120,250,500,1000,2500.
3. **Primary stable functional.** At q=0.90,0.95,0.975,0.99, average all ordered off-diagonal absolute joint-exceedance probability errors, divide each by alpha=1-q, average thresholds, then subtract the equal-size oracle-primary versus oracle-reference value.
4. **Why more stable.** It uses every process row for joint Bernoulli means, never divides by random coordinate-specific exceedance counts, aggregates four pre-specified thresholds, and explicitly benchmarks equal-size oracle-oracle L1 noise. It still measures extreme co-exceedance, not covariance.
5. **Primary estimate/uncertainty.** Point={gate['primary_point_estimate']:.6f}, MC SE={gate['primary_mc_standard_error']:.6f}, 95% CI=[{gate['primary_ci_low']:.6f}, {gate['primary_ci_high']:.6f}].
6. **Balanced comparisons.** Saved in `balanced_tail_comparison.csv`; the primary metric was evaluated n-versus-n with M4 cluster-balanced at 120,250,500,1000,2500.
7. **Center.** RMS mean difference/truth SD={center.mean_difference_RMS_over_truth_SD:.6f}; pass={gate['domain_checks']['center']}.
8. **Covariance.** Pointwise SD ratio={center.pointwise_SD_ratio:.6f}, leading eigenvalue ratio={center.leading_eigenvalue_ratio:.6f}, effective-rank ratio={center.effective_rank_ratio:.6f}, correlation Frobenius error={center.correlation_frobenius_relative_error:.6f}, off-diagonal correlation MAE={center.offdiagonal_correlation_MAE:.6f}; pass={gate['domain_checks']['covariance']}.
9. **Marginal tails.** q90/q95/q97.5/q99 mean coordinate ratios={_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.90):.6f}/{_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.95):.6f}/{_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.975):.6f}/{_value(marginal,'mean_coordinate_abs_quantile_ratio',threshold=0.99):.6f}; pass={gate['domain_checks']['marginal_tail']}.
10. **D4 KS.** {d4.KS:.6f}.
11. **D4 q95 ratio.** {d4.q95_ratio:.6f}.
12. **D4 empirical size.** At alpha=.05: {d4_type.size_005:.6f}, Wilson 95%=[{d4_type.wilson_low_005:.6f}, {d4_type.wilson_high_005:.6f}].
13. **Stable joint-tail discrepancy.** The formal answer is the final gate above. q95 tail-dependence MAE={q95.tail_dependence_offdiagonal_MAE:.6f}; q95 joint-exceedance MAE={q95.joint_exceedance_offdiagonal_MAE:.6f}.
14. **Evidence to develop M5.** Only `M4_STABLE_JOINT_TAIL_FAIL` would supply stable primary evidence for M5; the present gate must be interpreted under that pre-registration.
15. **Phase 2R-F gate.** `{gate['gate']}`.
16. **Fresh global confirmation eligibility.** {gate['qualifies_future_Phase2R_G']}; if true, this means eligibility only for a new Phase 2R-G with unseen seeds.
17. **Real data / later stages.** OhioT1DM, Phase 3, and transfer-null remain prohibited.
18. **Post-hoc tuning.** None; `M4_modified=false`, `posthoc_tuning=false`.
19. **Seed integrity.** All registered Phase 2R-F seeds and M4 process subseeds were unique and disjoint from prior registry values before formal computation.
20. **Main outputs.** Protocols are in `report/phase2rf_protocol.*`; all numerical CSV/JSON/NPZ outputs and registries are in `results_rs_cusum/phase2rf`; this report is `report/rs_cusum_phase2rf_report.md`.

## Interpretation boundary

Even a confirmation gate supplies computational evidence only for unchanged M4 under the current correctly specified synthetic N0 family. It does not revise Phase 2R-D or E, prove bootstrap validity, establish version 1.0.2, or justify OhioT1DM inference.
"""
    REPORT.write_text(text, encoding="utf-8")


def finalize() -> dict[str, object]:
    verify_hash(PROTOCOL_HASH)
    verify_hash(PRE_RUN_HASH)
    write_report()
    required = [
        CONFIG, PROTOCOL_MD, PROTOCOL_JSON, REPORT,
        OUTPUT / "sample_size_precision_plan.csv", OUTPUT / "oracle_convergence.csv",
        OUTPUT / "balanced_tail_comparison.csv", OUTPUT / "stable_joint_tail_metrics.csv",
        OUTPUT / "legacy_p95_tail_metric.csv", OUTPUT / "center_covariance_summary.csv",
        OUTPUT / "marginal_tail_summary.csv", OUTPUT / "max_distribution_summary.csv",
        OUTPUT / "type1_summary.csv", OUTPUT / "mc_uncertainty_summary.csv",
        OUTPUT / "phase2rf_gate.json", OUTPUT / "seed_registry.npz", OUTPUT / "seed_registry.json",
        PROTOCOL_HASH, PRE_RUN_HASH, OUTPUT / "test_results.txt", OUTPUT / "process_banks.npz",
        OUTPUT / "mc_uncertainty_draws.npz", OUTPUT / "joint_tail_structure_summary.csv",
        OUTPUT / "engineering_summary.json", OUTPUT / "runtime_summary.csv",
        OUTPUT / "oracle_primary_run_status.json", OUTPUT / "oracle_reference_run_status.json",
        OUTPUT / "m4_run_status.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source_files = sorted((WORKSPACE / "src" / "rs_cusum_phase2rf").glob("*.py"))
    test_files = sorted((WORKSPACE / "tests").glob("test_phase2rf*.py"))
    final_files = [*required, *source_files, *test_files]
    final_payload = {
        "stage": "PHASE2RF_FINAL",
        "hash_algorithm": "SHA-256",
        "files": {
            str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path)
            for path in final_files
        },
        "M4_modified": False,
        "phase2rd_gate_unchanged": "M4_MECHANISM_JOINT_TAIL_FAIL",
        "phase2re_gate_unchanged": "TAIL_METRIC_UNSTABLE",
    }
    FINAL_HASH.write_text(json.dumps(final_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_files = [*final_files, FINAL_HASH]
    gate = json.loads((OUTPUT / "phase2rf_gate.json").read_text(encoding="utf-8"))["gate"]
    manifest = {
        "phase": "Phase 2R-F",
        "gate": gate,
        "hash_algorithm": "SHA-256",
        "files": {
            str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path)
            for path in manifest_files
        },
        "formal_outputs_complete": True,
        "M4_modified": False,
        "posthoc_tuning": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
