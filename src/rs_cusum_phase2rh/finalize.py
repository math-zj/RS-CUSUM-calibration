"""Write the Phase 2R-H report, final hashes, and complete manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .protocol import (
    CONFIG, HISTORICAL_AUDIT, OUTPUT, PRECISION_PLAN, PRE_RUN_HASH, PROTOCOL_HASH,
    PROTOCOL_JSON, PROTOCOL_MD, SCENARIO_CSV, SCENARIO_JSON, TEST_HASH, WORKSPACE,
    sha256, verify_hash,
)


REPORT = WORKSPACE / "report" / "rs_cusum_phase2rh_report.md"
FINAL_HASH = OUTPUT / "hash_registry_final.json"
MANIFEST = OUTPUT / "manifest.json"


def write_report() -> None:
    gate = json.loads((OUTPUT / "phase2rh_gate.json").read_text(encoding="utf-8"))
    classification = pd.read_csv(OUTPUT / "scenario_classification.csv")
    type1 = pd.read_csv(OUTPUT / "scenario_type1_summary.csv")
    maximum = pd.read_csv(OUTPUT / "scenario_max_calibration.csv")
    center = pd.read_csv(OUTPUT / "scenario_center_covariance.csv")
    marginal = pd.read_csv(OUTPUT / "scenario_marginal_tail.csv")
    joint = pd.read_csv(OUTPUT / "scenario_joint_tail.csv")
    engineering = pd.read_csv(OUTPUT / "scenario_engineering.csv")
    registry = pd.read_csv(SCENARIO_CSV)
    lines = []
    for scenario in registry.scenario_id:
        label = classification[classification.scenario_id == scenario].iloc[0]
        d4 = type1[(type1.scenario_id == scenario) & (type1.layer == "D4")].iloc[0]
        d4max = maximum[(maximum.scenario_id == scenario) & (maximum.layer == "D4")].iloc[0]
        q95 = joint[(joint.scenario_id == scenario) & (joint.comparison == "M4_balanced") & (joint.threshold == .95)].iloc[0]
        eng = engineering[engineering.scenario_id == scenario].iloc[0]
        lines.append(
            f"- `{scenario}` ({label['role']}): **{label.classification}**; "
            f"D4 size={d4.size_005:.4f} ({int(d4.rejections_005)}/{int(d4.testable_replicates)}), "
            f"95% Wilson=[{d4.wilson_low_005:.4f},{d4.wilson_high_005:.4f}], "
            f"q95 ratio={d4max.q95_ratio:.4f}, KS={d4max.KS:.4f}, "
            f"EISJEM={q95.primary_EISJEM:.5f}, draw failure={eng.m4_draw_failure_rate:.4f}."
        )
    d4_table = type1[type1.layer == "D4"].copy()
    hardest_id = d4_table.sort_values(["size_005", "scenario_id"], ascending=[False, True]).iloc[0].scenario_id
    hardest_max = maximum[maximum.layer == "D4"].sort_values(["KS", "scenario_id"], ascending=[False, True]).iloc[0].scenario_id
    systematic = gate["systematic_mechanism_failure_domains"] or []
    family_answers = []
    for keyword, label in (
        ("reward_t5", "heavy-tailed reward"), ("transition_t5", "heavy-tailed transition"),
        ("heterogeneity", "patient heterogeneity"), ("imbalanced", "imbalance"),
        ("AR1", "temporal dependence"), ("weaker_overlap", "weaker overlap"),
    ):
        row = classification[classification.scenario_id.str.contains(keyword, case=False)].iloc[0]
        family_answers.append(f"{label}={row.classification}")
    claims = (
        "The frozen M4 controlled the pre-registered primary inference gates throughout the entire H1–H8 slate; a later, separately authorized Alternative/Power Validation is eligible."
        if gate["gate"] == "M4_TRANSFER_NULL_PASS"
        else "The frozen M4 did not earn an unrestricted transfer-null robustness claim; classifications must be reported scenario by scenario."
    )
    text = f"""# RS-CUSUM-RL Phase 2R-H: Transfer-Null and Null-Misspecification Validation

## Scope and final gate

Final gate: **`{gate['gate']}`**. M4 was unchanged (`{gate['m4_source_sha256']}`), no scenario-specific correction or post-hoc tuning was made, and all laws were stationary nulls with complete observation. Historical D/E/F/G gates remain immutable.

## Scenario-level results

{chr(10).join(lines)}

## Required answers

1. **Scenarios actually run.** H0 reference plus H1–H8 exactly as frozen in `scenario_registry.csv`; no scenario was added, removed, softened, or replaced.
2. **Differences from N0.** They respectively change patient random-effect law/scale, phenotype and treatment balance, reward tails, transition tails, reward variance, reward serial dependence, overlap, and a precombined moderate subset. Exact parameters are in the registry and protocol.
3. **Why still null.** Every coefficient and innovation law is invariant across all 240 transitions, observation is complete, and `true_change_point=None`; there is no time-indexed mean, variance, policy, or composition break.
4. **D0–D4 size.** All per-layer values and counts are in `scenario_type1_summary.csv`.
5. **D4 Wilson intervals.** Listed above; both ordinary 95% and pre-registered Bonferroni simultaneous sensitivity intervals are in the CSV.
6. **D4 q95 ratios.** Listed above; D0–D4 are in `scenario_max_calibration.csv`.
7. **D4 KS.** Listed above; D0–D4 are in the same file.
8. **Center/covariance.** Full mean, SD, eigenvalue, effective-rank and correlation diagnostics are in `scenario_center_covariance.csv`; diagnostic failures are recorded without redefining inference failure.
9. **Marginal tails.** q90/q95/q97.5/q99, exceedance, skewness and kurtosis diagnostics are in `scenario_marginal_tail.csv`.
10. **EISJEM.** Listed above and reported across thresholds in `scenario_joint_tail.csv`; it is mechanism-diagnostic in this phase.
11. **Systematic type-I inflation.** Primary-scenario failure present={gate['primary_null_validity_failure_present']}; stress failure present={gate['stress_null_validity_failure_present']}.
12. **Most difficult scenario.** Largest empirical D4 size: `{hardest_id}`; largest D4 KS: `{hardest_max}`. These rankings are descriptive and did not select or alter scenarios.
13–17. **Family results.** {'; '.join(family_answers)}.
18. **Evidence of N0-only validity.** The overall gate and scenario classifications, rather than a selected successful subset, answer this: `{gate['gate']}`.
19. **Overall Phase 2R-H gate.** `{gate['gate']}`.
20. **Paper-level robustness claim.** {claims}
21. **Limitations.** N=12, T=240, complete observation, only nine finite synthetic laws, 200 outer replicates, 199 inner draws, and about ten expected q99 exceedances per oracle coordinate. H6 deliberately violates the fitted innovation-independence/observed-state Markov specification but does not establish robustness to arbitrary dependence. Simulation PASS is not theoretical proof or clinical transportability.
22. **Alternative/power eligibility.** `{gate['alternative_power_validation_eligible']}`; eligibility is not automatic execution.
23. **OhioT1DM eligibility.** No.
24. **RS-CUSUM-RL 1.0.2 established.** No.
25. **Files.** Protocol=`report/phase2rh_protocol.md`; registry=`results_rs_cusum/phase2rh/scenario_registry.csv`; gate=`results_rs_cusum/phase2rh/phase2rh_gate.json`; summaries, hashes, seed registry, statuses and manifest are under `results_rs_cusum/phase2rh`; this report=`report/rs_cusum_phase2rh_report.md`.

## Gate audit

Scenario classes: `{json.dumps(gate['scenario_classifications'], sort_keys=True)}`. Diagnostic-only degradation count among H1–H8={gate['mild_degradation_count_H1_H8']}; systematic diagnostic domains={json.dumps(systematic)}. Duplicate/extra/unregistered seeds={gate['duplicate_seed_count']}/{gate['extra_seed_count']}/{gate['unregistered_seed_count']}.

This phase stops here. It did not run alternatives, power, changepoint, OhioT1DM, Phase 3, or establish version 1.0.2.
"""
    REPORT.write_text(text, encoding="utf-8")


def finalize() -> dict[str, object]:
    verify_hash(PROTOCOL_HASH); verify_hash(PRE_RUN_HASH); verify_hash(TEST_HASH)
    write_report()
    required = [
        CONFIG, PROTOCOL_MD, PROTOCOL_JSON, SCENARIO_CSV, SCENARIO_JSON,
        PRECISION_PLAN, HISTORICAL_AUDIT, REPORT,
        OUTPUT / "seed_registry.npz", OUTPUT / "seed_registry.json",
        PROTOCOL_HASH, PRE_RUN_HASH, TEST_HASH, OUTPUT / "test_results.txt",
        OUTPUT / "scenario_type1_summary.csv", OUTPUT / "scenario_max_calibration.csv",
        OUTPUT / "scenario_center_covariance.csv", OUTPUT / "scenario_marginal_tail.csv",
        OUTPUT / "scenario_joint_tail.csv", OUTPUT / "scenario_engineering.csv",
        OUTPUT / "scenario_dgp_summary.csv", OUTPUT / "scenario_classification.csv",
        OUTPUT / "phase2rh_gate.json", OUTPUT / "oracle_primary_run_status.json",
        OUTPUT / "oracle_reference_run_status.json", OUTPUT / "m4_run_status.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    temporary = list(OUTPUT.rglob("*.tmp")) + list(OUTPUT.rglob("*.tmp.npz"))
    if temporary:
        raise RuntimeError(f"temporary/incomplete outputs remain: {temporary}")
    checkpoints = sorted((OUTPUT / "checkpoints").rglob("*.npz"))
    expected = 9 * (50 + 50 + 200)
    if len(checkpoints) != expected:
        raise RuntimeError(f"checkpoint count {len(checkpoints)} != {expected}")
    sources = sorted((WORKSPACE / "src" / "rs_cusum_phase2rh").glob("*.py"))
    tests = sorted((WORKSPACE / "tests").glob("test_phase2rh*.py"))
    final_files = [*required, *sources, *tests]
    final_payload = {
        "stage": "PHASE2RH_FINAL", "hash_algorithm": "SHA-256",
        "files": {str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path) for path in final_files},
        "M4_modified": False, "posthoc_tuning": False,
        "formal_scenario_count": 9, "formal_checkpoints": len(checkpoints),
        "historical_gates_unchanged": True,
    }
    FINAL_HASH.write_text(json.dumps(final_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate = json.loads((OUTPUT / "phase2rh_gate.json").read_text(encoding="utf-8"))["gate"]
    manifest_files = [*final_files, FINAL_HASH, *checkpoints]
    manifest = {
        "phase": "Phase 2R-H", "scope": "Transfer-Null and Null-Misspecification Validation",
        "gate": gate, "hash_algorithm": "SHA-256",
        "files": {str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path) for path in manifest_files},
        "formal_outputs_complete": True, "checkpoint_count": len(checkpoints),
        "M4_modified": False, "posthoc_tuning": False, "replacement_scenario_or_seed": False,
        "alternatives_power_changepoint_run": False, "OhioT1DM_run": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
