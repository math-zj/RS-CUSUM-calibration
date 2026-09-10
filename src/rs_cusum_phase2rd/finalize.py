"""Create the immutable Phase 2R-D final gate, report, and manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .protocol import (
    CONFIG,
    D0_HASH,
    OUTPUT,
    PRE_D1_HASH,
    PROTOCOL_JSON,
    PROTOCOL_MD,
    WORKSPACE,
    load_config,
    sha256,
)


def _row(frame: pd.DataFrame, arm: str, **selectors: str) -> pd.Series:
    selected = frame[frame["arm"] == arm]
    for key, value in selectors.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {arm} row for {selectors}, found {len(selected)}")
    return selected.iloc[0]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _stage_tables(prefix: str = "m4") -> dict[str, pd.DataFrame]:
    return {
        "size": pd.read_csv(OUTPUT / f"{prefix}_size_summary.csv"),
        "cov": pd.read_csv(OUTPUT / f"{prefix}_covariance_process.csv"),
        "tail": pd.read_csv(OUTPUT / f"{prefix}_tail_diagnostics.csv"),
        "joint": pd.read_csv(OUTPUT / f"{prefix}_joint_exceedance.csv"),
        "runtime": pd.read_csv(OUTPUT / f"{prefix}_runtime_summary.csv"),
    }


def _mechanism_lines(tables: dict[str, pd.DataFrame], arm: str) -> dict[str, Any]:
    size = tables["size"]
    cov = _row(tables["cov"], arm, scope="joint_448")
    tail = _row(tables["tail"], arm)
    joint = _row(tables["joint"], arm)
    runtime = _row(tables["runtime"], arm)
    layers = {
        layer: _row(size, arm, layer=layer) for layer in ("D0", "D1", "D2", "D3", "D4")
    }
    return {
        "size": {key: float(value["size_005"]) for key, value in layers.items()},
        "q95": {key: float(value["q95_ratio"]) for key, value in layers.items()},
        "d4_wilson": (
            float(layers["D4"]["wilson_low_005"]),
            float(layers["D4"]["wilson_high_005"]),
        ),
        "center": float(cov["conditional_mean_RMS_over_truth_SD"]),
        "sd": float(cov["pointwise_SD_ratio"]),
        "eigen": float(cov["leading_eigenvalue_ratio"]),
        "effective_rank": float(cov["effective_rank_ratio"]),
        "corr_fro": float(cov["correlation_frobenius_relative_error"]),
        "corr_mae": float(cov["offdiagonal_correlation_MAE"]),
        "d2_ks": float(tail["D2_KS_distance"]),
        "d4_ks": float(tail["D4_KS_distance"]),
        "q95_coordinate": float(tail["mean_coordinate_abs_q95_ratio"]),
        "q99_coordinate": float(tail["mean_coordinate_abs_q99_ratio"]),
        "skew": float(tail["skewness_RMSE"]),
        "kurtosis": float(tail["excess_kurtosis_RMSE"]),
        "max_exceed": float(tail["probability_bootstrap_max_exceeds_oracle_q95"]),
        "joint_fro": float(joint["joint_exceedance_frobenius_relative_error"]),
        "joint_mae": float(joint["joint_exceedance_offdiagonal_MAE"]),
        "tail_mae": float(joint["tail_dependence_offdiagonal_MAE"]),
        "tail_p95_error": float(joint["tail_dependence_p95_abs_error"]),
        "failure": float(runtime["draw_failure_rate"]),
    }


def _render_map(values: dict[str, float]) -> str:
    return ", ".join(f"{key}={_fmt(value)}" for key, value in values.items())


def write_final_gate() -> dict[str, Any]:
    d1 = json.loads((OUTPUT / "d1_gate.json").read_text(encoding="utf-8"))
    d2 = json.loads((OUTPUT / "d2_gate.json").read_text(encoding="utf-8"))
    if d1["gate"] != "D1_ENGINEERING_PASS":
        final = d1["gate"]
        d3_run = False
    elif not d2.get("D3_allowed", False):
        final = d2["gate"]
        d3_run = False
    else:
        path = OUTPUT / "m4_holdout_gate.json"
        if not path.is_file():
            raise RuntimeError("D2 passed, so the single locked D3 run must finish before finalization")
        d3 = json.loads(path.read_text(encoding="utf-8"))
        final = d3["gate"]
        d3_run = True
    payload = {
        "gate": final,
        "D1_gate": d1["gate"],
        "D2_gate": d2["gate"],
        "D3_run": d3_run,
        "establish_RS_CUSUM_RL_1_0_2": False,
        "phase3_allowed": False,
        "transfer_null_allowed": False,
        "ohiot1dm_allowed": False,
        "alternatives_power_changepoint_allowed": False,
        "posthoc_tuning": False,
    }
    (OUTPUT / "phase2rd_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def write_report(final_gate: dict[str, Any]) -> None:
    cfg = load_config("D2")
    d2_gate = json.loads((OUTPUT / "d2_gate.json").read_text(encoding="utf-8"))
    d2 = _mechanism_lines(_stage_tables(), "M4")
    d3 = None
    if final_gate["D3_run"]:
        d3 = _mechanism_lines(_stage_tables("m4_holdout"), "M4")
    design_pass = bool(d2_gate["domain_checks"]["center"] and d2_gate["domain_checks"]["covariance"])
    tail_pass = bool(d2_gate["domain_checks"]["joint_tail"])
    overall_pass = all(d2_gate["domain_checks"].values())
    holdout_text = (
        f"D3 ran once and its locked gate was `{final_gate['gate']}`."
        if d3 is not None
        else "D3 did not run because the frozen D2 mechanism gate did not grant eligibility."
    )
    lines = [
        "# RS-CUSUM-RL Phase 2R-D: M4 Random-Design Joint-Process Bootstrap",
        "",
        "## Scope and frozen protocol",
        "",
        "Phase 2R-D evaluated only the complete balanced stationary null N0. It did not run OhioT1DM, Phase 3, transfer-null, alternatives, power, p-value calibration patches, or changepoint analysis. D0 froze the algorithm, stage sizes, gates, seeds, and protocol before any M4 output; pre-D1 separately froze the implementation and tests.",
        "",
        "M4 fits one stationary parametric null model to each outer N0 dataset. Each inner draw independently regenerates initial states, patient policy random effects, actions, Markov state innovations, and reward innovations; it then rebuilds the panel, re-screens support, refits both FQI sides at all seven candidates, recomputes W/Delta/V, and returns the direct full 7x64 null process. Phenotype balance, N=12, T=240, analysis start 48, the frozen evaluation grid, estimator, gamma, Ridge penalty, support rule, and candidate fractions remain fixed.",
        "",
        "## Gates",
        "",
        f"- D1: `{final_gate['D1_gate']}`",
        f"- D2: `{final_gate['D2_gate']}`; domains = `{json.dumps(d2_gate['domain_checks'], sort_keys=True)}`",
        f"- D3: {holdout_text}",
        f"- Final Phase 2R-D gate: `{final_gate['gate']}`",
        "",
        "## D2 M4 numerical summary",
        "",
        f"- D0-D4 size at 0.05: {_render_map(d2['size'])}",
        f"- D0-D4 q95 ratios: {_render_map(d2['q95'])}",
        f"- D4 Wilson 95% interval: [{_fmt(d2['d4_wilson'][0])}, {_fmt(d2['d4_wilson'][1])}]",
        f"- Center RMS/truth SD={_fmt(d2['center'])}; pointwise SD ratio={_fmt(d2['sd'])}; leading-eigenvalue ratio={_fmt(d2['eigen'])}; effective-rank ratio={_fmt(d2['effective_rank'])}",
        f"- Correlation Frobenius relative error={_fmt(d2['corr_fro'])}; off-diagonal correlation MAE={_fmt(d2['corr_mae'])}",
        f"- D2/D4 KS={_fmt(d2['d2_ks'])}/{_fmt(d2['d4_ks'])}; coordinate q95/q99 ratios={_fmt(d2['q95_coordinate'])}/{_fmt(d2['q99_coordinate'])}",
        f"- Skewness/kurtosis RMSE={_fmt(d2['skew'])}/{_fmt(d2['kurtosis'])}",
        f"- Joint exceedance Frobenius error={_fmt(d2['joint_fro'])}; off-diagonal MAE={_fmt(d2['joint_mae'])}; tail-dependence MAE={_fmt(d2['tail_mae'])}; tail p95 error={_fmt(d2['tail_p95_error'])}",
        f"- P(bootstrap D4 max > oracle q95)={_fmt(d2['max_exceed'])}; draw failure rate={_fmt(d2['failure'])}",
        "",
        "## Required questions",
        "",
        "1. **Exact M4 definition.** Fitted-parametric stationary-null random-design full-pipeline bootstrap, exactly as frozen above and in `phase2rd_protocol.md`.",
        "2. **What M4 randomizes beyond M2.** M4 regenerates state/action/history design and reward noise; M2 fixes all realized states/actions and perturbs only a restricted-null pseudo reward.",
        "3. **Structural difference from M3.** M3 keeps the observed trajectory/reward and exchangeably reweights patients; M4 assigns no patient weights and simulates a new complete trajectory before a full unweighted refit.",
        f"4. **Between-design variation.** {'Recovered within the frozen center/covariance tolerances.' if design_pass else 'Not recovered within all frozen center/covariance tolerances.'}",
        f"5. **Conditional-mean distortion.** Center RMS/truth SD was {_fmt(d2['center'])}; the frozen center gate {'passed' if d2_gate['domain_checks']['center'] else 'failed'}.",
        f"6. **Pointwise SD ratio.** {_fmt(d2['sd'])}.",
        f"7. **Leading eigenvalue ratio.** {_fmt(d2['eigen'])}.",
        f"8. **D0-D4 empirical size.** {_render_map(d2['size'])}.",
        f"9. **D0-D4 q95 ratio.** {_render_map(d2['q95'])}.",
        f"10. **D4 Wilson interval.** [{_fmt(d2['d4_wilson'][0])}, {_fmt(d2['d4_wilson'][1])}].",
        f"11. **D2/D4 KS.** {_fmt(d2['d2_ks'])}/{_fmt(d2['d4_ks'])}.",
        f"12. **Coordinate q95/q99.** Mean absolute-quantile ratios were {_fmt(d2['q95_coordinate'])}/{_fmt(d2['q99_coordinate'])}.",
        f"13. **Skewness/kurtosis.** RMSE values were {_fmt(d2['skew'])}/{_fmt(d2['kurtosis'])}.",
        f"14. **Joint exceedance.** Frobenius/off-diagonal errors were {_fmt(d2['joint_fro'])}/{_fmt(d2['joint_mae'])}.",
        f"15. **Tail dependence.** Off-diagonal MAE/p95 error were {_fmt(d2['tail_mae'])}/{_fmt(d2['tail_p95_error'])}; the joint-tail domain {'passed' if tail_pass else 'failed'}.",
        f"16. **Covariance versus max distribution.** Covariance domain {'passed' if d2_gate['domain_checks']['covariance'] else 'failed'} and joint-tail domain {'passed' if tail_pass else 'failed'}; D4 KS={_fmt(d2['d4_ks'])} and max-exceedance probability={_fmt(d2['max_exceed'])}.",
        f"17. **Joint improvement over center+covariance+tail.** {'Yes under every frozen D2 domain.' if overall_pass else 'No; at least one frozen D2 domain failed.'}",
        f"18. **Draw failure rate.** {_fmt(d2['failure'])}.",
        f"19. **Mechanism screen.** `{final_gate['D2_gate']}`.",
        f"20. **Fresh locked holdout eligibility.** {'Granted by D2.' if d2_gate.get('D3_allowed') else 'Not granted by D2.'}",
        f"21. **Locked holdout result.** {holdout_text}",
        "22. **Post-hoc tuning.** None.",
        "23. **Previously viewed seeds.** None were reused; all D1/D2/D3 arrays were frozen and checked disjoint from prior seed registries before D1.",
        f"24. **Final Phase 2R-D gate.** `{final_gate['gate']}`.",
        "25. **Establish RS-CUSUM-RL 1.0.2.** NO.",
        "26. **Allow Phase 3.** NO.",
        "27. **Allow transfer-null.** NO.",
        "28. **Allow OhioT1DM.** NO.",
        "",
        "## Interpretation limits and allowed next step",
        "",
        "This is a computational null-validation study under one correctly specified synthetic N0 family, not a proof of bootstrap validity and not evidence for OhioT1DM. The final gate controls only whether the pre-registered D3 confirmation was allowed. Regardless of outcome, the frozen protocol forbids automatically upgrading the method or entering the excluded analyses in this phase.",
        "",
        "## Traceability",
        "",
        f"Parent method: `{cfg['protocol']['parent_method']}`. D0 and pre-D1 hashes are recorded in `{D0_HASH.name}` and `{PRE_D1_HASH.name}`. Exact input/output hashes are in `manifest.json`.",
    ]
    if d3 is not None:
        lines.extend(
            [
                "",
                "## Locked D3 numerical summary",
                "",
                f"D0-D4 size: {_render_map(d3['size'])}; D0-D4 q95: {_render_map(d3['q95'])}; center={_fmt(d3['center'])}; SD ratio={_fmt(d3['sd'])}; leading eigen ratio={_fmt(d3['eigen'])}; D4 KS={_fmt(d3['d4_ks'])}; failure rate={_fmt(d3['failure'])}.",
            ]
        )
    (WORKSPACE / cfg["outputs"]["report"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest() -> dict[str, Any]:
    report = WORKSPACE / "report" / "rs_cusum_phase2rd_report.md"
    source = sorted((WORKSPACE / "src" / "rs_cusum_phase2rd").glob("*.py"))
    tests = sorted((WORKSPACE / "tests").glob("test_phase2rd*.py"))
    results = sorted(
        path for path in OUTPUT.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    files = [CONFIG, PROTOCOL_MD, PROTOCOL_JSON, report, *source, *tests, *results]
    payload = {
        "phase": "Phase 2R-D",
        "hash_algorithm": "SHA-256",
        "posthoc_tuning": False,
        "files": {
            str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    final = write_final_gate()
    write_report(final)
    manifest = write_manifest()
    print(json.dumps({"gate": final["gate"], "manifest_files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
