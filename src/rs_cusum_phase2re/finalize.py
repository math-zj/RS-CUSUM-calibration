"""Final report and manifest for Phase 2R-E."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .protocol import (
    CONFIG, OUTPUT, PRE_HYBRID_HASH, PROTOCOL_HASH, PROTOCOL_JSON, PROTOCOL_MD,
    WORKSPACE, load_config, sha256,
)


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _future_direction(best_arm: str, gate: str) -> str:
    if gate == "TAIL_METRIC_UNSTABLE":
        return "Before defining M5, increase independent oracle-process replication and replace the raw p95-of-conditional-probabilities diagnostic with a pre-registered uncertainty-aware tail functional. A method change is not justified by the unstable metric alone."
    if best_arm in {"true_policy", "true_random_effect"}:
        return "A future M5 should preserve patient-level latent policy heterogeneity semiparametrically, for example by resampling patient random effects or whole patient policy histories before full trajectory regeneration."
    if best_arm in {"true_transition", "true_reward", "true_innovation_scales"}:
        return "A future M5 should use joint residual-vector or trajectory-block innovation resampling, retaining cross-coordinate and temporal residual structure rather than independently drawing Gaussian innovations."
    if best_arm in {"true_initial", "true_design", "true_structural_coefficients"}:
        return "A future M5 should use a semiparametric whole-design/trajectory bootstrap that preserves the empirical initial-state and dynamic design law."
    return "A future method requires a new protocol; the present evidence does not identify a unique resampling construction."


def write_report() -> dict[str, Any]:
    cfg = load_config(); gate = json.loads((OUTPUT / "phase2re_gate.json").read_text(encoding="utf-8"))
    threshold = pd.read_csv(OUTPUT / "threshold_sensitivity.csv"); distribution = pd.read_csv(OUTPUT / "tail_error_distribution.csv")
    structure = pd.read_csv(OUTPUT / "tail_error_by_structure.csv"); top = pd.read_csv(OUTPUT / "top_offending_pairs.csv")
    mc = pd.read_csv(OUTPUT / "mc_uncertainty.csv"); stability = pd.read_csv(OUTPUT / "subsample_stability.csv")
    hybrid = pd.read_csv(OUTPUT / "hybrid_component_diagnostics.csv"); fit = pd.read_csv(OUTPUT / "component_fit_diagnostics.csv")
    primary = threshold[(threshold.threshold == 0.95) & (threshold.event_type == "absolute")].iloc[0]
    dist = distribution[(distribution.threshold == 0.95) & (distribution.event_type == "absolute")].iloc[0]
    combined = mc[mc.scheme == "both"].iloc[0]
    rules = cfg["localization_rules"]
    widespread = bool(dist.proportion_abs_gt_0_10 >= 0.25 or dist.top_5pct_share_total_absolute_error < 0.30)
    sparse = bool(dist.proportion_abs_gt_0_10 < 0.10 and dist.top_5pct_share_total_absolute_error >= 0.40)
    shape = "widespread" if widespread else "sparse-extreme" if sparse else "mixed/concentrated"
    if dist.under_fraction >= rules["direction_under_if_fraction_negative_at_least"]:
        direction = "predominantly under-dependence"
    elif dist.over_fraction >= rules["direction_over_if_fraction_positive_at_least"]:
        direction = "predominantly over-dependence"
    else:
        direction = "bidirectional"
    abs_rows = threshold[threshold.event_type == "absolute"].sort_values("threshold")
    gaps = abs_rows.conditional_distribution_p95_abs_gap.to_numpy()
    worsens = bool(gaps[-1] - gaps[0] >= 0.05 and np.sum(np.diff(gaps) >= 0) >= 3)
    q95 = abs_rows[abs_rows.threshold == 0.95].iloc[0]; q99 = abs_rows[abs_rows.threshold == 0.99].iloc[0]
    marginal_ok = bool(
        max(q95.marginal_probability_MAE, q99.marginal_probability_MAE) <= rules["marginal_tail_basic_correct_if"]["mean_absolute_marginal_probability_error_max"]
        and all(rules["marginal_tail_basic_correct_if"]["mean_abs_quantile_ratio_range"][0] <= value <= rules["marginal_tail_basic_correct_if"]["mean_abs_quantile_ratio_range"][1] for value in (q95.mean_coordinate_quantile_ratio, q99.mean_coordinate_quantile_ratio))
    )
    best_groups = structure[structure.group_type.isin(["candidate_distance", "value_action_pair", "structural_group"])].nlargest(8, "absolute_p95")
    fitted = hybrid[hybrid.arm == "fitted_all"].iloc[0]
    singles = hybrid[hybrid.arm.isin(["true_initial", "true_policy", "true_random_effect", "true_transition", "true_reward"])].sort_values("p95_error_reduction_vs_fitted_all", ascending=False)
    best = singles.iloc[0]
    future = _future_direction(str(best.arm), gate["gate"])
    fit_re = fit[(fit.diagnostic_type == "parameter") & (fit.component == "action_random_effect_sd")]
    lines = [
        "# Phase 2R-E Joint-Tail Failure Localization Report", "",
        "## Immutable starting point", "",
        "Phase 2R-D remains `M4_MECHANISM_JOINT_TAIL_FAIL`. Phase 2R-E performed mechanism diagnostics only. It did not modify M4, revise the old gate, run D3, or implement M5.", "",
        "## Existing D2 pairwise decomposition", "",
        f"The frozen 0.1418 diagnostic is a difference between distributional p95 values: oracle={_fmt(primary.oracle_conditional_distribution_p95)}, M4={_fmt(primary.m4_conditional_distribution_p95)}, absolute gap={_fmt(primary.conditional_distribution_p95_abs_gap)}. It is not the p95 of pairwise absolute errors, which is {_fmt(primary.conditional_absolute_p95)}.",
        f"Pair-error shape is **{shape}**. Median/p75/p90/p95/p99/max absolute pair errors are {_fmt(dist.absolute_median)}, {_fmt(dist.absolute_p75)}, {_fmt(dist.absolute_p90)}, {_fmt(dist.absolute_p95)}, {_fmt(dist.absolute_p99)}, {_fmt(dist.absolute_max)}. The fraction above 0.10 is {_fmt(dist.proportion_abs_gt_0_10)} and the top 5% contribute {_fmt(dist.top_5pct_share_total_absolute_error)} of total absolute error.",
        f"Direction is **{direction}**: under/over fractions are {_fmt(dist.under_fraction)}/{_fmt(dist.over_fraction)}.", "",
        "Most severe frozen structural groups:", "",
    ]
    for _, row in best_groups.iterrows():
        lines.append(f"- {row.group_type}={row.group_value}: pairs={int(row.pairs)}, absolute p95={_fmt(row.absolute_p95)}, mean={_fmt(row.absolute_mean)}, under fraction={_fmt(row.under_fraction)}")
    first = top.iloc[0]
    lines.extend([
        "", f"Largest individual pair error links candidate/evaluation ({int(first.candidate_i)}, {int(first.evaluation_i)}) to ({int(first.candidate_j)}, {int(first.evaluation_j)}): oracle conditional={_fmt(first.truth_conditional_probability)}, M4={_fmt(first.m4_conditional_probability)}, absolute error={_fmt(first.conditional_absolute_error)}.", "",
        "## Margins, thresholds, and Monte Carlo uncertainty", "",
        f"Marginal-tail calibration is {'basically correct under the frozen rule' if marginal_ok else 'not fully correct under the frozen rule'}. At 95%/99%, marginal probability MAE is {_fmt(q95.marginal_probability_MAE)}/{_fmt(q99.marginal_probability_MAE)} and coordinate quantile ratios are {_fmt(q95.mean_coordinate_quantile_ratio)}/{_fmt(q99.mean_coordinate_quantile_ratio)}.",
        "Absolute-tail conditional-distribution p95 gaps by threshold: " + ", ".join(f"{100*row.threshold:g}%={_fmt(row.conditional_distribution_p95_abs_gap)}" for _, row in abs_rows.iterrows()) + f". The frozen worsening rule is {'met' if worsens else 'not met'}.",
        f"The combined outer-resampling uncertainty estimate has MC SE={_fmt(combined.mc_standard_error)}, 95% interval=[{_fmt(combined.ci_low)}, {_fmt(combined.ci_high)}], width={_fmt(combined.ci_high-combined.ci_low)}. Maximum median gap range across nested sample-size axes is {_fmt(gate['max_subsample_median_range'])}.",
        f"The resulting stability checks are `{json.dumps(gate['mc_instability_checks'], sort_keys=True)}`.", "",
        "## Hybrid source decomposition", "",
        f"The internally replicated fitted-all versus true-all p95 error is {_fmt(fitted.tail_dependence_p95_abs_error)}. The strongest single truth replacement is `{best.arm}`: error={_fmt(best.tail_dependence_p95_abs_error)}, reduction={_fmt(best.p95_error_reduction_vs_fitted_all)}.",
        "", "All hybrid arms:", "",
    ])
    for _, row in hybrid.sort_values("p95_error_reduction_vs_fitted_all", ascending=False).iterrows():
        lines.append(f"- {row.arm}: p95 error={_fmt(row.tail_dependence_p95_abs_error)}, reduction={_fmt(row.p95_error_reduction_vs_fitted_all)}, tail MAE={_fmt(row.tail_dependence_offdiagonal_MAE)}, SD ratio={_fmt(row.pointwise_SD_ratio)}, D4 KS={_fmt(row.D4_KS)}")
    if len(fit_re):
        row = fit_re.iloc[0]
        lines.extend(["", f"Patient policy random-effect SD: mean fitted={_fmt(row.estimate_mean)}, truth={_fmt(row.truth)}, bias={_fmt(row.bias)}, RMSE={_fmt(row.rmse)}."])
    lines.extend([
        "", "## Required answers", "",
        f"1. **Widespread or few pairs?** {shape}; the quantitative concentration measures are reported above.",
        f"2. **Over- or under-dependence?** {direction}.",
        f"3. **Worst coordinate pairs?** The leading structural groups are listed above; full pair metadata are in `top_offending_pairs.csv` and `pairwise_error_heatmap.csv`.",
        f"4. **Does error worsen with threshold?** {'Yes' if worsens else 'No under the frozen monotonic-worsening rule'}.",
        f"5. **Are marginal tails basically correct?** {'Yes' if marginal_ok else 'No'} under the pre-registered marginal rule.",
        f"6. **MC uncertainty?** SE={_fmt(combined.mc_standard_error)}, interval=[{_fmt(combined.ci_low)}, {_fmt(combined.ci_high)}].",
        f"7. **Is the p95 error stable?** {'No' if gate['gate']=='TAIL_METRIC_UNSTABLE' else 'Yes under the frozen stability checks'}.",
        f"8. **Most likely component?** `{best.arm}` among single-component truth replacements, with reduction {_fmt(best.p95_error_reduction_vs_fitted_all)}; this attribution is subordinate to the stability gate.",
        f"9. **Uniquely localized by hybrids?** {'No, because the primary metric is unstable.' if gate['gate']=='TAIL_METRIC_UNSTABLE' else 'See the localization gate and arm separation above.'}",
        f"10. **Final localization gate.** `{gate['gate']}`.",
        f"11. **Most defensible future M5 direction.** {future} Phase 2R-E did not implement it.",
        "12. **Are D3 / Ohio / Phase 3 still prohibited?** YES. Transfer-null and method upgrade are also prohibited.", "",
        "## Interpretation", "",
        "Hybrid arms are interventions on a known synthetic DGP, not candidate inference procedures. No size-based selection was performed. Even a clear component rescue would only motivate a separately pre-registered future method stage.",
    ])
    report = WORKSPACE / cfg["outputs"]["report"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"gate": gate["gate"], "shape": shape, "direction": direction, "best_arm": str(best.arm), "marginal_ok": marginal_ok, "worsens": worsens}


def write_manifest() -> dict[str, Any]:
    cfg = load_config(); report = WORKSPACE / cfg["outputs"]["report"]
    sources = sorted((WORKSPACE / "src" / "rs_cusum_phase2re").glob("*.py")); tests = sorted((WORKSPACE / "tests").glob("test_phase2re*.py"))
    results = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "manifest.json")
    files = [CONFIG, PROTOCOL_MD, PROTOCOL_JSON, report, *sources, *tests, *results]
    payload = {
        "phase": "Phase 2R-E", "hash_algorithm": "SHA-256",
        "phase2rd_gate_unchanged": "M4_MECHANISM_JOINT_TAIL_FAIL", "M4_modified": False,
        "files": {str(path.relative_to(WORKSPACE)).replace("\\", "/"): sha256(path) for path in files},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    summary = write_report(); manifest = write_manifest(); print(json.dumps({**summary, "manifest_files": len(manifest["files"])}, sort_keys=True))

