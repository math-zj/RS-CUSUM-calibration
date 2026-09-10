"""Existing-draw pairwise tail forensics and Monte Carlo uncertainty."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .protocol import OUTPUT, WORKSPACE, load_config, load_seeds


def coordinate_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    grid_path = WORKSPACE / cfg["existing_draw_forensics"]["evaluation_grid"]
    with np.load(grid_path, allow_pickle=False) as grid:
        states = np.asarray(grid["states"], float)
        actions = np.asarray(grid["actions"], int)
        patients = np.asarray(grid["source_patient"], str)
        elapsed = np.asarray(grid["source_elapsed"], int)
    candidates = np.asarray(cfg["existing_draw_forensics"]["candidates"], int)
    rows = []
    for c_index, candidate in enumerate(candidates):
        for z in range(len(actions)):
            rows.append({
                "flat_index": c_index * len(actions) + z,
                "candidate_index": c_index,
                "candidate": int(candidate),
                "evaluation_index": z,
                "base_state_index": z % 32,
                "value_action": int(actions[z]),
                "value_component": f"Q_action_{int(actions[z])}",
                "phenotype": float(states[z, 1]),
                "phenotype_class": "negative" if states[z, 1] < 0 else "positive",
                "source_patient": patients[z],
                "source_elapsed": int(elapsed[z]),
                "state_l2_norm": float(np.linalg.norm(states[z])),
                "state_x0": float(states[z, 0]),
            })
    return pd.DataFrame(rows)


def load_existing(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = WORKSPACE / cfg["existing_draw_forensics"]["input"]
    with np.load(path, allow_pickle=False) as data:
        truth = np.asarray(data[cfg["existing_draw_forensics"]["truth_key"]], float)
        clustered = np.asarray(data[cfg["existing_draw_forensics"]["m4_key"]], float)
    m4 = clustered.reshape(-1, clustered.shape[-1])
    if truth.shape != (120, 448) or clustered.shape != (120, 100, 448):
        raise RuntimeError(f"unexpected frozen process shapes: {truth.shape}, {clustered.shape}")
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(clustered)):
        raise FloatingPointError("non-finite frozen process draw")
    return truth, m4, clustered


def _events(values: np.ndarray, threshold: np.ndarray, event_type: str) -> np.ndarray:
    if event_type == "absolute":
        return np.abs(values) > threshold
    if event_type == "upper":
        return values > threshold
    raise ValueError(event_type)


def _tail_matrices(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(events, np.float32)
    marginal = np.mean(x, axis=0, dtype=np.float64)
    joint = (x.T @ x).astype(np.float64) / len(x)
    conditional = joint / np.maximum(marginal[:, None], 1e-15)
    return marginal, joint, conditional


def _spearman(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=0, method="average")
    return np.corrcoef(ranks, rowvar=False)


def _distance_bin(values: np.ndarray, kind: str) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    if kind == "eval":
        bounds = [(0, 0, "0"), (1, 3, "1-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, 31, "16-31"), (32, 63, "32-63")]
    else:
        bounds = [(1, 7, "1-7"), (8, 31, "8-31"), (32, 63, "32-63"), (64, 127, "64-127"), (128, 255, "128-255"), (256, 10**9, "256+")]
    for low, high, label in bounds:
        selected = (values >= low) & (values <= high)
        result[selected] = label
    return result


def _pair_table(
    metadata: pd.DataFrame,
    truth_joint: np.ndarray,
    m4_joint: np.ndarray,
    truth_cond: np.ndarray,
    m4_cond: np.ndarray,
    truth_rank: np.ndarray,
    m4_rank: np.ndarray,
) -> pd.DataFrame:
    dimension = len(metadata)
    i, j = np.where(~np.eye(dimension, dtype=bool))
    meta = metadata.set_index("flat_index")
    ci = meta.candidate_index.to_numpy()[i]; cj = meta.candidate_index.to_numpy()[j]
    ei = meta.evaluation_index.to_numpy()[i]; ej = meta.evaluation_index.to_numpy()[j]
    bi = meta.base_state_index.to_numpy()[i]; bj = meta.base_state_index.to_numpy()[j]
    ai = meta.value_action.to_numpy()[i]; aj = meta.value_action.to_numpy()[j]
    pi = meta.phenotype.to_numpy()[i]; pj = meta.phenotype.to_numpy()[j]
    spi = meta.source_patient.to_numpy()[i]; spj = meta.source_patient.to_numpy()[j]
    signed = m4_cond[i, j] - truth_cond[i, j]
    rank_signed = m4_rank[i, j] - truth_rank[i, j]
    eval_distance = np.abs(ei - ej); global_distance = np.abs(i - j)
    table = pd.DataFrame({
        "flat_i": i, "flat_j": j,
        "candidate_index_i": ci, "candidate_index_j": cj,
        "candidate_i": meta.candidate.to_numpy()[i], "candidate_j": meta.candidate.to_numpy()[j],
        "candidate_distance": np.abs(ci - cj),
        "evaluation_i": ei, "evaluation_j": ej,
        "evaluation_distance": eval_distance,
        "evaluation_distance_bin": _distance_bin(eval_distance, "eval"),
        "global_distance": global_distance,
        "global_distance_bin": _distance_bin(global_distance, "global"),
        "base_state_i": bi, "base_state_j": bj,
        "value_action_i": ai, "value_action_j": aj,
        "phenotype_i": pi, "phenotype_j": pj,
        "source_patient_i": spi, "source_patient_j": spj,
        "same_candidate": ci == cj,
        "same_evaluation_coordinate": ei == ej,
        "same_base_state": bi == bj,
        "same_value_action": ai == aj,
        "same_phenotype": pi == pj,
        "same_source_patient": spi == spj,
        "truth_joint_probability": truth_joint[i, j],
        "m4_joint_probability": m4_joint[i, j],
        "joint_signed_error": m4_joint[i, j] - truth_joint[i, j],
        "truth_conditional_probability": truth_cond[i, j],
        "m4_conditional_probability": m4_cond[i, j],
        "conditional_signed_error": signed,
        "conditional_absolute_error": np.abs(signed),
        "truth_spearman": truth_rank[i, j],
        "m4_spearman": m4_rank[i, j],
        "spearman_signed_error": rank_signed,
        "spearman_absolute_error": np.abs(rank_signed),
    })
    table["value_action_pair"] = table.value_action_i.astype(str) + "->" + table.value_action_j.astype(str)
    table["phenotype_pair"] = table.phenotype_i.astype(int).astype(str) + "->" + table.phenotype_j.astype(int).astype(str)
    table["structural_group"] = np.select(
        [
            table.same_candidate & table.same_base_state & ~table.same_value_action,
            ~table.same_candidate & table.same_evaluation_coordinate,
            table.same_candidate & table.same_value_action,
            table.same_candidate & ~table.same_value_action,
            ~table.same_candidate & table.same_value_action,
        ],
        [
            "same_candidate_same_state_cross_action",
            "cross_candidate_same_evaluation",
            "same_candidate_same_action",
            "same_candidate_cross_action",
            "cross_candidate_same_action",
        ],
        default="cross_candidate_cross_action",
    )
    return table


def _distribution_row(values: np.ndarray, prefix: str) -> dict[str, float]:
    quantiles = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(("min", "p25", "median", "p75", "p90", "p95", "p99", "max"), quantiles)
    }


def _structure_summary(pair: pd.DataFrame) -> pd.DataFrame:
    groupings = {
        "same_candidate": "same_candidate",
        "candidate_distance": "candidate_distance",
        "same_evaluation_coordinate": "same_evaluation_coordinate",
        "same_base_state": "same_base_state",
        "same_value_action": "same_value_action",
        "value_action_pair": "value_action_pair",
        "same_phenotype": "same_phenotype",
        "phenotype_pair": "phenotype_pair",
        "same_source_patient": "same_source_patient",
        "evaluation_distance_bin": "evaluation_distance_bin",
        "global_distance_bin": "global_distance_bin",
        "structural_group": "structural_group",
    }
    total_abs = float(pair.conditional_absolute_error.sum())
    rows = []
    for group_type, column in groupings.items():
        for value, frame in pair.groupby(column, observed=True, sort=True):
            signed = frame.conditional_signed_error.to_numpy(); absolute = np.abs(signed)
            row = {
                "group_type": group_type, "group_value": str(value), "pairs": len(frame),
                "signed_mean": float(np.mean(signed)), "signed_median": float(np.median(signed)),
                "under_fraction": float(np.mean(signed < 0)), "over_fraction": float(np.mean(signed > 0)),
                "absolute_mean": float(np.mean(absolute)), "proportion_abs_gt_0_10": float(np.mean(absolute > 0.10)),
                "share_total_absolute_error": float(np.sum(absolute) / total_abs),
                "joint_probability_MAE": float(np.mean(np.abs(frame.joint_signed_error))),
                "spearman_MAE": float(np.mean(frame.spearman_absolute_error)),
            }
            row.update(_distribution_row(absolute, "absolute")); rows.append(row)
    return pd.DataFrame(rows)


def _threshold_row(
    truth: np.ndarray, m4: np.ndarray, quantile: float, event_type: str,
    truth_rank: np.ndarray, m4_rank: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    transformed_truth = np.abs(truth) if event_type == "absolute" else truth
    transformed_m4 = np.abs(m4) if event_type == "absolute" else m4
    threshold = np.quantile(transformed_truth, quantile, axis=0)
    te = _events(truth, threshold, event_type); be = _events(m4, threshold, event_type)
    tm, tj, tc = _tail_matrices(te); bm, bj, bc = _tail_matrices(be)
    mask = ~np.eye(truth.shape[1], dtype=bool); signed = bc[mask] - tc[mask]; absolute = np.abs(signed)
    truth_counts = te.sum(axis=1); m4_counts = be.sum(axis=1)
    coordinate_quantile = np.quantile(transformed_m4, quantile, axis=0) / np.maximum(np.quantile(transformed_truth, quantile, axis=0), 1e-15)
    row = {
        "threshold": quantile, "event_type": event_type,
        "oracle_draws": len(truth), "m4_draws": len(m4),
        "oracle_mean_marginal_exceedance": float(np.mean(tm)),
        "m4_mean_marginal_exceedance": float(np.mean(bm)),
        "marginal_probability_MAE": float(np.mean(np.abs(bm - tm))),
        "mean_coordinate_quantile_ratio": float(np.mean(coordinate_quantile)),
        "joint_probability_MAE": float(np.mean(np.abs(bj[mask] - tj[mask]))),
        "conditional_signed_mean": float(np.mean(signed)),
        "conditional_signed_median": float(np.median(signed)),
        "conditional_under_fraction": float(np.mean(signed < 0)),
        "conditional_over_fraction": float(np.mean(signed > 0)),
        "conditional_absolute_mean": float(np.mean(absolute)),
        "conditional_absolute_median": float(np.median(absolute)),
        "conditional_absolute_p90": float(np.quantile(absolute, 0.90)),
        "conditional_absolute_p95": float(np.quantile(absolute, 0.95)),
        "conditional_absolute_p99": float(np.quantile(absolute, 0.99)),
        "conditional_absolute_max": float(np.max(absolute)),
        "oracle_conditional_distribution_p95": float(np.quantile(tc[mask], 0.95)),
        "m4_conditional_distribution_p95": float(np.quantile(bc[mask], 0.95)),
        "conditional_distribution_p95_abs_gap": float(abs(np.quantile(bc[mask], 0.95) - np.quantile(tc[mask], 0.95))),
        "spearman_MAE": float(np.mean(np.abs(m4_rank[mask] - truth_rank[mask]))),
        "oracle_extreme_coordinates_mean": float(np.mean(truth_counts)),
        "m4_extreme_coordinates_mean": float(np.mean(m4_counts)),
        "oracle_extreme_coordinates_p95": float(np.quantile(truth_counts, 0.95)),
        "m4_extreme_coordinates_p95": float(np.quantile(m4_counts, 0.95)),
        "oracle_fraction_with_multiple_extremes": float(np.mean(truth_counts >= 2)),
        "m4_fraction_with_multiple_extremes": float(np.mean(m4_counts >= 2)),
    }
    distribution = {"threshold": quantile, "event_type": event_type, "pairs": int(mask.sum())}
    distribution.update(_distribution_row(signed, "signed")); distribution.update(_distribution_row(absolute, "absolute"))
    distribution.update({
        "proportion_abs_gt_0_05": float(np.mean(absolute > 0.05)),
        "proportion_abs_gt_0_10": float(np.mean(absolute > 0.10)),
        "proportion_abs_gt_0_15": float(np.mean(absolute > 0.15)),
        "proportion_abs_gt_0_20": float(np.mean(absolute > 0.20)),
        "under_fraction": float(np.mean(signed < 0)), "over_fraction": float(np.mean(signed > 0)),
    })
    order = np.sort(absolute)[::-1]; top = max(1, int(np.ceil(0.05 * len(order))))
    distribution["top_5pct_share_total_absolute_error"] = float(np.sum(order[:top]) / np.sum(order))
    return row, distribution


def _conditional_for_pairs(events: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    numerator = np.sum(events[:, pair_i] & events[:, pair_j], axis=0, dtype=np.int64)
    denominator = np.sum(events[:, pair_i], axis=0, dtype=np.int64)
    return numerator / np.maximum(denominator, 1)


def _mc_uncertainty(
    truth: np.ndarray, clustered: np.ndarray, threshold: np.ndarray,
    pair_i: np.ndarray, pair_j: np.ndarray, cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = np.abs(truth) > threshold
    be = np.abs(clustered) > threshold
    p = len(pair_i); n_truth = len(te); n_outer, n_inner, _ = be.shape
    truth_product = (te[:, pair_i] & te[:, pair_j]).astype(np.float32)
    truth_events = te.astype(np.float32)
    boot_numerator = np.empty((n_outer, p), np.float32)
    boot_marginal = np.sum(be, axis=1, dtype=np.int16).astype(np.float32)
    for outer in range(n_outer):
        current = be[outer]
        boot_numerator[outer] = np.sum(current[:, pair_i] & current[:, pair_j], axis=0)
    fixed_truth = _conditional_for_pairs(te, pair_i, pair_j)
    fixed_boot = np.sum(boot_numerator, axis=0) / np.maximum(np.sum(boot_marginal, axis=0)[pair_i], 1)
    fixed_truth_p95 = float(np.quantile(fixed_truth, 0.95)); fixed_boot_p95 = float(np.quantile(fixed_boot, 0.95))
    point = abs(fixed_boot_p95 - fixed_truth_p95)
    seeds = load_seeds("mc_bootstrap"); b = len(seeds)
    wt = np.empty((b, n_truth), np.float32); wb = np.empty((b, n_outer), np.float32)
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        wt[index] = rng.multinomial(n_truth, np.full(n_truth, 1.0 / n_truth))
        wb[index] = rng.multinomial(n_outer, np.full(n_outer, 1.0 / n_outer))
    truth_num = wt @ truth_product
    truth_den = (wt @ truth_events)[:, pair_i]
    boot_num = wb @ boot_numerator
    boot_den = (wb @ boot_marginal)[:, pair_i]
    truth_cond = truth_num / np.maximum(truth_den, 1)
    boot_cond = boot_num / np.maximum(boot_den, 1)
    truth_p95 = np.quantile(truth_cond, 0.95, axis=1)
    boot_p95 = np.quantile(boot_cond, 0.95, axis=1)
    metrics = {
        "oracle_rows_only": np.abs(fixed_boot_p95 - truth_p95),
        "M4_outer_clusters_only": np.abs(boot_p95 - fixed_truth_p95),
        "both": np.abs(boot_p95 - truth_p95),
    }
    rows = []
    for scheme, values in metrics.items():
        rows.append({
            "metric": "conditional_distribution_p95_abs_gap", "scheme": scheme,
            "point_estimate_full_pairs": np.nan, "point_estimate_pair_sample": point,
            "replicates": len(values), "mc_standard_error": float(np.std(values, ddof=1)),
            "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
            "mean": float(np.mean(values)), "median": float(np.median(values)),
        })
    # Nested sample-size stability. Inner-draw stability uses the frozen draw order.
    stability = []
    sub_seeds = load_seeds("subsample")
    fixed_boot_p95 = float(np.quantile(fixed_boot, 0.95))
    for size in cfg["mc_uncertainty"]["oracle_subsample_sizes"]:
        values = []
        for seed in sub_seeds:
            selected = np.random.default_rng(int(seed)).choice(n_truth, int(size), replace=False)
            cond = _conditional_for_pairs(te[selected], pair_i, pair_j)
            values.append(abs(fixed_boot_p95 - np.quantile(cond, 0.95)))
        stability.append(_stability_row("oracle_draws", int(size), values))
    for size in cfg["mc_uncertainty"]["m4_outer_subsample_sizes"]:
        values = []
        for seed in sub_seeds:
            selected = np.random.default_rng(int(seed)).choice(n_outer, int(size), replace=False)
            cond = np.sum(boot_numerator[selected], axis=0) / np.maximum(np.sum(boot_marginal[selected], axis=0)[pair_i], 1)
            values.append(abs(np.quantile(cond, 0.95) - fixed_truth_p95))
        stability.append(_stability_row("M4_outer_clusters", int(size), values))
    for size in cfg["mc_uncertainty"]["m4_nested_draws_per_outer"]:
        selected = be[:, : int(size)]
        cond = _conditional_for_pairs(selected.reshape(-1, selected.shape[-1]), pair_i, pair_j)
        stability.append(_stability_row("M4_inner_draws_per_outer", int(size), [abs(np.quantile(cond, 0.95) - fixed_truth_p95)]))
    return pd.DataFrame(rows), pd.DataFrame(stability)


def _stability_row(axis: str, size: int, values: list[float]) -> dict[str, Any]:
    values_array = np.asarray(values, float)
    return {
        "subsample_axis": axis, "size": size, "replicates": len(values_array),
        "mean_p95_gap": float(np.mean(values_array)), "median_p95_gap": float(np.median(values_array)),
        "sd_p95_gap": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else np.nan,
        "p025_p95_gap": float(np.quantile(values_array, 0.025)),
        "p975_p95_gap": float(np.quantile(values_array, 0.975)),
        "min_p95_gap": float(np.min(values_array)), "max_p95_gap": float(np.max(values_array)),
    }


def analyze_existing() -> dict[str, Any]:
    cfg = load_config(); OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = coordinate_metadata(cfg); metadata.to_csv(OUTPUT / "coordinate_metadata.csv", index=False)
    truth, m4, clustered = load_existing(cfg)
    truth_rank = _spearman(truth); m4_rank = _spearman(m4)
    threshold_rows = []; distribution_rows = []; primary = None
    for quantile in cfg["existing_draw_forensics"]["thresholds"]:
        for event_type in ("absolute", "upper"):
            row, distribution = _threshold_row(truth, m4, float(quantile), event_type, truth_rank, m4_rank)
            threshold_rows.append(row); distribution_rows.append(distribution)
            if float(quantile) == 0.95 and event_type == "absolute":
                threshold = np.quantile(np.abs(truth), quantile, axis=0)
                tm, tj, tc = _tail_matrices(np.abs(truth) > threshold)
                bm, bj, bc = _tail_matrices(np.abs(m4) > threshold)
                primary = (threshold, tj, bj, tc, bc)
    pd.DataFrame(threshold_rows).to_csv(OUTPUT / "threshold_sensitivity.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUTPUT / "tail_error_distribution.csv", index=False)
    if primary is None:
        raise RuntimeError("primary threshold was not evaluated")
    threshold, tj, bj, tc, bc = primary
    pairs = _pair_table(metadata, tj, bj, tc, bc, truth_rank, m4_rank)
    pairs.to_csv(OUTPUT / "pairwise_error_heatmap.csv", index=False)
    _structure_summary(pairs).to_csv(OUTPUT / "tail_error_by_structure.csv", index=False)
    top = pairs.nlargest(int(cfg["existing_draw_forensics"]["top_pairs_saved"]), "conditional_absolute_error").copy()
    top.to_csv(OUTPUT / "top_offending_pairs.csv", index=False)
    all_i, all_j = np.where(~np.eye(truth.shape[1], dtype=bool))
    rng = np.random.default_rng(int(load_seeds("mc_pair_sample")[0]))
    selected = rng.choice(len(all_i), int(cfg["mc_uncertainty"]["pair_sample_size"]), replace=False)
    mc, stability = _mc_uncertainty(truth, clustered, threshold, all_i[selected], all_j[selected], cfg)
    full_gap = float(abs(np.quantile(bc[~np.eye(448, dtype=bool)], 0.95) - np.quantile(tc[~np.eye(448, dtype=bool)], 0.95)))
    mc["point_estimate_full_pairs"] = full_gap
    mc.to_csv(OUTPUT / "mc_uncertainty.csv", index=False)
    stability.to_csv(OUTPUT / "subsample_stability.csv", index=False)
    payload = {
        "truth_draws": len(truth), "m4_draws": len(m4), "ordered_pairs": len(pairs),
        "primary_p95_distribution_gap": full_gap,
        "primary_pair_sample_size": int(cfg["mc_uncertainty"]["pair_sample_size"]),
    }
    (OUTPUT / "existing_forensics_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(analyze_existing(), sort_keys=True))
