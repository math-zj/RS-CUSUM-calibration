"""Pre-registered Phase 2R-F joint-tail and multi-domain diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _off_diagonal(dimension: int) -> np.ndarray:
    return ~np.eye(dimension, dtype=bool)


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, float).ravel())
    right = np.sort(np.asarray(right, float).ravel())
    values = np.sort(np.unique(np.r_[left, right]))
    return float(
        np.max(
            np.abs(
                np.searchsorted(left, values, side="right") / len(left)
                - np.searchsorted(right, values, side="right") / len(right)
            )
        )
    )


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def tail_matrices(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event = np.asarray(events, np.float32)
    marginal = np.mean(event, axis=0, dtype=np.float64)
    joint = np.asarray(event.T @ event, np.float64) / len(event)
    conditional = joint / np.maximum(marginal[:, None], 1e-15)
    return marginal, joint, conditional


def threshold_tail_metrics(
    reference: np.ndarray,
    comparison: np.ndarray,
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    reference = np.asarray(reference, np.float32)
    comparison = np.asarray(comparison, np.float32)
    mask = _off_diagonal(reference.shape[1])
    rows: list[dict[str, float]] = []
    for q in thresholds:
        coordinate_threshold = np.quantile(np.abs(reference), q, axis=0)
        ref_event = np.abs(reference) > coordinate_threshold
        cmp_event = np.abs(comparison) > coordinate_threshold
        ref_marginal, ref_joint, ref_conditional = tail_matrices(ref_event)
        cmp_marginal, cmp_joint, cmp_conditional = tail_matrices(cmp_event)
        joint_error = np.abs(cmp_joint - ref_joint)
        conditional_error = np.abs(cmp_conditional - ref_conditional)
        pair_values = conditional_error[mask]
        rows.append(
            {
                "threshold": float(q),
                "alpha": float(1 - q),
                "reference_draws": int(len(reference)),
                "comparison_draws": int(len(comparison)),
                "marginal_exceedance_MAE": float(np.mean(np.abs(cmp_marginal - ref_marginal))),
                "joint_exceedance_offdiagonal_MAE": float(np.mean(joint_error[mask])),
                "standardized_joint_exceedance_MAE": float(np.mean(joint_error[mask]) / (1 - q)),
                "joint_exceedance_frobenius_relative_error": float(
                    np.linalg.norm(cmp_joint - ref_joint) / max(np.linalg.norm(ref_joint), 1e-15)
                ),
                "tail_dependence_offdiagonal_MAE": float(np.mean(pair_values)),
                "reference_conditional_p95": float(np.quantile(ref_conditional[mask], 0.95)),
                "comparison_conditional_p95": float(np.quantile(cmp_conditional[mask], 0.95)),
                "legacy_conditional_p95_abs_gap": float(
                    abs(np.quantile(cmp_conditional[mask], 0.95) - np.quantile(ref_conditional[mask], 0.95))
                ),
                "pair_abs_error_median": float(np.quantile(pair_values, 0.50)),
                "pair_abs_error_p90": float(np.quantile(pair_values, 0.90)),
                "pair_abs_error_p95": float(np.quantile(pair_values, 0.95)),
                "pair_abs_error_p99": float(np.quantile(pair_values, 0.99)),
                "pair_abs_error_max": float(np.max(pair_values)),
            }
        )
    return rows


def integrated_standardized_joint_mae(rows: list[dict[str, float]]) -> float:
    return float(np.mean([row["standardized_joint_exceedance_MAE"] for row in rows]))


def primary_contrast(
    oracle_primary: np.ndarray,
    m4: np.ndarray,
    oracle_reference: np.ndarray,
    thresholds: Iterable[float],
) -> dict[str, float]:
    thresholds = tuple(map(float, thresholds))
    m4_rows = threshold_tail_metrics(oracle_primary, m4, thresholds)
    noise_rows = threshold_tail_metrics(oracle_primary, oracle_reference, thresholds)
    raw = integrated_standardized_joint_mae(m4_rows)
    noise = integrated_standardized_joint_mae(noise_rows)
    q95_index = thresholds.index(0.95)
    return {
        "raw_ISJEM_oracle_vs_M4": raw,
        "oracle_oracle_noise_floor": noise,
        "primary_EISJEM": raw - noise,
        "legacy_p95_gap_oracle_vs_M4": m4_rows[q95_index]["legacy_conditional_p95_abs_gap"],
        "legacy_p95_gap_oracle_vs_oracle": noise_rows[q95_index]["legacy_conditional_p95_abs_gap"],
    }


def center_covariance_metrics(reference: np.ndarray, comparison: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, float)
    comparison = np.asarray(comparison, float)
    reference_cov = np.cov(reference, rowvar=False, ddof=1)
    comparison_cov = np.cov(comparison, rowvar=False, ddof=1)
    reference_sd = np.sqrt(np.maximum(np.diag(reference_cov), 1e-15))
    comparison_sd = np.sqrt(np.maximum(np.diag(comparison_cov), 0))
    reference_eigen = np.linalg.eigvalsh(reference_cov)
    comparison_eigen = np.linalg.eigvalsh(comparison_cov)
    reference_corr = reference_cov / np.maximum(reference_sd[:, None] * reference_sd[None, :], 1e-15)
    comparison_corr = comparison_cov / np.maximum(comparison_sd[:, None] * comparison_sd[None, :], 1e-15)
    mask = _off_diagonal(reference.shape[1])
    return {
        "scope": "joint_448",
        "reference_draws": len(reference),
        "comparison_draws": len(comparison),
        "mean_difference_RMS_over_truth_SD": float(
            np.sqrt(np.mean(((np.mean(comparison, axis=0) - np.mean(reference, axis=0)) / reference_sd) ** 2))
        ),
        "pointwise_SD_ratio": float(np.mean(comparison_sd) / np.mean(reference_sd)),
        "leading_eigenvalue_ratio": float(comparison_eigen[-1] / reference_eigen[-1]),
        "leading_10_eigenvalue_ratios": (comparison_eigen[-10:] / np.maximum(reference_eigen[-10:], 1e-15))[::-1].tolist(),
        "effective_rank_ratio": float(
            (comparison_eigen.sum() ** 2 / np.sum(comparison_eigen**2))
            / (reference_eigen.sum() ** 2 / np.sum(reference_eigen**2))
        ),
        "correlation_frobenius_relative_error": float(
            np.linalg.norm(comparison_corr - reference_corr) / np.linalg.norm(reference_corr)
        ),
        "offdiagonal_correlation_MAE": float(np.mean(np.abs((comparison_corr - reference_corr)[mask]))),
    }


def marginal_tail_metrics(
    reference: np.ndarray, comparison: np.ndarray, thresholds: Iterable[float]
) -> list[dict[str, float]]:
    reference = np.asarray(reference, float)
    comparison = np.asarray(comparison, float)
    ref_center = reference - np.mean(reference, axis=0)
    cmp_center = comparison - np.mean(comparison, axis=0)
    ref_z = ref_center / np.maximum(np.std(ref_center, axis=0), 1e-15)
    cmp_z = cmp_center / np.maximum(np.std(cmp_center, axis=0), 1e-15)
    skew_rmse = float(np.sqrt(np.mean((np.mean(cmp_z**3, axis=0) - np.mean(ref_z**3, axis=0)) ** 2)))
    kurt_rmse = float(
        np.sqrt(np.mean(((np.mean(cmp_z**4, axis=0) - 3) - (np.mean(ref_z**4, axis=0) - 3)) ** 2))
    )
    rows: list[dict[str, float]] = []
    for q in thresholds:
        coordinate_threshold = np.quantile(np.abs(reference), q, axis=0)
        reference_quantile = coordinate_threshold
        comparison_quantile = np.quantile(np.abs(comparison), q, axis=0)
        rows.append(
            {
                "threshold": float(q),
                "mean_coordinate_abs_quantile_ratio": float(
                    np.mean(comparison_quantile / np.maximum(reference_quantile, 1e-15))
                ),
                "median_coordinate_abs_quantile_ratio": float(
                    np.median(comparison_quantile / np.maximum(reference_quantile, 1e-15))
                ),
                "marginal_exceedance_MAE": float(
                    np.mean(np.abs(np.mean(np.abs(comparison) > coordinate_threshold, axis=0) - (1 - q)))
                ),
                "skewness_RMSE": skew_rmse,
                "excess_kurtosis_RMSE": kurt_rmse,
            }
        )
    return rows


def max_distribution_metrics(reference_layers: np.ndarray, comparison_layers: np.ndarray) -> list[dict[str, float]]:
    reference_layers = np.asarray(reference_layers, float)
    comparison_layers = np.asarray(comparison_layers, float)
    rows: list[dict[str, float]] = []
    for index in range(reference_layers.shape[1]):
        reference = reference_layers[:, index]
        comparison = comparison_layers[:, index]
        reference_q95 = float(np.quantile(reference, 0.95))
        comparison_q95 = float(np.quantile(comparison, 0.95))
        rows.append(
            {
                "layer": f"D{index}",
                "reference_draws": len(reference),
                "comparison_draws": len(comparison),
                "KS": _ks(reference, comparison),
                "reference_q90": float(np.quantile(reference, 0.90)),
                "comparison_q90": float(np.quantile(comparison, 0.90)),
                "reference_q95": reference_q95,
                "comparison_q95": comparison_q95,
                "q95_ratio": comparison_q95 / reference_q95,
                "reference_q99": float(np.quantile(reference, 0.99)),
                "comparison_q99": float(np.quantile(comparison, 0.99)),
                "probability_comparison_exceeds_reference_q95": float(np.mean(comparison > reference_q95)),
            }
        )
    return rows
