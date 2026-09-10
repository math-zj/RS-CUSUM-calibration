"""Numerical FQI sanity checks only; this file cannot run CUSUM or p-values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
@dataclass
class LinearBinaryFQI:
    model: Any
    converged: bool
    iterations: int
    final_delta: float
    history: list[dict[str, float]]


@dataclass
class ClosedFormRidge:
    """Minimal predictor carrying the exact closed-form Ridge coefficients."""

    coef_: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef_


def _base_features(states: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(states)), states])


def _action_design(base: np.ndarray, actions: np.ndarray) -> np.ndarray:
    a = actions.astype(int)
    return np.hstack([base * (a == 0)[:, None], base * (a == 1)[:, None]])


def _pseudo_design(base: np.ndarray, action: int) -> np.ndarray:
    zero = np.zeros_like(base)
    return np.hstack([base, zero]) if action == 0 else np.hstack([zero, base])


def fit_linear_fqi(
    states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
    gamma: float, alpha: float, max_iter: int, tolerance: float,
) -> LinearBinaryFQI:
    """Action-blocked linear FQI matching the two-action Bellman max."""
    base0, base1 = _base_features(states), _base_features(next_states)
    x = _action_design(base0, actions)
    xn0, xn1 = _pseudo_design(base1, 0), _pseudo_design(base1, 1)
    gram = x.T @ x
    penalty = np.eye(gram.shape[0]) * alpha
    # Keep the two action-block intercepts regularized exactly like the legacy
    # Ridge(fit_intercept=False) implementation.
    q0_old = np.zeros(len(states))
    q1_old = np.zeros(len(states))
    history = []
    converged = False
    delta = np.inf
    for iteration in range(1, max_iter + 1):
        target = rewards + gamma * np.maximum(q0_old, q1_old)
        coef = np.linalg.solve(gram + penalty, x.T @ target)
        model = ClosedFormRidge(coef)
        q0_new = model.predict(xn0)
        q1_new = model.predict(xn1)
        delta = float(max(np.max(np.abs(q0_new - q0_old)), np.max(np.abs(q1_new - q1_old))))
        history.append({
            "iteration": iteration, "max_q_update": delta,
            "q0_mean": float(np.mean(q0_new)), "q1_mean": float(np.mean(q1_new)),
            "target_mean": float(np.mean(target)), "coef_l2": float(np.linalg.norm(model.coef_)),
        })
        q0_old, q1_old = q0_new, q1_new
        if not np.all(np.isfinite(model.coef_)) or not np.isfinite(delta):
            break
        if delta < tolerance:
            converged = True
            break
    return LinearBinaryFQI(model=model, converged=converged, iterations=iteration, final_delta=delta, history=history)


def predict_both(model: Any, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = _base_features(states)
    return model.predict(_pseudo_design(base, 0)), model.predict(_pseudo_design(base, 1))


def _quantile_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
    qs = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    labels = ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return {f"{prefix}_{label}": float(v) for label, v in zip(labels, qs)}


def run_fqi_sanity(
    flat5: pd.DataFrame, propensity_predictions: pd.DataFrame, cfg: dict[str, Any], seed: int = 42,
) -> dict[str, pd.DataFrame]:
    feature_names = cfg["state"]["feature_names"]
    s_cols = [f"state__{x}" for x in feature_names]
    sn_cols = [f"next_state__{x}" for x in feature_names]
    train = flat5[flat5["split"] == "training"].reset_index(drop=True)
    test = flat5[flat5["split"] == "testing"].reset_index(drop=True)
    train_p = propensity_predictions[propensity_predictions["split"] == "training"].reset_index(drop=True)["propensity"].to_numpy(float)
    test_p = propensity_predictions[propensity_predictions["split"] == "testing"].reset_index(drop=True)["propensity"].to_numpy(float)
    if len(train_p) != len(train) or len(test_p) != len(test):
        raise AssertionError("Propensity and FQI rows are not aligned")

    gamma = float(cfg["discount"]["gamma_5min"])
    max_iter = int(cfg["qc"]["fqi_max_iter"])
    tol = float(cfg["qc"]["fqi_tolerance"])
    alphas = [float(a) for a in cfg["qc"]["fqi_ridge_alphas"]]
    rewards = {
        "binary_tir": ("reward_binary_tir", 0.0, 1.0),
        "clinically_weighted": ("reward_clinically_weighted", -4.0, 1.0),
    }

    summary_rows, history_rows, support_rows, sample_rows = [], [], [], []
    gaps_by_setting: dict[tuple[str, float, str], np.ndarray] = {}
    rng = np.random.default_rng(seed)

    all_a = train["action"].to_numpy(int)
    positive = np.flatnonzero(all_a == 1)
    negative = np.flatnonzero(all_a == 0)
    n0 = min(len(negative), int(cfg["qc"]["fqi_action0_per_action1"]) * len(positive))
    sample_rng = np.random.default_rng(int(cfg["qc"]["fqi_sampling_seed"]))
    sampled_negative = sample_rng.choice(negative, size=n0, replace=False)
    fit_idx = np.sort(np.concatenate([positive, sampled_negative]))
    fit_frame = train.iloc[fit_idx].reset_index(drop=True)
    x0 = fit_frame[s_cols].to_numpy(float)
    x1 = fit_frame[sn_cols].to_numpy(float)
    a = fit_frame["action"].to_numpy(int)
    for reward_name, (reward_col, rmin, rmax) in rewards.items():
        r = fit_frame[reward_col].to_numpy(float)
        lower_bound, upper_bound = rmin / (1.0 - gamma), rmax / (1.0 - gamma)
        for alpha in alphas:
            fit = fit_linear_fqi(x0, x1, a, r, gamma, alpha, max_iter, tol)
            for h in fit.history:
                history_rows.append({"reward": reward_name, "alpha": alpha, **h})
            for split, frame, prop in (("training", train, train_p), ("testing", test, test_p)):
                states = frame[s_cols].to_numpy(float)
                q0, q1 = predict_both(fit.model, states)
                gap = q1 - q0
                gaps_by_setting[(reward_name, alpha, split)] = gap
                summary_rows.append({
                    "reward": reward_name, "alpha": alpha, "split": split,
                    "converged": fit.converged, "iterations": fit.iterations,
                    "final_max_q_update": fit.final_delta,
                    "fqi_fit_n": len(fit_frame), "fqi_fit_action_1": int(a.sum()),
                    "fqi_fit_action_1_rate": float(a.mean()),
                    "coef_l2": float(np.linalg.norm(fit.model.coef_)),
                    "theoretical_q_lower": lower_bound, "theoretical_q_upper": upper_bound,
                    "q0_outside_theoretical_bounds": float(((q0 < lower_bound) | (q0 > upper_bound)).mean()),
                    "q1_outside_theoretical_bounds": float(((q1 < lower_bound) | (q1 > upper_bound)).mean()),
                    **_quantile_fields("q0", q0), **_quantile_fields("q1", q1), **_quantile_fields("qgap", gap),
                })

                low_cut = float(cfg["qc"]["propensity_extreme_low"])
                masks = {
                    "p_below_extreme_low": prop < low_cut,
                    "p_0.005_to_0.02": (prop >= low_cut) & (prop < 0.02),
                    "p_0.02_to_0.10": (prop >= 0.02) & (prop < 0.10),
                    "p_at_least_0.10": prop >= 0.10,
                }
                for region, mask in masks.items():
                    if not np.any(mask):
                        continue
                    support_rows.append({
                        "reward": reward_name, "alpha": alpha, "split": split, "support_region": region,
                        "n": int(mask.sum()), "n_action_1": int(frame.loc[mask, "action"].sum()),
                        "action_1_rate": float(frame.loc[mask, "action"].mean()),
                        "qgap_mean": float(np.mean(gap[mask])), "qgap_sd": float(np.std(gap[mask])),
                        "abs_qgap_p95": float(np.quantile(np.abs(gap[mask]), .95)),
                    })

                take = min(2500, len(frame))
                idx = np.sort(rng.choice(len(frame), size=take, replace=False)) if take else np.array([], dtype=int)
                for k in idx:
                    sample_rows.append({
                        "reward": reward_name, "alpha": alpha, "split": split,
                        "patient_id": frame.loc[k, "patient_id"], "tau_t": frame.loc[k, "tau_t"],
                        "action": int(frame.loc[k, "action"]), "propensity": float(prop[k]),
                        "q0": float(q0[k]), "q1": float(q1[k]), "qgap": float(gap[k]),
                    })

    sensitivity_rows = []
    for reward_name in rewards:
        for split in ("training", "testing"):
            reference = gaps_by_setting[(reward_name, 1.0, split)] if (reward_name, 1.0, split) in gaps_by_setting else gaps_by_setting[(reward_name, alphas[0], split)]
            for alpha in alphas:
                gap = gaps_by_setting[(reward_name, alpha, split)]
                sensitivity_rows.append({
                    "reward": reward_name, "split": split, "reference_alpha": 1.0,
                    "alpha": alpha, "qgap_correlation_with_reference": float(np.corrcoef(reference, gap)[0, 1]),
                    "qgap_mean_absolute_difference": float(np.mean(np.abs(reference - gap))),
                    "qgap_max_absolute_difference": float(np.max(np.abs(reference - gap))),
                })

    return {
        "summary": pd.DataFrame(summary_rows),
        "convergence_history": pd.DataFrame(history_rows),
        "support_regions": pd.DataFrame(support_rows),
        "ridge_sensitivity": pd.DataFrame(sensitivity_rows),
        "q_sample": pd.DataFrame(sample_rows),
    }
