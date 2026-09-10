"""Checkpointed Phase 2R-A N0-only forensic runner."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.support import load_support_rule, screen_candidates
from src.rs_cusum.types import RiskSetPanel
from src.rs_cusum_phase2.protocol import load_phase2_config

from .core import (
    CandidateFit,
    decomposition_statistics,
    evaluate_candidate,
    finite_p,
    fit_candidate,
    fit_side,
    greedy_diagnostics,
    jackknife_pseudo_contributions,
    linearized_error,
    make_n0_dataset,
    multiplier_draws,
    n0_panel,
    numerical_negative_jacobian,
    patient_centered_residual,
    reference_model,
    replace_rewards,
    summary_condition,
)
from .protocol import OUTPUT, WORKSPACE, load_config, load_fixed_grid, stable_seed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty result {path}")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parallel_map(
    worker: Callable[[Any], Any], tasks: list[Any], workers: int, label: str
) -> list[Any]:
    started = time.perf_counter()
    results: list[Any] = []
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            results.append(worker(task))
            if index % max(1, len(tasks) // 20) == 0 or index == len(tasks):
                print(f"{label}: {index}/{len(tasks)}, {time.perf_counter()-started:.1f}s", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % max(1, len(tasks) // 20) == 0 or index == len(tasks):
                print(f"{label}: {index}/{len(tasks)}, {time.perf_counter()-started:.1f}s", flush=True)
    return results


def _common_numbers(cfg: dict) -> tuple[float, int, float]:
    dgp = cfg["dgp"]
    return float(dgp["gamma"]), int(dgp["fqi_max_iterations"]), float(dgp["fqi_tolerance"])


def _fit_reference(cfg: dict, parent: dict) -> tuple[np.ndarray, np.ndarray]:
    reference = cfg["reference_target"]
    gamma, max_iterations, tolerance = _common_numbers(cfg)
    dataset = make_n0_dataset(
        parent,
        int(reference["seed"]),
        int(reference["patients"]),
        int(reference["horizon_transitions"]),
        int(reference["analysis_start"]),
    )
    side = reference_model(
        dataset,
        gamma,
        float(cfg["dgp"]["ridge_alpha_primary"]),
        max_iterations,
        tolerance,
    )
    residual_norm = float(
        np.linalg.norm(
            side.design.T @ (side.weights * side.residual)
            - float(cfg["dgp"]["ridge_alpha_primary"]) * side.model.beta
        )
    )
    np.savez_compressed(
        OUTPUT / "reference_target.npz",
        beta0=side.model.beta,
        W0=side.model.influence_matrix,
        transition_weights=side.weights,
    )
    summary = {
        "seed": int(reference["seed"]),
        "patients": int(reference["patients"]),
        "horizon": int(reference["horizon_transitions"]),
        "transitions": len(side.records),
        "converged": bool(side.model.converged),
        "iterations": int(side.model.iterations),
        "beta_norm": float(np.linalg.norm(side.model.beta)),
        "W_condition": float(np.linalg.cond(side.model.influence_matrix)),
        "estimating_equation_residual_norm": residual_norm,
    }
    (OUTPUT / "reference_target_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return side.model.beta, side.model.influence_matrix


def _truth_worker(task: dict[str, Any]) -> dict[str, Any]:
    cfg = task["cfg"]
    parent = task["parent"]
    replicate = int(task["replicate"])
    seed = stable_seed(cfg["dgp"]["master_seed"], "truth", replicate)
    dataset = make_n0_dataset(parent, seed, 12, 240, 48)
    panel = n0_panel(dataset)
    gamma, max_iterations, tolerance = _common_numbers(cfg)
    alpha = float(cfg["dgp"]["ridge_alpha_primary"])
    fit = fit_candidate(
        panel, 144, 48, 240, gamma, alpha, max_iterations, tolerance,
        "patient_balanced", "patient",
    )
    single_design = np.asarray(task["single_design"], dtype=float)
    contrast, contributions, variance = evaluate_candidate(fit, single_design)
    beta0 = np.asarray(task["beta0"], dtype=float)
    w0 = np.asarray(task["w0"], dtype=float)
    approx_left = linearized_error(fit.left, beta0, w0, gamma, alpha)
    approx_right = linearized_error(fit.right, beta0, w0, gamma, alpha)
    directions = np.asarray(task["directions"], dtype=float)
    greedy_rows = []
    for side_name, side in (("left", fit.left), ("right", fit.right)):
        gaps, switches = greedy_diagnostics(
            side,
            cfg["greedy_nonsmoothness"]["perturbation_epsilons"],
            cfg["greedy_nonsmoothness"]["q_gap_thresholds"],
            directions,
        )
        for threshold, fraction in gaps.items():
            greedy_rows.append({
                "replicate": replicate, "seed": seed, "side": side_name,
                "diagnostic": "q_gap", "level": threshold,
                "mean_switch_fraction": "", "maximum_switch_fraction": "",
                "fraction": fraction,
            })
        for epsilon, (mean_switch, max_switch) in switches.items():
            greedy_rows.append({
                "replicate": replicate, "seed": seed, "side": side_name,
                "diagnostic": "perturbation", "level": epsilon,
                "mean_switch_fraction": mean_switch,
                "maximum_switch_fraction": max_switch,
                "fraction": "",
            })
    jacobian_rows = []
    if replicate < int(cfg["jacobian"]["checked_truth_replicates"]):
        step = float(cfg["jacobian"]["finite_difference_step"])
        for side_name, side in (("left", fit.left), ("right", fit.right)):
            numeric = numerical_negative_jacobian(side, side.model.beta, gamma, alpha, step)
            analytic = side.model.influence_matrix
            hessian = side.design.T @ (side.weights[:, None] * side.design) + alpha * np.eye(len(analytic))
            sv_a = np.linalg.svd(analytic, compute_uv=False)
            sv_n = np.linalg.svd(numeric, compute_uv=False)
            jacobian_rows.append({
                "replicate": replicate, "seed": seed, "side": side_name,
                "max_absolute_difference": float(np.max(np.abs(analytic - numeric))),
                "frobenius_relative_error": float(np.linalg.norm(analytic-numeric) / np.linalg.norm(numeric)),
                "max_relative_singular_value_error": float(np.max(np.abs(sv_a-sv_n) / np.maximum(np.abs(sv_n), 1e-15))),
                "analytic_min_singular": float(np.min(sv_a)),
                "numeric_min_singular": float(np.min(sv_n)),
                "analytic_condition": float(np.linalg.cond(analytic)),
                "numeric_condition": float(np.linalg.cond(numeric)),
                "ridge_hessian_frobenius_difference": float(np.linalg.norm(analytic-hessian)),
            })
    patient_ids = fit.left.patient_ids
    scores = np.stack([
        np.vstack([side.model.patient_scores[pid] for pid in patient_ids])
        for side in (fit.left, fit.right)
    ])
    eta = np.stack([
        np.vstack([side.model.patient_influences[pid] for pid in patient_ids])
        for side in (fit.left, fit.right)
    ])
    # The explicit count below avoids inferring action identity from block sparsity by reshape.
    next_actions = []
    block = len(fit.left.model.beta) // 2
    for side in (fit.left, fit.right):
        next_actions.append(float(np.mean(np.any(side.next_greedy_design[:, block:] != 0, axis=1))))
    left_condition = summary_condition(fit.left, alpha)
    right_condition = summary_condition(fit.right, alpha)
    return {
        "replicate": replicate,
        "seed": seed,
        "D": float(contrast[0]),
        "V": float(variance[0]),
        "sqrtV": float(np.sqrt(variance[0])),
        "Z": float(contrast[0] / np.sqrt(variance[0])),
        "beta": np.stack([fit.left.model.beta, fit.right.model.beta]),
        "W": np.stack([fit.left.model.influence_matrix, fit.right.model.influence_matrix]),
        "scores": scores,
        "eta": eta,
        "coefficient_contributions": fit.coefficient_contributions,
        "single_contributions": contributions[:, 0],
        "actual_error": np.stack([fit.left.model.beta-beta0, fit.right.model.beta-beta0]),
        "linearized_error": np.stack([approx_left, approx_right]),
        "design_left": fit.left.design.astype(np.float32),
        "design_right": fit.right.design.astype(np.float32),
        "greedy_action1_fraction": np.asarray(next_actions),
        "left_condition": left_condition,
        "right_condition": right_condition,
        "greedy_rows": greedy_rows,
        "jacobian_rows": jacobian_rows,
    }


def run_truth(workers: int) -> None:
    cfg = load_config()
    parent = load_phase2_config()
    states, actions = load_fixed_grid()
    single_index = int(cfg["fixed_point"]["single_point_index"])
    single_design = action_feature(states[[single_index]], actions[[single_index]])
    beta0, w0 = _fit_reference(cfg, parent)
    rng = np.random.default_rng(int(cfg["greedy_nonsmoothness"]["direction_seed"]))
    directions = rng.choice([-1.0, 1.0], size=(int(cfg["greedy_nonsmoothness"]["directions"]), len(beta0)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    repetitions = int(cfg["decomposition"]["truth_replicates"])
    tasks = [{
        "cfg": cfg, "parent": parent, "replicate": replicate,
        "single_design": single_design, "beta0": beta0, "w0": w0,
        "directions": directions,
    } for replicate in range(repetitions)]
    results = _parallel_map(_truth_worker, tasks, workers, "truth")
    results.sort(key=lambda value: value["replicate"])
    first = results[0]
    design_left = np.lib.format.open_memmap(
        OUTPUT / "truth_design_left.npy", mode="w+", dtype=np.float32,
        shape=(repetitions,) + first["design_left"].shape,
    )
    design_right = np.lib.format.open_memmap(
        OUTPUT / "truth_design_right.npy", mode="w+", dtype=np.float32,
        shape=(repetitions,) + first["design_right"].shape,
    )
    truth_rows = []
    greedy_rows: list[dict[str, Any]] = []
    jacobian_rows: list[dict[str, Any]] = []
    array_keys = [
        "beta", "W", "scores", "eta", "coefficient_contributions",
        "single_contributions", "actual_error", "linearized_error",
        "greedy_action1_fraction",
    ]
    arrays = {key: np.stack([value[key] for value in results]) for key in array_keys}
    for index, value in enumerate(results):
        design_left[index] = value["design_left"]
        design_right[index] = value["design_right"]
        row = {
            "replicate": value["replicate"], "seed": value["seed"],
            "D": value["D"], "V_hat": value["V"], "sqrt_V_hat": value["sqrtV"],
            "Z": value["Z"],
            "left_W_condition": value["left_condition"]["W_condition"],
            "right_W_condition": value["right_condition"]["W_condition"],
            "left_W_min_singular": value["left_condition"]["W_min_singular"],
            "right_W_min_singular": value["right_condition"]["W_min_singular"],
            "left_W_max_singular": value["left_condition"]["W_max_singular"],
            "right_W_max_singular": value["right_condition"]["W_max_singular"],
            "left_ridge_condition": value["left_condition"]["ridge_condition"],
            "right_ridge_condition": value["right_condition"]["ridge_condition"],
            "left_W_minus_hessian_fro": value["left_condition"]["W_minus_hessian_fro"],
            "right_W_minus_hessian_fro": value["right_condition"]["W_minus_hessian_fro"],
            "left_beta_norm": value["left_condition"]["beta_norm"],
            "right_beta_norm": value["right_condition"]["beta_norm"],
            "left_q_abs_max": value["left_condition"]["q_abs_max"],
            "right_q_abs_max": value["right_condition"]["q_abs_max"],
            "left_greedy_action1_fraction": value["greedy_action1_fraction"][0],
            "right_greedy_action1_fraction": value["greedy_action1_fraction"][1],
        }
        truth_rows.append(row)
        greedy_rows.extend(value["greedy_rows"])
        jacobian_rows.extend(value["jacobian_rows"])
    design_left.flush(); design_right.flush()
    del design_left, design_right
    np.savez_compressed(OUTPUT / "truth_forensic_arrays.npz", **arrays)
    _write_csv(OUTPUT / "truth_replicates.csv", truth_rows)
    _write_csv(OUTPUT / "jacobian_check.csv", jacobian_rows)
    _write_csv(OUTPUT / "greedy_nonsmoothness.csv", greedy_rows)


def _decomposition_worker(task: dict[str, Any]) -> dict[str, Any]:
    cfg, parent, replicate = task["cfg"], task["parent"], int(task["replicate"])
    seed = stable_seed(cfg["dgp"]["master_seed"], "decomposition", replicate)
    dataset = make_n0_dataset(parent, seed, 12, 240, 48)
    panel = n0_panel(dataset)
    gamma, max_iterations, tolerance = _common_numbers(cfg)
    alpha = float(cfg["dgp"]["ridge_alpha_primary"])
    evaluation_design = np.asarray(task["evaluation_design"], dtype=float)
    candidates = [int(value) for value in cfg["candidate_grid"]["candidates_for_base_window"]]
    rule = load_support_rule(WORKSPACE / "configs" / "rs_cusum_sensitivity.yaml", "primary")
    screen = screen_candidates(panel.support_view(), candidates, 48, 240, rule)
    if tuple(screen.admissible_candidates) != tuple(candidates):
        raise RuntimeError(f"N0 primary support unexpectedly rejected candidates: {screen.admissible_candidates}")
    rows = []
    draw_arrays = []
    for method, estimator, cluster in (
        ("RS", "patient_balanced", "patient"),
        ("original_compatible", "pooled", "transition"),
    ):
        fits = {
            candidate: fit_candidate(
                panel, candidate, 48, 240, gamma, alpha, max_iterations, tolerance,
                estimator, cluster,
            )
            for candidate in candidates
        }
        observed, bootstrap = decomposition_statistics(
            fits, 144, evaluation_design, 48, 240,
            int(cfg["decomposition"]["decomposition_bootstrap_draws"]),
            stable_seed(seed, method, "gaussian"), "gaussian",
        )
        method_draws = []
        for layer in ("D0", "D1", "D2", "D3", "D4"):
            values = bootstrap[layer]
            p_value = finite_p(observed[layer], values)
            rows.append({
                "replicate": replicate, "seed": seed, "method": method, "layer": layer,
                "status": "TESTABLE", "observed": observed[layer], "p_value": p_value,
                "reject_005": p_value <= 0.05,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_sd": float(np.std(values, ddof=1)),
                "bootstrap_q95": float(np.quantile(values, 0.95)),
            })
            method_draws.append(values)
        draw_arrays.append(np.stack(method_draws))
    return {"replicate": replicate, "rows": rows, "draws": np.stack(draw_arrays)}


def run_decomposition(workers: int) -> None:
    cfg = load_config(); parent = load_phase2_config()
    states, actions = load_fixed_grid()
    evaluation_design = action_feature(states, actions)
    repetitions = int(cfg["decomposition"]["max_decomposition_replicates"])
    tasks = [{"cfg": cfg, "parent": parent, "replicate": r, "evaluation_design": evaluation_design}
             for r in range(repetitions)]
    results = _parallel_map(_decomposition_worker, tasks, workers, "decomposition")
    results.sort(key=lambda value: value["replicate"])
    rows = [row for value in results for row in value["rows"]]
    _write_csv(OUTPUT / "decomposition_replicates.csv", rows)
    np.savez_compressed(
        OUTPUT / "decomposition_bootstrap_draws.npz",
        draws=np.stack([value["draws"] for value in results]),
        methods=np.asarray(["RS", "original_compatible"]),
        layers=np.asarray(["D0", "D1", "D2", "D3", "D4"]),
    )


def _scaling_worker(task: dict[str, Any]) -> dict[str, Any]:
    cfg, parent = task["cfg"], task["parent"]
    kind, value, replicate = task["kind"], task["value"], int(task["replicate"])
    patients, horizon, alpha = 12, 240, float(cfg["dgp"]["ridge_alpha_primary"])
    if kind == "N": patients = int(value)
    elif kind == "T": horizon = int(value)
    elif kind == "alpha": alpha = float(value)
    else: raise ValueError(kind)
    seed = stable_seed(cfg["dgp"]["master_seed"], "scaling", kind, value, replicate)
    row = {"diagnostic": kind, "value": value, "replicate": replicate, "seed": seed}
    try:
        dataset = make_n0_dataset(parent, seed, patients, horizon, 48)
        panel = n0_panel(dataset)
        candidate = 48 + int(np.floor(0.5 * (horizon - 48) + 0.5))
        gamma, max_iterations, tolerance = _common_numbers(cfg)
        fit = fit_candidate(
            panel, candidate, 48, horizon, gamma, alpha, max_iterations, tolerance,
            "patient_balanced", "patient",
        )
        single_design = np.asarray(task["single_design"], dtype=float)
        contrast, contributions, variance = evaluate_candidate(fit, single_design)
        B = int(task["bootstrap_draws"])
        xi = multiplier_draws("gaussian", B, len(fit.cluster_ids), stable_seed(seed, "bootstrap"))
        raw = np.abs(xi @ contributions[:, 0])
        observed_raw = float(abs(contrast[0]))
        observed_z = float(observed_raw / np.sqrt(variance[0]))
        boot_z = raw / np.sqrt(variance[0])
        cond_l = summary_condition(fit.left, alpha); cond_r = summary_condition(fit.right, alpha)
        row.update({
            "status": "TESTABLE", "patients": patients, "horizon": horizon, "alpha": alpha,
            "D": float(contrast[0]), "V_hat": float(variance[0]), "Z": float(contrast[0]/np.sqrt(variance[0])),
            "p_D0": finite_p(observed_raw, raw), "p_D1": finite_p(observed_z, boot_z),
            "reject_D0_005": finite_p(observed_raw, raw) <= 0.05,
            "reject_D1_005": finite_p(observed_z, boot_z) <= 0.05,
            "bootstrap_raw_sd": float(np.std(xi @ contributions[:, 0], ddof=1)),
            "bootstrap_raw_q95_abs": float(np.quantile(raw, 0.95)),
            "left_converged": fit.left.model.converged, "right_converged": fit.right.model.converged,
            "left_iterations": fit.left.model.iterations, "right_iterations": fit.right.model.iterations,
            "mean_W_condition": float(np.mean([cond_l["W_condition"], cond_r["W_condition"]])),
            "mean_ridge_condition": float(np.mean([cond_l["ridge_condition"], cond_r["ridge_condition"]])),
            "mean_beta_norm": float(np.mean([cond_l["beta_norm"], cond_r["beta_norm"]])),
            "max_abs_Q": float(max(cond_l["q_abs_max"], cond_r["q_abs_max"])),
            "error": "",
        })
    except Exception as error:
        row.update({
            "status": "NUMERICAL_FAILURE", "patients": patients, "horizon": horizon, "alpha": alpha,
            "D": "", "V_hat": "", "Z": "", "p_D0": "", "p_D1": "",
            "reject_D0_005": "", "reject_D1_005": "", "bootstrap_raw_sd": "",
            "bootstrap_raw_q95_abs": "", "left_converged": "", "right_converged": "",
            "left_iterations": "", "right_iterations": "", "mean_W_condition": "",
            "mean_ridge_condition": "", "mean_beta_norm": "", "max_abs_Q": "",
            "error": f"{type(error).__name__}: {error}",
        })
    return row


def run_scaling(workers: int) -> None:
    cfg = load_config(); parent = load_phase2_config()
    states, actions = load_fixed_grid(); idx = int(cfg["fixed_point"]["single_point_index"])
    single_design = action_feature(states[[idx]], actions[[idx]])
    tasks = []
    for kind, section, key in (
        ("N", "n_scaling", "patient_counts"),
        ("T", "t_scaling", "horizons"),
        ("alpha", "alpha_diagnostic", "values"),
    ):
        settings = cfg[section]
        for value in settings[key]:
            for replicate in range(int(settings["replicates"])):
                tasks.append({
                    "cfg": cfg, "parent": parent, "kind": kind, "value": value,
                    "replicate": replicate, "single_design": single_design,
                    "bootstrap_draws": int(settings["bootstrap_draws"]),
                })
    rows = _parallel_map(_scaling_worker, tasks, workers, "scaling")
    rows.sort(key=lambda r: (r["diagnostic"], float(r["value"]), r["replicate"]))
    _write_csv(OUTPUT / "scaling_replicates.csv", rows)


def _jackknife_worker(task: dict[str, Any]) -> dict[str, Any]:
    cfg, parent, replicate = task["cfg"], task["parent"], int(task["replicate"])
    seed = stable_seed(cfg["dgp"]["master_seed"], "truth", replicate)
    dataset = make_n0_dataset(parent, seed, 12, 240, 48); panel = n0_panel(dataset)
    gamma, max_iterations, tolerance = _common_numbers(cfg)
    pseudo, variance = jackknife_pseudo_contributions(
        panel, 144, 48, 240, gamma, float(cfg["dgp"]["ridge_alpha_primary"]),
        max_iterations, tolerance, np.asarray(task["single_design"], dtype=float),
    )
    return {"replicate": replicate, "pseudo": pseudo, "variance": variance}


def run_bootstrap_forensics(workers: int) -> None:
    cfg = load_config(); parent = load_phase2_config()
    states, actions = load_fixed_grid(); idx = int(cfg["fixed_point"]["single_point_index"])
    single_design = action_feature(states[[idx]], actions[[idx]])
    truth = list(csv.DictReader((OUTPUT / "truth_replicates.csv").open("r", encoding="utf-8")))
    with np.load(OUTPUT / "truth_forensic_arrays.npz", allow_pickle=False) as data:
        contributions = np.asarray(data["single_contributions"], dtype=float)
    count = int(cfg["bootstrap_forensics"]["datasets"])
    B = int(cfg["bootstrap_forensics"]["draws"])
    tasks = [{"cfg": cfg, "parent": parent, "replicate": r, "single_design": single_design}
             for r in range(count)]
    jackknife = _parallel_map(_jackknife_worker, tasks, workers, "jackknife")
    jackknife = {value["replicate"]: value for value in jackknife}
    distributions = list(cfg["bootstrap_forensics"]["multiplier_distributions"])
    versions = ["D0_raw", "B0_fixed", "B1_draw_specific", "B2_jackknife", "B3_cr2_inspired"]
    saved = np.full((count, len(distributions), len(versions), B), np.nan)
    rows = []
    for replicate in range(count):
        D = float(truth[replicate]["D"]); V = float(truth[replicate]["V_hat"])
        delta = contributions[replicate]
        G = len(delta)
        h = delta**2 / max(float(np.sum(delta**2)), np.finfo(float).tiny)
        adjusted = delta / np.sqrt(np.maximum(1.0-h, 1e-8))
        adjusted -= np.mean(adjusted)
        V_adj = G/(G-1.0)*float(np.sum(adjusted**2))
        pseudo = jackknife[replicate]["pseudo"]; V_jk = float(jackknife[replicate]["variance"])
        for distribution_index, distribution in enumerate(distributions):
            xi = multiplier_draws(
                distribution, B, G,
                stable_seed(cfg["dgp"]["master_seed"], "highB", replicate, distribution),
            )
            raw_signed = xi @ delta
            weighted = xi * delta[None, :]
            centered = weighted - np.mean(weighted, axis=1, keepdims=True)
            V_draw = G/(G-1.0)*np.sum(centered**2, axis=1)
            signed_arrays = {
                "D0_raw": raw_signed,
                "B0_fixed": raw_signed/np.sqrt(V),
                "B1_draw_specific": raw_signed/np.sqrt(np.maximum(V_draw, 1e-15)),
                "B2_jackknife": (xi @ pseudo)/np.sqrt(V_jk),
                "B3_cr2_inspired": (xi @ adjusted)/np.sqrt(V_adj),
            }
            arrays = {
                "D0_raw": np.abs(raw_signed),
                "B0_fixed": np.abs(raw_signed)/np.sqrt(V),
                "B1_draw_specific": np.abs(raw_signed)/np.sqrt(np.maximum(V_draw, 1e-15)),
                "B2_jackknife": np.abs(xi @ pseudo)/np.sqrt(V_jk),
                "B3_cr2_inspired": np.abs(xi @ adjusted)/np.sqrt(V_adj),
            }
            observed = {
                "D0_raw": abs(D),
                "B0_fixed": abs(D)/np.sqrt(V),
                "B1_draw_specific": abs(D)/np.sqrt(V),
                "B2_jackknife": abs(D)/np.sqrt(V_jk),
                "B3_cr2_inspired": abs(D)/np.sqrt(V_adj),
            }
            for version_index, version in enumerate(versions):
                values = arrays[version]
                saved[replicate, distribution_index, version_index] = values
                p_value = finite_p(observed[version], values)
                rows.append({
                    "replicate": replicate, "distribution": distribution, "variance_version": version,
                    "observed": observed[version], "p_value": p_value, "reject_005": p_value <= 0.05,
                    "bootstrap_mean": float(np.mean(values)), "bootstrap_sd": float(np.std(values, ddof=1)),
                    "bootstrap_signed_mean": float(np.mean(signed_arrays[version])),
                    "bootstrap_signed_sd": float(np.std(signed_arrays[version], ddof=1)),
                    "bootstrap_q025": float(np.quantile(values, 0.025)),
                    "bootstrap_q05": float(np.quantile(values, 0.05)),
                    "bootstrap_q95": float(np.quantile(values, 0.95)),
                    "bootstrap_q975": float(np.quantile(values, 0.975)),
                    "unique_values": int(len(np.unique(values))),
                    "observed_rank": float(np.mean(values <= observed[version])),
                })
    np.savez_compressed(
        OUTPUT / "highB_bootstrap_draws.npz", draws=saved,
        distributions=np.asarray(distributions), versions=np.asarray(versions),
    )
    _write_csv(OUTPUT / "bootstrap_forensic_replicates.csv", rows)


def _full_refit_worker(task: dict[str, Any]) -> dict[str, Any]:
    cfg, parent, replicate = task["cfg"], task["parent"], int(task["replicate"])
    seed = stable_seed(cfg["dgp"]["master_seed"], "full_refit", replicate)
    dataset = make_n0_dataset(parent, seed, 12, 240, 48); panel = n0_panel(dataset)
    gamma, max_iterations, tolerance = _common_numbers(cfg)
    alpha = float(cfg["dgp"]["ridge_alpha_primary"])
    records = tuple(panel.records)
    common = fit_side(records, gamma, alpha, max_iterations, tolerance, "patient_balanced")
    gram = common.design.T @ (common.weights[:, None] * common.design)
    ridge_offset = np.linalg.solve(gram, alpha * common.model.beta)
    centered_residual = patient_centered_residual(common)
    base_reward = (
        common.design @ common.model.beta
        - gamma * np.max(
            np.column_stack([
                action_feature(
                    np.vstack([record.next_state for record in records]),
                    np.full(len(records), action),
                ) @ common.model.beta
                for action in (0, 1)
            ]), axis=1
        )
        + common.design @ ridge_offset
    )
    zero_panel = RiskSetPanel(
        replace_rewards(records, base_reward), panel.feature_names, panel.clock_type,
        panel.interval_min, panel.edge_gap_min,
    )
    zero_common = fit_side(zero_panel.records, gamma, alpha, max_iterations, tolerance, "patient_balanced")
    imposed_error = float(np.linalg.norm(zero_common.model.beta-common.model.beta))
    observed_fit = fit_candidate(
        panel, 144, 48, 240, gamma, alpha, max_iterations, tolerance,
        "patient_balanced", "patient",
    )
    single_design = np.asarray(task["single_design"], dtype=float)
    contrast, _, variance = evaluate_candidate(observed_fit, single_design)
    observed = float(abs(contrast[0])/np.sqrt(variance[0]))
    patient_ids = np.asarray([record.patient_id for record in records], dtype=object)
    unique_ids = tuple(sorted(set(patient_ids.tolist())))
    xi = multiplier_draws(
        "gaussian", int(cfg["full_refit_prototype"]["draws"]), len(unique_ids),
        stable_seed(seed, "multipliers"),
    )
    location = {pid: index for index, pid in enumerate(unique_ids)}
    record_columns = np.asarray([location[pid] for pid in patient_ids], dtype=int)
    draws = np.full(len(xi), np.nan)
    errors = []
    for draw_index, multiplier in enumerate(xi):
        pseudo_reward = base_reward + multiplier[record_columns] * centered_residual
        pseudo_panel = RiskSetPanel(
            replace_rewards(records, pseudo_reward), panel.feature_names, panel.clock_type,
            panel.interval_min, panel.edge_gap_min,
        )
        try:
            pseudo_fit = fit_candidate(
                pseudo_panel, 144, 48, 240, gamma, alpha, max_iterations, tolerance,
                "patient_balanced", "patient",
            )
            d_star, _, v_star = evaluate_candidate(pseudo_fit, single_design)
            draws[draw_index] = abs(d_star[0])/np.sqrt(v_star[0])
        except Exception as error:
            errors.append(f"{draw_index}:{type(error).__name__}:{error}")
    valid = np.isfinite(draws)
    p_value = finite_p(observed, draws[valid]) if np.any(valid) else np.nan
    return {
        "replicate": replicate, "seed": seed, "observed": observed,
        "p_value": p_value, "reject_005": bool(p_value <= 0.05) if np.isfinite(p_value) else "",
        "valid_draws": int(np.sum(valid)), "failed_draws": int(np.sum(~valid)),
        "imposed_null_common_beta_error": imposed_error,
        "bootstrap_mean": float(np.nanmean(draws)), "bootstrap_sd": float(np.nanstd(draws, ddof=1)),
        "bootstrap_q95": float(np.nanquantile(draws, 0.95)),
        "errors": json.dumps(errors), "draws": draws,
    }


def run_full_refit(workers: int) -> None:
    required = [
        OUTPUT / "variance_forensics.csv", OUTPUT / "decomposition_summary.csv",
        OUTPUT / "multiplier_diagnostic.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"full-refit is sequenced after earlier summaries; missing {missing}")
    cfg = load_config(); parent = load_phase2_config()
    states, actions = load_fixed_grid(); idx = int(cfg["fixed_point"]["single_point_index"])
    single_design = action_feature(states[[idx]], actions[[idx]])
    tasks = [{"cfg": cfg, "parent": parent, "replicate": r, "single_design": single_design}
             for r in range(int(cfg["full_refit_prototype"]["datasets"]))]
    results = _parallel_map(_full_refit_worker, tasks, workers, "full-refit")
    results.sort(key=lambda value: value["replicate"])
    draws = np.stack([value.pop("draws") for value in results])
    np.savez_compressed(OUTPUT / "full_refit_bootstrap_draws.npz", draws=draws)
    _write_csv(OUTPUT / "full_refit_replicates.csv", results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["truth", "decomposition", "scaling", "bootstrap", "full_refit"]
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    {
        "truth": run_truth,
        "decomposition": run_decomposition,
        "scaling": run_scaling,
        "bootstrap": run_bootstrap_forensics,
        "full_refit": run_full_refit,
    }[args.stage](args.workers)


if __name__ == "__main__":
    main()
