"""Run RB0/RB1/sensitivity/RB2/transfer with frozen seeds and explicit gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from src.rs_cusum_phase2.protocol import load_phase2_config

from .engine import prepare_analysis, run_full_refit_arm, run_influence_arm
from .protocol import OUTPUT, WORKSPACE, load_config, load_seed_list, stable_seed


FIELDS = (
    "stage", "scenario", "replicate", "dataset_seed", "arm", "distribution", "layer",
    "status", "observed", "p_value", "reject_001", "reject_005", "reject_010",
    "bootstrap_q90", "bootstrap_q95", "bootstrap_q99", "bootstrap_mean", "bootstrap_sd",
    "valid_draws", "failed_draws", "draw_failure_rate", "admissible_candidates",
    "runtime_seconds", "diagnostics",
)


def _fixed_grid() -> tuple[np.ndarray, np.ndarray]:
    cfg = load_config()
    path = WORKSPACE / cfg["frozen_observed_method"]["evaluation_grid_file"]
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["states"], dtype=float), np.asarray(data["actions"], dtype=int)


def _arm_seed(dataset_seed: int, stage: str, arm: str, distribution: str) -> int:
    return stable_seed("phase2rb-bootstrap", dataset_seed, stage, arm, distribution)


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    cfg, parent = task["cfg"], task["parent"]
    stage, scenario = task["stage"], task["scenario"]
    replicate, dataset_seed = int(task["replicate"]), int(task["dataset_seed"])
    states, actions = np.asarray(task["states"]), np.asarray(task["actions"])
    prepared = prepare_analysis(scenario, dataset_seed, cfg, parent, states, actions)
    if not prepared.fits:
        rows = []
        for arm, distribution in task["arms"]:
            rows.append({
                "stage": stage, "scenario": scenario, "replicate": replicate,
                "dataset_seed": dataset_seed, "arm": arm, "distribution": distribution,
                "layer": "D4", "status": "NOT_TESTABLE_SUPPORT", "observed": "",
                "p_value": "", "reject_001": "", "reject_005": "", "reject_010": "",
                "bootstrap_q90": "", "bootstrap_q95": "", "bootstrap_q99": "",
                "bootstrap_mean": "", "bootstrap_sd": "", "valid_draws": 0,
                "failed_draws": 0, "draw_failure_rate": "",
                "admissible_candidates": "[]", "runtime_seconds": 0.0,
                "diagnostics": json.dumps({"support": "empty"}),
            })
        return {
            "replicate": replicate, "rows": rows, "observed_joint": None,
            "observed_midpoint": None, "moments": {},
        }
    rows = []; moments = {}
    observed_joint = None; observed_midpoint = None
    for arm, distribution in task["arms"]:
        started = time.perf_counter()
        seed = _arm_seed(dataset_seed, stage, arm, distribution)
        if arm == "M2":
            result = run_full_refit_arm(prepared, cfg, int(task["draws"]), seed, distribution)
        else:
            result = run_influence_arm(
                prepared, arm, int(task["draws"]), seed, distribution,
                float(cfg["arms"]["M0_global_scale"]["frozen_factor"]),
            )
        runtime = time.perf_counter()-started
        if observed_joint is None:
            # The signed observed joint process is reconstructed from frozen fits only.
            from .engine import observed_process
            from src.rs_cusum_phase2r.core import evaluate_candidate
            _, observed_process_array, _ = observed_process(prepared)
            observed_joint = observed_process_array.reshape(-1)
            midpoint_fit = prepared.fits.get(prepared.midpoint)
            if midpoint_fit is not None:
                midpoint_contrast, _, midpoint_variance = evaluate_candidate(
                    midpoint_fit, prepared.evaluation_design
                )
                observed_midpoint = midpoint_contrast/np.sqrt(midpoint_variance)
        valid_joint = result.joint_process[np.all(np.isfinite(result.joint_process), axis=1)]
        valid_mid = result.midpoint_process[np.all(np.isfinite(result.midpoint_process), axis=1)] if result.midpoint_process.shape[1] else np.empty((0, 0))
        key = f"{arm}__{distribution}"
        moments[key] = {
            "joint_cov": np.cov(valid_joint, rowvar=False, ddof=1).astype(np.float32) if len(valid_joint)>1 else None,
            "mid_cov": np.cov(valid_mid, rowvar=False, ddof=1).astype(np.float32) if len(valid_mid)>1 else None,
            "joint_mean": np.mean(valid_joint, axis=0).astype(np.float32) if len(valid_joint) else None,
            "mid_mean": np.mean(valid_mid, axis=0).astype(np.float32) if len(valid_mid) else None,
        }
        draw_hash = hashlib.sha256(
            np.asarray(result.bootstrap["D4"], dtype=np.float64).tobytes()
        ).hexdigest().upper()
        diagnostics = dict(result.diagnostics)
        diagnostics["D4_draw_sha256"] = draw_hash
        diagnostics["conditional_joint_mean_norm"] = float(np.linalg.norm(moments[key]["joint_mean"])) if moments[key]["joint_mean"] is not None else None
        for layer in sorted(result.bootstrap):
            values = result.bootstrap[layer]
            finite = values[np.isfinite(values)]
            p_value = result.p_values[layer]
            rows.append({
                "stage": stage, "scenario": scenario, "replicate": replicate,
                "dataset_seed": dataset_seed, "arm": arm, "distribution": distribution,
                "layer": layer, "status": "TESTABLE" if np.isfinite(p_value) else "BOOTSTRAP_FAILURE",
                "observed": result.observed[layer], "p_value": p_value,
                "reject_001": bool(p_value<=0.01) if np.isfinite(p_value) else "",
                "reject_005": bool(p_value<=0.05) if np.isfinite(p_value) else "",
                "reject_010": bool(p_value<=0.10) if np.isfinite(p_value) else "",
                "bootstrap_q90": float(np.quantile(finite, 0.90)) if len(finite) else "",
                "bootstrap_q95": float(np.quantile(finite, 0.95)) if len(finite) else "",
                "bootstrap_q99": float(np.quantile(finite, 0.99)) if len(finite) else "",
                "bootstrap_mean": float(np.mean(finite)) if len(finite) else "",
                "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite)>1 else "",
                "valid_draws": result.valid_draws, "failed_draws": result.failed_draws,
                "draw_failure_rate": result.failed_draws/(result.valid_draws+result.failed_draws),
                "admissible_candidates": json.dumps(sorted(prepared.support)),
                "runtime_seconds": runtime, "diagnostics": json.dumps(diagnostics, sort_keys=True),
            })
    return {
        "replicate": replicate, "rows": rows, "observed_joint": observed_joint,
        "observed_midpoint": observed_midpoint, "moments": moments,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _execute(
    stage: str,
    scenario_seeds: list[tuple[str, int, int]],
    arms: list[tuple[str, str]],
    draws: int,
    workers: int,
    output_stem: str,
) -> list[dict[str, Any]]:
    cfg = load_config(); parent = load_phase2_config(); states, actions = _fixed_grid()
    tasks = [{
        "cfg": cfg, "parent": parent, "stage": stage, "scenario": scenario,
        "replicate": replicate, "dataset_seed": seed, "states": states, "actions": actions,
        "arms": arms, "draws": draws,
    } for scenario, replicate, seed in scenario_seeds]
    started = time.perf_counter(); results = []
    if workers==1:
        iterator = ((index, _run_one(task)) for index, task in enumerate(tasks, 1))
        for index, result in iterator:
            results.append(result)
            print(f"{stage}: {index}/{len(tasks)}, {time.perf_counter()-started:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if index % max(1, len(tasks)//20)==0 or index==len(tasks):
                    print(f"{stage}: {index}/{len(tasks)}, {time.perf_counter()-started:.1f}s", flush=True)
    results.sort(key=lambda value: value["replicate"])
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (row["scenario"], int(row["replicate"]), row["arm"], row["distribution"], row["layer"]))
    _write_csv(OUTPUT/f"{output_stem}_replicate_results.csv", rows)
    observed_values = [result["observed_joint"] for result in results if result["observed_joint"] is not None]
    same_observed_shape = bool(observed_values) and len({value.shape for value in observed_values})==1
    observed = np.stack(observed_values) if same_observed_shape else np.empty((0, 0))
    midpoint_values = [
        result["observed_midpoint"] for result in results
        if result["observed_midpoint"] is not None
    ]
    same_midpoint_shape = bool(midpoint_values) and len({value.shape for value in midpoint_values})==1
    observed_midpoint = np.stack(midpoint_values) if same_midpoint_shape else np.empty((0, 0))
    keys = sorted({key for result in results for key in result["moments"]})
    arrays: dict[str, np.ndarray] = {
        "observed_joint": observed, "observed_midpoint": observed_midpoint,
    }
    for key in keys:
        joint_cov = [result["moments"][key]["joint_cov"] for result in results if key in result["moments"] and result["moments"][key]["joint_cov"] is not None]
        mid_cov = [result["moments"][key]["mid_cov"] for result in results if key in result["moments"] and result["moments"][key]["mid_cov"] is not None]
        joint_mean = [result["moments"][key]["joint_mean"] for result in results if key in result["moments"] and result["moments"][key]["joint_mean"] is not None]
        mid_mean = [result["moments"][key]["mid_mean"] for result in results if key in result["moments"] and result["moments"][key]["mid_mean"] is not None]
        if joint_cov and len({value.shape for value in joint_cov})==1: arrays[f"{key}__mean_joint_cov"] = np.mean(joint_cov, axis=0)
        if mid_cov and len({value.shape for value in mid_cov})==1: arrays[f"{key}__mean_mid_cov"] = np.mean(mid_cov, axis=0)
        if joint_mean and len({value.shape for value in joint_mean})==1:
            arrays[f"{key}__mean_joint_mean"] = np.mean(joint_mean, axis=0)
            arrays[f"{key}__conditional_joint_means"] = np.stack(joint_mean)
        if mid_mean and len({value.shape for value in mid_mean})==1:
            arrays[f"{key}__mean_mid_mean"] = np.mean(mid_mean, axis=0)
            arrays[f"{key}__conditional_mid_means"] = np.stack(mid_mean)
    np.savez_compressed(OUTPUT/f"{output_stem}_process_moments.npz", **arrays)
    return results


def run_rb0(workers: int) -> None:
    cfg = load_config(); seeds = load_seed_list("rb0")
    tasks = [("N0_complete_balanced_null", index, int(seed)) for index, seed in enumerate(seeds)]
    results = _execute("RB0", tasks, [("M0","gaussian"),("M1","gaussian"),("M2","gaussian")],
                       int(cfg["stages"]["RB0"]["bootstrap_draws"]), workers, "rb0")
    # Re-run the first dataset once to verify exact deterministic statistics/draw hashes.
    parent = load_phase2_config(); states, actions = _fixed_grid()
    first_task = {
        "cfg": cfg, "parent": parent, "stage": "RB0", "scenario": "N0_complete_balanced_null",
        "replicate": 0, "dataset_seed": int(seeds[0]), "states": states, "actions": actions,
        "arms": [("M0","gaussian"),("M1","gaussian"),("M2","gaussian")],
        "draws": int(cfg["stages"]["RB0"]["bootstrap_draws"]),
    }
    repeat = _run_one(first_task)
    original = results[0]
    signature = lambda result: [
        (row["arm"],row["layer"],row["p_value"],json.loads(row["diagnostics"])["D4_draw_sha256"])
        for row in result["rows"]
    ]
    deterministic = signature(original)==signature(repeat)
    rows = [row for result in results for row in result["rows"]]
    m2 = [row for row in rows if row["arm"]=="M2" and row["layer"]=="D4"]
    failure_rate = sum(int(row["failed_draws"]) for row in m2)/sum(int(row["valid_draws"])+int(row["failed_draws"]) for row in m2)
    diagnostics = [json.loads(row["diagnostics"]) for row in m2]
    checks = {
        "zero_crash": all(row["status"]=="TESTABLE" for row in rows),
        "M2_draw_valid_fraction": 1-failure_rate,
        "M2_draw_validity_at_least_099": failure_rate<=0.01,
        "same_seed_deterministic": deterministic,
        "all_statistics_finite": all(np.isfinite(float(row["observed"])) for row in rows),
        "support_unchanged": all(d["support_candidates_before"]==d["support_candidates_after"] for d in diagnostics),
        "full_refit_executed": all(d["refit_count"]>0 for d in diagnostics),
        "D4_pvalues_legal": all(0<float(row["p_value"])<=1 for row in m2),
    }
    gate = "RB0_ENGINEERING_PASS" if all(value is True or (key=="M2_draw_valid_fraction" and value>=0.99) for key,value in checks.items()) else "RB0_ENGINEERING_FAIL"
    (OUTPUT/"rb0_smoke.json").write_text(json.dumps({"gate":gate,"checks":checks},indent=2,sort_keys=True)+"\n",encoding="utf-8")


def run_rb1(workers: int) -> None:
    cfg = load_config(); seeds = load_seed_list("rb1")
    smoke = json.loads((OUTPUT/"rb0_smoke.json").read_text(encoding="utf-8"))
    if smoke["gate"]!="RB0_ENGINEERING_PASS": raise RuntimeError("RB1 blocked by RB0")
    tasks = [("N0_complete_balanced_null", index, int(seed)) for index,seed in enumerate(seeds)]
    _execute("RB1", tasks,
             [("M0","gaussian"),("M1","gaussian"),("M2","gaussian"),("M0_global_scale","gaussian")],
             int(cfg["stages"]["RB1"]["bootstrap_draws"]), workers, "rb1")


def _gate_candidates(filename: str, key: str) -> list[str]:
    gate = json.loads((OUTPUT/filename).read_text(encoding="utf-8"))
    return list(gate[key])


def run_sensitivity(workers: int) -> None:
    cfg=load_config(); candidates=_gate_candidates("rb1_gate.json","continuing_candidates")
    seeds=load_seed_list("rb1"); tasks=[("N0_complete_balanced_null",i,int(seed)) for i,seed in enumerate(seeds)]
    arms=[(arm,dist) for arm in candidates for dist in ("rademacher","webb_six_point")]
    if not arms:
        raise RuntimeError("no RB1-passing candidate for sensitivity")
    _execute("RB1_SENSITIVITY",tasks,arms,int(cfg["stages"]["RB1"]["bootstrap_draws"]),workers,"multiplier_sensitivity")


def run_rb2(workers: int) -> None:
    cfg=load_config(); candidates=_gate_candidates("rb1_gate.json","continuing_candidates")
    if not candidates: raise RuntimeError("no candidate may enter RB2")
    seeds=load_seed_list("rb2"); tasks=[("N0_complete_balanced_null",i,int(seed)) for i,seed in enumerate(seeds)]
    _execute("RB2",tasks,[(arm,"gaussian") for arm in candidates],
             int(cfg["stages"]["RB2"]["bootstrap_draws"]),workers,"rb2_holdout")


def run_transfer(workers: int) -> None:
    cfg=load_config(); candidates=_gate_candidates("rb2_gate.json","confirmed_candidates")
    if not candidates: raise RuntimeError("no RB2-confirmed candidate for transfer")
    tasks=[]
    for scenario in cfg["stages"]["transfer_null"]["scenarios"]:
        for replicate,seed in enumerate(load_seed_list(f"transfer__{scenario}")):
            tasks.append((scenario,replicate,int(seed)))
    _execute("TRANSFER",tasks,[("M0","gaussian")]+[(candidate,"gaussian") for candidate in candidates],
             int(cfg["stages"]["transfer_null"]["bootstrap_draws"]),workers,"transfer_null")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["rb0","rb1","sensitivity","rb2","transfer"]); parser.add_argument("--workers",type=int,default=1); args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    {"rb0":run_rb0,"rb1":run_rb1,"sensitivity":run_sensitivity,"rb2":run_rb2,"transfer":run_transfer}[args.stage](args.workers)


if __name__=="__main__": main()
