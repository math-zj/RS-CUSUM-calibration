"""Run frozen D1/D2/D3 Phase 2R-D stages with streamed process moments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path
from typing import Any

import numpy as np

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2r.core import finite_p
from src.rs_cusum_phase2rb.engine import prepare_analysis
from src.rs_cusum_phase2rc.c0_engine import ProcessDraws,full_refit_process,influence_process,observed_arrays
from src.rs_cusum_phase2rc.m3_engine import run_m3

from .engine import run_m4
from .protocol import OUTPUT,WORKSPACE,load_config,load_seed_list


LAYERS=("D0","D1","D2","D3","D4")
MOMENT_NAMES=("joint_cov","mid_cov","joint_mean","mid_mean","coordinate_abs_q95","coordinate_abs_q99","coordinate_skew","coordinate_excess_kurtosis")


def _grid(cfg: dict[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    with np.load(WORKSPACE/cfg["frozen_observed_method"]["evaluation_grid_file"],allow_pickle=False) as data: return np.asarray(data["states"],float),np.asarray(data["actions"],int)


def _layer_draws(process: ProcessDraws,midpoint: int) -> dict[str,np.ndarray]:
    return {"D0":np.abs(process.raw[:,midpoint,0]),"D1":np.abs(process.normalized[:,midpoint,0]),"D2":np.max(np.abs(process.normalized[:,midpoint]),axis=1),
            "D3":np.max(np.abs(process.joint[:,:,0]),axis=1),"D4":np.max(np.abs(process.joint),axis=(1,2))}


def _observed_layers(raw: np.ndarray,normalized: np.ndarray,joint: np.ndarray,midpoint: int) -> dict[str,float]:
    return {"D0":float(abs(raw[midpoint,0])),"D1":float(abs(normalized[midpoint,0])),"D2":float(np.max(np.abs(normalized[midpoint]))),
            "D3":float(np.max(np.abs(joint[:,0]))),"D4":float(np.max(np.abs(joint)))}


def _moments(process: ProcessDraws,midpoint: int,save_draws: int) -> dict[str,np.ndarray]:
    valid=process.valid; direct=process.joint[valid].reshape(int(np.sum(valid)),-1); mid_direct=process.normalized[valid,midpoint]
    if len(direct)<2: raise FloatingPointError("fewer than two valid process draws")
    mean=np.mean(direct,axis=0); centered=direct-mean; mid_mean=np.mean(mid_direct,axis=0); mid_centered=mid_direct-mid_mean
    sd=np.std(centered,axis=0,ddof=0); standardized=centered/np.maximum(sd,1e-15)
    saved=np.full((save_draws,direct.shape[1]),np.nan,dtype=np.float32)
    if save_draws: saved[:min(save_draws,len(direct))]=direct[:save_draws]
    return {"joint_cov":(centered.T@centered/(len(centered)-1)).astype(np.float32),"mid_cov":(mid_centered.T@mid_centered/(len(mid_centered)-1)).astype(np.float32),
            "joint_mean":mean.astype(np.float32),"mid_mean":mid_mean.astype(np.float32),
            "coordinate_abs_q95":np.quantile(np.abs(direct),.95,axis=0).astype(np.float32),"coordinate_abs_q99":np.quantile(np.abs(direct),.99,axis=0).astype(np.float32),
            "coordinate_skew":np.mean(standardized**3,axis=0).astype(np.float32),"coordinate_excess_kurtosis":(np.mean(standardized**4,axis=0)-3).astype(np.float32),
            "saved_direct_process":saved}


def _worker(task: dict[str,Any]) -> dict[str,Any]:
    cfg,parent=task["cfg"],task["parent"]; replicate,seed=int(task["replicate"]),int(task["dataset_seed"])
    prepared=prepare_analysis("N0_complete_balanced_null",seed,cfg,parent,task["states"],task["actions"])
    if tuple(sorted(prepared.fits))!=tuple(cfg["frozen_observed_method"]["base_candidates"]): raise RuntimeError("outer N0 support changed")
    observed_raw,observed_norm,observed_joint=observed_arrays(prepared); midpoint=sorted(prepared.fits).index(prepared.midpoint); observed=_observed_layers(observed_raw,observed_norm,observed_joint,midpoint)
    processes={}; runtimes={}; draws=int(task["draws"])
    for arm in task["arms"]:
        started=time.perf_counter(); arm_seed=int(task["arm_seeds"][arm])
        if arm=="M0": process=influence_process(prepared,"M0",draws,arm_seed)
        elif arm=="M1": process=influence_process(prepared,"M1",draws,arm_seed)
        elif arm=="M2": process=full_refit_process(prepared,cfg,draws,arm_seed)
        elif arm=="M3": process=run_m3(prepared,cfg,draws,arm_seed,"normalized_exponential")
        elif arm=="M4": process=run_m4(prepared,cfg,parent,task["states"],task["actions"],draws,arm_seed)
        else: raise ValueError(arm)
        processes[arm]=process; runtimes[arm]=time.perf_counter()-started
    rows=[]; moments={}; maxima={}; save=int(task["save_draws"])
    for arm,process in processes.items():
        layers=_layer_draws(process,midpoint); draw_hash=hashlib.sha256(np.asarray(process.joint,np.float64).tobytes()).hexdigest().upper(); diagnostics=dict(process.diagnostics); diagnostics["joint_draw_sha256"]=draw_hash
        for layer,values in layers.items():
            finite=values[np.isfinite(values)]; p=finite_p(observed[layer],finite) if len(finite) else np.nan
            rows.append({"stage":task["stage"],"replicate":replicate,"dataset_seed":seed,"arm":arm,"layer":layer,"status":"TESTABLE" if np.isfinite(p) else "BOOTSTRAP_FAILURE",
                         "observed":observed[layer],"p_value":p,"reject_001":bool(p<=.01) if np.isfinite(p) else "","reject_005":bool(p<=.05) if np.isfinite(p) else "","reject_010":bool(p<=.10) if np.isfinite(p) else "",
                         "bootstrap_q90":float(np.quantile(finite,.90)) if len(finite) else "","bootstrap_q95":float(np.quantile(finite,.95)) if len(finite) else "","bootstrap_q99":float(np.quantile(finite,.99)) if len(finite) else "",
                         "valid_draws":int(np.sum(process.valid)),"failed_draws":int(np.sum(~process.valid)),"draw_failure_rate":float(np.mean(~process.valid)),"runtime_seconds":runtimes[arm],"diagnostics":json.dumps(diagnostics,sort_keys=True)})
        moments[arm]=_moments(process,midpoint,save); maxima[arm]={"D2":layers["D2"].astype(np.float32),"D4":layers["D4"].astype(np.float32)}
    m4_diag=dict(processes["M4"].diagnostics) if "M4" in processes else None
    return {"replicate":replicate,"rows":rows,"observed_joint":observed_joint.reshape(-1).astype(np.float32),"observed_midpoint":observed_norm[midpoint].astype(np.float32),
            "moments":moments,"maxima":maxima,"m4_diagnostics":m4_diag}


def _write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    fields=sorted({key for row in rows for key in row}); temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _signature(rows: list[dict[str,Any]]) -> list[tuple]:
    return sorted((row["arm"],row["layer"],float(row["p_value"]),json.loads(row["diagnostics"])["joint_draw_sha256"]) for row in rows)


def _task(stage: str,index: int,dataset_seed: int,arms: list[str],draws: int,save: int,cfg: dict,parent: dict,states: np.ndarray,actions: np.ndarray) -> dict[str,Any]:
    arm_seeds={arm:int(load_seed_list(f"{stage.lower()}_{arm.lower()}_bootstrap",stage)[index]) for arm in arms}
    return {"cfg":cfg,"parent":parent,"stage":stage,"replicate":index,"dataset_seed":int(dataset_seed),"states":states,"actions":actions,"arms":arms,"arm_seeds":arm_seeds,"draws":draws,"save_draws":save}


def execute(stage: str,workers: int) -> list[tuple]:
    cfg=load_config(stage); parent=load_phase2_config(); states,actions=_grid(cfg); spec=cfg["stages"][stage]; arms=list(spec["arms"]); draws=int(spec["bootstrap_draws"]); save=int(spec.get("saved_joint_draws_per_dataset",0))
    dataset_seeds=load_seed_list(f"{stage.lower()}_dataset",stage); tasks=[_task(stage,i,int(seed),arms,draws,save,cfg,parent,states,actions) for i,seed in enumerate(dataset_seeds)]
    R=len(tasks); rows=[]; model_rows=[]; arrays:dict[str,np.ndarray]={}; sums={}; keys=[]; first_signature=[]; started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(_worker,task) for task in tasks}; count=0
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                result=future.result(); idx=int(result["replicate"]); count+=1
                if not keys:
                    keys=sorted(result["moments"]); arrays["observed_joint"]=np.empty((R,len(result["observed_joint"])),np.float32); arrays["observed_midpoint"]=np.empty((R,len(result["observed_midpoint"])),np.float32)
                    for key in keys:
                        sums[key]={name:np.zeros_like(result["moments"][key][name],dtype=np.float64) for name in MOMENT_NAMES}
                        arrays[f"{key}__conditional_joint_means"]=np.empty((R,len(result["moments"][key]["joint_mean"])),np.float32); arrays[f"{key}__conditional_mid_means"]=np.empty((R,len(result["moments"][key]["mid_mean"])),np.float32)
                        arrays[f"{key}__D2_draws"]=np.empty((R,draws),np.float32); arrays[f"{key}__D4_draws"]=np.empty((R,draws),np.float32)
                        if save: arrays[f"{key}__saved_direct_process"]=np.empty((R,save,len(result["moments"][key]["joint_mean"])),np.float32)
                arrays["observed_joint"][idx]=result["observed_joint"]; arrays["observed_midpoint"][idx]=result["observed_midpoint"]; rows.extend(result["rows"])
                for key in keys:
                    for name in MOMENT_NAMES: sums[key][name]+=result["moments"][key][name]
                    arrays[f"{key}__conditional_joint_means"][idx]=result["moments"][key]["joint_mean"]; arrays[f"{key}__conditional_mid_means"][idx]=result["moments"][key]["mid_mean"]
                    arrays[f"{key}__D2_draws"][idx]=result["maxima"][key]["D2"]; arrays[f"{key}__D4_draws"][idx]=result["maxima"][key]["D4"]
                    if save: arrays[f"{key}__saved_direct_process"][idx]=result["moments"][key]["saved_direct_process"]
                if result["m4_diagnostics"] is not None: model_rows.append({"replicate":idx,"diagnostics":json.dumps(result["m4_diagnostics"],sort_keys=True)})
                if idx==0: first_signature=_signature(result["rows"])
                del result
                if count%max(1,R//20)==0 or count==R: print(f"{stage}: {count}/{R}, {time.perf_counter()-started:.1f}s",flush=True)
    rows.sort(key=lambda row:(int(row["replicate"]),row["arm"],row["layer"])); stem=stage.lower(); _write_csv(OUTPUT/f"{stem}_replicate_results.csv",rows)
    for key in keys:
        for name in MOMENT_NAMES: arrays[f"{key}__mean_{name}"]=(sums[key][name]/R).astype(np.float32)
    np.savez_compressed(OUTPUT/f"{stem}_process_forensics.npz",**arrays)
    if model_rows: _write_csv(OUTPUT/f"{stem}_m4_model_diagnostics.csv",sorted(model_rows,key=lambda row:int(row["replicate"])))
    return first_signature


def run_d1(workers: int) -> None:
    cfg=load_config("D1"); original=execute("D1",workers); parent=load_phase2_config(); states,actions=_grid(cfg); seed=int(load_seed_list("d1_dataset","D1")[0]); repeat=_worker(_task("D1",0,seed,["M4"],int(cfg["stages"]["D1"]["bootstrap_draws"]),0,cfg,parent,states,actions))
    rows=[]
    with (OUTPUT/"d1_replicate_results.csv").open("r",encoding="utf-8",newline="") as handle: rows=list(csv.DictReader(handle))
    d4=[row for row in rows if row["layer"]=="D4"]; diagnostics=[json.loads(row["diagnostics"]) for row in d4]; gate_cfg=cfg["engineering_gate"]
    checks={"testable_fraction":sum(row["status"]=="TESTABLE" for row in d4)/len(d4)>=gate_cfg["testable_replicate_fraction_min"],
            "valid_draw_fraction":sum(int(row["valid_draws"]) for row in d4)/sum(int(row["valid_draws"])+int(row["failed_draws"]) for row in d4)>=gate_cfg["valid_draw_fraction_min"],
            "deterministic":_signature(repeat["rows"])==original,"finite_process":all(np.isfinite(float(row["p_value"])) for row in rows),
            "design_changed":min(float(item["design_changed_fraction"]) for item in diagnostics)>=gate_cfg["design_changed_fraction_min"],
            "reward_changed":min(float(item["reward_changed_fraction"]) for item in diagnostics)>=gate_cfg["reward_changed_fraction_min"],
            "all_seven_candidates":min(float(item["all_seven_candidates_fraction"]) for item in diagnostics)>=gate_cfg["all_seven_candidates_fraction_min"],
            "full_refit_count":all(int(item["refit_count"])==int(row["valid_draws"])*2*len(cfg["frozen_observed_method"]["base_candidates"]) for item,row in zip(diagnostics,d4)),
            "no_pseudo_reward":all(not item["pseudo_reward_used"] for item in diagnostics),"draw_specific_W_V":all(item["W_recomputed_count"]>0 and item["V_recomputed_count"]>0 for item in diagnostics)}
    payload={"gate":"D1_ENGINEERING_PASS" if all(checks.values()) else "M4_ENGINEERING_FAIL","checks":checks}
    (OUTPUT/"d1_gate.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def run_d2(workers: int) -> None:
    if json.loads((OUTPUT/"d1_gate.json").read_text(encoding="utf-8"))["gate"]!="D1_ENGINEERING_PASS": raise RuntimeError("D2 blocked by D1")
    execute("D2",workers)


def run_d3(workers: int) -> None:
    gate=json.loads((OUTPUT/"d2_gate.json").read_text(encoding="utf-8"))
    if not gate.get("D3_allowed",False): raise RuntimeError("D3 blocked by D2")
    execute("D3",workers)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["d1","d2","d3"]); parser.add_argument("--workers",type=int,default=1); args=parser.parse_args(); {"d1":run_d1,"d2":run_d2,"d3":run_d3}[args.stage](args.workers)


if __name__=="__main__": main()
