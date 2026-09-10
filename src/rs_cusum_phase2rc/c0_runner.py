"""Run fresh C0 process forensics and the nested design/reward decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from typing import Any

import numpy as np

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2r.core import finite_p
from src.rs_cusum_phase2rb.engine import prepare_analysis

from .c0_engine import (
    full_refit_process,
    influence_process,
    nested_design_process,
    observed_arrays,
    patient_design_rows,
)
from .protocol import OUTPUT,WORKSPACE,load_config,load_seed_list,stable_seed


def _grid(cfg: dict[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    with np.load(WORKSPACE/cfg["frozen_observed_method"]["evaluation_grid_file"],allow_pickle=False) as data:
        return np.asarray(data["states"],float),np.asarray(data["actions"],int)


def _arm_summary(arm: str, process: Any, observed_norm: np.ndarray, observed_joint: np.ndarray, midpoint: int) -> dict[str,Any]:
    valid=process.valid; normalized=process.normalized[valid]; joint=process.joint[valid]
    mean_norm=np.mean(normalized,axis=0); mean_joint=np.mean(joint,axis=0)
    centered_norm=normalized-mean_norm; centered_joint=joint-mean_joint
    unc_d2=np.max(np.abs(normalized[:,midpoint]),axis=1); cen_d2=np.max(np.abs(centered_norm[:,midpoint]),axis=1)
    unc_d4=np.max(np.abs(joint),axis=(1,2)); cen_d4=np.max(np.abs(centered_joint),axis=(1,2))
    observed_d2=float(np.max(np.abs(observed_norm[midpoint]))); observed_d4=float(np.max(np.abs(observed_joint)))
    row={"arm":arm,"valid_draws":len(joint),"failed_draws":int(np.sum(~valid)),
         "draw_failure_rate":float(np.mean(~valid)),"observed_D2":observed_d2,"observed_D4":observed_d4}
    for layer,unc,cen,observed in (("D2",unc_d2,cen_d2,observed_d2),("D4",unc_d4,cen_d4,observed_d4)):
        row.update({
            f"uncentered_{layer}_p":finite_p(observed,unc),f"centered_{layer}_p":finite_p(observed,cen),
            f"uncentered_{layer}_reject005":finite_p(observed,unc)<=0.05,
            f"centered_{layer}_reject005":finite_p(observed,cen)<=0.05,
        })
        for quantile in (0.90,0.95,0.99):
            label=int(quantile*100); row[f"uncentered_{layer}_q{label}"]=float(np.quantile(unc,quantile)); row[f"centered_{layer}_q{label}"]=float(np.quantile(cen,quantile))
    return row


def _c0_worker(task: dict[str,Any]) -> dict[str,Any]:
    cfg,parent,replicate,seed=task["cfg"],task["parent"],int(task["replicate"]),int(task["seed"])
    prepared=prepare_analysis("N0_complete_balanced_null",seed,cfg,parent,task["states"],task["actions"])
    if tuple(sorted(prepared.fits))!=tuple(cfg["frozen_observed_method"]["base_candidates"]): raise RuntimeError("C0 support/candidates changed")
    observed_raw,observed_norm,observed_joint=observed_arrays(prepared); midpoint=sorted(prepared.fits).index(prepared.midpoint)
    shared_seed=stable_seed(cfg["fresh_seeds"]["master_seed"],"c0-bootstrap",seed,"M01")
    m0=influence_process(prepared,"M0",int(cfg["c0"]["bootstrap_draws"]),shared_seed)
    m1=influence_process(prepared,"M1",int(cfg["c0"]["bootstrap_draws"]),shared_seed)
    m2=full_refit_process(prepared,cfg,int(cfg["c0"]["bootstrap_draws"]),stable_seed(cfg["fresh_seeds"]["master_seed"],"c0-bootstrap",seed,"M2"))
    arms={"M0":m0,"M1":m1,"M2":m2}; summaries=[]; moments={}; means={}
    for arm,process in arms.items():
        row=_arm_summary(arm,process,observed_norm,observed_joint,midpoint); row.update({"replicate":replicate,"dataset_seed":seed})
        if arm=="M2": row.update(process.diagnostics)
        summaries.append(row); valid=process.valid
        raw=process.raw[valid]; norm=process.normalized[valid]; joint=process.joint[valid]
        joint_centered=joint-np.mean(joint,axis=0); mid_centered=norm[:,midpoint]-np.mean(norm[:,midpoint],axis=0)
        flat=joint.reshape(len(joint),-1); flat_centered=joint_centered.reshape(len(joint),-1)
        moments[arm]={
            "joint_cov":(flat_centered.T@flat_centered/(len(flat)-1)).astype(np.float32),
            "joint_second":(flat.T@flat/len(flat)).astype(np.float32),
            "mid_cov":(mid_centered.T@mid_centered/(len(mid_centered)-1)).astype(np.float32),
            "mid_second":(norm[:,midpoint].T@norm[:,midpoint]/len(norm)).astype(np.float32),
        }
        means[arm]={"raw":np.mean(raw,axis=0).astype(np.float32),"normalized":np.mean(norm,axis=0).astype(np.float32),"joint":np.mean(joint,axis=0).astype(np.float32)}
    patients=patient_design_rows(prepared,m1,m2)
    for row in patients: row.update({"replicate":replicate,"dataset_seed":seed})
    return {"replicate":replicate,"observed_raw":observed_raw.astype(np.float32),"observed_normalized":observed_norm.astype(np.float32),"observed_joint":observed_joint.astype(np.float32),
            "means":means,"moments":moments,"zero_raw":m2.zero_raw.astype(np.float32),"zero_normalized":m2.zero_normalized.astype(np.float32),"summaries":summaries,"patients":patients}


def _write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for row in rows for key in row}); temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def run_c0(workers: int) -> None:
    cfg=load_config("c0"); parent=load_phase2_config(); states,actions=_grid(cfg); seeds=load_seed_list("c0","c0")
    tasks=[{"cfg":cfg,"parent":parent,"replicate":i,"seed":int(seed),"states":states,"actions":actions} for i,seed in enumerate(seeds)]
    R=len(tasks); C=len(cfg["frozen_observed_method"]["base_candidates"]); Z=int(cfg["frozen_observed_method"]["evaluation_points"])
    arrays={name:np.empty((R,C,Z),dtype=np.float32) for name in ("observed_raw","observed_normalized","observed_joint","zero_raw","zero_normalized")}
    for arm in cfg["c0"]["arms"]:
        for kind in ("raw","normalized","joint"): arrays[f"{arm}_{kind}_conditional_mean"]=np.empty((R,C,Z),dtype=np.float32)
    moment_sums={arm:{kind:np.zeros((C*Z,C*Z) if kind.startswith("joint") else (Z,Z),dtype=float) for kind in ("joint_cov","joint_second","mid_cov","mid_second")} for arm in cfg["c0"]["arms"]}
    summary_rows=[]; patient_rows=[]; started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_c0_worker,task) for task in tasks]
        for count,future in enumerate(as_completed(futures),1):
            result=future.result(); index=result["replicate"]
            for name in ("observed_raw","observed_normalized","observed_joint","zero_raw","zero_normalized"): arrays[name][index]=result[name]
            for arm in cfg["c0"]["arms"]:
                for kind in ("raw","normalized","joint"): arrays[f"{arm}_{kind}_conditional_mean"][index]=result["means"][arm][kind]
                for kind in moment_sums[arm]: moment_sums[arm][kind]+=result["moments"][arm][kind]
            summary_rows.extend(result["summaries"]); patient_rows.extend(result["patients"])
            if count%15==0 or count==R: print(f"C0: {count}/{R}, {time.perf_counter()-started:.1f}s",flush=True)
    for arm in cfg["c0"]["arms"]:
        for kind,value in moment_sums[arm].items(): arrays[f"{arm}_mean_{kind}"]=(value/R).astype(np.float32)
    OUTPUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUTPUT/"c0_process_forensics.npz",**arrays)
    _write_csv(OUTPUT/"c0_replicates.csv",sorted(summary_rows,key=lambda row:(row["replicate"],row["arm"])))
    _write_csv(OUTPUT/"c0_patient_decomposition.csv",sorted(patient_rows,key=lambda row:(row["replicate"],row["patient_id"])))


def _nested_worker(task: dict[str,Any]) -> dict[str,Any]:
    process=nested_design_process(int(task["seed"]),task["inner_seeds"],task["cfg"],task["parent"],task["states"],task["actions"])
    return {"replicate":int(task["replicate"]),"seed":int(task["seed"]),"process":process.astype(np.float32)}


def run_nested(workers: int) -> None:
    cfg=load_config("c0"); parent=load_phase2_config(); states,actions=_grid(cfg)
    seeds=load_seed_list("c0_nested","c0"); inner=load_seed_list("c0_nested_inner","c0")
    O=len(seeds); C=len(cfg["frozen_observed_method"]["base_candidates"]); Z=int(cfg["frozen_observed_method"]["evaluation_points"]); I=inner.shape[1]
    processes=np.empty((O,I,C,Z),dtype=np.float32); rows=[]; started=time.perf_counter()
    tasks=[{"cfg":cfg,"parent":parent,"states":states,"actions":actions,"replicate":i,"seed":int(seeds[i]),"inner_seeds":inner[i]} for i in range(O)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_nested_worker,task) for task in tasks]
        for count,future in enumerate(as_completed(futures),1):
            result=future.result(); processes[result["replicate"]]=result["process"]; rows.append({"outer_replicate":result["replicate"],"design_seed":result["seed"],"inner_replicates":I})
            if count%10==0 or count==O: print(f"C0-nested: {count}/{O}, {time.perf_counter()-started:.1f}s",flush=True)
    np.savez_compressed(OUTPUT/"c0_nested_process.npz",joint_process=processes)
    _write_csv(OUTPUT/"c0_nested_designs.csv",sorted(rows,key=lambda row:row["outer_replicate"]))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["means","nested"]); parser.add_argument("--workers",type=int,default=1); args=parser.parse_args()
    {"means":run_c0,"nested":run_nested}[args.stage](args.workers)


if __name__=="__main__": main()
