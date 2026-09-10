"""Run C1/C2/MULT/C3 with frozen M0-M3 definitions and streamed moments."""

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
import yaml

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2r.core import finite_p
from src.rs_cusum_phase2rb.engine import prepare_analysis

from .c0_engine import full_refit_process,influence_process,observed_arrays
from .m3_engine import all_one_identity,run_m3
from .protocol import CONFIG,OUTPUT,WORKSPACE,load_config,load_seed_list,stable_seed


SUPPLEMENT=WORKSPACE/"configs"/"rs_cusum_phase2rc_m3.yaml"
LAYERS=("D0","D1","D2","D3","D4")


def load_supplement() -> dict[str,Any]: return yaml.safe_load(SUPPLEMENT.read_text(encoding="utf-8"))


def _grid(cfg: dict[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    with np.load(WORKSPACE/cfg["frozen_observed_method"]["evaluation_grid_file"],allow_pickle=False) as data: return np.asarray(data["states"],float),np.asarray(data["actions"],int)


def _layer_draws(process: Any,midpoint: int) -> dict[str,np.ndarray]:
    raw=process.raw; norm=process.normalized; joint=process.joint
    return {"D0":np.abs(raw[:,midpoint,0]),"D1":np.abs(norm[:,midpoint,0]),"D2":np.max(np.abs(norm[:,midpoint]),axis=1),
            "D3":np.max(np.abs(joint[:,:,0]),axis=1),"D4":np.max(np.abs(joint),axis=(1,2))}


def _observed_layers(raw: np.ndarray,norm: np.ndarray,joint: np.ndarray,midpoint: int) -> dict[str,float]:
    return {"D0":float(abs(raw[midpoint,0])),"D1":float(abs(norm[midpoint,0])),"D2":float(np.max(np.abs(norm[midpoint]))),
            "D3":float(np.max(np.abs(joint[:,0]))),"D4":float(np.max(np.abs(joint)))}


def _process_moments(process: Any,midpoint: int,saved_m1: int) -> dict[str,Any]:
    valid=process.valid; joint=process.joint[valid].reshape(np.sum(valid),-1); mid=process.normalized[valid,midpoint]
    mean=np.mean(joint,axis=0); centered=joint-mean; mid_mean=np.mean(mid,axis=0); mid_centered=mid-mid_mean
    sd=np.std(centered,axis=0,ddof=0); standardized=centered/np.maximum(sd,1e-15)
    return {"joint_cov":(centered.T@centered/(len(centered)-1)).astype(np.float32),"mid_cov":(mid_centered.T@mid_centered/(len(mid_centered)-1)).astype(np.float32),
            "joint_mean":mean.astype(np.float32),"mid_mean":mid_mean.astype(np.float32),
            "coordinate_abs_q95":np.quantile(np.abs(centered),.95,axis=0).astype(np.float32),"coordinate_abs_q99":np.quantile(np.abs(centered),.99,axis=0).astype(np.float32),
            "coordinate_skew":np.mean(standardized**3,axis=0).astype(np.float32),"coordinate_excess_kurtosis":(np.mean(standardized**4,axis=0)-3).astype(np.float32),
            "saved_centered_process":centered[:saved_m1].astype(np.float32)}


def _worker(task: dict[str,Any]) -> dict[str,Any]:
    cfg,parent,stage=task["cfg"],task["parent"],task["stage"]; replicate,seed=int(task["replicate"]),int(task["seed"])
    prepared=prepare_analysis("N0_complete_balanced_null",seed,cfg,parent,task["states"],task["actions"])
    if tuple(sorted(prepared.fits))!=tuple(cfg["frozen_observed_method"]["base_candidates"]): raise RuntimeError("M3 stage support changed")
    observed_raw,observed_norm,observed_joint=observed_arrays(prepared); midpoint=sorted(prepared.fits).index(prepared.midpoint); observed=_observed_layers(observed_raw,observed_norm,observed_joint,midpoint)
    processes={}; runtimes={}; draws=int(task["draws"])
    for arm,law in task["arms"]:
        started=time.perf_counter(); arm_seed=stable_seed(cfg["fresh_seeds"]["master_seed"],stage,seed,arm,law)
        if arm=="M0": result=influence_process(prepared,"M0",draws,arm_seed)
        elif arm=="M1": result=influence_process(prepared,"M1",draws,arm_seed)
        elif arm=="M2": result=full_refit_process(prepared,cfg,draws,arm_seed)
        elif arm=="M3": result=run_m3(prepared,cfg,draws,arm_seed,law)
        else: raise ValueError(arm)
        processes[f"{arm}__{law}"]=result; runtimes[f"{arm}__{law}"]=time.perf_counter()-started
    rows=[]; moments={}; maxima={}; saved={}; save_count=int(task["save_m1_draws"])
    for key,process in processes.items():
        arm,law=key.split("__",1); layer_draws=_layer_draws(process,midpoint)
        draw_hash=hashlib.sha256(np.asarray(layer_draws["D4"],np.float64).tobytes()).hexdigest().upper()
        diagnostics=dict(process.diagnostics); diagnostics["D4_draw_sha256"]=draw_hash
        for layer,values in layer_draws.items():
            finite=values[np.isfinite(values)]; p=finite_p(observed[layer],finite) if len(finite) else np.nan
            rows.append({"stage":stage,"replicate":replicate,"dataset_seed":seed,"arm":arm,"weight_law":law,"layer":layer,
                         "status":"TESTABLE" if np.isfinite(p) else "BOOTSTRAP_FAILURE","observed":observed[layer],"p_value":p,
                         "reject_001":bool(p<=.01) if np.isfinite(p) else "","reject_005":bool(p<=.05) if np.isfinite(p) else "","reject_010":bool(p<=.10) if np.isfinite(p) else "",
                         "bootstrap_q90":float(np.quantile(finite,.90)) if len(finite) else "","bootstrap_q95":float(np.quantile(finite,.95)) if len(finite) else "","bootstrap_q99":float(np.quantile(finite,.99)) if len(finite) else "",
                         "valid_draws":int(np.sum(process.valid)),"failed_draws":int(np.sum(~process.valid)),"draw_failure_rate":float(np.mean(~process.valid)),
                         "runtime_seconds":runtimes[key],"admissible_candidates":json.dumps(sorted(prepared.support)),"diagnostics":json.dumps(diagnostics,sort_keys=True)})
        moments[key]=_process_moments(process,midpoint,save_count if arm=="M1" else 0)
        maxima[key]={"D2":layer_draws["D2"].astype(np.float32),"D4":layer_draws["D4"].astype(np.float32)}
        if arm=="M1" and save_count: saved[key]=moments[key]["saved_centered_process"]
    identity=all_one_identity(prepared,cfg) if task["check_identity"] else None
    return {"replicate":replicate,"rows":rows,"observed_joint":observed_joint.reshape(-1).astype(np.float32),"observed_midpoint":observed_norm[midpoint].astype(np.float32),
            "moments":moments,"maxima":maxima,"saved":saved,"identity":identity}


def _write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    fields=sorted({key for row in rows for key in row}); temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _execute(stage: str,seed_name: str,arms: list[tuple[str,str]],draws: int,workers: int,stem: str,check_identity: bool=False) -> list[tuple]:
    cfg=load_config("m3"); supplement=load_supplement(); parent=load_phase2_config(); states,actions=_grid(cfg); seeds=load_seed_list(seed_name,"m3")
    save_m1=int(supplement["c2_forensics"]["saved_M1_process_draws_per_dataset"]) if stage=="C2" else 0
    tasks=[{"cfg":cfg,"parent":parent,"stage":stage,"replicate":i,"seed":int(seed),"states":states,"actions":actions,"arms":arms,"draws":draws,"save_m1_draws":save_m1,"check_identity":check_identity} for i,seed in enumerate(seeds)]
    R=len(tasks); started=time.perf_counter(); rows=[]; identities=[]; first_signature=[]
    arrays: dict[str,np.ndarray]={}; moment_sums: dict[str,dict[str,np.ndarray]]={}; keys: list[str]=[]
    moment_names=("joint_cov","mid_cov","joint_mean","mid_mean","coordinate_abs_q95","coordinate_abs_q99","coordinate_skew","coordinate_excess_kurtosis")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(_worker,task) for task in tasks}; count=0
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                result=future.result(); idx=int(result["replicate"]); count+=1
                if not keys:
                    keys=sorted(result["moments"])
                    arrays["observed_joint"]=np.empty((R,len(result["observed_joint"])),dtype=np.float32)
                    arrays["observed_midpoint"]=np.empty((R,len(result["observed_midpoint"])),dtype=np.float32)
                    for key in keys:
                        moment_sums[key]={name:np.zeros_like(result["moments"][key][name],dtype=np.float64) for name in moment_names}
                        arrays[f"{key}__conditional_joint_means"]=np.empty((R,len(result["moments"][key]["joint_mean"])),dtype=np.float32)
                        arrays[f"{key}__conditional_mid_means"]=np.empty((R,len(result["moments"][key]["mid_mean"])),dtype=np.float32)
                        arrays[f"{key}__D2_draws"]=np.empty((R,draws),dtype=np.float32)
                        arrays[f"{key}__D4_draws"]=np.empty((R,draws),dtype=np.float32)
                        if key.startswith("M1__") and save_m1:
                            shape=result["saved"][key].shape
                            arrays[f"{key}__saved_centered_process"]=np.empty((R,*shape),dtype=np.float32)
                if sorted(result["moments"])!=keys: raise RuntimeError("Inconsistent method keys across replicates")
                arrays["observed_joint"][idx]=result["observed_joint"]; arrays["observed_midpoint"][idx]=result["observed_midpoint"]
                rows.extend(result["rows"])
                for key in keys:
                    for name in moment_names: moment_sums[key][name]+=result["moments"][key][name]
                    arrays[f"{key}__conditional_joint_means"][idx]=result["moments"][key]["joint_mean"]
                    arrays[f"{key}__conditional_mid_means"][idx]=result["moments"][key]["mid_mean"]
                    arrays[f"{key}__D2_draws"][idx]=result["maxima"][key]["D2"]
                    arrays[f"{key}__D4_draws"][idx]=result["maxima"][key]["D4"]
                    if key.startswith("M1__") and save_m1: arrays[f"{key}__saved_centered_process"][idx]=result["saved"][key]
                if result["identity"] is not None: identities.append({"replicate":idx,**result["identity"]})
                if idx==0:
                    first_signature=sorted((row["arm"],row["layer"],float(row["p_value"]),json.loads(row["diagnostics"])["D4_draw_sha256"]) for row in result["rows"])
                del result
                if count%max(1,R//20)==0 or count==R: print(f"{stage}: {count}/{R}, {time.perf_counter()-started:.1f}s",flush=True)
    rows.sort(key=lambda row:(int(row["replicate"]),row["arm"],row["weight_law"],row["layer"])); _write_csv(OUTPUT/f"{stem}_replicate_results.csv",rows)
    for key in keys:
        for name in moment_names: arrays[f"{key}__mean_{name}"]=(moment_sums[key][name]/R).astype(np.float32)
    np.savez_compressed(OUTPUT/f"{stem}_process_forensics.npz",**arrays)
    if identities: _write_csv(OUTPUT/f"{stem}_all_one_identity.csv",sorted(identities,key=lambda row:int(row["replicate"])))
    return first_signature


def run_c1(workers: int) -> None:
    cfg=load_config("m3"); arms=[("M0","gaussian"),("M1","gaussian"),("M2","gaussian"),("M3","normalized_exponential")]
    original=_execute("C1","c1",arms,int(cfg["stages"]["C1"]["bootstrap_draws"]),workers,"c1",True)
    # Exact deterministic repeat of the first dataset.
    parent=load_phase2_config(); states,actions=_grid(cfg); seed=int(load_seed_list("c1","m3")[0])
    task={"cfg":cfg,"parent":parent,"stage":"C1","replicate":0,"seed":seed,"states":states,"actions":actions,"arms":arms,"draws":int(cfg["stages"]["C1"]["bootstrap_draws"]),"save_m1_draws":0,"check_identity":True}
    repeat=_worker(task)
    signature=sorted((row["arm"],row["layer"],row["p_value"],json.loads(row["diagnostics"])["D4_draw_sha256"]) for row in repeat["rows"])
    deterministic=signature==original
    frame=read_csv_rows(OUTPUT/"c1_replicate_results.csv"); m3=[row for row in frame if row["arm"]=="M3" and row["layer"]=="D4"]
    identity=list(csv.DictReader((OUTPUT/"c1_all_one_identity.csv").open("r",encoding="utf-8")))
    checks={"zero_crash":all(row["status"]=="TESTABLE" for row in frame),"M3_draw_valid_fraction":1-sum(float(row["failed_draws"]) for row in m3)/sum(float(row["valid_draws"])+float(row["failed_draws"]) for row in m3),
            "deterministic":deterministic,"support_unchanged":all(json.loads(row["diagnostics"])["support_candidates_before"]==json.loads(row["diagnostics"])["support_candidates_after"] for row in m3),
            "weighted_FQI_converged":all(float(row["draw_failure_rate"])<=.01 for row in m3),
            "all_one_beta_identity":max(float(row["maximum_beta_error"]) for row in identity)<=2e-10,
            "all_one_increment_identity":max(float(row["maximum_centered_increment"]) for row in identity)<=2e-10}
    gate={"gate":"C1_ENGINEERING_PASS" if all(value is True or (key=="M3_draw_valid_fraction" and value>=.99) for key,value in checks.items()) else "C1_ENGINEERING_FAIL","checks":checks}
    (OUTPUT/"c1_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))


def pd_read_rows(path: Path,replicate: int) -> list[tuple]:
    rows=read_csv_rows(path); return sorted((row["arm"],row["layer"],float(row["p_value"]),json.loads(row["diagnostics"])["D4_draw_sha256"]) for row in rows if int(row["replicate"])==replicate)


def run_c2(workers: int) -> None:
    if json.loads((OUTPUT/"c1_gate.json").read_text())["gate"]!="C1_ENGINEERING_PASS": raise RuntimeError("C2 blocked")
    cfg=load_config("m3"); arms=[("M0","gaussian"),("M1","gaussian"),("M2","gaussian"),("M3","normalized_exponential")]
    _execute("C2","c2",arms,int(cfg["stages"]["C2"]["bootstrap_draws"]),workers,"c2")


def run_mult(workers: int) -> None:
    gate=json.loads((OUTPUT/"c2_gate.json").read_text());
    if not gate.get("M3_continue"): raise RuntimeError("M3-MULT blocked by C2")
    cfg=load_config("m3"); _execute("C2_MULT","c2",[("M3","multinomial_cluster_counts")],int(cfg["stages"]["C2"]["bootstrap_draws"]),workers,"c2_mult")


def run_c3(workers: int) -> None:
    gate=json.loads((OUTPUT/"c2_gate.json").read_text());
    if not gate.get("M3_continue"): raise RuntimeError("C3 blocked by C2")
    cfg=load_config("m3"); _execute("C3","c3",[("M3","normalized_exponential")],int(cfg["stages"]["C3"]["bootstrap_draws"]),workers,"c3")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["c1","c2","mult","c3"]); parser.add_argument("--workers",type=int,default=1); args=parser.parse_args()
    {"c1":run_c1,"c2":run_c2,"mult":run_mult,"c3":run_c3}[args.stage](args.workers)


if __name__=="__main__": main()
