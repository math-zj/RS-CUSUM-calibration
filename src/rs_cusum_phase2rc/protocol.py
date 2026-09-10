"""Freeze Phase 2R-C C0 protocol and all future stage seeds before C0 runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKSPACE=Path(__file__).resolve().parents[2]
CONFIG=WORKSPACE/"configs"/"rs_cusum_phase2rc.yaml"
PROTOCOL=WORKSPACE/"report"/"rs_cusum_phase2rc_protocol.md"
OUTPUT=WORKSPACE/"results_rs_cusum"/"phase2rc"
PRE_C0_HASH=OUTPUT/"pre_c0_hashes.json"
PRE_M3_HASH=OUTPUT/"pre_m3_hashes.json"


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest().upper()


def stable_seed(*parts: object) -> int:
    digest=hashlib.sha256(":".join(map(str,parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8],"little",signed=False)


def load_raw_config() -> dict[str,Any]:
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8")); p=cfg["protocol"]
    if p["label"]!="phase2rc-design-variation" or p["parent_method"]!="1.0.1-phase1": raise RuntimeError("unexpected Phase 2R-C identity")
    if p["phase3_gate"]!="CLOSED" or p["formal_ohiot1dm_gate"]!="CLOSED": raise RuntimeError("closed gates changed")
    if any(p[key] for key in ("alternatives_allowed","power_allowed","changepoint_performance_allowed","transfer_null_allowed")): raise RuntimeError("forbidden Phase 2R-C analysis enabled")
    return cfg


def _prior_seeds() -> set[int]:
    prior:set[int]=set()
    phase2r=WORKSPACE/"results_rs_cusum"/"phase2r"
    for filename in ("truth_replicates.csv","decomposition_replicates.csv","scaling_replicates.csv","full_refit_replicates.csv"):
        path=phase2r/filename
        if path.is_file():
            with path.open("r",encoding="utf-8",newline="") as handle:
                for row in csv.DictReader(handle): prior.add(int(row["seed"]))
    phase2rb=WORKSPACE/"results_rs_cusum"/"phase2rb"/"frozen_seed_lists.npz"
    if phase2rb.is_file():
        with np.load(phase2rb,allow_pickle=False) as data:
            for name in data.files: prior.update(map(int,np.asarray(data[name]).ravel()))
    return prior


def create_seed_lists(cfg: dict[str,Any]) -> Path:
    master=int(cfg["fresh_seeds"]["master_seed"]); arrays:dict[str,np.ndarray]={}
    counts={"c0":int(cfg["c0"]["replicates"]),"c0_nested":int(cfg["c0"]["nested_design_reward"]["outer_designs"])}
    counts.update({stage.lower():int(cfg["stages"][stage]["replicates"]) for stage in ("C1","C2","C3")})
    for name,count in counts.items():
        arrays[name]=np.asarray([stable_seed(master,name,index) for index in range(count)],dtype=np.uint64)
    # Independent inner seeds are fixed per outer design and inner index.
    outer=counts["c0_nested"]; inner=int(cfg["c0"]["nested_design_reward"]["inner_reward_replicates"])
    arrays["c0_nested_inner"]=np.asarray([[stable_seed(master,"c0_nested_inner",i,j) for j in range(inner)] for i in range(outer)],dtype=np.uint64)
    flat=np.concatenate([value.ravel() for value in arrays.values()])
    if len(np.unique(flat))!=len(flat): raise RuntimeError("Phase 2R-C seed collision")
    overlap=set(map(int,flat))&_prior_seeds()
    if overlap: raise RuntimeError(f"Phase 2R-C seeds overlap prior phases: {sorted(overlap)[:5]}")
    OUTPUT.mkdir(parents=True,exist_ok=True); target=WORKSPACE/cfg["fresh_seeds"]["seed_file"]
    np.savez_compressed(target,**arrays); return target


def _verify(hash_file: Path) -> None:
    frozen=json.loads(hash_file.read_text(encoding="utf-8")); bad=[]
    for relative,expected in frozen["files"].items():
        path=WORKSPACE/relative
        if not path.is_file() or sha256(path)!=expected: bad.append(relative)
    if bad: raise RuntimeError(f"Phase 2R-C hash mismatch: {bad}")


def load_config(stage: str="c0") -> dict[str,Any]:
    cfg=load_raw_config(); _verify(PRE_C0_HASH)
    if stage.lower()!="c0": _verify(PRE_M3_HASH)
    return cfg


def load_seed_list(name: str,stage: str="c0") -> np.ndarray:
    cfg=load_config(stage)
    with np.load(WORKSPACE/cfg["fresh_seeds"]["seed_file"],allow_pickle=False) as data: return np.asarray(data[name])


def freeze_pre_c0() -> dict[str,Any]:
    if PRE_C0_HASH.exists(): raise RuntimeError("pre-C0 hash already exists")
    cfg=load_raw_config(); seeds=create_seed_lists(cfg)
    paths=[CONFIG,PROTOCOL,seeds,WORKSPACE/"configs"/"rs_cusum_phase2_pilot.yaml",WORKSPACE/"configs"/"rs_cusum_sensitivity.yaml",WORKSPACE/"results_rs_cusum"/"phase2r"/"fixed_evaluation_grid.npz",WORKSPACE/"results_rs_cusum"/"phase2rb"/"manifest.json"]
    paths.extend(sorted((WORKSPACE/"src"/"rs_cusum_phase2rc").glob("*.py")))
    paths.extend(sorted((WORKSPACE/"tests").glob("test_phase2rc_c0*.py")))
    rel=lambda path:str(path.relative_to(WORKSPACE)).replace("\\","/")
    payload={"stage":"pre-C0","parent_method":"1.0.1-phase1","all_future_seed_lists_frozen":True,"hash_algorithm":"SHA-256","files":{rel(path):sha256(path) for path in paths}}
    OUTPUT.mkdir(parents=True,exist_ok=True); PRE_C0_HASH.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return payload


if __name__=="__main__": print(json.dumps({"frozen_files":len(freeze_pre_c0()["files"])},sort_keys=True))
