"""Protocol, seed registry and immutable-hash controls for Phase 2R-D."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKSPACE=Path(__file__).resolve().parents[2]
CONFIG=WORKSPACE/"configs"/"rs_cusum_phase2rd.yaml"
PROTOCOL_MD=WORKSPACE/"report"/"phase2rd_protocol.md"
PROTOCOL_JSON=WORKSPACE/"report"/"phase2rd_protocol.json"
OUTPUT=WORKSPACE/"results_rs_cusum"/"phase2rd"
SEEDS=OUTPUT/"seed_registry.npz"
SEED_META=OUTPUT/"seed_registry.json"
D0_HASH=OUTPUT/"hash_registry_d0.json"
PRE_D1_HASH=OUTPUT/"hash_registry_pre_d1.json"
OLD_PRE_D1_HASH=PRE_D1_HASH
PRE_D1_HASH=OUTPUT/"hash_registry_pre_d1_v2.json"
BUG_LOG=OUTPUT/"engineering_bug_log.json"


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
    if p["label"]!="phase2rd-m4-random-design" or p["version"]!="1.0.0-d0": raise RuntimeError("unexpected Phase 2R-D identity")
    if p["parent_method"]!="1.0.1-phase1" or p["parent_gate"]!="DESIGN_VARIATION_HYPOTHESIS_SUPPORTED_BUT_M3_PARTIAL": raise RuntimeError("unexpected Phase 2R-D parent")
    forbidden=("phase3_allowed","ohiot1dm_allowed","transfer_null_allowed","alternatives_allowed","power_allowed","changepoint_allowed","posthoc_patch_allowed")
    if any(bool(p[key]) for key in forbidden): raise RuntimeError("a prohibited Phase 2R-D analysis was enabled")
    if not cfg["m4"]["primary_only"] or cfg["m4"]["fallback"]!="none; any fit/simulation/support/FQI failure is explicit": raise RuntimeError("M4 primary/fallback rule changed")
    return cfg


def _prior_seeds() -> set[int]:
    prior:set[int]=set()
    for path in (WORKSPACE/"results_rs_cusum").glob("**/*seed*.npz"):
        if path.resolve()==SEEDS.resolve(): continue
        try:
            with np.load(path,allow_pickle=False) as data:
                for name in data.files:
                    array=np.asarray(data[name])
                    if np.issubdtype(array.dtype,np.integer): prior.update(map(int,array.ravel()))
        except (OSError,ValueError):
            continue
    for path in (WORKSPACE/"results_rs_cusum").glob("**/*replicate*.csv"):
        try:
            with path.open("r",encoding="utf-8",newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("seed","dataset_seed","bootstrap_seed"):
                        if row.get(key) not in (None,""): prior.add(int(row[key]))
        except (OSError,ValueError):
            continue
    return prior


def create_seed_registry() -> dict[str,Any]:
    if SEEDS.exists() or SEED_META.exists(): raise RuntimeError("Phase 2R-D seed registry already exists")
    cfg=load_raw_config(); master=int(cfg["seeds"]["master_seed"]); arrays={}
    for stage in ("D1","D2","D3"):
        count=int(cfg["stages"][stage]["replicates"]); key=stage.lower()
        arrays[f"{key}_dataset"]=np.asarray([stable_seed(master,key,"dataset",i) for i in range(count)],dtype=np.uint64)
        for arm in cfg["stages"][stage]["arms"]:
            arrays[f"{key}_{arm.lower()}_bootstrap"]=np.asarray([stable_seed(master,key,arm,"bootstrap",i) for i in range(count)],dtype=np.uint64)
    flat=np.concatenate([value.ravel() for value in arrays.values()])
    if len(np.unique(flat))!=len(flat): raise RuntimeError("within-Phase 2R-D seed collision")
    overlap=set(map(int,flat))&_prior_seeds()
    if overlap: raise RuntimeError(f"Phase 2R-D seed overlap: {sorted(overlap)[:5]}")
    OUTPUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(SEEDS,**arrays)
    payload={"master_seed":master,"algorithm":"SHA-256 first 64 bits","all_values_unique":True,"disjoint_from_prior":True,
             "D3_values_unread_until_D2_pass":True,"arrays":{key:{"count":int(value.size),"sha256":hashlib.sha256(value.tobytes()).hexdigest().upper()} for key,value in arrays.items()}}
    SEED_META.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return payload


def _relative(path: Path) -> str: return str(path.relative_to(WORKSPACE)).replace("\\","/")


def _write_hash(path: Path,stage: str,paths: list[Path],extra: dict[str,Any]) -> dict[str,Any]:
    if path.exists(): raise RuntimeError(f"{path.name} already exists")
    unique=[]; seen=set()
    for item in paths:
        item=item.resolve()
        if item not in seen: seen.add(item); unique.append(item)
    missing=[str(item) for item in unique if not item.is_file()]
    if missing: raise FileNotFoundError(missing)
    payload={"stage":stage,"hash_algorithm":"SHA-256","files":{_relative(item):sha256(item) for item in unique},**extra}
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return payload


def freeze_d0() -> dict[str,Any]:
    cfg=load_raw_config(); create_seed_registry()
    parents=[WORKSPACE/"report"/"rs_cusum_phase2rb_inference_revision_report.md",WORKSPACE/"report"/"rs_cusum_phase2rc_report.md",
             WORKSPACE/"results_rs_cusum"/"phase2rb"/"manifest.json",WORKSPACE/"results_rs_cusum"/"phase2rc"/"manifest.json",
             WORKSPACE/"results_rs_cusum"/"phase2rb"/"phase2rb_gate.json",WORKSPACE/"results_rs_cusum"/"phase2rc"/"phase2rc_gate.json",
             WORKSPACE/"results_rs_cusum"/"phase2r"/"fixed_evaluation_grid.npz"]
    return _write_hash(D0_HASH,"D0_PROTOCOL_FREEZE",[CONFIG,PROTOCOL_MD,PROTOCOL_JSON,SEEDS,SEED_META,*parents],
                       {"protocol_frozen_before_any_M4_output":True,"all_D3_seeds_frozen_before_D1":True})


def verify_hash(path: Path) -> None:
    payload=json.loads(path.read_text(encoding="utf-8")); bad=[]
    for relative,expected in payload["files"].items():
        target=WORKSPACE/relative
        if not target.is_file() or sha256(target)!=expected: bad.append(relative)
    if bad: raise RuntimeError(f"Phase 2R-D hash mismatch: {bad}")


def freeze_pre_d1() -> dict[str,Any]:
    verify_hash(D0_HASH)
    sources=sorted((WORKSPACE/"src"/"rs_cusum_phase2rd").glob("*.py")); tests=sorted((WORKSPACE/"tests").glob("test_phase2rd*.py"))
    return _write_hash(PRE_D1_HASH,"PRE_D1_ENGINEERING_FREEZE_V2",[D0_HASH,OLD_PRE_D1_HASH,BUG_LOG,*sources,*tests],
                       {"implementation_revision":"engineering-r1","supersedes":"hash_registry_pre_d1.json",
                        "implementation_frozen_before_engineering_execution":True,"bug_policy":"invalidate, version, new hashes and unused seeds"})


def load_config(stage: str="d0") -> dict[str,Any]:
    cfg=load_raw_config(); verify_hash(D0_HASH)
    if stage.lower()!="d0": verify_hash(PRE_D1_HASH)
    return cfg


def load_seed_list(name: str,stage: str) -> np.ndarray:
    cfg=load_config(stage)
    if name.startswith("d3_"):
        gate_path=OUTPUT/"d2_gate.json"
        if not gate_path.is_file() or not json.loads(gate_path.read_text(encoding="utf-8")).get("D3_allowed",False): raise RuntimeError("D3 seeds are locked until D2 passes")
    with np.load(SEEDS,allow_pickle=False) as data: return np.asarray(data[name])


if __name__=="__main__": print(json.dumps({"frozen_files":len(freeze_d0()["files"])},sort_keys=True))
