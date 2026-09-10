"""Second Phase 2R-C freeze after C0 support and before any M3 stage run."""

from __future__ import annotations

import json

from .protocol import OUTPUT,PRE_C0_HASH,PRE_M3_HASH,WORKSPACE,load_config,sha256


def freeze() -> dict:
    if PRE_M3_HASH.exists(): raise RuntimeError("pre-M3 hash already exists")
    load_config("c0")
    gate=json.loads((OUTPUT/"c0_gate.json").read_text(encoding="utf-8"))
    if gate.get("gate")!="DESIGN_VARIATION_HYPOTHESIS_SUPPORTED" or not gate.get("M3_implementation_allowed"): raise RuntimeError("C0 did not authorize M3")
    paths=[PRE_C0_HASH,OUTPUT/"frozen_seed_lists.npz",OUTPUT/"c0_gate.json",WORKSPACE/"report"/"rs_cusum_phase2rc_c0_report.md",WORKSPACE/"configs"/"rs_cusum_phase2rc_m3.yaml"]
    paths.extend(sorted(path for path in OUTPUT.glob("c0_*") if path.is_file()))
    paths.extend(sorted((WORKSPACE/"src"/"rs_cusum_phase2rc").glob("m3*.py")))
    paths.extend(sorted((WORKSPACE/"tests").glob("test_phase2rc_m3*.py")))
    unique=[]; seen=set()
    for path in paths:
        if path not in seen: seen.add(path); unique.append(path)
    rel=lambda path:str(path.relative_to(WORKSPACE)).replace("\\","/")
    payload={"stage":"pre-M3","C0_gate":"DESIGN_VARIATION_HYPOTHESIS_SUPPORTED","M3_is_diagnostic_candidate_only":True,
             "C2_and_C3_seeds_were_frozen_before_C0":True,"hash_algorithm":"SHA-256","files":{rel(path):sha256(path) for path in unique}}
    PRE_M3_HASH.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return payload


if __name__=="__main__": print(json.dumps({"frozen_files":len(freeze()["files"])},sort_keys=True))
