"""Complete balanced alternatives using the frozen Phase-2 data law."""
from __future__ import annotations
from math import log
from typing import Any
import numpy as np
from src.rs_cusum_phase2.types import SyntheticDataset

def generate(scenario: dict[str, Any], seed: int, parent: dict[str, Any], fraction: float, transition_multiplier: float) -> SyntheticDataset:
    sim=parent["simulation"]; base=parent["null_scenarios"]["N0_complete_balanced_null"]
    n,p,h=int(sim["patients"]),int(sim["state_dimension"]),int(base["horizon"])
    start=int(sim["initial_record_history_transitions"]); tau=start+int(np.floor(fraction*(h-start)+.5)); effect=float(scenario["effect_size"])
    rng=np.random.default_rng(int(seed)); ids=tuple(f"P{i:02d}" for i in range(n)); phenotype=np.r_[-np.ones(n//2),np.ones(n-n//2)]
    states=np.zeros((n,h+1,p)); actions=np.zeros((n,h),int); rewards=np.zeros((n,h)); states[:,0]=rng.normal(0,.7,(n,p)); states[:,0,0]+=.5*phenotype; states[:,0,1]=phenotype
    patient_effect=rng.normal(0,.15,n); target=float(base["action_target"]); intercept=log(target/(1-target))
    for t in range(h):
        logits=intercept+patient_effect+float(sim["action_state_coefficient"])*states[:,t,0]+.20*states[:,t,1]
        probability=1/(1+np.exp(-np.clip(logits,-30,30))); actions[:,t]=rng.binomial(1,probability)
        shift = effect*transition_multiplier if scenario["change_type"]=="transition" and t>=tau else 0.
        states[:,t+1,0]=float(sim["state_ar"])*states[:,t,0]+.12*states[:,t,1]+.18*actions[:,t]+shift+rng.normal(0,float(sim["nonphenotype_state_noise_sd"]),n)
        states[:,t+1,1]=phenotype
        if p>2: states[:,t+1,2:]=.55*states[:,t,2:]+.04*states[:,t,[0]]+rng.normal(0,float(sim["nonphenotype_state_noise_sd"]),(n,p-2))
        rshift=effect if scenario["change_type"]=="reward" and t>=tau else 0.
        rewards[:,t]=.40*states[:,t,0]+.10*states[:,t,1]+.05*states[:,t,2]-.20*actions[:,t]+rshift+rng.normal(0,float(sim["reward_noise_sd"]),n)
    data=SyntheticDataset("N0_complete_balanced_null", f"representative_altloc::{scenario['id']}", False, str(scenario["effect_level"]), effect, int(seed), states, actions, rewards, np.ones((n,h),bool), ids, phenotype, start, h, tau)
    data.validate(); return data
