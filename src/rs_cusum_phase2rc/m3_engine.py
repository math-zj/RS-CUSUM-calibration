"""Centered exchangeably weighted whole-patient full-refit bootstrap."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any,Sequence

import numpy as np

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.types import TransitionRecord
from src.rs_cusum_phase2r.core import evaluate_candidate
from src.rs_cusum_phase2rb.engine import PreparedAnalysis

from .c0_engine import ProcessDraws


@dataclass(frozen=True)
class WeightedSideResult:
    beta: np.ndarray
    patient_ids: tuple[str,...]
    patient_objective_weights: np.ndarray
    transition_weights: np.ndarray
    contributions: np.ndarray
    W: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    coefficient_delta: np.ndarray


def draw_patient_weights(law: str,draws: int,patients: int,seed: int) -> np.ndarray:
    if draws<=0 or patients<=1: raise ValueError("draws must be positive and at least two patients are required")
    rng=np.random.default_rng(int(seed))
    if law=="normalized_exponential":
        raw=rng.exponential(1.0,size=(draws,patients)); return raw/np.mean(raw,axis=1,keepdims=True)
    if law=="multinomial_cluster_counts":
        return rng.multinomial(patients,np.full(patients,1.0/patients),size=draws).astype(float)
    raise ValueError(f"unknown M3 weight law {law!r}")


def weighted_side_refit(
    records: Sequence[TransitionRecord],
    global_patient_ids: tuple[str,...],
    patient_weights: np.ndarray,
    cfg: dict[str,Any],
    *,
    allow_zero: bool,
) -> WeightedSideResult:
    if not records: raise ValueError("weighted FQI side requires transitions")
    patient_weights=np.asarray(patient_weights,float)
    if patient_weights.ndim!=2 or patient_weights.shape[1]!=len(global_patient_ids): raise ValueError("patient weight shape mismatch")
    if np.any(~np.isfinite(patient_weights)) or np.any(patient_weights<0): raise ValueError("patient weights must be finite and nonnegative")
    if not allow_zero and np.any(patient_weights<=0): raise ValueError("this weight law requires strictly positive patient weights")
    gamma=float(cfg["frozen_observed_method"]["gamma"]); alpha=float(cfg["frozen_observed_method"]["ridge_alpha"])
    max_iterations=int(cfg["frozen_observed_method"]["fqi_max_iterations"]); tolerance=float(cfg["frozen_observed_method"]["fqi_tolerance"])
    states=np.vstack([record.state for record in records]); next_states=np.vstack([record.next_state for record in records])
    actions=np.asarray([record.action for record in records],int); rewards=np.asarray([record.reward_binary for record in records],float)
    X=action_feature(states,actions)
    phi=np.column_stack([np.ones(len(records)),states]); phi_next=np.column_stack([np.ones(len(records)),next_states])
    block=phi.shape[1]
    record_ids=np.asarray([record.patient_id for record in records],object); side_ids=tuple(sorted(set(record_ids.tolist())))
    global_location={pid:index for index,pid in enumerate(global_patient_ids)}
    if any(pid not in global_location for pid in side_ids): raise ValueError("side contains unknown patient")
    side_columns=np.asarray([global_location[pid] for pid in side_ids],int); raw_side=patient_weights[:,side_columns]
    totals=np.sum(raw_side,axis=1)
    if np.any(totals<=0): raise ValueError("a bootstrap draw has zero total patient weight on a side")
    omega=raw_side/totals[:,None]
    counts=Counter(record_ids.tolist()); side_location={pid:index for index,pid in enumerate(side_ids)}
    record_side=np.asarray([side_location[pid] for pid in record_ids],int)
    transition_weights=omega[:,record_side]/np.asarray([counts[pid] for pid in record_ids],float)[None,:]
    if not np.allclose(np.sum(transition_weights,axis=1),1.0,rtol=0,atol=1e-12): raise FloatingPointError("transition weights do not sum to one")
    action0=actions==0; action1=~action0
    gram0=np.einsum("bn,np,nq->bpq",transition_weights[:,action0],phi[action0],phi[action0])
    gram1=np.einsum("bn,np,nq->bpq",transition_weights[:,action1],phi[action1],phi[action1])
    H0=gram0+alpha*np.eye(block)[None,:,:]; H1=gram1+alpha*np.eye(block)[None,:,:]
    H0_inverse=np.linalg.inv(H0); H1_inverse=np.linalg.inv(H1)
    B=patient_weights.shape[0]; beta=np.zeros((B,X.shape[1])); converged=np.zeros(B,bool); iterations=np.zeros(B,int); coefficient_delta=np.full(B,np.inf)
    for iteration in range(1,max_iterations+1):
        active=~converged
        q0=beta[:,:block]@phi_next.T; q1=beta[:,block:]@phi_next.T; next_max=np.maximum(q0,q1); target=rewards[None,:]+gamma*next_max
        rhs0=np.einsum("bn,bn,np->bp",transition_weights[:,action0],target[:,action0],phi[action0])
        rhs1=np.einsum("bn,bn,np->bp",transition_weights[:,action1],target[:,action1],phi[action1])
        beta_new=np.column_stack([np.einsum("bpq,bq->bp",H0_inverse,rhs0),np.einsum("bpq,bq->bp",H1_inverse,rhs1)])
        new_delta=np.linalg.norm(beta_new-beta,axis=1); coefficient_delta[active]=new_delta[active]
        newly=active&(new_delta<=tolerance); iterations[newly]=iteration; beta[active]=beta_new[active]; converged|=newly
        if np.all(converged): break
    iterations[~converged]=max_iterations
    q0=beta[:,:block]@phi_next.T; q1=beta[:,block:]@phi_next.T; greedy=q1>q0
    residual=rewards[None,:]+gamma*np.maximum(q0,q1)-beta@X.T
    W=np.zeros((B,2*block,2*block)); W[:,:block,:block]=H0; W[:,block:,block:]=H1
    for current_action,current_mask in ((0,action0),(1,action1)):
        current_phi=phi[current_mask]; next_phi=phi_next[current_mask]; weights_current=transition_weights[:,current_mask]
        row=slice(current_action*block,(current_action+1)*block)
        for next_action in (0,1):
            selected=greedy[:,current_mask]==next_action
            cross=np.einsum("bn,np,nq->bpq",weights_current*selected,current_phi,next_phi)
            column=slice(next_action*block,(next_action+1)*block); W[:,row,column]-=gamma*cross
    raw_scores=np.empty((B,len(side_ids),X.shape[1]))
    for index,pid in enumerate(side_ids):
        patient_mask=record_ids==pid; raw_scores[:,index]=0.0; patient_count=np.sum(patient_mask)
        for action in (0,1):
            selected=patient_mask&(actions==action); column=slice(action*block,(action+1)*block)
            raw_scores[:,index,column]=residual[:,selected]@phi[selected]/patient_count
    weighted_mean=np.einsum("bg,bgp->bp",omega,raw_scores); centered=raw_scores-weighted_mean[:,None,:]
    eta=np.linalg.solve(W,np.swapaxes(centered,1,2)).swapaxes(1,2)
    contributions=omega[:,:,None]*eta
    return WeightedSideResult(beta,side_ids,omega,transition_weights,contributions,W,converged,iterations,coefficient_delta)


def run_m3(
    prepared: PreparedAnalysis,
    cfg: dict[str,Any],
    draws: int,
    seed: int,
    law: str="normalized_exponential",
    patient_weights: np.ndarray|None=None,
) -> ProcessDraws:
    global_ids=prepared.global_patient_ids
    if patient_weights is None: patient_weights=draw_patient_weights(law,draws,len(global_ids),seed)
    else:
        patient_weights=np.asarray(patient_weights,float); draws=len(patient_weights)
    allow_zero=law=="multinomial_cluster_counts"
    if law=="normalized_exponential" and np.any(patient_weights<=0): raise ValueError("EXP weights must be strictly positive")
    records=tuple(sorted(prepared.panel.records,key=lambda record:(record.patient_id,record.elapsed_index)))
    elapsed=np.asarray([record.elapsed_index for record in records]); candidates=sorted(prepared.fits); C=len(candidates); Z=prepared.evaluation_design.shape[0]
    raw=np.full((draws,C,Z),np.nan); normalized=np.full_like(raw,np.nan); valid=np.ones(draws,bool); iterations=[]; w_checksum=np.zeros(draws)
    global_location={pid:index for index,pid in enumerate(global_ids)}
    for c_index,candidate in enumerate(candidates):
        li=np.flatnonzero((elapsed>=prepared.dataset.analysis_start)&(elapsed<candidate)); ri=np.flatnonzero((elapsed>=candidate)&(elapsed<prepared.dataset.analysis_end))
        left=weighted_side_refit(tuple(records[index] for index in li),global_ids,patient_weights,cfg,allow_zero=allow_zero)
        right=weighted_side_refit(tuple(records[index] for index in ri),global_ids,patient_weights,cfg,allow_zero=allow_zero)
        valid&=left.converged&right.converged; iterations.extend([left.iterations,right.iterations]); w_checksum+=np.linalg.norm(left.W,axis=(1,2))+np.linalg.norm(right.W,axis=(1,2))
        observed=prepared.fits[candidate].beta_difference
        increment=(left.beta-right.beta-observed[None,:])@prepared.evaluation_design.T; raw[:,c_index]=increment
        coefficient=np.zeros((draws,len(global_ids),left.beta.shape[1]))
        for index,pid in enumerate(left.patient_ids): coefficient[:,global_location[pid]]+=left.contributions[:,index]
        for index,pid in enumerate(right.patient_ids): coefficient[:,global_location[pid]]-=right.contributions[:,index]
        active=np.asarray([global_location[pid] for pid in sorted(set(left.patient_ids)|set(right.patient_ids))])
        delta=np.einsum("bgp,zp->bgz",coefficient,prepared.evaluation_design); variance=len(active)/(len(active)-1.0)*np.sum(delta[:,active]**2,axis=1)
        normalized[:,c_index]=increment/np.sqrt(np.maximum(variance,1e-15))
    raw[~valid]=np.nan; normalized[~valid]=np.nan
    boundaries=np.asarray([prepared.boundaries[candidate] for candidate in candidates]); joint=normalized*boundaries[None,:,None]
    return ProcessDraws("M3",raw,normalized,joint,valid,patient_weights,None,None,np.empty((0,C,Z)),
                        {"weight_law":law,"refit_count":draws*2*C,"W_recomputed_count":draws*2*C,"V_recomputed_count":draws*C,
                         "valid_draws":int(np.sum(valid)),"failed_draws":int(np.sum(~valid)),"mean_iterations":float(np.mean(np.concatenate(iterations))),
                         "W_checksum_sd":float(np.std(w_checksum,ddof=1)) if draws>1 else 0.0,"support_candidates_before":tuple(sorted(prepared.support)),"support_candidates_after":tuple(sorted(prepared.support)),
                         "pseudo_reward_used":False,"same_patient_weight_all_candidates":True})


def all_one_identity(prepared: PreparedAnalysis,cfg: dict[str,Any]) -> dict[str,float]:
    weights=np.ones((1,len(prepared.global_patient_ids))); result=run_m3(prepared,cfg,1,0,"normalized_exponential",weights)
    max_increment=float(np.max(np.abs(result.raw)))
    beta_error=0.0; records=tuple(sorted(prepared.panel.records,key=lambda record:(record.patient_id,record.elapsed_index))); elapsed=np.asarray([record.elapsed_index for record in records])
    for candidate in sorted(prepared.fits):
        li=np.flatnonzero((elapsed>=prepared.dataset.analysis_start)&(elapsed<candidate)); ri=np.flatnonzero((elapsed>=candidate)&(elapsed<prepared.dataset.analysis_end))
        left=weighted_side_refit(tuple(records[index] for index in li),prepared.global_patient_ids,weights,cfg,allow_zero=False)
        right=weighted_side_refit(tuple(records[index] for index in ri),prepared.global_patient_ids,weights,cfg,allow_zero=False)
        beta_error=max(beta_error,float(np.linalg.norm(left.beta[0]-prepared.fits[candidate].left.model.beta)),float(np.linalg.norm(right.beta[0]-prepared.fits[candidate].right.model.beta)))
    return {"maximum_beta_error":beta_error,"maximum_centered_increment":max_increment}
