"""C0 conditional-mean, zero-center, nested-design, and patient diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum_phase2.generator import generate_dataset
from src.rs_cusum_phase2r.core import evaluate_candidate
from src.rs_cusum_phase2rb.engine import (
    PreparedAnalysis,
    _align_candidate_contributions,
    _vectorized_side_refit,
    build_common_null,
    observed_process,
    prepare_analysis,
    pseudo_rewards,
)
from src.rs_cusum_phase2r.core import multiplier_draws


@dataclass(frozen=True)
class ProcessDraws:
    arm: str
    raw: np.ndarray
    normalized: np.ndarray
    joint: np.ndarray
    valid: np.ndarray
    multipliers: np.ndarray
    zero_raw: np.ndarray | None
    zero_normalized: np.ndarray | None
    patient_vectors: np.ndarray
    diagnostics: dict[str,Any]


def observed_arrays(prepared: PreparedAnalysis) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    candidates=sorted(prepared.fits); C=len(candidates); Z=prepared.evaluation_design.shape[0]
    raw=np.empty((C,Z)); normalized=np.empty((C,Z)); joint=np.empty((C,Z))
    for index,candidate in enumerate(candidates):
        contrast,_,variance=evaluate_candidate(prepared.fits[candidate],prepared.evaluation_design)
        raw[index]=contrast; normalized[index]=contrast/np.sqrt(variance)
        joint[index]=prepared.boundaries[candidate]*normalized[index]
    return raw,normalized,joint


def influence_process(
    prepared: PreparedAnalysis, arm: str, draws: int, seed: int
) -> ProcessDraws:
    if arm not in {"M0","M1"}: raise ValueError(arm)
    candidates=sorted(prepared.fits); C=len(candidates); Z=prepared.evaluation_design.shape[0]
    G=len(prepared.global_patient_ids)
    xi=multiplier_draws("gaussian",draws,G,seed)
    raw=np.empty((draws,C,Z)); normalized=np.empty_like(raw); patient_vectors=np.empty((G,C,Z))
    for c_index,candidate in enumerate(candidates):
        fit=prepared.fits[candidate]
        coefficient=_align_candidate_contributions(prepared,fit)
        delta=coefficient@prepared.evaluation_design.T
        raw[:,c_index]=xi@delta
        active=np.asarray([prepared.global_patient_ids.index(pid) for pid in fit.cluster_ids],dtype=int)
        if arm=="M1":
            weighted=xi[:,:,None]*delta[None,:,:]
            centered=weighted[:,active]-np.mean(weighted[:,active],axis=1,keepdims=True)
            variance=len(active)/(len(active)-1.0)*np.sum(centered**2,axis=1)
        else:
            _,_,observed_variance=evaluate_candidate(fit,prepared.evaluation_design)
            variance=np.broadcast_to(observed_variance,(draws,Z))
        normalized[:,c_index]=raw[:,c_index]/np.sqrt(np.maximum(variance,1e-15))
        _,_,observed_variance=evaluate_candidate(fit,prepared.evaluation_design)
        patient_vectors[:,c_index]=(
            prepared.boundaries[candidate]*delta/np.sqrt(observed_variance)[None,:]
        )
    boundaries=np.asarray([prepared.boundaries[candidate] for candidate in candidates])
    joint=normalized*boundaries[None,:,None]
    return ProcessDraws(arm,raw,normalized,joint,np.ones(draws,dtype=bool),xi,None,None,
                        patient_vectors,{"refit_count":0})


def _side_indices(prepared: PreparedAnalysis, records: tuple, candidate: int) -> tuple[np.ndarray,np.ndarray]:
    elapsed=np.asarray([record.elapsed_index for record in records],dtype=int)
    left=np.flatnonzero((elapsed>=prepared.dataset.analysis_start)&(elapsed<candidate))
    right=np.flatnonzero((elapsed>=candidate)&(elapsed<prepared.dataset.analysis_end))
    return left,right


def full_refit_process(
    prepared: PreparedAnalysis, cfg: dict[str,Any], draws: int, seed: int
) -> ProcessDraws:
    common=build_common_null(prepared,cfg); xi=multiplier_draws("gaussian",draws,len(common.patient_ids),seed)
    random_rewards=pseudo_rewards(common,xi)
    # The first row is the exact zero-perturbation restricted-null reward.
    rewards=np.vstack([common.base_reward[None,:],random_rewards])
    candidates=sorted(prepared.fits); C=len(candidates); Z=prepared.evaluation_design.shape[0]
    raw=np.full((draws,C,Z),np.nan); normalized=np.full_like(raw,np.nan)
    zero_raw=np.empty((C,Z)); zero_normalized=np.empty((C,Z)); valid=np.ones(draws,dtype=bool)
    iteration_values=[]; w_checks=np.zeros(draws)
    for c_index,candidate in enumerate(candidates):
        left_indices,right_indices=_side_indices(prepared,common.records,candidate)
        left_records=tuple(common.records[index] for index in left_indices)
        right_records=tuple(common.records[index] for index in right_indices)
        left=_vectorized_side_refit(left_records,rewards[:,left_indices],cfg)
        right=_vectorized_side_refit(right_records,rewards[:,right_indices],cfg)
        if not (left.converged[0] and right.converged[0]): raise FloatingPointError("zero-perturbation side failed")
        beta_diff=left.beta-right.beta
        contrast=beta_diff@prepared.evaluation_design.T
        locations={pid:index for index,pid in enumerate(prepared.global_patient_ids)}
        coefficient=np.zeros((draws+1,len(prepared.global_patient_ids),beta_diff.shape[1]))
        for index,pid in enumerate(left.patient_ids): coefficient[:,locations[pid]]+=left.contributions[:,index]
        for index,pid in enumerate(right.patient_ids): coefficient[:,locations[pid]]-=right.contributions[:,index]
        active=np.asarray([locations[pid] for pid in sorted(set(left.patient_ids)|set(right.patient_ids))])
        delta=np.einsum("bgp,zp->bgz",coefficient,prepared.evaluation_design)
        variance=len(active)/(len(active)-1.0)*np.sum(delta[:,active]**2,axis=1)
        zero_raw[c_index]=contrast[0]; zero_normalized[c_index]=contrast[0]/np.sqrt(np.maximum(variance[0],1e-15))
        raw[:,c_index]=contrast[1:]; normalized[:,c_index]=contrast[1:]/np.sqrt(np.maximum(variance[1:],1e-15))
        valid&=left.converged[1:]&right.converged[1:]
        iteration_values.extend([left.iterations[1:],right.iterations[1:]])
        w_checks+=np.linalg.norm(left.W[1:],axis=(1,2))+np.linalg.norm(right.W[1:],axis=(1,2))
    raw[~valid]=np.nan; normalized[~valid]=np.nan
    boundaries=np.asarray([prepared.boundaries[candidate] for candidate in candidates])
    joint=normalized*boundaries[None,:,None]
    # Least-squares patient projection of the nonlinear full-refit process on cluster multipliers.
    usable=np.flatnonzero(valid); xc=xi[usable]-np.mean(xi[usable],axis=0,keepdims=True)
    yc=joint[usable].reshape(len(usable),-1); yc-=np.mean(yc,axis=0,keepdims=True)
    projection=np.linalg.lstsq(xc,yc,rcond=None)[0].reshape(len(common.patient_ids),C,Z)
    return ProcessDraws(
        "M2",raw,normalized,joint,valid,xi,zero_raw,zero_normalized,projection,
        {"refit_count":(draws+1)*2*C,"valid_draws":int(np.sum(valid)),
         "failed_draws":int(np.sum(~valid)),"mean_iterations":float(np.mean(np.concatenate(iteration_values))),
         "W_checksum_sd":float(np.std(w_checks,ddof=1)),"zero_refit_error":common.zero_refit_error,
         "ridge_score_error":common.ridge_score_error},
    )


def patient_design_rows(prepared: PreparedAnalysis, m1: ProcessDraws, m2: ProcessDraws) -> list[dict[str,Any]]:
    records=prepared.panel.records; ids=prepared.global_patient_ids
    states=np.vstack([record.state for record in records]); actions=np.asarray([record.action for record in records])
    design=action_feature(states,actions); global_state_mean=np.mean(states,axis=0)
    global_gram=design.T@design/len(design); dataset_pid={pid:index for index,pid in enumerate(prepared.dataset.patient_ids)}
    rows=[]
    for patient_index,pid in enumerate(ids):
        selected=np.asarray([record.patient_id==pid for record in records]); s=states[selected]; a=actions[selected]; x=design[selected]
        action_patterns=[]
        patient_records=[record for record in records if record.patient_id==pid]
        for candidate in sorted(prepared.fits):
            for side in ("left","right"):
                values=[record.action for record in patient_records if (record.elapsed_index<candidate if side=="left" else record.elapsed_index>=candidate)]
                action_patterns.append(float(np.mean(values)))
        covariance=np.cov(s,rowvar=False,ddof=1)
        patient_gram=x.T@x/len(x)
        rows.append({
            "patient_id":pid,"phenotype":float(prepared.dataset.phenotypes[dataset_pid[pid]]),
            "transitions":int(np.sum(selected)),"action_rate":float(np.mean(a)),
            "state_cov_trace":float(np.trace(covariance)),
            "state_mean_distance":float(np.linalg.norm(np.mean(s,axis=0)-global_state_mean)),
            "state_range_mean":float(np.mean(np.ptp(s,axis=0))),
            "candidate_action_composition_range":float(np.ptp(action_patterns)),
            "gram_frobenius_distance":float(np.linalg.norm(patient_gram-global_gram)),
            "gram_trace":float(np.trace(patient_gram)),
            "M1_patient_process_norm":float(np.linalg.norm(m1.patient_vectors[patient_index])),
            "M2_patient_projection_norm":float(np.linalg.norm(m2.patient_vectors[patient_index])),
        })
    return rows


def regenerate_reward_matrix(dataset: Any, inner_seeds: np.ndarray, records: tuple, reward_sd: float) -> np.ndarray:
    means=(0.40*dataset.states[:,:-1,0]+0.10*dataset.states[:,:-1,1]
           +(0.05*dataset.states[:,:-1,2] if dataset.states.shape[2]>2 else 0.0)-0.20*dataset.actions)
    pid_location={pid:index for index,pid in enumerate(dataset.patient_ids)}
    patient=np.asarray([pid_location[record.patient_id] for record in records]); elapsed=np.asarray([record.elapsed_index for record in records])
    rows=[]
    for seed in np.asarray(inner_seeds).ravel():
        rng=np.random.default_rng(int(seed)); reward=means+rng.normal(0.0,reward_sd,size=means.shape)
        rows.append(reward[patient,elapsed])
    return np.asarray(rows)


def nested_design_process(
    dataset_seed: int, inner_seeds: np.ndarray, cfg: dict[str,Any], parent: dict[str,Any],
    evaluation_states: np.ndarray, evaluation_actions: np.ndarray,
) -> np.ndarray:
    prepared=prepare_analysis("N0_complete_balanced_null",dataset_seed,cfg,parent,evaluation_states,evaluation_actions)
    records=tuple(sorted(prepared.panel.records,key=lambda record:(record.patient_id,record.elapsed_index)))
    rewards=regenerate_reward_matrix(dataset=prepared.dataset,inner_seeds=inner_seeds,records=records,
                                     reward_sd=float(parent["simulation"]["reward_noise_sd"]))
    C=len(prepared.fits); Z=prepared.evaluation_design.shape[0]; B=len(rewards)
    joint=np.empty((B,C,Z)); elapsed=np.asarray([record.elapsed_index for record in records])
    for c_index,candidate in enumerate(sorted(prepared.fits)):
        li=np.flatnonzero((elapsed>=prepared.dataset.analysis_start)&(elapsed<candidate)); ri=np.flatnonzero((elapsed>=candidate)&(elapsed<prepared.dataset.analysis_end))
        left=_vectorized_side_refit(tuple(records[index] for index in li),rewards[:,li],cfg)
        right=_vectorized_side_refit(tuple(records[index] for index in ri),rewards[:,ri],cfg)
        if not np.all(left.converged&right.converged): raise FloatingPointError("nested conditional FQI failure")
        contrast=(left.beta-right.beta)@prepared.evaluation_design.T
        locations={pid:index for index,pid in enumerate(prepared.global_patient_ids)}
        coefficient=np.zeros((B,len(prepared.global_patient_ids),left.beta.shape[1]))
        for index,pid in enumerate(left.patient_ids): coefficient[:,locations[pid]]+=left.contributions[:,index]
        for index,pid in enumerate(right.patient_ids): coefficient[:,locations[pid]]-=right.contributions[:,index]
        active=np.asarray([locations[pid] for pid in sorted(set(left.patient_ids)|set(right.patient_ids))])
        delta=np.einsum("bgp,zp->bgz",coefficient,prepared.evaluation_design)
        variance=len(active)/(len(active)-1.0)*np.sum(delta[:,active]**2,axis=1)
        joint[:,c_index]=prepared.boundaries[candidate]*contrast/np.sqrt(np.maximum(variance,1e-15))
    return joint
