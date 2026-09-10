"""M4 fitted-parametric random-design full-pipeline bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum.statistic import boundary_factor
from src.rs_cusum.support import load_support_rule,screen_candidates
from src.rs_cusum_phase2.grids import candidate_grid
from src.rs_cusum_phase2.panels import build_panel
from src.rs_cusum_phase2.types import SyntheticDataset
from src.rs_cusum_phase2.protocol import WORKSPACE
from src.rs_cusum_phase2r.core import fit_candidate
from src.rs_cusum_phase2rb.engine import PreparedAnalysis
from src.rs_cusum_phase2rc.c0_engine import ProcessDraws,observed_arrays

from .protocol import stable_seed


@dataclass(frozen=True)
class FittedGenerativeNull:
    patient_ids: tuple[str,...]
    phenotypes: np.ndarray
    horizon: int
    analysis_start: int
    state_dimension: int
    initial_x0_beta: np.ndarray
    initial_x0_sd: float
    initial_other_mean: float
    initial_other_sd: float
    action_beta: np.ndarray
    action_random_effect_sd: float
    transition_x0_beta: np.ndarray
    transition_x0_sd: float
    transition_other_beta: np.ndarray
    transition_other_sd: float
    reward_beta: np.ndarray
    reward_sd: float
    diagnostics: dict[str,Any]


def _ols(y: np.ndarray,x: np.ndarray,minimum_sd: float) -> tuple[np.ndarray,float,np.ndarray]:
    y=np.asarray(y,float); x=np.asarray(x,float)
    beta=np.linalg.lstsq(x,y,rcond=None)[0]; residual=y-x@beta
    dof=max(1,len(y)-np.linalg.matrix_rank(x)); sd=max(float(np.sqrt(np.sum(residual**2)/dof)),minimum_sd)
    if np.any(~np.isfinite(beta)) or not np.isfinite(sd): raise FloatingPointError("non-finite OLS null model")
    return beta,sd,residual


def _fit_action_model(states: np.ndarray,actions: np.ndarray,penalty: float) -> tuple[np.ndarray,float,dict[str,Any]]:
    n,t=actions.shape; x=np.column_stack([np.ones(n*t),states[:,:-1,0].reshape(-1),states[:,:-1,1].reshape(-1)])
    y=actions.reshape(-1).astype(float); patient=np.repeat(np.arange(n),t)
    def objective(theta: np.ndarray) -> tuple[float,np.ndarray]:
        beta=theta[:3]; effects=theta[3:]; eta=x@beta+effects[patient]
        value=float(np.sum(np.logaddexp(0.0,eta)-y*eta)+0.5*penalty*np.sum(effects**2))
        difference=1.0/(1.0+np.exp(-np.clip(eta,-35,35)))-y
        gradient=np.r_[x.T@difference,np.bincount(patient,weights=difference,minlength=n)+penalty*effects]
        return value,gradient
    initial=np.zeros(3+n); mean=float(np.clip(np.mean(y),1e-6,1-1e-6)); initial[0]=np.log(mean/(1-mean))
    result=minimize(lambda value:objective(value)[0],initial,jac=lambda value:objective(value)[1],method="L-BFGS-B",options={"maxiter":2000,"ftol":1e-12,"gtol":1e-8})
    if not result.success or np.any(~np.isfinite(result.x)): raise FloatingPointError(f"action null model failed: {result.message}")
    beta=np.asarray(result.x[:3],float); effects=np.asarray(result.x[3:],float)
    eta=x@beta+effects[patient]; probability=1/(1+np.exp(-np.clip(eta,-35,35)))
    information=np.bincount(patient,weights=probability*(1-probability),minlength=n)
    measurement=float(np.mean(1.0/np.maximum(information+penalty,1e-12)))
    variance=max(float(np.var(effects,ddof=1))-measurement,0.0); random_sd=float(np.sqrt(variance))
    diagnostics={"optimizer_success":bool(result.success),"optimizer_iterations":int(result.nit),"penalized_effect_mean":float(np.mean(effects)),
                 "penalized_effect_sd":float(np.std(effects,ddof=1)),"measurement_variance_subtracted":measurement,
                 "estimated_random_effect_sd":random_sd,"probability_min":float(np.min(probability)),"probability_max":float(np.max(probability)),
                 "score_norm":float(np.linalg.norm(result.jac))}
    return beta,random_sd,diagnostics


def fit_generative_null(dataset: SyntheticDataset,cfg: dict[str,Any]) -> FittedGenerativeNull:
    dataset.validate()
    if dataset.scenario!="N0_complete_balanced_null" or not np.all(dataset.observed_mask): raise ValueError("M4 D-stage accepts complete stationary N0 only")
    states=np.asarray(dataset.states,float); actions=np.asarray(dataset.actions,int); rewards=np.asarray(dataset.rewards,float)
    n,t=actions.shape; p=states.shape[2]; minimum=float(cfg["generative_model"]["minimum_sd"])
    phenotype=states[:,0,1].copy()
    if not np.all(states[:,:,1]==phenotype[:,None]): raise ValueError("phenotype coordinate is not time invariant")
    initial_x0=np.column_stack([np.ones(n),phenotype]); initial_x0_beta,initial_x0_sd,_=_ols(states[:,0,0],initial_x0,minimum)
    other_initial=states[:,0,2:].reshape(-1); initial_other_mean=float(np.mean(other_initial)); initial_other_sd=max(float(np.std(other_initial,ddof=1)),minimum)
    action_beta,action_sd,action_diag=_fit_action_model(states,actions,float(cfg["generative_model"]["action_patient_penalty"]))
    current=states[:,:-1]; following=states[:,1:]
    x0=np.column_stack([np.ones(n*t),current[:,:,0].reshape(-1),current[:,:,1].reshape(-1),actions.reshape(-1)])
    transition_x0_beta,transition_x0_sd,_=_ols(following[:,:,0].reshape(-1),x0,minimum)
    current_other=current[:,:,2:].reshape(-1); following_other=following[:,:,2:].reshape(-1); repeated_x0=np.repeat(current[:,:,0].reshape(-1),p-2)
    xo=np.column_stack([np.ones(len(current_other)),current_other,repeated_x0]); transition_other_beta,transition_other_sd,_=_ols(following_other,xo,minimum)
    reward_x2=current[:,:,2] if p>2 else np.zeros((n,t)); xr=np.column_stack([np.ones(n*t),current[:,:,0].reshape(-1),current[:,:,1].reshape(-1),reward_x2.reshape(-1),actions.reshape(-1)])
    reward_beta,reward_sd,_=_ols(rewards.reshape(-1),xr,minimum)
    values=(initial_x0_sd,initial_other_sd,transition_x0_sd,transition_other_sd,reward_sd)
    if any(not np.isfinite(value) or value<minimum for value in values): raise FloatingPointError("invalid fitted innovation scale")
    diagnostics={"patients":n,"horizon":t,"state_dimension":p,"phenotype_counts":{str(value):int(np.sum(phenotype==value)) for value in np.unique(phenotype)},
                 "initial_x0_beta":initial_x0_beta.tolist(),"initial_x0_sd":initial_x0_sd,"initial_other_mean":initial_other_mean,"initial_other_sd":initial_other_sd,
                 "action_beta":action_beta.tolist(),"action":action_diag,"transition_x0_beta":transition_x0_beta.tolist(),"transition_x0_sd":transition_x0_sd,
                 "transition_other_beta":transition_other_beta.tolist(),"transition_other_sd":transition_other_sd,"reward_beta":reward_beta.tolist(),"reward_sd":reward_sd,
                 "fit_uses_rewards_only_in_reward_model":True,"candidate_specific_parameters":False}
    return FittedGenerativeNull(tuple(dataset.patient_ids),phenotype,t,dataset.analysis_start,p,initial_x0_beta,initial_x0_sd,initial_other_mean,initial_other_sd,
                                action_beta,action_sd,transition_x0_beta,transition_x0_sd,transition_other_beta,transition_other_sd,reward_beta,reward_sd,diagnostics)


def simulate_design(model: FittedGenerativeNull,seed: int,cfg: dict[str,Any]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    rng=np.random.default_rng(int(seed)); n=len(model.patient_ids); t=model.horizon; p=model.state_dimension
    states=np.empty((n,t+1,p),float); actions=np.empty((n,t),int); phenotype=model.phenotypes
    states[:,0,0]=model.initial_x0_beta[0]+model.initial_x0_beta[1]*phenotype+rng.normal(0,model.initial_x0_sd,n)
    states[:,0,1]=phenotype
    states[:,0,2:]=rng.normal(model.initial_other_mean,model.initial_other_sd,size=(n,p-2))
    patient_effect=rng.normal(0,model.action_random_effect_sd,n)
    lower,upper=map(float,cfg["generative_model"]["action_probability_clip"]); pmin=1.0; pmax=0.0
    for time in range(t):
        eta=model.action_beta[0]+model.action_beta[1]*states[:,time,0]+model.action_beta[2]*states[:,time,1]+patient_effect
        probability=np.clip(1/(1+np.exp(-np.clip(eta,-35,35))),lower,upper); pmin=min(pmin,float(np.min(probability))); pmax=max(pmax,float(np.max(probability)))
        actions[:,time]=rng.binomial(1,probability)
        states[:,time+1,0]=(model.transition_x0_beta[0]+model.transition_x0_beta[1]*states[:,time,0]+model.transition_x0_beta[2]*states[:,time,1]
                            +model.transition_x0_beta[3]*actions[:,time]+rng.normal(0,model.transition_x0_sd,n))
        states[:,time+1,1]=phenotype
        states[:,time+1,2:]=(model.transition_other_beta[0]+model.transition_other_beta[1]*states[:,time,2:]
                             +model.transition_other_beta[2]*states[:,time,[0]]+rng.normal(0,model.transition_other_sd,size=(n,p-2)))
    if np.any(~np.isfinite(states)): raise FloatingPointError("non-finite simulated M4 state")
    return states,actions,{"action_rate":float(np.mean(actions)),"action_probability_min":pmin,"action_probability_max":pmax,
                           "state_mean":float(np.mean(states[:,:-1,0])),"state_sd":float(np.std(states[:,:-1,0],ddof=1)),"patient_effect_sd":float(np.std(patient_effect,ddof=1))}


def simulate_rewards(model: FittedGenerativeNull,states: np.ndarray,actions: np.ndarray,seed: int) -> tuple[np.ndarray,dict[str,float]]:
    rng=np.random.default_rng(int(seed)); current=states[:,:-1]; x2=current[:,:,2] if model.state_dimension>2 else np.zeros(actions.shape)
    mean=(model.reward_beta[0]+model.reward_beta[1]*current[:,:,0]+model.reward_beta[2]*current[:,:,1]+model.reward_beta[3]*x2+model.reward_beta[4]*actions)
    innovation=rng.normal(0,model.reward_sd,size=actions.shape); rewards=mean+innovation
    if np.any(~np.isfinite(rewards)): raise FloatingPointError("non-finite simulated M4 reward")
    return rewards,{"reward_mean":float(np.mean(rewards)),"reward_sd":float(np.std(rewards,ddof=1)),"innovation_mean":float(np.mean(innovation)),"innovation_sd":float(np.std(innovation,ddof=1))}


def simulate_dataset(model: FittedGenerativeNull,bootstrap_seed: int,draw: int,cfg: dict[str,Any]) -> tuple[SyntheticDataset,dict[str,Any]]:
    design_seed=stable_seed(bootstrap_seed,"M4","design",draw); reward_seed=stable_seed(bootstrap_seed,"M4","reward",draw)
    states,actions,design_diag=simulate_design(model,design_seed,cfg); rewards,reward_diag=simulate_rewards(model,states,actions,reward_seed)
    dataset=SyntheticDataset("N0_complete_balanced_null","N0_complete_balanced",True,"none",0.0,int(stable_seed(bootstrap_seed,"M4","dataset",draw)),states,actions,rewards,
                             np.ones_like(actions,dtype=bool),model.patient_ids,model.phenotypes.copy(),model.analysis_start,model.horizon,None)
    dataset.validate(); return dataset,{"design_seed":int(design_seed),"reward_seed":int(reward_seed),**design_diag,**reward_diag}


def prepare_bootstrap_analysis(dataset: SyntheticDataset,cfg: dict[str,Any],parent_cfg: dict[str,Any],evaluation_states: np.ndarray,evaluation_actions: np.ndarray) -> PreparedAnalysis:
    panel=build_panel(dataset,"rs_reentry",int(parent_cfg["simulation"]["post_gap_burnin_transitions"])); candidates=candidate_grid(dataset.analysis_start,dataset.analysis_end,cfg["frozen_observed_method"]["candidate_fractions"])
    rule=load_support_rule(WORKSPACE/"configs"/"rs_cusum_sensitivity.yaml","primary"); screen=screen_candidates(panel.support_view(),candidates,dataset.analysis_start,dataset.analysis_end,rule)
    support={item.candidate:item for item in screen.candidates if item.admissible}
    if tuple(sorted(support))!=tuple(cfg["frozen_observed_method"]["base_candidates"]): raise ValueError(f"M4 bootstrap support lost frozen N0 candidates: {sorted(support)}")
    f=cfg["frozen_observed_method"]
    fits={candidate:fit_candidate(panel,candidate,dataset.analysis_start,dataset.analysis_end,float(f["gamma"]),float(f["ridge_alpha"]),int(f["fqi_max_iterations"]),float(f["fqi_tolerance"]),"patient_balanced","patient") for candidate in sorted(support)}
    midpoint=dataset.analysis_start+int(np.floor(.5*(dataset.analysis_end-dataset.analysis_start)+.5)); ids=tuple(sorted(panel.patient_ids))
    boundaries={candidate:boundary_factor(item,dataset.analysis_start,dataset.analysis_end,len(ids),"harmonic_active_patients") for candidate,item in support.items()}
    return PreparedAnalysis(dataset,panel,candidates,support,fits,action_feature(evaluation_states,evaluation_actions),midpoint,ids,boundaries)


def run_m4(prepared: PreparedAnalysis,cfg: dict[str,Any],parent_cfg: dict[str,Any],evaluation_states: np.ndarray,evaluation_actions: np.ndarray,draws: int,seed: int) -> ProcessDraws:
    if draws<=0: raise ValueError("M4 draws must be positive")
    model=fit_generative_null(prepared.dataset,cfg); candidates=tuple(cfg["frozen_observed_method"]["base_candidates"]); c=len(candidates); z=len(evaluation_actions)
    raw=np.full((draws,c,z),np.nan); normalized=np.full_like(raw,np.nan); joint=np.full_like(raw,np.nan); valid=np.zeros(draws,bool); errors={}; design_changed=np.zeros(draws,bool); reward_changed=np.zeros(draws,bool)
    all_candidate=np.zeros(draws,bool); design_rows=[]; original=prepared.dataset; w_checks=[]; iteration_checks=[]
    for draw in range(draws):
        try:
            dataset,diag=simulate_dataset(model,seed,draw,cfg); design_changed[draw]=not(np.array_equal(dataset.states,original.states) and np.array_equal(dataset.actions,original.actions)); reward_changed[draw]=not np.array_equal(dataset.rewards,original.rewards)
            boot=prepare_bootstrap_analysis(dataset,cfg,parent_cfg,evaluation_states,evaluation_actions); all_candidate[draw]=tuple(sorted(boot.fits))==candidates
            br,bn,bj=observed_arrays(boot); raw[draw]=br; normalized[draw]=bn; joint[draw]=bj; valid[draw]=np.all(np.isfinite(bj));
            for fit in boot.fits.values():
                w_checks.extend([float(np.linalg.norm(fit.left.model.influence_matrix)),float(np.linalg.norm(fit.right.model.influence_matrix))]); iteration_checks.extend([fit.left.model.iterations,fit.right.model.iterations])
            design_rows.append(diag)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            errors[str(draw)]=f"{type(error).__name__}: {error}"
    raw[~valid]=np.nan; normalized[~valid]=np.nan; joint[~valid]=np.nan
    diagnostics={"algorithm":"M4_stationary_fitted_parametric_random_design","valid_draws":int(np.sum(valid)),"failed_draws":int(np.sum(~valid)),"failure_messages":errors,
                 "design_changed_fraction":float(np.mean(design_changed)),"reward_changed_fraction":float(np.mean(reward_changed)),"all_seven_candidates_fraction":float(np.mean(all_candidate)),
                 "refit_count":int(np.sum(valid))*2*c,"W_recomputed_count":int(np.sum(valid))*2*c,"V_recomputed_count":int(np.sum(valid))*c,
                 "mean_FQI_iterations":float(np.mean(iteration_checks)) if iteration_checks else np.nan,"W_checksum_sd":float(np.std(w_checks,ddof=1)) if len(w_checks)>1 else 0.0,
                 "design_action_rate_sd":float(np.std([row["action_rate"] for row in design_rows],ddof=1)) if len(design_rows)>1 else 0.0,
                 "design_state_mean_sd":float(np.std([row["state_mean"] for row in design_rows],ddof=1)) if len(design_rows)>1 else 0.0,
                 "conditional_reward_innovation_sd_mean":float(np.mean([row["innovation_sd"] for row in design_rows])) if design_rows else np.nan,
                 "pseudo_reward_used":False,"patient_weight_used":False,"support_rescreened":True,"generative_model":model.diagnostics}
    return ProcessDraws("M4",raw,normalized,joint,valid,np.empty((draws,0)),None,None,np.empty((0,c,z)),diagnostics)
