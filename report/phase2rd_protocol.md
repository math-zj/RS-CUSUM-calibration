# Phase 2R-D Frozen Protocol: M4 Random-Design Joint-Process Bootstrap

Status at entry: `DESIGN_VARIATION_HYPOTHESIS_SUPPORTED_BUT_M3_PARTIAL`. Parent estimator remains `1.0.1-phase1`. This phase is null-inference research only: 1.0.2, Phase 3, transfer-null, OhioT1DM, alternatives, power and changepoint are closed.

## D0 audit conclusion

The N0 generator has four distinct random sources: initial non-phenotype state; a patient policy random intercept; recursive Bernoulli action and Markov state innovations; and conditional Gaussian reward innovation. M0/M1 condition on the complete realized trajectory and linearize fixed observed patient contributions. M2 fixes states/actions/history and perturbs only restricted-null rewards. M3 fixes every realized trajectory and varies only patient objective weights. Hence neither M2 nor M3 generates new within-patient state/action paths.

C0 identifies the missing component: fixed-design reward variation accounts for 0.497 of pointwise variance and 0.322 of the leading eigenvalue, while between-design variation accounts for 0.694 of the leading eigenvalue. M1 separately shows that second-order covariance recovery does not imply recovery of the maximum distribution.

## Frozen M4 definition

M4 is the sole primary candidate: a fitted-parametric, stationary-null, random-design full-pipeline bootstrap. For each observed N0 dataset, one common generative model is fitted over all t=0,...,239. It is never split by candidate or left/right side.

The model fixes the observed balanced phenotype strata and estimates: Gaussian initial-state laws; a penalized logistic action model with Gaussian patient policy heterogeneity; Gaussian Markov transition regressions; and a Gaussian conditional reward regression. Each draw uses disjoint deterministic sub-seeds to regenerate the complete state/action design and reward innovations. It then rebuilds the panel, reruns primary support, refits both side FQI models for all seven candidates, recomputes W/Delta/V and evaluates the complete 7x64 studentized CUSUM path. D0-D4 are derived from that path. M4 uses the direct stationary-null bootstrap statistic, not an observed-estimator increment, multiplier, Gaussian covariance surrogate, global scale, or D4 calibration factor.

This is a parametric model-based prototype. It is conditional on the fitted stationary generative parameters and is not asserted to have a ready-made RS-CUSUM-RL theorem. Its structural motivation is that recursive design regeneration supplies `Var(E[T|D])`, reward regeneration supplies `E Var(T|D)`, and fitting all candidates/points on each common trajectory draw preserves nonlinear joint-path and tail dependence.

## Frozen stages

- D1 engineering: R=10, B=49, M4 only.
- D2 development mechanism screen: R=120, B=199, paired M0/M1/M2/M3/M4.
- D3 locked fresh holdout: R=300, B=499, M4 only; it is run once only if every D2 domain passes.

All D1/D2/D3 outer seeds are generated before D1. D3 values remain unloaded until a machine-readable D2 pass. No fallback method, multiplier, scale, alternate generative model, or replacement holdout is permitted.

## Frozen gates and rationale

The exact numerical gates are in `configs/rs_cusum_phase2rd.yaml` and `phase2rd_protocol.json`. They jointly cover type-I/critical values, centering/covariance, higher-order joint tails and engineering stability. Passing D4 alone is insufficient.

New tail tolerances use already-existing independent C0-versus-C2 truth samples (300 each), not M4 output. Oracle-to-oracle discrepancies are: pointwise SD 0.992, leading eigenvalue 0.983, D4 q95 0.968, D4 KS 0.060, correlation Frobenius 0.292, joint-exceedance Frobenius 0.875 and kurtosis RMSE 0.918. The high-dimensional pairwise and fourth-moment gates therefore use scientifically meaningful broad anti-pathology tolerances, while center, scale and max-distribution gates remain tighter.

Any post-freeze bug invalidates affected output and requires an explicit new protocol version, new hashes and unused seeds. There is no silent overwrite.

