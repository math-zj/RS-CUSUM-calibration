# Phase 2R-E Frozen Protocol: Joint-Tail Failure Localization

Phase 2R-E starts from the immutable Phase 2R-D gate `M4_MECHANISM_JOINT_TAIL_FAIL`. It is mechanism forensics, not method optimization. It cannot modify M4, create M4.1/M4.2/M5, alter the old gate, run D3, or enter OhioT1DM, Phase 3, transfer-null, alternatives, power, or changepoint analysis.

## Existing-draw decomposition

The primary input is the already-saved D2 process: 120 oracle N0 processes and 120 outer clusters of 100 saved M4 processes, each with seven candidate coordinates by 64 evaluation coordinates. Coordinates are candidate-major. The 64 evaluation points are 32 frozen base states duplicated for value actions 0 and 1; metadata include phenotype, source patient, and source elapsed index.

The primary event is the two-sided absolute exceedance used by D4. Signed upper-tail exceedance is secondary. At 90%, 95%, 97.5%, and 99% oracle coordinate-wise thresholds, the analysis compares marginal exceedance, pairwise joint exceedance, directional conditional exceedance `P(E_j|E_i)`, Spearman dependence, and extreme co-occurrence. All 448x447 ordered off-diagonal pairs are retained. The original 0.1418 quantity is explicitly distinguished from the distribution of pairwise absolute errors: it is the absolute difference between the oracle and M4 95th percentiles of their conditional-dependence distributions.

Structure is frozen before computation: same/different candidate, candidate-index distance, same/different evaluation point, same base state, same/cross value-action component, phenotype pairing, source-patient pairing, evaluation-coordinate distance, and global flattened distance. A 448x448 numeric heatmap table and the top 500 offending pairs are saved.

## Monte Carlo uncertainty

Uncertainty uses 25,000 frozen uniformly sampled ordered pairs and 500 resamples. Oracle rows and M4 outer clusters are resampled separately and jointly; the 100 saved inner draws per M4 cluster remain grouped. Nested stability uses oracle sample sizes 30/60/90/120, M4 outer-cluster counts 30/60/90/120, and 25/50/75/100 saved inner draws per outer cluster. The metric is classified unstable if its MC SE exceeds 0.04, its 95% interval width exceeds 0.16, or its nested-subsample range exceeds 0.10. This classification cannot revise the Phase 2R-D failure.

## Frozen hybrid factorial

The hybrid experiment uses 32 new diagnostic outer N0 datasets and 79 draws per outer dataset per arm. All arms use common random numbers within outer replicate and draw. `true_all` uses the exact N0 parameters. `fitted_all` reproduces the M4 generative construction. Single replacements set initial-state, full policy, policy random-effect SD alone, transition, or reward components to truth. `true_design` replaces initial, policy, and transition together. `true_innovation_scales` replaces Gaussian scales only; `true_structural_coefficients` replaces regression means/slopes while retaining fitted scales and random-effect SD.

Every generated dataset goes through the unchanged Phase 2R-D panel, support screen, seven left/right FQI fits, W/Delta/V construction, and full 7x64 statistic. These arms are diagnostic interventions, not candidate inference methods. No arm may be chosen by empirical size, and no fallback is permitted.

The true N0 family is Gaussian and innovation-independent. Therefore residual skewness, kurtosis, cross-coordinate correlation, lag-one dependence, patient heterogeneity, and extreme clustering are audited rather than presumed harmless. The factorial is intended to separate plug-in/conditioning effects from policy heterogeneity, design dynamics, reward fitting, and innovation-scale estimation.

## Frozen localization decision

`TAIL_SOURCE_LOCALIZED` requires a stable metric, at least 50% error reduction from one single-component truth replacement, residual p95 error at most 0.08, and at least 20 percentage points more reduction than the next single component. `TAIL_SOURCE_PARTIALLY_LOCALIZED` requires a stable metric and at least 25% reduction for one component/family without meeting the unique-source rule. `TAIL_SOURCE_DIFFUSE` requires a stable metric and at least two distinct single components each reducing error by 20%. `TAIL_METRIC_UNSTABLE` is triggered by any frozen MC instability rule and overrides component attribution. Otherwise the gate is `FORENSIC_INCONCLUSIVE`.

No localization outcome upgrades M4 or authorizes D3.

