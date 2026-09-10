# Phase 2R-H frozen protocol

## Question and scope

This phase asks whether unchanged M4 controls false positives when the data are stationary and contain no change point, but their data-generating law differs from M4's fitted Gaussian stationary parametric null. It is a null-robustness study, not power, changepoint, OhioT1DM, or method development.

The immutable historical record is Phase 2R-D=`M4_MECHANISM_JOINT_TAIL_FAIL`, Phase 2R-E=`TAIL_METRIC_UNSTABLE`, Phase 2R-F=`M4_JOINT_TAIL_CONFIRMED`, and Phase 2R-G=`M4_FRESH_GLOBAL_NULL_PASS`. The M4 engine SHA-256 is frozen at `EC7E7DA7D8625736CEAC2B872CD739DEBA29792CD535B8AD533D5266251AF977`.

## Historical scenario audit

The original simulator already defined N0 and missingness/action-support scenarios N1–N8, and those definitions were inspected during Phase 2 and Phase 2R-B planning. Phase 2R-B's planned transfer run was stopped and its transfer summary is `NOT_RUN_BY_GATE`, but the families are not called fresh. Phase 2R-D/F/G formally used N0. Therefore H0 is an assay reference, while H1–H8 below are newly fixed stationary complete-data misspecification laws. Claims use “pre-registered transfer-null,” not “never conceived before.”

All Phase-H laws use 12 independent patients, 240 transitions, 48 burn-in transitions, 21 state variables, complete observation, fixed parameters over time, and `true_change_point=None`. The conditional reward and transition means are unchanged except for the explicitly listed stationary policy/composition differences; no scenario contains a time-index indicator or a structural break.

## Frozen scenarios

- H0 reference N0: balanced 6/6 phenotype, action target .50, Gaussian policy random effect SD .15, Gaussian state SD .45, Gaussian reward SD .60.
- H1 patient heterogeneity: policy random effect is standardized Student-t(5), SD .30.
- H2 imbalanced design: 8 negative and 4 positive phenotypes; action target .35.
- H3 reward heavy tail: standardized Student-t(5) reward innovation, SD .60.
- H4 transition heavy tail: standardized Student-t(5) innovations for X0 and other dynamic state coordinates, SD .45.
- H5 reward heteroskedasticity: Gaussian reward innovation with conditional SD `.60*(.75+.35*abs(tanh(X0))+.20*A)`, hence .45–.78, with the same conditional mean.
- H6 mild temporal dependence: per-patient stationary Gaussian AR(1) reward innovation with rho=.30 and marginal SD .60.
- H7 weaker overlap stress: action target .25, X0 policy coefficient .75, Gaussian policy random-effect SD .20.
- H8 combined moderate stress: 8/4 phenotype composition, action target .35, standardized t(7) policy random effect SD .225, and standardized t(7) reward innovation SD .60.

`SyntheticDataset.scenario` must remain `N0_complete_balanced_null` solely because frozen M4 rejects other labels. The true scenario identifier is mandatory in the Phase-H registry, task, checkpoint, status, summary, and manifest. This compatibility label must never be reported as the actual DGP.

## Sample size and balance

For every scenario, two independent oracle banks contain 1000 processes each. There are 200 independent outer datasets and 199 M4 draws per outer. Max/process comparisons use exactly the first five registered draw positions per outer, yielding 1000 cluster-balanced M4 processes; a failed selected position is not replaced. Type-I uses all 199 draws within each outer.

At a .05 rate and n=200, MCSE is .0154; 10/200 has a 95% Wilson interval approximately [.0274,.0896]. At a .10 rate MCSE is .0212. This plan distinguishes gross inflation beyond the frozen D4 .10 tolerance without an after-the-fact increase. Each oracle bank has about 10 expected q99 exceedances per coordinate, so q99 and joint-process results are explicitly diagnostic rather than precision claims comparable to Phase F/G.

## Gates, multiplicity, and stop rules

Primary per-scenario inference requires: D4 size at .05 <=.10; every D0–D4 size at .05 <=.13; D4 size at .01 <=.04; D4 nominal .05 lies in its 95% Wilson interval or the observed size is within one MCSE; D2 and D4 KS <=.20; D0–D3 q95 ratios in [.80,1.20], D4 in [.88,1.12]; M4 probability above the oracle D4 q95 in [.02,.09]; and testable fraction >=.99.

Center, covariance, marginal-tail, EISJEM, and joint-tail thresholds are mechanism diagnostics. Passing inference with a diagnostic failure is `MILD_DEGRADATION`, not `NULL_VALIDITY_FAIL`. A primary inference failure is `NULL_VALIDITY_FAIL`. Technical/protocol/hash/seed corruption or failure rates beyond 1% is `ENGINEERING_INVALID`.

The primary family H1–H6 is evaluated by an intersection-union rule: all must pass, which does not require an alpha adjustment to protect an erroneous overall PASS. Per-scenario 95% Wilson intervals are descriptive; simultaneous Bonferroni-Wilson intervals at confidence `1-.05/6` are also reported as sensitivity and are not an additional gate.

Overall `PASS` requires H0 and H1–H8 primary inference validity, at most two diagnostic-only degradations among H1–H8, and no common diagnostic domain failing in three or more H1–H8 scenarios. `PARTIAL` applies when H1–H6 retain inference validity but a stress scenario fails or diagnostic degradation is systematic. Any H1–H6 (or H0 assay) null-validity failure is `FAIL`. Any required technical invalidity is `INVALID`.

Only `M4_TRANSFER_NULL_PASS` grants eligibility for a separately authorized Alternative/Power Validation. This phase never grants OhioT1DM, Phase 3, changepoint, or version 1.0.2 authority and stops after its report and manifest.
