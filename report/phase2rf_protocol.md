# Phase 2R-F Frozen Protocol: Independent Oracle and Stable Joint-Tail Confirmation

Phase 2R-F is a measurement/validation stage. It leaves the Phase 2R-D result `M4_MECHANISM_JOINT_TAIL_FAIL` and the Phase 2R-E result `TAIL_METRIC_UNSTABLE` immutable. It does not modify M4, calibrate a critical value, create M4.1/M4.2/M5, run the old D3, or enter OhioT1DM, Phase 3, transfer-null, alternatives, power, or changepoint analysis.

## Precision plan frozen before new validation draws

The planning calculation used only the already-saved Phase 2R-E `true_all` and `fitted_all` process arrays. At equal bank size 2,500, the planned primary excess metric had bootstrap SD 0.00119 and 95% width 0.00475, while the 99% threshold retained about 25 exceedances per coordinate. Phase 2R-F therefore freezes two independent oracle banks of 2,500 processes each (5,000 total). M4 generates 300 independent outer N0 datasets with 199 unchanged M4 draws per outer dataset (59,700 draws); the primary balanced bank contains exactly 2,500 processes and represents every outer cluster, using 9 draws from outer 0–99 and 8 from outer 100–299.

## Primary metric

For threshold levels q in {0.90, 0.95, 0.975, 0.99}, oracle-primary coordinate thresholds define two-sided events E_j(q). For every ordered off-diagonal pair, the absolute error in joint exceedance probability is averaged and divided by alpha=1-q. The four standardized means receive equal weight; this is ISJEM. The primary EISJEM contrast is:

`ISJEM(oracle-primary, balanced M4) - ISJEM(oracle-primary, oracle-reference)`.

The contrast is not truncated at zero. It remains a joint extreme-event functional rather than covariance, while avoiding division by noisy coordinate-specific conditional-event counts. The independent oracle-reference term measures the same finite-sample L1 noise floor and does not alter M4 or its critical values.

The equivalence margin is 0.015. Confirmation requires the cluster-bootstrap 95% upper bound to be no larger than 0.015, MC SE no larger than 0.0075, CI width no larger than 0.03, and the median primary range over balanced n=1,000 through 2,500 no larger than 0.02. A lower 95% bound above 0.015 gives stable failure. Precision/convergence failure gives instability; all remaining mixed outcomes are inconclusive.

## Secondary and multi-domain requirements

The old q95 conditional-dependence distribution p95 gap remains mandatory, together with pair-error median/p90/p95/p99/max, tail-dependence MAE, joint-exceedance MAE/Frobenius discrepancy, and threshold sensitivity. A stable old gap whose lower 95% bound exceeds 0.12 blocks confirmation. Center, covariance, marginal tails, D0–D4 maxima, empirical type-I error, Wilson intervals, and engineering failures are evaluated under the frozen numerical gates in `configs/rs_cusum_phase2rf.yaml`, predominantly inherited from the Phase 2R-D D2 screen.

Balanced comparisons use n={120,250,500,1000,2500}. Oracle rows are sampled without replacement; M4 selection is cluster-balanced and covers as many distinct outer datasets as possible before adding another draw per cluster. Uncertainty resamples oracle process rows independently and resamples M4 outer datasets as clusters, retaining their selected inner blocks.

## Frozen computation and storage

Every oracle process seed and every M4 design/reward/dataset subseed is registered before formal computation and checked against all prior seed registries. Oracle work is checkpointed in 25-process chunks; M4 is checkpointed by outer dataset. Writes are atomic, completed checkpoints are validated and not repeated, and invalid checkpoints hard-fail. Saved sufficient objects are the full 448-coordinate joint process and D0–D4 reductions; M4 raw/normalized 7x64 arrays are discarded only after their D0–D4 values are recorded. Estimated wall time at 12 workers is approximately 8 minutes for oracle and 60 minutes for M4, with about 140 MB of checkpoint storage.

Even `M4_JOINT_TAIL_CONFIRMED` supplies only independent computational support under synthetic N0. It can qualify M4 for a new Phase 2R-G fresh locked global-null confirmation; it cannot revise Phase 2R-D, establish RS-CUSUM-RL 1.0.2, or open any real-data analysis.

