# Phase 2R-G Frozen Protocol: Fresh Locked N0 Confirmation

Phase 2R-G is a one-time holdout confirmation of the unchanged Phase 2R-D M4 under only the complete balanced synthetic N0 family. Historical conclusions remain immutable: Phase 2R-D=`M4_MECHANISM_JOINT_TAIL_FAIL`, Phase 2R-E=`TAIL_METRIC_UNSTABLE`, and Phase 2R-F=`M4_JOINT_TAIL_CONFIRMED`. This phase does not reinterpret the old gates and is not the unused Phase 2R-D D3.

## Frozen method and scope

The M4 implementation is exactly `src/rs_cusum_phase2rd/engine.py` with SHA-256 `EC7E7DA7D8625736CEAC2B872CD739DEBA29792CD535B8AD533D5266251AF977`. Its fitted parametric stationary-null model, random-design regeneration, patient random effects, state/action/reward simulation, FQI, support screen, CUSUM process, critical values, and 199 inner draws are unchanged. The stable joint-tail implementation is exactly the Phase 2R-F metric source with SHA-256 `1C45FB63F5D3429C3A8BDA24355CD55BF81F823BEF0B20D0A2AE4AD988567F98`.

Only `N0_complete_balanced_null` is run. No additional null family is genuinely fresh and pre-defined, so the result must be called `Fresh Locked N0 Confirmation`, not cross-distribution robustness or general global-null validation.

## Pre-run precision plan

The frozen design uses two new independent oracle banks of 2,500 processes each, 300 new outer N0 datasets, and 199 M4 inner draws per outer. The primary comparison uses a pre-specified cluster-balanced M4 bank of 2,500 rows: nine draws from outer 0–99 and eight from outer 100–299. This is the same sample design validated in Phase 2R-F, where EISJEM MC SE was 0.001035 and CI width was 0.004070, q99 had about 25 exceedances per coordinate, the legacy q95 interval width was 0.011971, and the D4 size MC SE was about 0.0113. These values are sufficiently smaller than the frozen precision/tolerance scales, so replication is neither reduced nor increased. The plan was fixed before any Phase 2R-G formal output.

## Frozen metrics and gates

Primary joint-tail evidence is EISJEM at q={0.90,0.95,0.975,0.99}: the equal-weight mean standardized off-diagonal joint-exceedance MAE for oracle-primary versus balanced M4, minus its oracle-primary versus oracle-reference finite-sample noise floor. It is not truncated at zero. Joint-tail PASS requires its 95% cluster-bootstrap upper bound to be at most 0.015, MC SE at most 0.0075, CI width at most 0.03, all inherited q95 secondary tolerances to pass, and no stable legacy lower bound above 0.12.

Center, covariance, marginal-tail, maximum-distribution, type-I, and engineering thresholds are copied without loosening from the Phase 2R-F frozen protocol. Oracle rows are resampled independently; M4 outer datasets are cluster-resampled while retaining their selected inner blocks. Type-I uses each fresh outer dataset's observed statistic against its own 199 M4 draws. The complete numeric definitions are in `configs/rs_cusum_phase2rg.yaml` and `phase2rg_protocol.json`.

The final gate is binary for scientifically valid output: every required domain PASS gives `M4_FRESH_GLOBAL_NULL_PASS`; any required scientific/inference domain FAIL gives `M4_FRESH_GLOBAL_NULL_FAIL`. `M4_FRESH_GLOBAL_NULL_INVALID` is reserved only for seed contamination, protocol violation, corruption, hash mismatch, or unrecoverable engineering invalidity. A scientific FAIL cannot be relabeled INVALID and cannot trigger a replacement holdout.

## Seed and execution lock

All outer, oracle, bootstrap-base, design, reward, dataset, and uncertainty seeds are generated before the formal run, recorded in NPZ/JSON, SHA-256 hashed, unique within Phase 2R-G, and checked against every historical integer seed value discoverable in Phase 2R-B/C/D/E/F registries and replicate files. Any collision hard-fails. Formal computation uses atomic checkpoints, validates exact registered seeds and shapes, skips completed checkpoints, and runs only missing frozen seeds after interruption.

After formal generation begins, the algorithm, metrics, thresholds, gates, seeds, and sample sizes cannot change. No M4.1/M4.2/M5, mechanism exploration, recalibration, replacement holdout, OhioT1DM, Phase 3, transfer-null, alternatives, power, changepoint, or version 1.0.2 establishment is permitted in this phase. Even a PASS ends this task and only supports a separately authorized decision about a later research stage.

