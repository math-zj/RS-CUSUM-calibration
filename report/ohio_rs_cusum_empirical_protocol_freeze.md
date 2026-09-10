# OhioT1DM empirical protocol freeze

## Gate

**`OHIO_PROTOCOL_NOT_FEASIBLE`**. This is an outcome-blind negative protocol freeze, not an Ohio analysis. No RS-CUSUM statistic, fitted Q function, bootstrap draw, p-value, or estimated changepoint was computed or inspected.

## Inherited, frozen material

The 5-minute one-step state, binary bolus-start action, binary-TIR primary reward, clinical scaling, gamma, and the primary 4.5--5.5-minute edge rule are inherited without modification from `configs/mdp_5min_protocol.yaml`. The patient-as-cluster multiplier unit and the primary support thresholds (66 observations of each action per side, at least 6 active patients/side, at least 5 both-side patients, at least 4 action-specific patients/side, A=1 dominance <=0.60, total dominance <=0.45) are inherited from the Phase-1 support protocol.

The previous landmark scan is usable only as outcome-blind availability evidence. Its earliest training-reset 6-hour landmark at which all 12 patients were eligible was 42 h. That scan deliberately used 4--6 minute timing to study jitter, while the frozen primary MDP uses 4.5--5.5 minutes. Consequently 42 h is recorded as a future **proposal**, never as an authorized formal landmark. The existing scan cannot certify a primary-rule formal trajectory.

Testing is independently disqualified even under the more permissive scan: at reset landmark 24 h it has 12 entry-eligible patients but only one active patient in the central follow-up region, a worst side with four A=1 transitions, and 100% maximum A=1 dominance. This violates the inherited primary support rule without reference to outcomes.

## Decisive M4 compatibility failure

The locked M4 source (`src/rs_cusum_phase2rd/engine.py`, SHA-256 `EC7E7DA7D8625736CEAC2B872CD739DEBA29792CD535B8AD533D5266251AF977`) explicitly accepts only a complete balanced `N0_complete_balanced_null` `SyntheticDataset`. It has no frozen or validated constructor for Ohio trajectories, no real-data stationary fitted-null specification, and no defined M4 bootstrap unit for varying termination/censoring. Supplying one now would create a new inference method, contrary to this protocol-freeze stage.

Thus neither split is eligible for formal Original-vs-M4 empirical inference: training has an unvalidated exact-time trajectory/support transition and no valid M4 calibration interface; testing has those failures plus objectively inadequate support. This is not evidence that Ohio lacks a change; it means the locked method cannot currently support a defensible confirmatory Ohio claim.

## Immutability

No state/action/reward definition was changed. No patient was selectively removed. No outcome-dependent landmark, candidate, seed, threshold, or bootstrap scheme was selected. A future real-data stage would require a separately authored and validated real-data M4 calibration method and a new pre-outcome protocol; it must not be introduced silently as a rerun.
