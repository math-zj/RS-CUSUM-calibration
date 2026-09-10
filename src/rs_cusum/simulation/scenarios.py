"""Pre-declared synthetic scenario families; none inspect statistical results."""

from __future__ import annotations

from .generator import SimulationDesign


def phase1_scenarios(seed: int = 20260828) -> dict[str, SimulationDesign]:
    base = dict(n_patients=12, n_transitions=120, state_dimension=4)
    return {
        "complete_balanced_null": SimulationDesign(
            "complete_balanced_null", seed=seed, missing_pattern="complete", alternative="null", **base
        ),
        "complete_balanced_one_change": SimulationDesign(
            "complete_balanced_one_change",
            seed=seed + 1,
            missing_pattern="complete",
            alternative="one_change",
            **base,
        ),
        "monotone_dropout": SimulationDesign(
            "monotone_dropout", seed=seed + 2, missing_pattern="monotone_dropout", **base
        ),
        "intermittent_missing_with_reentry": SimulationDesign(
            "intermittent_missing_with_reentry", seed=seed + 3, missing_pattern="intermittent_reentry", **base
        ),
        "unequal_length": SimulationDesign(
            "unequal_length", seed=seed + 4, missing_pattern="unequal_length", **base
        ),
        "single_patient_dominance": SimulationDesign(
            "single_patient_dominance", seed=seed + 5, missing_pattern="single_patient_dominance", **base
        ),
        "rare_action1": SimulationDesign(
            "rare_action1", seed=seed + 6, missing_pattern="complete", rare_action1=True, **base
        ),
        "dropout_plus_rare_action1": SimulationDesign(
            "dropout_plus_rare_action1",
            seed=seed + 7,
            missing_pattern="dropout_plus_intermittent",
            rare_action1=True,
            **base,
        ),
        "intermittent_plus_dominance": SimulationDesign(
            "intermittent_plus_dominance",
            seed=seed + 8,
            missing_pattern="dropout_plus_intermittent",
            action_intercept=-2.0,
            **base,
        ),
    }
