import numpy as np
import pytest
from src.rs_cusum_realdata_bridge_v0.runner import M4, M4HASH, cfg, draw_statistics, fit_null, make_data, prepare, seed, sha, simulate


def test_fixed_seed_and_frozen_locked_core_are_reproducible():
    c = cfg()
    d1, t1 = make_data('C_action_imbalance_patient_heterogeneity', 17, c)
    d2, t2 = make_data('C_action_imbalance_patient_heterogeneity', 17, c)
    # fixed seed / configured scenario, state, actions and binary reward agree
    assert np.array_equal(t1, t2)
    assert np.array_equal(d1.states, d2.states)
    assert np.array_equal(d1.actions, d2.actions)
    assert np.array_equal(d1.rewards, d2.rewards)
    assert sha(M4) == M4HASH


def test_termination_ids_binary_reward_and_recursive_regeneration():
    c = cfg()
    d, T = make_data('B_unequal_followup_monotone_censoring', 17, c)
    assert T.tolist() == [144] * 4 + [192] * 4 + [240] * 4
    assert set(d.rewards.ravel()) <= {0.0, 1.0}
    assert np.array_equal(d.observed_mask.sum(1), T)  # no post-termination observation
    assert np.all(d.actions[~d.observed_mask] == 0)
    assert np.all(d.rewards[~d.observed_mask] == 0)
    prep, ev = prepare(d, c)
    model = fit_null(d, T, c)
    x1 = simulate(model, 18, c)
    x2 = simulate(model, 18, c)
    x3 = simulate(model, 19, c)
    # same cluster IDs/T_i and one trajectory per patient; new seed changes design.
    assert x1.patient_ids == d.patient_ids
    assert np.array_equal(x1.observed_mask.sum(1), T)
    assert np.array_equal(x1.actions, x2.actions)
    assert np.array_equal(x1.states, x2.states)
    assert not np.array_equal(x1.actions, x3.actions)
    assert set(x1.rewards.ravel()) <= {0.0, 1.0}
    # Each bootstrap panel is rebuilt/refit against the original fixed U_adm.
    rebuilt, _ = prepare(x1, c, ev, tuple(sorted(prep.fits)))
    assert tuple(sorted(rebuilt.fits)) == tuple(sorted(prep.fits))
    assert len(rebuilt.fits) == len(prep.fits)


def test_lost_observed_candidate_is_invalid_not_replaced():
    c = cfg()
    d, _ = make_data('A_complete_balanced_baseline', 31, c)
    prep, ev = prepare(d, c)
    # An impossible fixed candidate is never silently substituted by a newly
    # admissible candidate in the resampled draw.
    with pytest.raises(ValueError, match='INVALID_SUPPORT_DRAW'):
        prepare(d, c, ev, tuple(sorted(prep.fits)) + (999999,))


def test_one_draw_exception_is_recorded_without_seed_replacement_or_abort():
    c = cfg()
    d, T = make_data('A_complete_balanced_baseline', 71, c)
    prep, ev = prepare(d, c)
    model = fit_null(d, T, c)
    task = {'scenario': 'A_complete_balanced_baseline', 'outer_id': 991,
            'dataset_seed': 71, 'bootstrap_seed': 72}
    failed_seed = seed(task['bootstrap_seed'], 'draw', 1)

    def injected_failure(_model, draw_seed, _c):
        if draw_seed == failed_seed:
            raise RuntimeError('deterministic injected draw failure')
        return simulate(_model, draw_seed, _c)

    values, failures, _ = draw_statistics(model, prep, ev, task, c, B=3, simulate_fn=injected_failure)
    assert np.isfinite(values[[0, 2]]).all()
    assert np.isnan(values[1]).all()
    assert len(failures) == 1
    failure = failures[0]
    assert failure['draw_id'] == 1 and failure['draw_seed'] == failed_seed
    assert failure['exception_type'] == 'RuntimeError'
    assert 'deterministic injected draw failure' in failure['exception_message']
    assert 'RuntimeError' in failure['traceback']
