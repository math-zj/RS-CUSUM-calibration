"""Regression test for the Ohio persistence-only repair.

This test intentionally simulates a total Bridge failure without generating any
trajectory.  It verifies that observed and Original outputs are durable before
Bridge calibration is examined.
"""
import json

import numpy as np

from src.ohio_final_empirical_application import runner


def test_all_bridge_draws_invalid_preserves_observed_and_original(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    observed = np.array([1., 2., 3., 4., 5.])
    raw = np.array([[.1, .2]])
    norm = np.array([[.3, .4]])
    joint = np.array([[.5, .6]])
    original_draws = np.ones((199, 5))
    critical = np.array([1., 1., 1., 1., 1.])
    pvalues = np.array([.1, .2, .3, .4, .5])
    support = runner.pd.DataFrame([{'candidate_transition': 72, 'admissible': True}])
    patients = runner.pd.DataFrame([{'patient_id': 'P01', 'T_i': 100}])

    runner.persist_pre_bridge(observed, 72, raw, norm, joint, (72,), original_draws, critical, pvalues, .05, support, patients)
    status = runner.bridge_status(199, 0, [{'failure_category': 'SUPPORT_REJECTION'} for _ in range(199)])
    runner.atomic(tmp_path / 'bridge_calibration_status.json', status)

    assert (tmp_path / 'ohio_final_observed_results.csv').is_file()
    assert (tmp_path / 'original_ohio_result.csv').is_file()
    assert (tmp_path / 'candidate_process.csv').is_file()
    assert (tmp_path / 'ohio_support_diagnostics.csv').is_file()
    assert json.loads((tmp_path / 'bridge_calibration_status.json').read_text())['calibration_available'] is False
    assert json.loads((tmp_path / 'bridge_calibration_status.json').read_text())['failure_reason'] == 'CALIBRATION_UNAVAILABLE_NO_VALID_DRAWS'
