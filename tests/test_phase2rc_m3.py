import unittest

import numpy as np

from src.rs_cusum.patient_balance import action_feature
from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2rb.engine import observed_process,prepare_analysis
from src.rs_cusum_phase2rc.m3_engine import (
    all_one_identity,
    draw_patient_weights,
    run_m3,
    weighted_side_refit,
)
from src.rs_cusum_phase2rc.protocol import WORKSPACE,load_raw_config


class Phase2RCM3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_raw_config(); cls.parent=load_phase2_config()
        with np.load(WORKSPACE/cls.cfg["frozen_observed_method"]["evaluation_grid_file"],allow_pickle=False) as data:
            states=np.asarray(data["states"],float); actions=np.asarray(data["actions"],int)
        cls.prepared=prepare_analysis("N0_complete_balanced_null",918273,cls.cfg,cls.parent,states,actions)
        cls.exp_weights=draw_patient_weights("normalized_exponential",3,12,123)
        cls.exp=run_m3(cls.prepared,cls.cfg,3,123,"normalized_exponential",cls.exp_weights)
        cls.identity=all_one_identity(cls.prepared,cls.cfg)

    def _left_records(self):
        return tuple(record for record in self.prepared.panel.records if record.elapsed_index<self.prepared.midpoint)

    def test_01_same_patient_weight_shared_across_all_transitions(self):
        side=weighted_side_refit(self._left_records(),self.prepared.global_patient_ids,self.exp_weights,self.cfg,allow_zero=False)
        record_ids=np.asarray([record.patient_id for record in self._left_records()],object)
        for index,pid in enumerate(side.patient_ids):
            self.assertAlmostEqual(float(side.transition_weights[0,record_ids==pid].sum()),float(side.patient_objective_weights[0,index]),places=13)

    def test_02_same_patient_weight_shared_across_candidates(self):
        self.assertTrue(self.exp.diagnostics["same_patient_weight_all_candidates"])
        np.testing.assert_array_equal(self.exp.multipliers,self.exp_weights)

    def test_03_all_weights_one_reproduce_observed_fqi(self):
        self.assertLessEqual(self.identity["maximum_beta_error"],2e-10)

    def test_04_all_weights_one_centered_increment_is_zero(self):
        self.assertLessEqual(self.identity["maximum_centered_increment"],2e-10)

    def test_05_weighted_objective_patient_totals_equal_normalized_weights(self):
        side=weighted_side_refit(self._left_records(),self.prepared.global_patient_ids,self.exp_weights,self.cfg,allow_zero=False)
        expected=self.exp_weights[:,[self.prepared.global_patient_ids.index(pid) for pid in side.patient_ids]]
        expected=expected/expected.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(side.patient_objective_weights,expected,rtol=0,atol=1e-14)

    def test_06_support_is_never_rescreened(self):
        self.assertEqual(self.exp.diagnostics["support_candidates_before"],self.exp.diagnostics["support_candidates_after"])

    def test_07_no_pseudo_reward_is_used(self):
        self.assertFalse(self.exp.diagnostics["pseudo_reward_used"])

    def test_08_trajectory_fields_remain_linked_and_unchanged(self):
        before=[(r.patient_id,r.elapsed_index,r.action,r.reward_binary,r.state.copy(),r.next_state.copy()) for r in self.prepared.panel.records]
        run_m3(self.prepared,self.cfg,2,456,"normalized_exponential")
        after=[(r.patient_id,r.elapsed_index,r.action,r.reward_binary,r.state,r.next_state) for r in self.prepared.panel.records]
        for left,right in zip(before,after):
            self.assertEqual(left[:4],right[:4]); np.testing.assert_array_equal(left[4],right[4]); np.testing.assert_array_equal(left[5],right[5])

    def test_09_every_draw_refits_left_and_right_for_every_candidate(self):
        self.assertEqual(self.exp.diagnostics["refit_count"],3*2*7)

    def test_10_every_draw_recomputes_W(self):
        self.assertEqual(self.exp.diagnostics["W_recomputed_count"],3*2*7)
        self.assertGreater(self.exp.diagnostics["W_checksum_sd"],0)

    def test_11_every_draw_recomputes_V(self):
        self.assertEqual(self.exp.diagnostics["V_recomputed_count"],3*7)

    def test_12_seed_is_exactly_deterministic(self):
        first=run_m3(self.prepared,self.cfg,3,999,"normalized_exponential")
        second=run_m3(self.prepared,self.cfg,3,999,"normalized_exponential")
        np.testing.assert_array_equal(first.raw,second.raw); np.testing.assert_array_equal(first.normalized,second.normalized)

    def test_13_invalid_or_nonpositive_exp_weight_hard_fails(self):
        bad=np.ones((1,12)); bad[0,0]=0
        with self.assertRaises(ValueError): run_m3(self.prepared,self.cfg,1,0,"normalized_exponential",bad)
        bad[0,0]=-1
        with self.assertRaises(ValueError): run_m3(self.prepared,self.cfg,1,0,"normalized_exponential",bad)

    def test_14_multinomial_zero_count_does_not_change_frozen_candidates(self):
        weights=draw_patient_weights("multinomial_cluster_counts",5,12,77)
        self.assertTrue(np.any(weights==0))
        result=run_m3(self.prepared,self.cfg,5,77,"multinomial_cluster_counts",weights)
        self.assertEqual(result.diagnostics["support_candidates_before"],tuple(sorted(self.prepared.support)))
        self.assertEqual(result.raw.shape[1],len(self.prepared.support))

    def test_15_observed_statistic_is_unchanged(self):
        before=observed_process(self.prepared)[0]
        run_m3(self.prepared,self.cfg,2,818,"normalized_exponential")
        after=observed_process(self.prepared)[0]
        self.assertEqual(before,after)

    def test_16_failure_is_exception_not_pvalue_one(self):
        with self.assertRaises(ValueError): draw_patient_weights("gaussian",2,12,1)

    def test_17_weight_laws_have_unit_mean_and_nonnegative_values(self):
        exp=draw_patient_weights("normalized_exponential",50,12,1); mult=draw_patient_weights("multinomial_cluster_counts",50,12,1)
        np.testing.assert_allclose(exp.mean(axis=1),1,rtol=0,atol=1e-14)
        np.testing.assert_allclose(mult.mean(axis=1),1,rtol=0,atol=1e-14)
        self.assertTrue(np.all(exp>0)); self.assertTrue(np.all(mult>=0))

    def test_18_block_sparse_weighted_fqi_matches_dense_reference(self):
        records=self._left_records(); weights=self.exp_weights[:2]
        block=weighted_side_refit(records,self.prepared.global_patient_ids,weights,self.cfg,allow_zero=False)
        states=np.vstack([r.state for r in records]); next_states=np.vstack([r.next_state for r in records]); actions=np.asarray([r.action for r in records]); rewards=np.asarray([r.reward_binary for r in records])
        X=action_feature(states,actions); X0=action_feature(next_states,np.zeros(len(records),int)); X1=action_feature(next_states,np.ones(len(records),int))
        for draw in range(2):
            tw=block.transition_weights[draw]; H=X.T@(tw[:,None]*X)+self.cfg["frozen_observed_method"]["ridge_alpha"]*np.eye(X.shape[1]); beta=np.zeros(X.shape[1])
            for _ in range(self.cfg["frozen_observed_method"]["fqi_max_iterations"]):
                target=rewards+self.cfg["frozen_observed_method"]["gamma"]*np.maximum(X0@beta,X1@beta); new=np.linalg.solve(H,X.T@(tw*target))
                delta=np.linalg.norm(new-beta); beta=new
                if delta<=self.cfg["frozen_observed_method"]["fqi_tolerance"]: break
            chosen=np.where(((X1@beta)>(X0@beta))[:,None],X1,X0)
            W=H-self.cfg["frozen_observed_method"]["gamma"]*(X.T@(tw[:,None]*chosen))
            np.testing.assert_allclose(block.beta[draw],beta,rtol=0,atol=3e-13)
            np.testing.assert_allclose(block.W[draw],W,rtol=0,atol=3e-13)


if __name__=="__main__": unittest.main()
