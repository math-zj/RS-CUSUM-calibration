import unittest

import numpy as np

from src.rs_cusum_phase2.protocol import load_phase2_config
from src.rs_cusum_phase2rb.engine import prepare_analysis
from src.rs_cusum_phase2rc.c0_engine import (
    full_refit_process,
    influence_process,
    nested_design_process,
    observed_arrays,
    patient_design_rows,
    regenerate_reward_matrix,
)
from src.rs_cusum_phase2rc.protocol import WORKSPACE,load_raw_config


class Phase2RCC0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_raw_config(); cls.parent=load_phase2_config()
        with np.load(WORKSPACE/cls.cfg["frozen_observed_method"]["evaluation_grid_file"],allow_pickle=False) as data:
            cls.states=np.asarray(data["states"],float); cls.actions=np.asarray(data["actions"],int)
        cls.prepared=prepare_analysis("N0_complete_balanced_null",918273,cls.cfg,cls.parent,cls.states,cls.actions)
        cls.m0=influence_process(cls.prepared,"M0",5,111)
        cls.m1=influence_process(cls.prepared,"M1",5,111)
        cls.m2=full_refit_process(cls.prepared,cls.cfg,3,222)

    def test_m0_m1_share_identical_raw_multiplier_process(self):
        np.testing.assert_array_equal(self.m0.raw,self.m1.raw)

    def test_process_shapes_are_full_candidate_by_state_grid(self):
        self.assertEqual(self.m2.raw.shape,(3,7,64))
        self.assertEqual(self.m2.joint.shape,(3,7,64))
        self.assertEqual(self.m2.zero_raw.shape,(7,64))

    def test_zero_perturbation_is_an_actual_extra_full_refit(self):
        self.assertEqual(self.m2.diagnostics["refit_count"],(3+1)*2*7)
        self.assertTrue(np.all(np.isfinite(self.m2.zero_raw)))

    def test_observed_arrays_keep_frozen_dimensions(self):
        raw,normalized,joint=observed_arrays(self.prepared)
        self.assertEqual(raw.shape,(7,64)); self.assertEqual(normalized.shape,(7,64)); self.assertEqual(joint.shape,(7,64))

    def test_patient_projection_retains_original_patient_identity(self):
        rows=patient_design_rows(self.prepared,self.m1,self.m2)
        self.assertEqual(len(rows),12)
        self.assertEqual({row["patient_id"] for row in rows},set(self.prepared.global_patient_ids))

    def test_conditional_reward_regeneration_changes_only_rewards(self):
        records=tuple(sorted(self.prepared.panel.records,key=lambda record:(record.patient_id,record.elapsed_index)))
        states_before=self.prepared.dataset.states.copy(); actions_before=self.prepared.dataset.actions.copy()
        rewards=regenerate_reward_matrix(self.prepared.dataset,np.asarray([10,11]),records,0.60)
        self.assertEqual(rewards.shape,(2,len(records)))
        self.assertGreater(np.linalg.norm(rewards[0]-rewards[1]),0)
        np.testing.assert_array_equal(self.prepared.dataset.states,states_before)
        np.testing.assert_array_equal(self.prepared.dataset.actions,actions_before)

    def test_nested_process_uses_all_seven_candidates_and_64_points(self):
        process=nested_design_process(99117,np.asarray([1,2,3]),self.cfg,self.parent,self.states,self.actions)
        self.assertEqual(process.shape,(3,7,64))
        self.assertTrue(np.all(np.isfinite(process)))


if __name__=="__main__": unittest.main()
