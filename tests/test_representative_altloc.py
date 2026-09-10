import numpy as np
from src.rs_cusum_representative_altloc.generator import generate
from src.rs_cusum_phase2.protocol import load_phase2_config
def test_representative_generator_has_fixed_complete_change():
 s={'id':'reward_moderate','change_type':'reward','effect_level':'moderate','effect_size':.60}
 d=generate(s,123,load_phase2_config(),.5,.25)
 assert d.true_change_point==144 and d.states.shape==(12,241,21) and np.all(d.observed_mask)
