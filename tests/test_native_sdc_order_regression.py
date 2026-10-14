"""Regression for original WOMD SDC not necessarily being track index zero."""
import numpy as np
from research_probe.native_womd import compact_selected_state,observational_metrics,STATE_ATTRS


def test_sdc_reordered_to_local_zero_and_observational_excludes_sdc():
    n=5
    s={'scenario_id':'test','ego_index':3,'agent_ids':np.array([10,11,12,13,14]),
       'types':np.ones(n,int)}
    for phase,t in [('past',10),('current',1),('future',80)]:
        for attr in STATE_ATTRS:
            d=np.tile(np.arange(n,dtype=float)[:,None],(1,t))
            if attr=='valid':d[:]=1
            if attr=='velocity_x':d[:]=3
            if attr=='velocity_y':d[:]=0
            s[f'{phase}_{attr}']=d
    f={'neighbor_indices':[4,1]}
    compact=compact_selected_state(s,f)
    assert compact['ego_index']==0
    assert compact['agent_ids'].tolist()==[13,14,11]
    assert compact['current_x'][:,0].tolist()==[3,4,1]
    observed=observational_metrics(s,f['neighbor_indices'])
    assert len(observed['observed_min_longitudinal_accel_mps2'])==3
