import json,copy,hashlib
import pytest
from ride.v15_stress_frozen import derive
from ride.v14_frozen_replay import load_frozen


def test_stress_reuse_changes_only_stress_settings(tmp_path):
    before={'type':'manifest','map':'Carla/Maps/Town04','archetype':'chain_cut_in',
        'seed':171,'anchor_protocol_version':'v12','anchors':[{'anchor':0,'key':[1,2,3],
        'coords':{'ego':[0,0,0]}}], 'cruise_mps':7.5,'cutin_rear_gap_m':35.,
        'chain_c_control_delay_s':0.0,'chain_c_brake_command_cap':1.,
        'families':['tm','behavior'],'profiles':['cruise']}
    src=tmp_path/'source.jsonl';src.write_text(json.dumps(before)+'\n')
    for d,c in [(0.3,1.),(0,.85),(.3,.85)]:
        altered=derive(src,delay_s=d,cap=c)
        assert altered['anchors']==before['anchors']
        assert altered['seed']==before['seed']
        assert altered['map']==before['map']
        assert altered['cruise_mps']==before['cruise_mps']
        assert altered['chain_c_control_delay_s']==d
        assert altered['chain_c_brake_command_cap']==c
        assert altered['v15_stress_parent_manifest_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
    with pytest.raises(ValueError):derive(src,delay_s=.33,cap=.85)
    before['chain_c_control_delay_s']=.3
    src.write_text(json.dumps(before)+'\n')
    with pytest.raises(ValueError,match='STRESS_BASE_MUST_BE_UNPERTURBED'):
        derive(src,delay_s=.3,cap=.85)
