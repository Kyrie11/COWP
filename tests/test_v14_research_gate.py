import json
from types import SimpleNamespace
import pytest

from ride.carla.lagged_controller import DelayedCController
from ride.carla.roles import per_role_response_modes
from ride.v14_counterfactual_audit import burdens,sustained_s,trace_integrity
from ride.v14_frozen_replay import derive, emit_env
from ride.v14_safety_envelope import required_gap,screen_trace


def test_acceleration_cap_reduces_single_physics_spike():
    tr=[{'t':round((i+1)*.05,2),'follower_c_accel':(-22.0 if i==45 else -3.0)}
        for i in range(100)]
    assert burdens(tr)>burdens(tr,cap=6)
    assert burdens(tr,median_window=5)<burdens(tr)
    assert trace_integrity(tr)['poststartup_extreme_accel_count']==1


def test_brake_duration_at_threshold_is_sensitive():
    tr=[{'t':round((i+1)*.05,2),'follower_c_accel':(-4.1 if 40<=i<44 else 0.0)}
        for i in range(100)]
    assert sustained_s(tr,braking=4.0)==.2
    assert sustained_s(tr,braking=4.5)==0


def test_analytic_margin_physics_and_validation():
    got=required_gap(8,7,reaction_s=.6,b_rear=3.5,b_front=5.0)
    expected=max(2.,2.+8*.6+64/7-49/10)
    assert got==pytest.approx(expected)
    with pytest.raises(ValueError): required_gap(8,7,reaction_s=.6,b_rear=-2,b_front=5)


def test_safety_screen_hypothetical_not_measured_collision():
    tr=[{'t':2.0,'ego_in_target_lane':True,'rear_pair_ttc_applicable':True,
         'follower_b_accel':-4,'rear_pair_gap_m':5,
         'follower_c_speed_mps':8,'follower_b_speed_mps':7}]
    got=screen_trace(tr,reaction_s=.9,b_rear=2.5,b_front=5)
    assert got['status']=='HYPOTHETICAL_STOP_MARGIN_DEFICIT'
    assert got['min_margin_m']<0


def test_frozen_seed_only_rebases_metadata_not_road_geometry(tmp_path):
    original={'type':'manifest','map':'Carla/Maps/Town04',
        'archetype':'chain_cut_in','seed':17,'anchor_protocol_version':'ride_lane_corridor_v12_20261028',
        'anchors':[{'anchor':3,'key':[1,2,3],'coords':{'ego':[1,2,3]}},
                   {'anchor':4,'key':[20,30,40],'coords':{'ego':[20,30,40]}}]}
    path=tmp_path/'original.jsonl';path.write_text(json.dumps(original)+'\n')
    v=derive(path,anchor_ids=['4'],seed=109)
    assert v['anchors']==original['anchors'][1:]
    assert v['seed']==109
    assert v['v14_frozen_replay_parent_seed']==17
    assert "export RIDE_SEED=109" in emit_env(v)
    with pytest.raises(ValueError):derive(path,anchor_ids=['6'],seed=109)


def _control(throttle=0.2,steer=0.0,brake=0.0):
    return SimpleNamespace(throttle=throttle,steer=steer,brake=brake,hand_brake=False,reverse=False,
                           manual_gear_shift=False,gear=1)


def test_delayed_c_controller_applies_exact_ticks_and_caps_brake():
    delayed=DelayedCController(_control(.4),delay_s=.1,brake_command_cap=.6,dt=.05)
    x=delayed.step(_control(.0,brake=1.0))
    y=delayed.step(_control(.0,brake=.8))
    z=delayed.step(_control(.0,brake=.9))
    assert x.throttle==0 and y.throttle==0
    assert z.brake==.6
    assert delayed.ticks==2


def test_new_response_family_only_works_for_chain():
    p=per_role_response_modes('chain_cut_in','mixed_b_tm_c_delayed_behavior')
    assert p['follower_b']=='tm' and p['follower_c']=='behavior'
    with pytest.raises(ValueError):
        per_role_response_modes('cut_in','mixed_b_tm_c_delayed_behavior')


def test_stress_compares_same_c_policy_not_all_tm(monkeypatch):
    from ride import v14_safety_stress_audit as audit
    manifest={'type':'manifest','archetype':'chain_cut_in','map':'Town04',
              'anchor_protocol_version':'v14'}
    trials=[]
    for fam in ('tm','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'):
        for profile in ('cruise','early_slow','late_slow'):
            trials.append({'type':'trial','anchor':'4','seed':17,'family':fam,'profile':profile,
                 'metrics':{'valid':True,'ego_collision':False,'non_ego_pair_collision':False},
                 'trace':[], 'ego_collision':False})
    monkeypatch.setattr(audit, 'read_jsonl',lambda _: [manifest]+trials)
    d=audit.analyze('mock.jsonl')
    # We compare *behavior C* against lagged *behavior C*, not all-TM C.
    assert 'same_controller_C_vs_lagged_C_parity' in d
    assert 'tm_vs_lagged_C_parity' not in d
    assert len(d['same_controller_C_vs_lagged_C_parity'])==3


def test_two_stage_merge_emits_verified_run_end(monkeypatch):
    from ride import v14_merge_carla_panels as m
    head={'type':'manifest','simulator':'CARLA','archetype':'chain_cut_in',
          'map':'Town04','seed':17, 'chain_c_control_delay_s':.3,
          'chain_c_brake_command_cap':.85}
    s=[dict(head),{'type':'trial','family':'mixed_b_tm_c_behavior','anchor':1,
                   'seed':17,'profile':'early_slow','archetype':'chain_cut_in'},
       {'type':'run_end','completed':True,'trial_count':1}]
    t=[dict(head),{'type':'trial','family':'mixed_b_tm_c_delayed_behavior','anchor':1,
                   'seed':17,'profile':'early_slow','archetype':'chain_cut_in'},
       {'type':'run_end','completed':True,'trial_count':1}]
    monkeypatch.setattr(m,'read_jsonl',lambda p:s if p=='source' else t)
    rows=m.merge('source','target')
    assert rows[-1]['type']=='run_end' and rows[-1]['completed'] is True
    assert rows[-1]['trial_count']==2
    assert rows[0]['families']==['mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior']
    t[0]['chain_c_control_delay_s']=.6
    with pytest.raises(ValueError,match='metadata mismatch'):m.merge('source','target')
