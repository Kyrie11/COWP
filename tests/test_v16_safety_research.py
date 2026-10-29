import json
import zipfile
from pathlib import Path
import pytest
from ride.carla.lagged_controller import DelayedCController
from ride.v16_cross_stress import load_panel,preflight_cap,summarize_panel,_check_completed
from ride.v16_transparent_selectors import choose,lock,SELECTORS


class Command:
    def __init__(self, brake=0.0):
        self.throttle=0.;self.steer=0.;self.brake=brake
        self.hand_brake=False;self.reverse=False;self.manual_gear_shift=False;self.gear=1


def test_delay_only_and_cap_are_separate_interventions():
    ctl=DelayedCController(Command(),delay_s=.1,brake_command_cap=.2,dt=.05)
    vals=[ctl.step(Command(brake=x)).brake for x in (.0,.3,.3,.1)]
    assert vals==[0.,0.,0.,.2]
    r=ctl.actuation_report()
    assert r['requested_delay_ticks']==2 and r['command_samples']==4
    assert r['command_cap_binding_ticks']==1
    assert r['delay_changed_command_ticks']>0
    assert r['effectiveness']=='CAP_AND_DELAY'


def test_inactive_cap_actuation_report():
    ctl=DelayedCController(Command(),delay_s=0,brake_command_cap=.85,dt=.05)
    for brake in (.1,.2,.3):ctl.step(Command(brake=brake))
    assert ctl.actuation_report()['command_cap_binding_ticks']==0
    assert ctl.actuation_report()['effectiveness']=='NO_EFFECT'


def test_noop_brake_cap_does_not_reduce_normal_command():
    ctl=DelayedCController(Command(),delay_s=0,brake_command_cap=.85)
    assert abs(ctl.step(Command(brake=.3)).brake-.3)<1e-9


def _minimal_panel(tmp_path,*,requested_cap=.85,normal_max=.30):
    root=tmp_path/'p'
    root.mkdir()
    manifest={'type':'manifest','simulator':'CARLA','archetype':'chain_cut_in','map':'Town04',
              'seed':1,'anchors':[], 'chain_c_brake_command_cap':requested_cap,
              'chain_c_control_delay_s':0}
    record={'type':'trial','archetype':'chain_cut_in','anchor':0,'seed':1,'family':'mixed_b_tm_c_behavior',
            'profile':'cruise','metrics':{'valid':True,'mechanism_executed':False,
            'min_rear_pair_ttc_s':8.0, 'follower_c_max_sustained_brake_s':0.0},
            'trace':[{'controls_by_role':{'follower_c':{'brake':normal_max}}}]}
    end={'type':'run_end','completed':True,'trial_count':1}
    f=root/'chain_cut_in.jsonl';f.write_text('\n'.join(json.dumps(r) for r in (manifest,record,end))+'\n')
    return f


def test_cap_preflight_rejects_untriggered_brake(tmp_path):
    f=_minimal_panel(tmp_path,normal_max=.3)
    out=preflight_cap(f,.85)
    assert out['eligible_arms_with_brake_exceeding_cap']==0
    assert out['status'].startswith('INACTIVE_CAP')


def test_cap_preflight_can_be_potentially_binding(tmp_path):
    f=_minimal_panel(tmp_path,normal_max=.30)
    out=preflight_cap(f,.20)
    assert out['eligible_arms_with_brake_exceeding_cap']==1
    assert out['status'].startswith('CAP_CAN_BIND')


def test_cap_preflight_does_not_claim_physical_deceleration(tmp_path):
    f=_minimal_panel(tmp_path)
    assert 'not validation of maximum physical deceleration' in preflight_cap(f,.2)['caveat']


def test_incomplete_carla_panel_rejected():
    with pytest.raises(ValueError,match='Incomplete CARLA panel'):
        _check_completed([{'type':'manifest','simulator':'CARLA'}, {'type':'trial'}])


def _s(B,C,progress=50,primary=None):
    return {'B_burden':B,'C_burden':C,'ego_progress_m':progress,
            'C_primary_risk':primary or [0,0,0]}


@pytest.mark.parametrize('selector',SELECTORS)
def test_selector_has_defined_behavior(selector):
    a=_s(.15,.9,50);b=_s(.1,.1,49.98)
    choice,why=choose(a,b,selector)
    assert choice in ('a','b',None)
    assert isinstance(why,str)


def test_progress_only_ignores_c():
    assert choose(_s(0,1,51),_s(0,0,50),'PROGRESS_ONLY')[0]=='a'


def test_b_only_courtesy_misses_c_danger():
    a=_s(.2,.9,50);b=_s(.2,.1,49.99)
    assert choose(a,b,'B_ONLY_COURTESY',budget=.4)[0]=='a'
    assert choose(a,b,'B_C_COURTESY',budget=.4)[0]=='b'


def test_both_courtesy_unsafe_must_be_unknown():
    assert choose(_s(.8,.8),_s(.9,.7),'B_C_COURTESY',budget=.4)[0] is None


def test_primary_safety_beats_comfort_cost():
    assert choose(_s(0,0,50,[0,0,1]),_s(0,1,49,[0,0,0]),'C_ONLY')[0]=='b'


def test_lock_does_not_allow_preexisting_target(tmp_path):
    src=tmp_path/'src';target=tmp_path/'target';frozen=tmp_path/'frozen';choices=tmp_path/'choices'
    target.write_text('should never read')
    with pytest.raises(ValueError,match='TARGET_ALREADY_EXISTS'):
        lock(src,target,frozen,choices,family='mixed_b_tm_c_behavior',
             target_family='mixed_b_tm_c_delayed_behavior')


def test_zip_completed_panel_can_be_read(tmp_path):
    p=_minimal_panel(tmp_path)
    zfile=tmp_path/'panel.zip'
    with zipfile.ZipFile(zfile,'w') as z:z.write(p,'p/carla/chain_cut_in.jsonl')
    panel=load_panel(zfile)
    assert len(panel['trials'])==1
    assert summarize_panel(panel)['cap_actuation_status'].startswith('UNTRIGGERED')
