import pytest
from ride.v15_protocol_check import check


def test_panel_check_nondelayed_is_identity():
    x=check('panel',0.,1.)
    assert not x['perturbation_nonzero']
    assert 'mixed_b_tm_c_delayed_behavior' in x['stages'][0]['families']

@pytest.mark.parametrize('d,c',[(.3,1),(.0,.85),(.3,.85)])
def test_stress_panel_and_locked_path_both_preflight(d,c):
    a=check('panel',d,c)
    b=check('locked',d,c)
    assert a['status']==b['status']=='OFFLINE_PROTOCOL_VALIDATED_NO_CARLA_EXECUTED'
    assert b['stages'][1]['inactive_parameter_metadata_only']
    assert b['stages'][2]['delayed_C_active']


def test_bad_family_is_refused():
    with pytest.raises(ValueError,match='V15_STRESS_PANEL_REQUIRES'):
        check('panel',.3,.85,['tm','behavior'])


def test_bad_delay_or_cap_is_refused():
    for d,c in [(0.32,.85),(-.1,1),(0.3,.02)]:
        with pytest.raises(ValueError):check('locked',d,c)
