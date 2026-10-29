"""Regression tests for freeze/locked-control split after the v14 error."""
from pathlib import Path
import pytest
from ride.carla.run import validate_v14_c_stress_family
from ride.v13_model_transfer import lock_choices,evaluate_lock
from test_v13_model_transfer import rows

ROOT=Path(__file__).resolve().parent.parent

@pytest.mark.parametrize('delay,cap',[(0.3,1.0),(0.0,0.85),(0.3,0.85)])
def test_manifest_accepts_precommitted_family_with_active_delayed_c(delay,cap):
    assert not validate_v14_c_stress_family('chain_cut_in',
        ['tm','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'],
        delay,cap,manifest_only=True)

@pytest.mark.parametrize('delay,cap',[(0.3,1.0),(0.0,0.85),(0.3,0.85)])
def test_default_frozen_families_must_not_silently_imply_delayed_c(delay,cap):
    with pytest.raises(ValueError,match='DELAYED_C_PARAMETERS_REQUIRE'):
        validate_v14_c_stress_family('chain_cut_in',['tm','behavior'],
                                    delay,cap,manifest_only=True)


def test_inactive_source_metadata_requires_explicit_opt_in():
    with pytest.raises(ValueError,match='DELAYED_C_PARAMETERS_REQUIRE'):
        validate_v14_c_stress_family('chain_cut_in',['mixed_b_tm_c_behavior'],.3,.85)
    assert validate_v14_c_stress_family('chain_cut_in',['mixed_b_tm_c_behavior'],.3,.85,
                                        inactive_metadata_only=True)
    for families,archetype in ((['tm'],'chain_cut_in'),
                               (['mixed_b_tm_c_behavior'],'cut_in'),
                               (['mixed_b_tm_c_delayed_behavior'],'chain_cut_in')):
        with pytest.raises(ValueError,match='INACTIVE_STRESS_METADATA'):
            validate_v14_c_stress_family(archetype,families,.3,.85,
                                         inactive_metadata_only=True)


def test_freeze_shell_passes_families_and_profiles_to_manifest_only():
    txt=(ROOT/'scripts/03_precommit_scenes.sh').read_text()
    assert '--families "${RIDE_FAMILY_LIST[@]}" --profiles "${RIDE_PROFILE_LIST[@]}"' in txt
    assert '--manifest-only' in txt


def test_two_stage_source_flag_is_explicit_and_removed_before_target():
    txt=(ROOT/'scripts/45_v14_locked_source_to_safety_stress.sh').read_text()
    assert "export RIDE_FAMILIES='mixed_b_tm_c_behavior mixed_b_tm_c_delayed_behavior'" in txt
    a=txt.index('export RIDE_V14_SOURCE_INACTIVE_STRESS_METADATA=1')
    b=txt.index('unset RIDE_V14_SOURCE_INACTIVE_STRESS_METADATA')
    c=txt.index('export RIDE_OUTPUT_ROOT="$RIDE_V14_LOCKED_STRESS_ROOT/target"')
    assert a<b<c


def test_v14_stress_choice_is_not_counted_as_response_model_error():
    src=rows('mixed_b_tm_c_behavior')
    tgt=rows('mixed_b_tm_c_delayed_behavior',True)
    src[0]['chain_c_control_delay_s']=.3
    src[0]['chain_c_brake_command_cap']=.85
    tgt[0]['chain_c_control_delay_s']=.3
    tgt[0]['chain_c_brake_command_cap']=.85
    choices=lock_choices(src,source_family='mixed_b_tm_c_behavior',
                         target_family='mixed_b_tm_c_delayed_behavior',
                         frozen_sha='a',source_sha='b',expected_target_path='/tmp/target')
    assert choices['controlled_C_policy_stress']
    assert choices['mode']=='V14_CONTROLLED_C_POLICY_STRESS_NOT_MODEL_ERROR'
    result=evaluate_lock(choices,src,tgt)
    assert result['summary']['diagnostic_model_induced_choice_errors']==0
    assert result['summary']['controlled_C_policy_avoidable_choices']>=1
    tgt[0]['chain_c_control_delay_s']=.5
    with pytest.raises(ValueError,match='manifest_mismatch'):
        evaluate_lock(choices,src,tgt)


def test_other_mixed_family_not_supported_for_v13_choice():
    with pytest.raises(ValueError,match='Unsupported response family'):
        lock_choices(rows('mixed_b_tm_c_behavior'),
            source_family='mixed_b_tm_c_behavior',target_family='tm',
            frozen_sha='a',source_sha='b',expected_target_path='/tmp/target')
