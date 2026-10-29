"""CARLA-independent preflight for the delayed-C v14/v15 experimental protocol.

Allows a dry-run of freeze, unmodified source and modified target policy rules
before a CARLA server is launched or an expensive output folder is created.
"""
from __future__ import annotations
import argparse,json,os,math
from .carla.run import validate_v14_c_stress_family

PANEL=('tm','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior')
SOURCE=('mixed_b_tm_c_behavior',)
TARGET=('mixed_b_tm_c_delayed_behavior',)


def check(mode,delay,cap,families=None):
    if mode=='panel':
        fam=list(families or PANEL)
        if fam!=list(PANEL):
            raise ValueError('V15_STRESS_PANEL_REQUIRES_FIXED_THREE_FAMILIES')
        validate_v14_c_stress_family('chain_cut_in',fam,delay,cap,manifest_only=True)
        validate_v14_c_stress_family('chain_cut_in',fam,delay,cap)
        ops=[{'stage':'freeze','families':fam,'delayed_C_active':False},
             {'stage':'panel_rollouts','families':fam,'delayed_C_active':True}]
    elif mode=='locked':
        validate_v14_c_stress_family('chain_cut_in',list(SOURCE+TARGET),delay,cap,manifest_only=True)
        validate_v14_c_stress_family('chain_cut_in',list(SOURCE),delay,cap,
                                    inactive_metadata_only=True)
        validate_v14_c_stress_family('chain_cut_in',list(TARGET),delay,cap)
        ops=[{'stage':'freeze','families':list(SOURCE+TARGET),'delayed_C_active':False},
             {'stage':'unmodified_source','families':list(SOURCE),'delayed_C_active':False,
              'inactive_parameter_metadata_only':True},
             {'stage':'perturbed_target','families':list(TARGET),'delayed_C_active':True}]
    else:raise ValueError('Unknown experimental mode')
    return {'status':'OFFLINE_PROTOCOL_VALIDATED_NO_CARLA_EXECUTED',
            'mode':mode,'C_command_brake_delay_s':delay,'C_brake_command_cap':cap,
            'perturbation_nonzero':bool(delay>0 or cap<1),
            'stages':ops,'warning':'No physics or safety outcome has been measured'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('panel','locked'),required=True)
    p.add_argument('--delay-s',type=float,default=float(os.getenv('RIDE_CHAIN_C_CONTROL_DELAY_S','0')))
    p.add_argument('--brake-cap',type=float,default=float(os.getenv('RIDE_CHAIN_C_BRAKE_COMMAND_CAP','1')))
    p.add_argument('--families',nargs='+')
    a=p.parse_args(argv)
    print(json.dumps(check(a.mode,a.delay_s,a.brake_cap,a.families),ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
