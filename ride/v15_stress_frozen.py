"""Strict same-geometry v14 C-brake stress condition derivation.

Only experimental C command parameters/family list may change. The upstream
CARLA anchor positions, speed, map, seed, route and timebase are preserved.
The original stress baseline remains immutable. No physics are executed.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math
from pathlib import Path
from .v14_frozen_replay import load_frozen
from .v15_protocol_check import PANEL,check


def derive(src,*,delay_s,cap):
    check('panel',delay_s,cap)
    original=load_frozen(src)
    if (original.get('chain_c_control_delay_s',0.0)!=0.0 or
        original.get('chain_c_brake_command_cap',1.0)!=1.0):
        raise ValueError('STRESS_BASE_MUST_BE_UNPERTURBED_DELAY0_CAP1')
    src_bytes=Path(src).read_bytes()
    new=copy.deepcopy(original)
    new['chain_c_control_delay_s']=float(delay_s)
    new['chain_c_brake_command_cap']=float(cap)
    new['families']=list(PANEL)
    new['profiles']=['cruise','early_slow','late_slow','mild_early','mild_late',
                     'speed_early','speed_late']
    new['v14_C_exploratory_protocol']='HYPOTHETICAL_DELAYED_C_CONTROL_V1'
    new['v15_stress_parent_manifest_sha256']=hashlib.sha256(src_bytes).hexdigest()
    new['v15_stress_derivation']='C_ONLY_DELAY_OR_BRAKE_COMMAND_CAP_CHANGED'
    new['v15_stress_shared_geometry']=True
    new['v15_stress_not_human_response']=True
    if (original['seed']!=new['seed'] or original['map']!=new['map'] or
        original['anchors']!=new['anchors']):
        raise RuntimeError('Frozen scene changed unexpectedly')
    return new


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--delay-s',type=float,required=True);p.add_argument('--brake-cap',type=float,required=True)
    a=p.parse_args(argv)
    m=derive(a.input,delay_s=a.delay_s,cap=a.brake_cap)
    dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('x',encoding='utf8') as fh:
        fh.write(json.dumps(m,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'status':'FROZEN_GEOMETRY_REUSED','output':str(dst),
      'parent_sha256':m['v15_stress_parent_manifest_sha256'],
      'anchors':len(m['anchors']),'seed':m['seed'],
      'c_delay_s':m['chain_c_control_delay_s'],
      'c_brake_command_cap':m['chain_c_brake_command_cap']},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
