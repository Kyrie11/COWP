"""Re-use exact CARLA v13 frozen anchor geometry for SAME-ROAD placebo repeats.

Geometry coordinates and map are not modified. A new seed changes the TM
random state, so it is a *new stochastic replication*, not exactly the same
full CARLA state. This is NOT a new physical-corridor holdout sample.
"""
from __future__ import annotations
import argparse,hashlib,json,math,shlex
from pathlib import Path

ENV_MAP = {
 'seed':'RIDE_SEED', 'cruise_mps':'RIDE_CRUISE_MPS',
 'rear_speed_delta_mps':'RIDE_REAR_SPEED_DELTA_MPS',
 'cutin_ego_gap_m':'RIDE_CUTIN_EGO_GAP_M','cutin_rear_gap_m':'RIDE_CUTIN_REAR_GAP_M',
 'chain_bc_gap_m':'RIDE_CHAIN_BC_GAP_M','chain_c_speed_delta_mps':'RIDE_CHAIN_C_SPEED_DELTA_MPS',
 'warmup_ticks':'RIDE_WARMUP_TICKS','cutin_late_switch_s':'RIDE_CUTIN_LATE_SWITCH_S',
 'cutin_lateral_lookahead_m':'RIDE_CUTIN_LATERAL_LOOKAHEAD_M',
 'cutin_paired_amplitude_mps':'RIDE_CUTIN_PAIRED_AMPLITUDE_MPS',
 'behavior_route_protocol':'RIDE_BEHAVIOR_ROUTE_PROTOCOL',
 'behavior_speed_protocol':'RIDE_BEHAVIOR_SPEED_PROTOCOL',
 'chain_c_control_delay_s':'RIDE_CHAIN_C_CONTROL_DELAY_S',
 'chain_c_brake_command_cap':'RIDE_CHAIN_C_BRAKE_COMMAND_CAP',
}


def load_frozen(src):
    path=Path(src)
    rows=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    if len(rows)!=1:raise ValueError('Expected one frozen CARLA manifest, not outcome file')
    m=rows[0]
    if m.get('type')!='manifest' or m.get('archetype')!='chain_cut_in' or not m.get('anchors'):
        raise ValueError('Expected frozen CARLA chain_cut_in manifest with anchors')
    if not m.get('anchor_protocol_version') or not m.get('map') or m.get('seed') is None:
        raise ValueError('Malformed frozen simulator manifest')
    if len({str(x['anchor']) for x in m['anchors']})!=len(m['anchors']):
        raise ValueError('Duplicate anchor identity')
    return m


def derive(src,*,anchor_ids,seed):
    if seed<0:raise ValueError('seed must be nonnegative')
    old=load_frozen(src)
    indexes={str(x['anchor']):x for x in old['anchors']}
    ids=[str(x) for x in anchor_ids]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Explicit, unique anchor ids are required')
    unknown=set(ids)-set(indexes)
    if unknown:raise ValueError('Unknown anchor(s): '+repr(sorted(unknown)))
    newer=dict(old)
    newer['anchors']=[indexes[x] for x in ids]
    newer['seed']=int(seed)
    newer['v14_frozen_replay_parent_sha256']=hashlib.sha256(Path(src).read_bytes()).hexdigest()
    newer['v14_frozen_replay_parent_seed']=old['seed']
    newer['v14_frozen_replay_original_anchor_ids']=ids
    newer['v14_frozen_replay_class']='REPEATED_FROZEN_GEOMETRY_NOT_NEW_CORRIDOR'
    return newer


def emit_env(m):
    info={}
    for key,name in ENV_MAP.items():
        if key in m:info[name]=m[key]
    info['CARLA_TOWN']=str(m['map']).split('/')[-1]
    info['RIDE_ANCHORS']=len(m['anchors'])
    info['RIDE_REPEATS']=1
    lines=[]
    for key,value in info.items():
        if not isinstance(value,(str,int,float)) or (isinstance(value,float) and not math.isfinite(value)):
            raise ValueError('Unsupported value in frozen manifest: '+key)
        lines.append('export '+key+'='+shlex.quote(str(value)))
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('derive',help='select subset, possibly change seed; NEVER changes geometry')
    d.add_argument('--input',required=True);d.add_argument('--output',required=True)
    d.add_argument('--anchors',nargs='+',required=True)
    d.add_argument('--seed',type=int,required=True)
    e=sub.add_parser('emit-env',help='shell-quoted export statements for frozen protocol')
    e.add_argument('--input',required=True)
    a=p.parse_args(argv)
    if a.command=='derive':
        newer=derive(a.input,anchor_ids=a.anchors,seed=a.seed)
        out=Path(a.output)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf8') as fp:
            fp.write(json.dumps(newer,ensure_ascii=False,sort_keys=True)+'\n')
        print(json.dumps({'output':str(out),'map':newer['map'],
             'seed':newer['seed'],'anchors':a.anchors,
             'parent_sha256':newer['v14_frozen_replay_parent_sha256']},ensure_ascii=False))
    else:print(emit_env(load_frozen(a.input)))
    return 0

if __name__=='__main__':raise SystemExit(main())
