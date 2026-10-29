"""Fail closed when the confirmation protocol drifts from frozen development.

Excludes seed/map (holdouts may change), but never silently changes
vehicle speeds, action timings, gaps, warmup or intervention horizon.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
FIELDS=('cutin_late_switch_s','cutin_lateral_lookahead_m',
        'cutin_rear_gap_m','cutin_ego_gap_m','rear_speed_delta_mps',
        'cruise_mps','warmup_ticks','chain_bc_gap_m','chain_c_speed_delta_mps',
        'cutin_paired_amplitude_mps')
TEXT_FIELDS=('behavior_speed_protocol','behavior_route_protocol')

def check(frozen,actual,required_version='ride_lane_corridor_v10_20261028'):
    if frozen.get('type')!='manifest':raise ValueError('Expected frozen manifest')
    differences={}
    for key in FIELDS:
        if key not in frozen: differences[key]='MISSING_FROZEN_FIELD';continue
        if float(frozen[key])!=float(actual[key]):differences[key]={'development':frozen[key],'confirmation':actual[key]}
    for key in TEXT_FIELDS:
        if key not in frozen: differences[key]='MISSING_FROZEN_FIELD'
        elif frozen[key]!=actual.get(key):
            differences[key]={'development':frozen[key],'confirmation':actual.get(key)}
    if frozen.get('anchor_protocol_version')!=required_version:
        differences['anchor_protocol_version']={'development':frozen.get('anchor_protocol_version'),'required':required_version}
    return differences

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',required=True,type=Path)
    p.add_argument('--required-version',default='ride_lane_corridor_v10_20261028')
    for name in FIELDS:p.add_argument('--'+name.replace('_','-'),type=float,required=True)
    for name in TEXT_FIELDS:p.add_argument('--'+name.replace('_','-'),required=True)
    a=p.parse_args(argv)
    frozen=json.loads(a.frozen.read_text().splitlines()[0]);actual={x:getattr(a,x) for x in FIELDS+TEXT_FIELDS}
    diff=check(frozen,actual,required_version=a.required_version)
    print(json.dumps({'locked_prospective_parameters':actual,'protocol_mismatch':diff},indent=2,ensure_ascii=False))
    if diff:raise SystemExit('STOP_CONFIRMATION: frozen controls differ from preregistered development; start a new named protocol instead.')
if __name__=='__main__':main()
