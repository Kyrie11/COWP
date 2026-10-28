"""Fail closed when the confirmation protocol drifts from frozen development.

Excludes seed/map (holdouts may change), but never silently changes
vehicle speeds, action timings, gaps, warmup or intervention horizon.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
FIELDS=('cutin_late_switch_s','cutin_lateral_lookahead_m',
        'cutin_rear_gap_m','cutin_ego_gap_m','rear_speed_delta_mps',
        'cruise_mps','warmup_ticks')

def check(frozen,actual,required_version='ride_lane_corridor_v8_20261028'):
    if frozen.get('type')!='manifest':raise ValueError('Expected frozen manifest')
    differences={}
    for key in FIELDS:
        if key not in frozen: differences[key]='MISSING_FROZEN_FIELD';continue
        if float(frozen[key])!=float(actual[key]):differences[key]={'development':frozen[key],'confirmation':actual[key]}
    if frozen.get('anchor_protocol_version')!=required_version:
        differences['anchor_protocol_version']={'development':frozen.get('anchor_protocol_version'),'required':required_version}
    return differences

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',required=True,type=Path)
    for name in FIELDS:p.add_argument('--'+name.replace('_','-'),type=float,required=True)
    a=p.parse_args(argv)
    frozen=json.loads(a.frozen.read_text().splitlines()[0]);actual={x:getattr(a,x) for x in FIELDS}
    diff=check(frozen,actual)
    print(json.dumps({'locked_prospective_parameters':actual,'protocol_mismatch':diff},indent=2,ensure_ascii=False))
    if diff:raise SystemExit('STOP_CONFIRMATION: frozen controls differ from preregistered development; start a new named protocol instead.')
if __name__=='__main__':main()
