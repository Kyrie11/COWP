"""Freeze outcome-blind WOMD IDs for a native Waymax pilot.

Uses manifest scenario_id+partition only. Never inspect observed_future_diagnostic
or COWP pseudo-labels when selecting or freezing IDs.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def freeze(cohort, n, partition='development', salt='ride-native-waymax-20261026'):
    man=json.loads((Path(cohort)/'cohort_manifest.json').read_text(encoding='utf8'))
    entries=[e for e in man['entries'] if e.get('evaluation_partition')==partition]
    unique={str(e['scenario_id']) for e in entries}
    if len(unique)!=len(entries):raise ValueError('Duplicate scenario IDs; cannot freeze independent cohort')
    ids=sorted(unique,key=lambda x:hashlib.sha256((salt+'|'+x).encode('utf8')).hexdigest())
    if n and len(ids)<n:raise ValueError(f'Only {len(ids)} eligible {partition} scenes, requested {n}')
    return ids[:n] if n else ids


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True)
    p.add_argument('--count',type=int,default=22);p.add_argument('--partition',default='development')
    p.add_argument('--output',required=True);a=p.parse_args(argv)
    if a.count<=0:raise ValueError('count must be positive')
    selected=freeze(a.cohort,a.count,a.partition)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Frozen scene IDs already exist: '+str(path))
    path.write_text('\n'.join(selected)+'\n',encoding='utf8')
    print(json.dumps({'frozen_ids':len(selected),'output':str(path),'partition':a.partition},indent=2))
if __name__=='__main__':main()
