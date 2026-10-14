"""Check ID leakage between validation and validation_interactive cohorts.

Scene-ID overlap is a LOWER BOUND on raw temporal overlap: two windows from one
log segment can differ in scenario_id. Don't report disjoint IDs as independent
trajectories unless parent log timestamps/identifiers were also checked.
"""
from __future__ import annotations
import json,argparse
from pathlib import Path

def report(m1,m2):
    a=json.loads(Path(m1).read_text());b=json.loads(Path(m2).read_text())
    x={e['scenario_id']:e for e in a['entries']};y={e['scenario_id']:e for e in b['entries']}
    shared=set(x)&set(y)
    return {'split_a':a.get('split'),'split_b':b.get('split'),
      'sampled_a':len(x),'sampled_b':len(y),'same_scenario_id_count':len(shared),
      'overlap_fraction_b':len(shared)/len(y) if y else None,
      'shared_sample_ids':sorted(shared)[:250],
      'shared_in_sealed_holdout':sum(x[s].get('evaluation_partition')=='sealed_holdout' for s in shared),
      'independence_proven':False,
      'warning':'No shared scenario_id does NOT establish no overlapping log segment/window.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--a',required=True,type=Path)
    p.add_argument('--b',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();r=report(args.a,args.b)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
