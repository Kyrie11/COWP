"""Merge a prospectively locked two-stage CARLA source/target *after* both run.

The output is ONLY for independent physical-stress forensics, never fed back to
the pretarget model-choice lock. Source/target provenance is retained.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .evidence import read_jsonl
from .v13_model_transfer import _manifest_parity


def merge(source,target):
    a,b=read_jsonl(source),read_jsonl(target)
    if sum(x.get('type')=='manifest' for x in a)!=1 or sum(x.get('type')=='manifest' for x in b)!=1:
        raise ValueError('Exactly one manifest required per stage')
    ma=next(x for x in a if x.get('type')=='manifest')
    mb=next(x for x in b if x.get('type')=='manifest')
    eq,reason=_manifest_parity(ma,mb)
    if not eq:
        if reason.endswith(('chain_c_control_delay_s','chain_c_brake_command_cap')):
            raise ValueError('Frozen stress-control metadata mismatch')
        raise ValueError(reason)
    if ma.get('archetype')!='chain_cut_in':raise ValueError('Four-car chain required')
    if ma.get('chain_c_control_delay_s')!=mb.get('chain_c_control_delay_s') or \
       ma.get('chain_c_brake_command_cap')!=mb.get('chain_c_brake_command_cap'):
        raise ValueError('Frozen stress-control metadata mismatch')
    if sum(x.get('type')=='run_end' for x in a)!=1 or sum(x.get('type')=='run_end' for x in b)!=1:
        raise ValueError('Source/target run_end missing')
    ra=[x for x in a if x.get('type')=='trial'];rb=[x for x in b if x.get('type')=='trial']
    if not ra or not rb:raise ValueError('Empty source/target trials')
    names={(x['family'],str(x['anchor']),str(x['seed']),x['profile']) for x in ra+rb}
    if len(names)!=len(ra)+len(rb):raise ValueError('Duplicate action trial')
    families=sorted({x['family'] for x in ra+rb})
    if len(families)!=2:raise ValueError('Exactly 2 distinct response families required')
    manifest=dict(ma)
    manifest['families']=families
    manifest['v14_merged_source_panel']=str(Path(source).resolve())
    manifest['v14_merged_target_panel']=str(Path(target).resolve())
    manifest['v14_observation_status']='POST_LOCK_SIMULATED_ROLLOUTS_NOT_HUMAN_GROUND_TRUTH'
    return [manifest]+ra+rb+[{'type':'run_end','completed':True,'trial_count':len(ra)+len(rb),'v14_merge_of_complete_stages':True}]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:
        for x in merge(a.source,a.target):f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print('COMPLETE_MERGED_STRESS_PANEL '+str(out))

if __name__=='__main__':main()
