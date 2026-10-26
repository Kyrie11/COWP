"""Narrow-vs-wide finite recourse coverage: never infer global infeasibility.

Requires a nested search bank, identical initial states, simulator versions and
shared candidate outcomes. If a shared candidate changes, flag inconclusive.
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
from .evidence import read_jsonl
from .science import same_initial, Protocol


def parse_bank(rows):
    manifest=next((x for x in rows if x.get('type')=='manifest'),None)
    if not manifest:raise ValueError('Missing manifest in finite bank')
    attempts=collections.defaultdict(dict)
    for r in rows:
        if r.get('type')!='trial' or r.get('family')!='joint_witness':continue
        key=(r.get('archetype','support_gap'),str(r['anchor']),str(r['seed']),r['profile'])
        target=tuple(r.get('joint_targets') or ())
        if len(target)!=2:raise ValueError('Missing joint responder control')
        if target in attempts[key]:raise ValueError('Duplicate joint control: '+str(key)+' '+str(target))
        attempts[key][target]=r
    return manifest,attempts


def is_witness(row):
    m=row.get('metrics') or {}
    return bool(m.get('valid') and not m.get('ego_collision') and
                not m.get('non_ego_collision') and not m.get('hard_brake'))


def compare(narrow,wide):
    mn,an=parse_bank(narrow);mw,aw=parse_bank(wide)
    if not an or not aw:raise ValueError('No joint-search trials in one bank')
    if an.keys()!=aw.keys():raise ValueError('Different scenario/profile identities across banks')
    for attr in ('carla_server_version','carla_client_version','map','cruise_mps','dt','archetype'):
        if mn.get(attr)!=mw.get(attr):
            raise ValueError('Different simulator setup for '+attr)
    rows=[];proto=Protocol()
    for k in sorted(an):
        small,large=an[k],aw[k]
        if not set(small).issubset(large):
            raise ValueError('Not nested: narrow search has candidates absent from wide bank')
        inconsistent=[]
        for a in small:
            x,y=small[a],large[a]
            ok,why=same_initial(x,y,proto)
            if not ok:inconsistent.append({'joint_targets':a,'reason':why})
            elif is_witness(x)!=is_witness(y):
                inconsistent.append({'joint_targets':a,'reason':'shared_candidate_witness_outcome_changed'})
        small_witness={a for a,r in small.items() if is_witness(r)}
        large_witness={a for a,r in large.items() if is_witness(r)}
        conclusive=not inconsistent
        rows.append({'archetype':k[0],'anchor':k[1],'seed':k[2],'profile':k[3],
                     'small_bank_attempts':len(small),'large_bank_attempts':len(large),
                     'small_bank_positive_witnesses':len(small_witness),
                     'large_bank_positive_witnesses':len(large_witness),
                     'new_verified_positive_witnesses':len(large_witness-small_witness),
                     'shared_candidate_inconsistencies':inconsistent,
                     'comparison_valid':conclusive,
                     'finite_bank_miss_observed':conclusive and not bool(small_witness) and bool(large_witness),
                     'small_bank_no_witness_status':'UNKNOWN' if not small_witness else 'POSITIVE_WITNESS',
                     'large_bank_no_witness_status':'UNKNOWN' if not large_witness else 'POSITIVE_WITNESS'})
    return {'summary':{'cases':len(rows),
                       'valid_nested_bank_comparisons':sum(q['comparison_valid'] for q in rows),
                       'inconclusive_shared_candidate_inconsistency':sum(not q['comparison_valid'] for q in rows),
                       'finite_bank_misses':sum(q['finite_bank_miss_observed'] for q in rows),
                       'global_UNSAT_proven':False,
                       'labels_valid_only_under_explicit_CARLA_models':True},'details':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--narrow',required=True);p.add_argument('--wide',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    r=compare(read_jsonl(a.narrow),read_jsonl(a.wide))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(r['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
