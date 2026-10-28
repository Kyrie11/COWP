"""Inventory ReactSim scenario log names against every local nuPlan split.

This is a filename-level diagnostic only. It never opens a DB or rewrites an
ID list. Its purpose is to distinguish an incomplete trainval staging view
from a genuinely missing nuPlan log before anyone downloads data.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SPLITS = ('val','train_boston','train_pittsburgh','train_singapore','train_vegas','test')


def _ids(path: Path):
    data=json.loads(path.read_text(encoding='utf8'))
    groups=data.get('groups')
    if not isinstance(groups,list):
        raise ValueError("Expected grouped scenario IDs with a 'groups' list")
    out=[]
    for g in groups:
        name=g.get('log_name'); tokens=g.get('scenario_tokens')
        if not isinstance(name,str) or not isinstance(tokens,list):
            raise ValueError('Malformed scenario group: '+repr(g)[:200])
        out.append({'log_name':name,'scenario_count':len(tokens)})
    return out


def inventory(nuplan_root: Path, ids_path: Path, staged_splits, all_splits=DEFAULT_SPLITS):
    base=nuplan_root/'nuplan-v1.1'/'splits'
    files=defaultdict(list); counts={}
    for split in all_splits:
        folder=base/split
        paths=sorted(folder.glob('*.db')) if folder.is_dir() else []
        counts[split]=len(paths)
        for p in paths: files[p.name].append(split)
    groups=_ids(ids_path)
    staged=set(staged_splits)
    status=Counter(); rows=[]
    for g in groups:
        name=g['log_name']+'.db'; present=files.get(name,[])
        staged_present=sorted(set(present)&staged)
        if not present: state='MISSING_IN_ALL_LOCAL_SPLITS'
        elif not staged_present: state='AVAILABLE_OUTSIDE_STAGED_SPLITS'
        else: state='AVAILABLE_IN_STAGED_SPLITS'
        status[state]+=1
        rows.append({**g,'db_name':name,'available_splits':sorted(present),
                     'staged_splits':staged_present,'status':state})
    return {'nuplan_root':str(nuplan_root),'ids_file':str(ids_path),
            'all_splits':list(all_splits),'staged_splits':sorted(staged),
            'db_counts_by_split':counts,'scenario_log_groups':len(groups),
            'status_counts':dict(status),'groups':rows,
            'warning':'Filename coverage is necessary but does not prove token joins, boundary eligibility, or cache success.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nuplan-root',type=Path,required=True)
    p.add_argument('--ids',type=Path,required=True)
    p.add_argument('--staged-splits',nargs='+',default=['val'])
    p.add_argument('--all-splits',nargs='+',default=list(DEFAULT_SPLITS))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    result=inventory(a.nuplan_root,a.ids,a.staged_splits,a.all_splits)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status_counts':result['status_counts'],
                      'db_counts_by_split':result['db_counts_by_split']},ensure_ascii=False,indent=2))
    if result['status_counts'].get('MISSING_IN_ALL_LOCAL_SPLITS',0):
        raise SystemExit(3)
if __name__=='__main__': main()
