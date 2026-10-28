"""Read-only inventory and opt-in symlink staging for ReactSim-Bench + nuPlan.

This script NEVER copies GB/TB of nuPlan data and never deletes existing files.
No pickles are unpickled; file names are not proof of scenario identity.
Examples:
 python -m ride.reactsim_setup --splits val
 python -m ride.reactsim_setup --splits val --apply
 python -m ride.reactsim_setup --splits val train_vegas train_boston train_pittsburgh train_singapore --apply
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from pathlib import Path

DEFAULT_NUPLAN='/data0/senzeyu2/dataset/CapPlan/data/nuplan'
DEFAULT_REPO='/data0/senzeyu2/dataset/RIDE/research_baselines/ReactSim_Bench'
DEFAULT_WORK='/data0/senzeyu2/dataset/RIDE/reactsim_workspace'
DEFAULT_BEHAVIOR='/data0/senzeyu2/dataset/RIDE/ReactSim-Bench/custom_behaviors'


def list_dbs(nuplan, splits):
    base=Path(nuplan)/'nuplan-v1.1'/'splits'
    result=[];counts={}
    for split in splits:
        if split=='test':
            raise ValueError('Do not put a hidden/heldout test split into training preprocessing')
        folder=base/split
        if not folder.is_dir():
            raise FileNotFoundError(f'nuPlan split missing: {folder}')
        paths=sorted(folder.glob('*.db'))
        counts[split]=len(paths)
        if not paths:
            raise ValueError('nuPlan split contains no .db files: '+str(folder))
        result.extend(paths)
    byname={}
    for path in result:
        if path.name in byname and path.resolve()!=byname[path.name].resolve():
            raise ValueError('Duplicate different DB log names across splits: '+path.name)
        byname[path.name]=path
    return byname,counts


def plan_links(nuplan, repo, work, behaviors, paths):
    root=Path(repo)
    dbroot=Path(work)/'nuplan_db_trainval_links'
    return ({dbroot/name:src for name,src in paths.items()}, {
        root/'data'/'nuplan'/'maps':Path(nuplan)/'maps',
        root/'data'/'nuplan'/'nuplan-v1.1'/'splits'/'trainval':dbroot,
        # Some nuplan-devkit configs use unsuffixed /nuplan-v1.1/trainval.
        root/'data'/'nuplan'/'nuplan-v1.1'/'trainval':dbroot,
        root/'data'/'nuplan_processed':Path(work)/'nuplan_processed',
        root/'data'/'custom_behaviors':Path(behaviors),
    })


def ensure_link(dst: Path, source: Path, apply=False):
    source=source.absolute()
    if not source.exists():
        return 'missing_source'
    if dst.is_symlink():
        return 'ready' if dst.resolve()==source.resolve() else 'conflict_different_symlink'
    if dst.exists():
        return 'conflict_existing_path'
    if not apply:
        return 'would_link'
    dst.parent.mkdir(parents=True,exist_ok=True)
    # exclusive symlink creation; no overwrite of unrelated data
    dst.symlink_to(source, target_is_directory=source.is_dir())
    return 'created'


def inspect(nuplan,repo,work,behaviors,splits,apply=False,max_dbs=0):
    nuplan,repo,work,behaviors=map(Path,(nuplan,repo,work,behaviors))
    if not repo.is_dir():
        raise FileNotFoundError('ReactSim-Bench repo missing: '+str(repo))
    maps=nuplan/'maps'
    manifest=maps/'nuplan-maps-v1.0.json'
    if not manifest.is_file():
        raise FileNotFoundError('Missing official nuPlan map manifest: '+str(manifest))
    cities=('sg-one-north','us-ma-boston','us-nv-las-vegas-strip','us-pa-pittsburgh-hazelwood')
    map_ok={c:any((maps/c).rglob('map.gpkg')) for c in cities}
    if not all(map_ok.values()):
        raise FileNotFoundError('Incomplete nuPlan map.gpkg tree: '+str(map_ok))
    behavior_num=sum(1 for _ in behaviors.glob('*.pkl')) if behaviors.is_dir() else 0
    if not behavior_num:
        raise FileNotFoundError('ReactSim custom behavior .pkl missing in '+str(behaviors)+
            ' (try --behaviors /data0/senzeyu2/dataset/ReactSim-Bench/custom_behaviors)')
    logs,counts=list_dbs(nuplan,splits)
    if max_dbs<0:
        raise ValueError('max_dbs cannot be negative')
    selected={n:logs[n] for n in sorted(logs)[:max_dbs or None]}
    db_links,root_links=plan_links(nuplan,repo,work,behaviors,selected)
    if apply:
        # processed dir is a workspace OUTPUT, not a download
        (work/'nuplan_processed').mkdir(parents=True,exist_ok=True)
        (work/'nuplan_db_trainval_links').mkdir(parents=True,exist_ok=True)
    db_status=Counter(ensure_link(t,s,apply) for t,s in db_links.items())
    root_status={}
    for target, source in root_links.items():
        if not apply and source in (work/'nuplan_processed',work/'nuplan_db_trainval_links') and not source.exists():
            root_status[str(target)]='would_link_after_workspace_creation'
        else:
            root_status[str(target)]=ensure_link(target,source,apply)
    problem={'missing_source','conflict_different_symlink','conflict_existing_path'}
    summary={'mode':'apply' if apply else 'dry_run','nuplan_root':str(nuplan),
             'reactsim_repo':str(repo),'workspace':str(work),
             'selected_db_splits':list(splits),'db_by_split':counts,
             'db_unique_available':len(logs),'db_selected':len(selected),
             'db_link_status':dict(db_status),'root_link_status':root_status,
             'map_gpkg_present':map_ok,'custom_behavior_file_count':behavior_num,
             'scientific_note':'Staged file presence does not prove ReactSim scenario-to-nuPlan token matching or valid checkpoint inference.',
             'ready_for_preprocess':(apply and not any(x in problem for x in db_status) and
                                     not any(x in problem for x in root_status.values()))}
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nuplan-root',default=DEFAULT_NUPLAN)
    p.add_argument('--reactsim-repo',default=DEFAULT_REPO)
    p.add_argument('--workspace',default=DEFAULT_WORK)
    p.add_argument('--behaviors',default=DEFAULT_BEHAVIOR)
    p.add_argument('--splits',nargs='+',default=['val'])
    p.add_argument('--max-dbs',type=int,default=0,help='0=all selected DBs, positive integer=path smoke only')
    p.add_argument('--apply',action='store_true',help='Actually create safe symlinks (dry-run default)')
    p.add_argument('--output',help='Save diagnostic JSON')
    args=p.parse_args(argv)
    output=inspect(args.nuplan_root,args.reactsim_repo,args.workspace,args.behaviors,
                   args.splits,args.apply,args.max_dbs)
    if args.output:
        dst=Path(args.output);dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    if any(v.startswith('conflict') or v=='missing_source' for v in output['root_link_status'].values()):
        raise SystemExit(3)
if __name__=='__main__':main()
