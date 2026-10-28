"""Non-mutating readiness assessment for Bosch SMART and GameFormer-Planner.

A cloned repo is NOT a reproduced planning baseline. Reports absent checkpoints,
interactive GUI invocation, version pinning and necessary nuPlan data.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path


def _sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),''):h.update(block)
    return h.hexdigest()


def inspect_repo(root):
    if not root.is_dir():return {'root':str(root),'present':False}
    git=None
    try:
        git=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,timeout=4)
    except (OSError,subprocess.TimeoutExpired):pass
    weights=[]
    for suffix in ('*.pth','*.pt','*.ckpt','*.pkl'):
        for p in root.rglob(suffix):
            if any(s in p.parts for s in ('.git','__pycache__')):continue
            if len(weights)>=100:break
            size=p.stat().st_size
            weights.append({'path':str(p),'size_bytes':size,
                            'sha256':_sha(p) if size > 256*1024 and suffix!='*.pkl' else None,
                            'suspicious_pointer_or_empty':size<=256})
    return {'root':str(root),'present':True,
      'git_commit':git.stdout.strip() if git is not None and git.returncode==0 else None,
      'weight_candidates':weights}


def inspect(baselines:Path,nuplan:Path):
    gf=next((baselines/p for p in ('GameFormer-Planner','GameFormer-Planner-main','GameFormer_Planner') if (baselines/p).is_dir()),baselines/'GameFormer-Planner')
    bosch=next((baselines/p for p in ('Bosch_SMART_reactive','interactive-closed-loop','bosch_reactive') if (baselines/p).is_dir()),baselines/'Bosch_SMART_reactive')
    react=next((baselines/p for p in ('ReactSim_Bench','ReactSim-Bench-main') if (baselines/p).is_dir()),baselines/'ReactSim_Bench')
    stats={}
    for s in ('val','train_boston','train_vegas','train_singapore','train_pittsburgh','test'):
        stats[s]=len(list((nuplan/'nuplan-v1.1'/'splits'/s).glob('*.db')))
    out={'baseline_roles':{'bosch': 'response environment','gameformer': 'ego planner','reactsim':'response environment'},
         'bosch':inspect_repo(bosch),'gameformer':inspect_repo(gf),'reactsim':inspect_repo(react),
         'nuplan_db_counts':stats,'maps_exist':(nuplan/'maps'/'nuplan-maps-v1.0.json').exists(),
         'gameformer_required_nuplan_devkit':'v1.2.2 (according to uploaded README)',
         'gameformer_headless_patch_required':gf.is_dir() and 'build_nuboard(builder, simulation_file)' in (gf/'run_nuplan_test.py').read_text(errors='replace') if (gf/'run_nuplan_test.py').is_file() else None,
         'rq3_scientific_status':'NOT_STARTED: requires frozen ex-ante candidate choices and same nuPlan scene/response/model budget'}
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--nuplan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);o=inspect(a.baselines,a.nuplan);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in o.items() if k not in ('bosch','gameformer','reactsim')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
