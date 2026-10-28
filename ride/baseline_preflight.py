"""Static, non-destructive inventory of external RQ3 repositories and checkpoints."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path


def digest(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()


def git_head(root):
    if not (root/'.git').exists():return None
    p=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else None


def inventory(gameformer:Path,reactsim:Path,bosch:Path|None,nuplan:Path):
    checks={}
    for name,root,files in [
        ('gameformer',gameformer,('Planner/planner.py','run_nuplan_test.py','requirements.txt')),
        ('reactsim',reactsim,('reactsim_data_pipeline/scenario_cache.py','SMART_CATK_TrajTok','MTR')),
        ('bosch',bosch,('environment.yml','README.md'))]:
        if root is None:continue
        checks[name]={'path':str(root),'source_exists':root.is_dir(),
                      'git_commit':git_head(root) if root.is_dir() else None,
                      'expected_files':{f:(root/f).exists() for f in files}}
    ck={}
    for name,root in [('gameformer_model',gameformer),('reactsim_models',reactsim)]:
        paths= (list(root.rglob('*.pth'))+list(root.rglob('*.ckpt'))) if root.is_dir() else []
        # Never hash arbitrary GB archives here; hash only small/available files on demand.
        ck[name]=[{'file':str(x),'bytes':x.stat().st_size,'looks_like_lfs_pointer':
                   (x.stat().st_size<1024 and 'git-lfs' in x.read_bytes()[:256].decode('utf8','replace'))}
                  for x in paths[:200] if x.is_file()]
    return {'gameformer':checks.get('gameformer'), 'reactsim':checks.get('reactsim'),
            'bosch':checks.get('bosch'),
            'nuplan_root':str(nuplan),
            'nuplan_map_present':(nuplan/'maps'/'nuplan-maps-v1.0.json').exists(),
            'nuplan_val_db_files':len(list((nuplan/'nuplan-v1.1/splits/val').glob('*.db'))),
            'checkpoint_inventory':ck,
            'scientific_readiness':{'RIDE_RQ3_main_table':'NOT_READY',
              'native_GameFormer_smoke':'REQUIRES_ORIGINAL_MODEL_PTH_AND_MATCHING_NUPLAN_DEVKIT',
              'native_Bosch_SMART_smoke':'REQUIRES_ISOLATED_NUPLAN_PLANNING_FORK_AND_CHECKPOINT',
              'ReactSim':'REACTIVE_ENVIRONMENT_NOT_PLANNER'},
            'notes':['Bosch SMART and ReactSim SMART may share model family/checkpoints; do not count as two independent learned response assumptions.',
                     'Do not combine Bosch modified planning/ directory with standard GameFormer nuPlan in one Python environment.',
                     'GameFormer main-table candidate selection requires instrumentation of ex-ante choices, not just nuPlan closed-loop score.']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gameformer',type=Path,required=True)
    p.add_argument('--reactsim',type=Path,required=True)
    p.add_argument('--bosch',type=Path)
    p.add_argument('--nuplan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    d=inventory(a.gameformer,a.reactsim,a.bosch,a.nuplan)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(d['scientific_readiness'],indent=2))
if __name__=='__main__':main()
