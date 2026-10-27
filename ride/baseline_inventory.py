"""Read-only inventory of independently installed baseline source/checkpoints.

Never pretends a cloned repo is a runnable reference implementation.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path

METHODS={
  'Bosch_SMART_reactive':{'role':'reactive_world_model_and_nuplan_evaluator',
     'checkpoint':'Repository README says planning/checkpoint; may require separate SMART repo and nuPlan DB/maps'},
  'GameFormer':{'role':'interactive_prediction_or_planner_check_actual_repo',
     'checkpoint':'GameFormer-Planner README uses --model_path trained model; no official model weight inferred'},
  'R3R_Julia':{'role':'decentralized_safety_controller_not_direct_car_planner',
     'checkpoint':'Julia source and project dependencies; no PyTorch .pth required'},
  'RA_CBF':{'role':'learned_responsibility_safety_control',
     'checkpoint':'Author says pretrained model available externally, but closed-loop code not released'},
  'ReactSim_Bench':{'role':'reactive_world_model_benchmark',
     'checkpoint':'Hugging Face zzzy0729/ReactSim-Bench-Models has clsft/ctg/mtr/smart/trajtok/vbd'},
}
EXTENSIONS={'.pt','.pth','.ckpt','.safetensors','.onnx','.jld2','.bson'}
SKIP={'.git','.venv','venv','env','__pycache__','node_modules','datasets','data','outputs'}

def inventory(root):
    base=Path(root).expanduser()
    result={'root':str(base),'directory_exists':base.is_dir(),'methods':{},
            'checkpoints_are_code_compatible_unverified':True}
    for name,meta in METHODS.items():
        d=base/name
        info={**meta,'exists':d.is_dir(),'path':str(d),'git_head':None,
              'checkpoints':[],'checked_file_count':0,'scan_truncated':False}
        if d.is_dir():
            try:
                ret=subprocess.run(['git','-C',str(d),'rev-parse','HEAD'],
                    capture_output=True,text=True,timeout=8)
                if ret.returncode==0:info['git_head']=ret.stdout.strip()
            except (OSError,subprocess.TimeoutExpired):pass
            for folder, dirs, files in os.walk(d):
                dirs[:]=[x for x in dirs if x not in SKIP]
                for filename in files:
                    info['checked_file_count']+=1
                    if info['checked_file_count']>=100000:
                        info['scan_truncated']=True;break
                    f=Path(folder)/filename
                    if f.suffix.lower() not in EXTENSIONS:continue
                    try:
                        size=f.stat().st_size
                        lfs = size < 300 and f.read_bytes().startswith(b'version https://git-lfs')
                        info['checkpoints'].append({'file':str(f.relative_to(d)),
                           'size_bytes':size,'likely_git_lfs_pointer':lfs})
                    except OSError:pass
                if info['scan_truncated']:break
        result['methods'][name]=info
    return result

def main(argv=None):
    pa=argparse.ArgumentParser();pa.add_argument('--root',default='/data0/senzeyu2/dataset/RIDE/research_baselines')
    pa.add_argument('--output',required=True);a=pa.parse_args(argv)
    result=inventory(a.root);o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True)
    o.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for name,info in result['methods'].items():
        print(name,'present='+str(info['exists']),'weights='+str(len(info['checkpoints'])),
              'git='+str(info['git_head']))

if __name__=='__main__':main()
