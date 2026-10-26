"""Export native WOMD cohorts into R3R-style planar initial/goal specification.

This ONLY exports inputs; it does not implement or execute the authors' R3R
algorithm. Official native Julia planner must be separately reproduced and then
verified for initialization, model equivalence, communication, and collision.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def export(cohort,output,max_scenes=30):
    cohort=Path(cohort); manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    out=[]
    for row in manifest['entries'][:max_scenes]:
        with np.load(cohort/row['file'],allow_pickle=False) as f:
            n=len(f['agent_ids']);start=np.c_[f['current_x'][:,0],f['current_y'][:,0]]
            dest=[]
            for i in range(n):
                valid=np.flatnonzero(f['future_valid'][i]>.5)
                if not len(valid):dest.append(start[i].tolist())
                else:dest.append([float(f['future_x'][i,valid[-1]]),float(f['future_y'][i,valid[-1]])])
            if not np.isfinite(start).all() or not np.isfinite(dest).all():continue
            out.append({'scenario_id':row['scenario_id'],'start_xy_m':start.tolist(),'goal_xy_m':dest,
               'yaw_rad':f['current_bbox_yaw'][:,0].astype(float).tolist(),
               'speed_mps':np.hypot(f['current_velocity_x'][:,0],f['current_velocity_y'][:,0]).tolist(),
               'length_m':f['current_length'][:,0].astype(float).tolist(),
               'width_m':f['current_width'][:,0].astype(float).tolist(),
               'model_conversion':'UNVALIDATED: native R3R uses Dubins agents, circular safety distance and communicated trajectories; WOMD participants do not share committed plans'})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'benchmark_status':'INPUT_EXPORT_ONLY_DO_NOT_REPORT_BASELINE_RESULTS',
      'inputs':out,'license_source':'User-provided R3R official code retained separately',
      'required_native_steps':['Verify Julia Project.toml and Manifest.toml from supplied official repository',
        'Run authors native benchmark first with original communication and backup assumptions',
        'Calibrate vehicle size and dynamics before comparing any result',
        'Report communication-based R3R separate from a no-V2X ablation, which is NOT original R3R']},indent=2))
    return len(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-scenes',type=int,default=30)
    a=p.parse_args();print('exported',export(a.cohort,a.output,a.max_scenes))
