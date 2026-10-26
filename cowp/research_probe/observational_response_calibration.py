"""Compare IDМ-family logged-ego replay to a constant-velocity null model.

This estimates descriptive fit to logged human/AV responses under the observed
single ego history, NOT a human intervention potential outcome. The ego's actual
future is explicitly replayed, never used for selecting counterfactual actions.
"""
from __future__ import annotations
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
from .reactive_stress import _load_scene,simulate,REGIMES,DT,_pair_conflicts


def _summarize(v):
    a=np.asarray(v,dtype=float)
    if not a.size:return {'n':0,'mean_m':None,'median_m':None,'p90_m':None}
    return {'n':int(len(a)),'mean_m':float(np.mean(a)),'median_m':float(np.median(a)),
            'p90_m':float(np.percentile(a,90))}


def examine(path,steps=10):
    data,st,L,W,valid,ego=_load_scene(path)
    if not valid[ego]:return {'scenario_id':Path(path).stem,'eligible':False,'reason':'ego_invalid'}
    if _pair_conflicts(st[:,:2],st[:,4],L,W,valid):
        return {'scenario_id':Path(path).stem,'eligible':False,'reason':'initial_overlap'}
    if data['future_x'].shape[1]<2*steps:
        return {'scenario_id':Path(path).stem,'eligible':False,'reason':'future_too_short'}
    times=2*np.arange(1,steps+1)-1
    if not np.all(data['future_valid'][ego,times]>.5):
        return {'scenario_id':Path(path).stem,'eligible':False,'reason':'ego_logged_future_invalid'}
    ego_path=np.vstack((st[ego,:2],np.stack((data['future_x'][ego,times],data['future_y'][ego,times]),axis=1)))
    valid_future=data['future_valid'][:,times]>.5
    eligible_agents=[int(i) for i in range(len(valid)) if i!=ego and valid[i] and valid_future[i,:5].sum()>=4]
    if not eligible_agents:
        return {'scenario_id':Path(path).stem,'eligible':False,'reason':'no_observed_responders'}
    result={'scenario_id':Path(path).stem,'eligible':True,'steps':steps,'n_observed_responders':len(eligible_agents),'models':{}}
    target=np.stack((data['future_x'][:,times],data['future_y'][:,times]),axis=-1)
    t=np.arange(1,steps+1,dtype=float)*DT
    cv=st[None,:,:2]+st[None,:,2:4]*t[:,None,None]
    predicted={'constant_velocity':np.transpose(cv,(1,0,2))}
    for regime in REGIMES:
        raw=simulate(st,L,W,valid,ego,data['agent_ids'].tolist(),0.0,regime,
                     steps=steps,logged_ego_path=ego_path)
        predicted[regime]=np.transpose(raw['paths_xy'][1:],(1,0,2))
    for key,p in predicted.items():
        err=np.linalg.norm(p-target,axis=-1)
        ok=valid_future & valid[:,None]
        ok[ego,:]=False
        first5=ok[:,:5]
        first10=ok[:,:min(10,steps)]
        result['models'][key]={
            'ade1_m':float(np.mean(err[:,:5][first5])) if first5.any() else None,
            'ade2_m':float(np.mean(err[:,:min(10,steps)][first10])) if first10.any() else None,
            'n_agent_timesteps_1s':int(first5.sum()),
            'n_agent_timesteps_2s':int(first10.sum())}
    return result


def run(cohort,output,partition='development',max_scenes=160,steps=10):
    man=json.loads((Path(cohort)/'cohort_manifest.json').read_text())
    entries=[e for e in man['entries'] if partition=='all' or e.get('evaluation_partition')==partition]
    entries.sort(key=lambda e:hashlib.sha256(('response-fit-v1|'+e['scenario_id']).encode()).hexdigest())
    if max_scenes:entries=entries[:max_scenes]
    rows=[];errors=[]
    for e in entries:
        try:rows.append(examine(Path(cohort)/e['file'],steps))
        except Exception as exc:errors.append({'scenario_id':e['scenario_id'],'error':repr(exc)})
    valid=[r for r in rows if r.get('eligible')]
    methods=('constant_velocity',)+REGIMES
    summary={'scientific_claim_allowed':False,'attempted':len(entries),'eligible':len(valid),
        'errors':errors,'partition':partition,'conditioning':'logged ego future, observational evaluation only',
        'steps':steps,'metrics':{k:{'ade1':_summarize([r['models'][k]['ade1_m'] for r in valid if r['models'][k]['ade1_m'] is not None]),
                                'ade2':_summarize([r['models'][k]['ade2_m'] for r in valid if r['models'][k]['ade2_m'] is not None])} for k in methods},
        'cv_better_ade1_scene_count':{k:sum(r['models']['constant_velocity']['ade1_m'] < r['models'][k]['ade1_m'] for r in valid) for k in REGIMES},
        'claim':'No measured do-effect; fit quality does not establish human counterfactual validity.'}
    o=Path(output);o.parent.mkdir(parents=True,exist_ok=True)
    o.write_text(json.dumps({'summary':summary,'rows':rows},indent=2));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--partition',choices=['all','development','sealed_holdout'],default='development')
    p.add_argument('--max-scenes',type=int,default=160);p.add_argument('--steps',type=int,default=10)
    a=p.parse_args();print(json.dumps(run(a.cohort,a.output,a.partition,a.max_scenes,a.steps),indent=2))
if __name__=='__main__':main()
