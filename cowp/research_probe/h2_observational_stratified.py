"""Interaction-local observation-fit audit; NOT a causal validation of an ego intervention.

Strata are selected from PAST/CURRENT ONLY, never observed future. In each
scene compare matched responders across CV and model candidates. Macro means
and confidence intervals use scenario as the unit, not correlated time steps.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .reactive_stress import _load_scene, simulate, REGIMES, DT, _pair_conflicts


def _geometry_groups(st,valid,ego,near_radius=30.):
    xy=st[:,:2]; d=xy-xy[ego]; ego_h=np.array([np.cos(st[ego,4]),np.sin(st[ego,4])])
    lateral=d@np.array([-ego_h[1],ego_h[0]])
    along=d@ego_h
    dist=np.linalg.norm(d,axis=1)
    speed=np.linalg.norm(st[:,2:4],axis=1)
    base=valid.copy();base[ego]=False
    return {'all':base,
            'near_ego_30m':base&(dist<=near_radius),
            'longitudinal_corridor':base&(dist<=near_radius)&(np.abs(lateral)<=5.)&(along>=-20)&(along<=30),
            'moving_near_ego':base&(dist<=near_radius)&(speed>=2.)}


def examine(path,steps=10):
    data,st,L,W,valid,ego=_load_scene(path)
    sid=Path(path).stem
    if not valid[ego] or _pair_conflicts(st[:,:2],st[:,4],L,W,valid):
        return {'scenario_id':sid,'eligible':False,'reason':'bad_initial_state'}
    if data['future_x'].shape[1]<2*steps:
        return {'scenario_id':sid,'eligible':False,'reason':'short_future'}
    indices=2*np.arange(1,steps+1)-1
    if not np.all(data['future_valid'][ego,indices]>.5):
        return {'scenario_id':sid,'eligible':False,'reason':'invalid_ego_future'}
    ego_path=np.vstack((st[ego,:2],np.stack((data['future_x'][ego,indices],data['future_y'][ego,indices]),axis=1)))
    target=np.stack((data['future_x'][:,indices],data['future_y'][:,indices]),axis=-1)
    observed=(data['future_valid'][:,indices]>.5)&valid[:,None]
    t=np.arange(1,steps+1)*DT
    pred={'CV':st[None,:,:2]+st[None,:,2:4]*t[:,None,None]}
    pred['CV']=np.transpose(pred['CV'],(1,0,2))
    ids=data['agent_ids'].tolist()
    for regime in REGIMES:
        sim=simulate(st,L,W,valid,ego,ids,0.,regime,steps=steps,logged_ego_path=ego_path)
        pred[regime]=np.transpose(sim['paths_xy'][1:],(1,0,2))
    groups=_geometry_groups(st,valid,ego)
    result={'scenario_id':sid,'eligible':True,'strata':{}}
    for name,agent_mask in groups.items():
        per_model={}; ok=observed&agent_mask[:,None]
        n_agents=int(np.sum(agent_mask & observed.any(axis=1)))
        for metric,n_time in [('ade1_m',min(5,steps)),('ade2_m',min(10,steps))]:
            mask=ok[:,:n_time]
            for method in pred:
                errs=np.linalg.norm(pred[method][:,:n_time]-target[:,:n_time],axis=-1)
                per_model.setdefault(method,{})[metric]=float(np.mean(errs[mask])) if mask.any() else None
        result['strata'][name]={'n_agents':n_agents,'models':per_model}
    return result


def _summarize(rows,key):
    eligible=[r for r in rows if r['strata'][key]['n_agents']>0 and r['strata'][key]['models']['CV']['ade1_m'] is not None]
    out={'n_scenes':len(eligible),'model_results':{}}
    if not eligible:return out
    rng=np.random.default_rng(20261015)
    for k in REGIMES:
        stats={}
        for metric in ('ade1_m','ade2_m'):
            valid_rows=[r for r in eligible if r['strata'][key]['models'][k][metric] is not None and r['strata'][key]['models']['CV'][metric] is not None]
            if not valid_rows: continue
            diff=np.asarray([r['strata'][key]['models'][k][metric]-r['strata'][key]['models']['CV'][metric] for r in valid_rows])
            idx=rng.integers(0,len(diff),size=(2000,len(diff)))
            ci=np.percentile(np.mean(diff[idx],axis=1),[2.5,97.5])
            stats[metric]={'n_scenes':len(diff),'mean_model_minus_cv_m':float(diff.mean()),
                           'paired_bootstrap95_m':[float(x) for x in ci],
                           'model_better_scenes':int(np.sum(diff<0))}
        out['model_results'][k]=stats
    return out


def run(cohort,output,partition='development',max_scenes=320,steps=10):
    cohort=Path(cohort);manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    entries=[e for e in manifest['entries'] if partition=='all' or e.get('evaluation_partition')==partition]
    entries.sort(key=lambda e:hashlib.sha256(('response-fit-v1|'+e['scenario_id']).encode()).hexdigest())
    if max_scenes: entries=entries[:max_scenes]
    rows=[];errors=[]
    for e in entries:
        try:rows.append(examine(cohort/e['file'],steps))
        except Exception as ex:errors.append({'scenario_id':e['scenario_id'],'error':repr(ex)})
    eligible=[r for r in rows if r['eligible']]
    summary={'scientific_claim_allowed':False,'attempted':len(entries),'eligible':len(eligible),'errors':errors,
             'strata':{k:_summarize(eligible,k) for k in ('all','near_ego_30m','longitudinal_corridor','moving_near_ego')},
             'limitations':['Observed logged ego future, not intervention ground truth','CV is short-horizon nonreactive null, not a decentralized response model',
                            'Stratum membership is based on CURRENT geometry only','Shared fixed-heading physics and limited agent subset']}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'summary':summary,'rows':rows},indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partition',default='development',choices=['development','sealed_holdout','all']);p.add_argument('--max-scenes',type=int,default=320);p.add_argument('--steps',type=int,default=10)
    a=p.parse_args();x=run(a.cohort,a.output,a.partition,a.max_scenes,a.steps);print(json.dumps({k:v for k,v in x.items() if k!='strata'},indent=2));print(json.dumps(x['strata'],indent=2))
if __name__=='__main__':main()
