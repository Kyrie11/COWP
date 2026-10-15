"""Positive joint-trajectory witness verifier (finite model only).

Input JSONL row should contain:
 scenario_id, ego_action_id, dt, object_ids, ego_object_id,
 xy[T,N,2], yaw[T,N], length[N], width[N], valid[T,N],
 road_valid[T,N], max_accel_mps2 (optional, default=6), source.

The geometric-and-kinematic SAT witness is verified on its explicit trajectory.
Negative result means UNKNOWN, NOT global UNSAT: finite proposal space.
No claim about human independent policies follows from SAT.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from .reactive_stress import _pair_conflicts

def check_row(r,default_max_accel=6.,max_yaw_rate=2.):
    name={'scenario_id':r.get('scenario_id'), 'ego_action_id':r.get('ego_action_id')}
    required=('dt','object_ids','ego_object_id','xy','yaw','length','width','valid','road_valid','source')
    missing=[k for k in required if k not in r]
    if missing:return {**name,'status':'UNKNOWN_INVALID_WITNESS','reasons':['missing '+','.join(missing)]}
    try:
        xy=np.asarray(r['xy'],dtype=float);yaw=np.asarray(r['yaw'],dtype=float)
        L=np.asarray(r['length'],dtype=float); W=np.asarray(r['width'],dtype=float)
        valid=np.asarray(r['valid'],dtype=bool);road=np.asarray(r['road_valid'],dtype=bool)
        ids=list(map(int,r['object_ids']));dt=float(r['dt'])
        if xy.ndim!=3 or xy.shape[-1]!=2:raise ValueError('xy must have shape T,N,2')
        T,N,_=xy.shape
        if T<3 or N<2 or yaw.shape!=(T,N) or valid.shape!=(T,N) or road.shape!=(T,N) or L.shape!=(N,) or W.shape!=(N,):
            raise ValueError('T,N shape inconsistency')
        if len(ids)!=N or len(set(ids))!=N or int(r['ego_object_id']) not in ids:raise ValueError('invalid original actor IDs')
        if not (np.isfinite(xy).all() and np.isfinite(yaw).all() and np.isfinite(L).all() and np.isfinite(W).all()):raise ValueError('non-finite')
        if np.min(L)<=0 or np.min(W)<=0 or not (0.01<=dt<=0.5):raise ValueError('invalid dimensions or timestep')
        active=valid.all(axis=0)
        if not active[ids.index(int(r['ego_object_id']))]:raise ValueError('ego not fully valid')
        if not np.all(road[valid]):return {**name,'status':'UNKNOWN_NOT_ROAD_VALID','reasons':['offroad reported by supplied map rule']}
        all_collisions=[]
        for t in range(T):
            pairs=_pair_conflicts(xy[t],yaw[t],L,W,valid[t]);all_collisions.extend((t,*p) for p in pairs)
        if all_collisions:
            return {**name,'status':'UNKNOWN_COLLIDING_PROPOSAL','first_collision':list(all_collisions[0]),
                    'reasons':['candidate itself collides; does not establish global UNSAT']}
        speed=np.linalg.norm(np.diff(xy,axis=0),axis=-1)/dt
        accel=np.diff(speed,axis=0)/dt
        if np.any(abs(accel[:,active])>float(r.get('max_accel_mps2',default_max_accel))+1e-6):
            return {**name,'status':'UNKNOWN_DYNAMICS_VIOLATION','reasons':['bounded-acceleration failure']}
        yaw_rate=np.abs((np.diff(yaw,axis=0)+math.pi)%(2*math.pi)-math.pi)/dt
        if np.any(yaw_rate[:,active]>max_yaw_rate+1e-6):
            return {**name,'status':'UNKNOWN_DYNAMICS_VIOLATION','reasons':['yaw-rate failure']}
        return {**name,'status':'SAT_WITNESS_WITHIN_DECLARED_MODEL','n_actors':N,'n_steps':T,
                'source':str(r['source']),
                'limitations':['road_valid mask is provided by external simulator; independently verify map provenance',
                               'finite sampled trajectory existence, not policy response probability or global solvability']}
    except Exception as exc:return {**name,'status':'UNKNOWN_INVALID_WITNESS','reasons':[str(exc)]}

def run(jsonl,out):
    rows=[json.loads(x) for x in Path(jsonl).read_text().splitlines() if x.strip()]
    result=[check_row(x) for x in rows]
    summary={'submitted':len(rows),'sat_model_witnesses':sum(x['status']=='SAT_WITNESS_WITHIN_DECLARED_MODEL' for x in result),
             'global_unsat_certificates':0,'scientific_claim_allowed':False,
             'principle':'Valid positive witness only; failed candidate cannot prove nonexistence in unsearched set.'}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps({'summary':summary,'rows':result},indent=2));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--jsonl',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.jsonl,a.output),indent=2))
if __name__=='__main__':main()
