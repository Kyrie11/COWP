"""Falsifiable H1 finite-bank counterexample search (NOT physical infeasibility proof).

For each ego acceleration action, enumerate each non-ego's admissible fixed
longitudinal controls. Test single-ego collision constraints and then an EXACT
CSP for pairwise other-other compatibility within the finite generated bank.
CSP UNSAT proves only bank-conditional unsatisfiability. Sparse/map-free,
constant-heading kinematics are hypothesis generation, not WOMD causality.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from .reactive_stress import _load_scene, _pair_conflicts, CONTROLS, DT

RESPONSES=(-6.,-3.,-1.,0.,1.5)


def exact_pairwise_csp(domains, conflicts, max_nodes=200000):
    """domains[i] lists candidate IDs; conflicts dict (i,j,p,q)->bool.

    Returns SAT/UNSAT/UNKNOWN (when max_nodes reached). No sampling or
    greedy-only feasibility claims.
    """
    agents=sorted(domains,key=lambda i:len(domains[i]))
    if any(not domains[i] for i in agents):return 'UNSAT',0,{}
    nodes=0;solution={}
    def dfs(depth):
        nonlocal nodes
        if depth==len(agents):return True
        i=agents[depth]
        for x in domains[i]:
            nodes+=1
            if nodes>max_nodes:return None
            if any(conflicts.get((min(i,j),max(i,j),x if i<j else y,y if i<j else x),False)
                   for j,y in solution.items()):continue
            solution[i]=x
            got=dfs(depth+1)
            if got is True:return True
            del solution[i]
            if got is None:return None
        return False
    result=dfs(0)
    return ('SAT' if result is True else 'UNKNOWN' if result is None else 'UNSAT'),nodes,(solution.copy() if result else {})


def roll_constant(st,L,W,valid,acc,steps):
    """Full trajectory array [steps+1, n, 2], with constant headings."""
    pos=st[:,:2].astype(float).copy()
    yaw=st[:,4].astype(float)
    vel=np.maximum(0, np.sum(st[:,2:4] * np.c_[np.cos(yaw),np.sin(yaw)],axis=1))
    traj=np.empty((steps+1,len(pos),2),dtype=float)
    traj[0]=pos
    for t in range(steps):
        new=np.clip(vel+acc*DT,0,37)
        pos=pos+(vel+new)[:,None]*np.c_[np.cos(yaw),np.sin(yaw)]*DT*.5
        vel=new;traj[t+1]=pos
    return traj


def pair_collision(ta,tb,yaw_i,yaw_j,l_i,l_j,w_i,w_j):
    # Uses same SAT footprint routine as all other screening experiments.
    from .reactive_stress import _collide
    return any(_collide(ta[t],tb[t],l_i,w_i,yaw_i,l_j,w_j,yaw_j)
               for t in range(1,min(len(ta),len(tb))))


def analyze_scene(path, *, steps=20, max_agents=6, controls=CONTROLS,
                  response_bank=RESPONSES,max_nodes=200000):
    d,st,L,W,valid,ego=_load_scene(Path(path))
    sid=Path(path).stem
    out={'scenario_id':sid,'status':'FINITE_BANK_HYPOTHESIS_ONLY','actions':[]}
    if _pair_conflicts(st[:,:2],st[:,4],L,W,valid):
        return {**out,'eligible':False,'reason':'initial_oriented_overlap'}
    # Restrict to the indicated subset; never claim all-scene H1 when truncated.
    agents=[i for i in range(len(st)) if i!=ego and valid[i]]
    dists={i:float(np.linalg.norm(st[i,:2]-st[ego,:2])) for i in agents}
    agents=sorted(agents,key=lambda x:dists[x])[:max_agents]
    if len(agents)<2:return {**out,'eligible':False,'reason':'too_few_responders'}
    out.update({'eligible':True,'agents_local':agents,'total_valid_responders':int(sum(valid))-1,
                'truncated_responders':int(sum(valid))-1>len(agents),
                'steps':steps,'response_bank':list(response_bank),
                'physical_infeasibility_certified':False})
    n=len(st);yaws=st[:,4]
    trajectories={}
    for i in agents:
        for ai,a in enumerate(response_bank):
            accelerations=np.zeros(n);accelerations[i]=a
            tr=roll_constant(st,L,W,valid,accelerations,steps)
            trajectories[i,ai]=tr[:,i,:]
    for ea in controls:
        accelerations=np.zeros(n);accelerations[ego]=ea
        te=roll_constant(st,L,W,valid,accelerations,steps)[:,ego,:]
        domains={i:[] for i in agents}
        for i in agents:
            for ai,_ in enumerate(response_bank):
                if not pair_collision(te,trajectories[i,ai],yaws[ego],yaws[i],L[ego],L[i],W[ego],W[i]):
                    domains[i].append(ai)
        conflicts={}
        for jidx,i in enumerate(agents):
            for j in agents[jidx+1:]:
                for ai in domains[i]:
                    for aj in domains[j]:
                        conflicts[(i,j,ai,aj)]=pair_collision(trajectories[i,ai],trajectories[j,aj],
                                          yaws[i],yaws[j],L[i],L[j],W[i],W[j])
        status,nodes,wit=exact_pairwise_csp(domains,conflicts,max_nodes=max_nodes)
        out['actions'].append({'ego_accel':float(ea),'individual_domains':{str(i):list(v) for i,v in domains.items()},
           'every_responder_individually_ego_safe':all(domains.values()),
           'joint_finite_bank_status':status,'nodes':nodes,
           'h1_bank_discordance':all(domains.values()) and status=='UNSAT',
           'joint_witness':{str(i):float(response_bank[a]) for i,a in wit.items()}})
    return out


def run(cohort,output,*,max_scenes=0,max_agents=6,steps=20):
    cohort=Path(cohort);manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    entries=sorted(manifest['entries'],key=lambda e:hashlib.sha256(('h1-v2|'+e['scenario_id']).encode()).hexdigest())
    if max_scenes:entries=entries[:max_scenes]
    rows=[];errors=[]
    for k,e in enumerate(entries,1):
        try:rows.append(analyze_scene(cohort/e['file'],max_agents=max_agents,steps=steps))
        except Exception as err:errors.append({'scenario_id':e['scenario_id'],'error':repr(err)})
        if k%25==0:print('H1 processed',k,'/',len(entries),flush=True)
    summary={'n_checked':len(rows),'errors':len(errors),'n_eligible':sum(x.get('eligible',False) for x in rows),
        'n_candidate_bank_unsat_scenes':sum(any(a['h1_bank_discordance'] for a in x['actions']) for x in rows),
        'n_truncated_scenes':sum(x.get('truncated_responders',False) for x in rows),
        'n_unknown_actions':sum(a['joint_finite_bank_status']=='UNKNOWN' for x in rows for a in x['actions']),
        'causal_claim_allowed':False,'infeasibility_certified':False}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'summary':summary,'rows':rows,'errors':errors},indent=2))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-scenes',type=int,default=0)
    p.add_argument('--max-agents',type=int,default=6);p.add_argument('--steps',type=int,default=20)
    a=p.parse_args();print(json.dumps(run(a.cohort,a.output,max_scenes=a.max_scenes,max_agents=a.max_agents,steps=a.steps),indent=2))
if __name__=='__main__':main()
