"""H3 EXECUTED PREFIX -> FINITE BACKUP OPTIONS diagnostic.

NOT a proof of viability-set collapse. Fix the same initial WOMD scene,
execute a 1s ego policy prefix while other agents react, then test whether
at least one *specific* pre-registered feedback-free backup trajectory
survives an additional 3s. Returns UNKNOWN/finite-bank absence, not 'forced'.
This approximates receding-contingency action sets with frozen policies;
Waymax/learned reactive model and closed-loop backups still required.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .reactive_stress import _load_scene, _pair_conflicts, simulate, CONTROLS, REGIMES, DT

BACKUPS=(-6.,-3.,-1.,0.,1.5)

def analyze_scene(path,*,prefix_steps=5,total_steps=20,progress_tolerance=3.):
    d,st,L,W,valid,ego=_load_scene(Path(path))
    row={'scenario_id':Path(path).stem,'eligible':False,'regimes':{}}
    if _pair_conflicts(st[:,:2],st[:,4],L,W,valid):row['reason']='initial_overlap';return row
    if sum(valid)<3:row['reason']='fewer_than_three';return row
    v=st[:,2:4];speed=np.linalg.norm(v,axis=1)
    h=np.c_[np.cos(st[:,4]),np.sin(st[:,4])]
    if np.any((speed>3)&(np.sum(v*h,axis=1)/np.maximum(speed,1e-5)<.65)&valid):
        row['reason']='velocity_heading_mismatch';return row
    row['eligible']=True
    for regime in REGIMES:
        actions=[]
        for a in CONTROLS:
            results=[];prefix_progress=None
            for backup in BACKUPS:
                control=[a]*prefix_steps+[backup]*(total_steps-prefix_steps)
                sim=simulate(st,L,W,valid,ego,d['agent_ids'],control,regime,steps=total_steps)
                # prefix XY identical within same regime for any backup.
                prefix_progress=float(np.linalg.norm(sim['paths_xy'][prefix_steps,ego]-sim['paths_xy'][0,ego]))
                first=sim['first_collision_step']
                prefix_safe=first is None or first>prefix_steps
                results.append({'backup_acceleration':backup,'collision_free':not sim['collision'],
                                'prefix_safe':prefix_safe,'first_collision_step':first})
            actions.append({'ego_prefix_acceleration':a,'prefix_progress_m':prefix_progress,
                            'safe_backup_count':sum(z['collision_free'] for z in results),
                            'prefix_safe':all(z['prefix_safe'] for z in results),
                            'backup_bank':results})
        cases=[]
        for a in actions:
            if not a['prefix_safe'] or a['safe_backup_count']!=0:continue
            for other in actions:
                if other['ego_prefix_acceleration']==a['ego_prefix_acceleration']:continue
                if not other['prefix_safe'] or other['safe_backup_count']==0:continue
                delta=other['prefix_progress_m']-a['prefix_progress_m']
                if abs(delta)<=progress_tolerance:
                    cases.append({'risk_prefix':a['ego_prefix_acceleration'],
                                  'backup_preserving_prefix':other['ego_prefix_acceleration'],
                                  'progress_difference_m':delta})
        row['regimes'][regime]={'actions':actions,'matched_finite_bank_backup_loss_cases':cases}
    common=set.intersection(*(set((x['risk_prefix'],x['backup_preserving_prefix'])
                         for x in row['regimes'][r]['matched_finite_bank_backup_loss_cases']) for r in REGIMES))
    row['shared_across_regime_pairs']=[list(x) for x in sorted(common)]
    row['claim_level']='MODEL_RELATIVE_PREFIX_BACKUP_SURVIVAL_ONLY'
    return row


def run(cohort,output,*,max_scenes=0,prefix_steps=5,total_steps=20,tolerance=3.):
    cohort=Path(cohort);m=json.loads((cohort/'cohort_manifest.json').read_text())
    entries=sorted(m['entries'],key=lambda e:hashlib.sha256(('h3-v1|'+e['scenario_id']).encode()).hexdigest())
    if max_scenes:entries=entries[:max_scenes]
    rows=[];errors=[]
    for k,e in enumerate(entries,1):
        try:rows.append(analyze_scene(cohort/e['file'],prefix_steps=prefix_steps,total_steps=total_steps,progress_tolerance=tolerance))
        except Exception as err:errors.append({'scenario_id':e['scenario_id'],'error':repr(err)})
        if k%20==0:print('H3 processed',k,'/',len(entries),flush=True)
    summary={'n_checked':len(rows),'n_eligible':sum(r.get('eligible',False) for r in rows),'errors':len(errors),
        'n_matched_backup_loss_scenes_by_regime':{r:sum(bool(s['regimes'][r]['matched_finite_bank_backup_loss_cases']) for s in rows if s['eligible']) for r in REGIMES},
        'n_shared_regime_scenes':sum(bool(s.get('shared_across_regime_pairs')) for s in rows),
        'true_successor_viability_established':False,'human_causal_effect_established':False}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'summary':summary,'rows':rows,'errors':errors},indent=2))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-scenes',type=int,default=0)
    p.add_argument('--prefix-steps',type=int,default=5);p.add_argument('--total-steps',type=int,default=20)
    p.add_argument('--progress-tolerance',type=float,default=3.)
    a=p.parse_args();print(json.dumps(run(a.cohort,a.output,max_scenes=a.max_scenes,prefix_steps=a.prefix_steps,total_steps=a.total_steps,tolerance=a.progress_tolerance),indent=2))
if __name__=='__main__':main()
