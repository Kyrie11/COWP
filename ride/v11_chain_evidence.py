"""Read-only, fail-closed V11 audit of the *mechanism*, not a RIDE proof.

Supports v10/v11 existing JSONL. Records intervention relative to matched
cruise and verifies a temporal B -> C response *screening signature*.
CARLA traffic agents are not observational human counterfactuals.

Primary RQ1 remains the pre-registered tuple (collision, hard brake, TTC),
this module never silently promotes continuous response metrics to primary.
"""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path
from ride.science import Protocol, same_initial, same_ego_execution, validate_pair
from ride.v10_mechanism_report import _risk_key, _lexicographic_order
from ride.carla.mechanism_audit import _treatment_time

# Mechanism screening only; strictly not significance thresholds or paper safety claims.
B_RESPONSE_DELTA_V_MPS=0.5
C_RESPONSE_DELTA_V_MPS=0.3
PERSISTENT_S=0.20
CHAIN_LAG_LIMIT_S=3.0
PRECOMMAND_EGO_XY_M=0.75
PRECOMMAND_RESPONDER_XY_M=1.0
PRECOMMAND_RESPONDER_DV_MPS=0.75


def sustained_first(times, signal, threshold, start=0.0, dt=0.05, persistence=PERSISTENT_S):
    """First interval of consistently elevated positive signals.

    Missing or nonfinite values break the interval. Not a model of reaction
    time. Assumes consecutive samples at fixed dt, verified by audit_pair.
    """
    consecutive=0
    for t,s in zip(times,signal):
        if t>=start and s is not None and math.isfinite(float(s)) and s>=threshold:
            consecutive+=1
            if consecutive >= math.ceil(persistence/dt-1e-9):
                return round(float(t)-(consecutive-1)*dt,5)
        else:consecutive=0
    return None


def audit_intervention(treatment, cruise):
    """Precondition: frozen chain roles and same initial states."""
    p=Protocol()
    if treatment.get('archetype')!='chain_cut_in' or cruise.get('archetype')!='chain_cut_in':
        return {'eligible':False,'reason':'not_chain_cut_in'}
    m=treatment.get('metrics') or {}; cm=cruise.get('metrics') or {}
    if not m.get('valid') or not cm.get('valid'):
        return {'eligible':False,'reason':'invalid_treatment_or_cruise'}
    ok,why=same_initial(treatment,cruise,p)
    if not ok:return {'eligible':False,'reason':why}
    tr=treatment.get('trace') or []; base=cruise.get('trace') or []
    if not tr or len(tr)!=len(base) or len(tr)<5:
        return {'eligible':False,'reason':'missing_or_unpaired_trace'}
    t0=_treatment_time(treatment,tr)
    if t0 is None:return {'eligible':False,'reason':'no_verified_intervention_command'}
    try:
        ts=[float(t['t']) for t in tr]; bt=[float(t['t']) for t in base]
        if any(abs(a-b)>1e-3 for a,b in zip(ts,bt)):
            return {'eligible':False,'reason':'timebase_not_identical'}
        dt=float(ts[1]-ts[0])
        if not 0.04<=dt<=0.11 or any(abs(ts[i]-ts[i-1]-dt)>1e-3 for i in range(2,len(ts))):
            return {'eligible':False,'reason':'irregular_timebase'}
        i_before=[i for i,t in enumerate(ts) if 1.0<=t<t0]
        i_after=[i for i,t in enumerate(ts) if t>=max(1.0,t0)]
        if not i_before or not i_after:
            return {'eligible':False,'reason':'no_pre_or_post_window'}
        def pos(row,role):return [float(v) for v in row['xy_by_role'][role]]
        pre_ego=max(math.dist(pos(tr[i],'ego'),pos(base[i],'ego')) for i in i_before)
        pre_agent=max(math.dist(pos(tr[i],r),pos(base[i],r)) for i in i_before
                      for r in ('follower_b','follower_c'))
        pre_speed=max(abs(float(tr[i][f'{r}_speed_mps'])-float(base[i][f'{r}_speed_mps']))
                      for i in i_before for r in ('follower_b','follower_c'))
        matched=pre_ego<=PRECOMMAND_EGO_XY_M and pre_agent<=PRECOMMAND_RESPONDER_XY_M and pre_speed<=PRECOMMAND_RESPONDER_DV_MPS
        diag={'pre_ego_max_xy_m':round(pre_ego,4),'pre_BC_max_xy_m':round(pre_agent,4),
              'pre_BC_max_speed_diff_mps':round(pre_speed,4)}
        if not matched:
            return {'eligible':False,'reason':'pre_intervention_drift',**diag}
        bdiff=[float(b['follower_b_speed_mps'])-float(a['follower_b_speed_mps'])
               for a,b in zip(tr,base)]
        cdiff=[float(b['follower_c_speed_mps'])-float(a['follower_c_speed_mps'])
               for a,b in zip(tr,base)]
        bfirst=sustained_first(ts,bdiff,B_RESPONSE_DELTA_V_MPS,start=t0,dt=dt)
        cfirst=sustained_first(ts,cdiff,C_RESPONSE_DELTA_V_MPS,start=t0,dt=dt)
        ordered=(bfirst is not None and cfirst is not None and
                 bfirst <= cfirst <= bfirst+CHAIN_LAG_LIMIT_S)
        post_speed_b=max(bdiff[i] for i in i_after)
        post_speed_c=max(cdiff[i] for i in i_after)
        ttc=[float(tr[i]['rear_pair_ttc_s']) for i in i_after
             if tr[i].get('rear_pair_ttc_s') is not None]
        gap=[float(tr[i]['rear_pair_gap_m']) for i in i_after
             if tr[i].get('rear_pair_gap_m') is not None]
        bc_risk=bool(m.get('non_ego_pair_collision')) or bool(m.get('hard_brake')) or any(z<1.5 for z in ttc)
        baseline_risk=bool(cm.get('non_ego_pair_collision')) or bool(cm.get('hard_brake')) or any(
            float(base[i]['rear_pair_ttc_s'])<1.5 for i in i_after
            if base[i].get('rear_pair_ttc_s') is not None)
        return {'eligible':True,'reason':'','t0_lateral_command_s':round(t0,3),**diag,
                'first_B_extra_slowdown_s':bfirst,'first_C_extra_slowdown_s':cfirst,
                'ordered_B_then_C_slowdown_proxy':ordered,
                'max_extra_B_speed_loss_mps':round(post_speed_b,4),
                'max_extra_C_speed_loss_mps':round(post_speed_c,4),
                'min_BC_gap_m_post':round(min(gap),3) if gap else None,
                'min_BC_ttc_s_post':round(min(ttc),3) if ttc else None,
                'TTC_applicable_post_steps':len(ttc),
                'B_C_primary_hazard_treatment':bc_risk,
                'B_C_primary_hazard_cruise':baseline_risk,
                'new_hazard_vs_cruise_not_task_matched':bool(bc_risk and not baseline_risk),
                'mechanism_stage':'ORDERED_RESPONSE_PROXY_ONLY' if ordered else 'NOT_OBSERVED',
                'causal_note':'Matched simulated negative control; neither chain causality nor human validity certified.'}
    except (ValueError,TypeError,KeyError,OverflowError) as ex:
        return {'eligible':False,'reason':'malformed_trace_'+type(ex).__name__}


def read_panel(path):
    rows=[];broken=0
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if not line.strip():continue
            try:rows.append(json.loads(line))
            except ValueError:broken+=1
    manifests=[r for r in rows if r.get('type')=='manifest']
    trials=[r for r in rows if r.get('type')=='trial' and not r.get('joint_targets') and not r.get('recovery')]
    ends=[r for r in rows if r.get('type')=='run_end']
    completed=(broken==0 and len(manifests)==1 and len(ends)==1 and ends[0].get('completed') is True
               and ends[0].get('trial_count')==len(trials))
    return manifests[0] if len(manifests)==1 else {}, trials, completed,broken


def count_spatial_corridors(keys, separation_m=90.0):
    """Count physical corridor clusters; seeds and action arms are not IID scenes."""
    keys=list(set(keys));parent=list(range(len(keys)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            if keys[i][0]==keys[j][0] and math.dist(keys[i][1:],keys[j][1:])<separation_m:
                pi,pj=find(i),find(j)
                if pi!=pj:parent[pi]=pj
    return len({find(i) for i in range(len(keys))})


def evaluate(paths):
    p=Protocol();panels=[];contrasts=[];negative=[];cross=[];counts=collections.Counter()
    unique_physical={};observed_contexts=[]
    for path in paths:
        m,ts,complete,broken=read_panel(path)
        details={'file':str(path),'complete':complete,'broken_json_lines':broken,
                 'map':m.get('map'), 'seed':m.get('seed'), 'anchors':len(m.get('anchors') or []),
                 'trials':len(ts),'valid_trials':sum(bool(t.get('metrics',{}).get('valid')) for t in ts),
                 'exceptions':collections.Counter(str(t.get('simulation_exception')).split(':')[0]
                       for t in ts if t.get('simulation_exception'))}
        details['exceptions']=dict(details['exceptions'])
        panels.append(details)
        counts['trials']+=len(ts)
        counts['valid_trials']+=details['valid_trials']
        counts['invalid_trials']+=len(ts)-details['valid_trials']
        counts['post_action_primary_pair_collisions_valid']+=sum(bool(t.get('metrics',{}).get('valid') and
            t.get('metrics',{}).get('non_ego_pair_collision')) for t in ts)
        counts['hard_brake_valid']+=sum(bool(t.get('metrics',{}).get('valid') and t.get('metrics',{}).get('hard_brake')) for t in ts)
        counts['simulator_exceptions']+=sum(bool(t.get('simulation_exception')) for t in ts)
        by={};duplicate=[]
        for t in ts:
            key=(str(t.get('anchor')),str(t.get('seed')),t.get('family'),t.get('profile'))
            if key in by:duplicate.append(key)
            by[key]=t
        details['duplicate_trial_keys']=[list(k) for k in duplicate]
        for anchor in m.get('anchors') or []:
            key=(m.get('map'),round(float(anchor['key'][0]),1),round(float(anchor['key'][1]),1))
            unique_physical[key]=True
        contexts={(k[0],k[1]) for k in by}
        for anchor, seed in sorted(contexts):
            fams={k[2] for k in by if k[:2]==(anchor,seed)}
            for fam in fams:
                cruise=by.get((anchor,seed,fam,'cruise'))
                if cruise is None:continue
                for action in ('early_slow','late_slow','mild_early','mild_late'):
                    a=by.get((anchor,seed,fam,action))
                    if a is None:continue
                    z=audit_intervention(a,cruise)
                    negative.append({'file':str(path),'map':m.get('map'),'anchor':anchor,'seed':seed,
                                     'family':fam,'action':action,**z})
            for a,b in p.contrast_actions:
                scores={};errors=[];paired=True
                for family in p.families:
                    x=by.get((anchor,seed,family,a));y=by.get((anchor,seed,family,b))
                    if x is None or y is None:
                        scores[family]={'eligible':False,'reason':'missing_arm'};paired=False;continue
                    ok,why,progress=validate_pair(x,y,p)
                    scores[family]={'eligible':ok,'reason':why,'progress_delta_m':progress}
                    if ok:
                        scores[family]['risk_order']=_lexicographic_order(_risk_key(x,p),_risk_key(y,p))
                        counts['within_family_pairs']+=1
                    else:paired=False
                if paired:
                    for action in (a,b):
                        x=by[(anchor,seed,'tm',action)];y=by[(anchor,seed,'behavior',action)]
                        ok,why=same_initial(x,y,p)
                        if not ok:errors.append(action+':'+why)
                        ok,why=same_ego_execution(x,y)
                        if not ok:errors.append(action+':'+why)
                matched=paired and not errors
                if matched:
                    counts['cross_family_pairs']+=1
                    k=(m.get('map'),round(float(m['anchors'][int(anchor)]['key'][0]),1),
                       round(float(m['anchors'][int(anchor)]['key'][1]),1))
                    observed_contexts.append(k)
                r1=scores.get('tm',{}).get('risk_order');r2=scores.get('behavior',{}).get('risk_order')
                reversal=bool(matched and r1 in (-1,1) and r1==-r2)
                if reversal:counts['strict_primary_rank_reversals']+=1
                contrasts.append({'file':str(path),'anchor':anchor,'seed':seed,'actions':[a,b],
                                  'cross_family_eligible':matched,'reversal':reversal,
                                  'family':scores,'errors':errors})
    counts.setdefault('strict_primary_rank_reversals',0)
    counts.setdefault('cross_family_pairs',0)
    counts.setdefault('within_family_pairs',0)
    counts['distinct_anchor_keys_in_files']=len(unique_physical)
    counts['distinct_anchor_keys_with_matched_rq1_pairs']=len(set(observed_contexts))
    counts['spatial_corridor_clusters_90m_all']=count_spatial_corridors(unique_physical)
    counts['spatial_corridor_clusters_90m_with_matched_rq1_pairs']=count_spatial_corridors(observed_contexts)
    counts['matched_cruise_negative_controls']=sum(x.get('eligible') is True for x in negative)
    counts['ordered_B_then_C_response_proxies']=sum(x.get('eligible') and x.get('ordered_B_then_C_slowdown_proxy') for x in negative)
    counts['new_primary_hazard_vs_cruise_exploratory']=sum(x.get('eligible') and x.get('new_hazard_vs_cruise_not_task_matched') for x in negative)
    out={'protocol':'V11_SCIENTIFIC_GATE_EXPLORATORY_WITH_FROZEN_PRIMARY_ENDPOINTS',
         'panels':panels,'counts':dict(counts), 'contrasts':contrasts,
         'negative_controls':negative,
         'rq1':'NOT_ESTABLISHED' if not counts['strict_primary_rank_reversals'] else 'SCREENING_POSITIVES_REQUIRE_HELDOUT',
         'rq2':'NOT_ESTABLISHED_REQUIRES_TASK_MATCHED_NEW_BC_HAZARD_AND_REPLICATION',
         'rq3':'NOT_TESTED_NO_PRE_ROLLOUT_PLANNER_CHOICES',
         'limits':['Physical map anchors/seeds are not IID trial observations.',
                   'A B-before-C response proxy is NOT a proven mediated causal effect.',
                   'Negative-control cruise does not satisfy task-matched alternative-action RQ2.',
                   'Engine-exception trials must remain in denominators.',
                   'No CARLA controller provides human reaction counterfactual ground truth.',
                   'Post-rollout risk ordering is not a real planner choice.']}
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    out=evaluate(a.input)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'counts':out['counts'],'RQ1':out['rq1'],'RQ2':out['rq2']},indent=2))
    return 0 if all(pn['complete'] for pn in out['panels']) else 4

if __name__=='__main__':raise SystemExit(main())
