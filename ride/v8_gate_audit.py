"""Read-only, repeatable v7/v8 RIDE evidence meta-audit.

Never treats invalid collisions as positive science; never treats a missing
cross-family comparator as a negative observation. This script also detects
protocol drift between discovery and confirmation panels.
"""
from __future__ import annotations
import argparse,collections,json,math,hashlib
from pathlib import Path
from ride.evidence import read_jsonl
from ride.ranking import audit_rankings
from ride.carla.mechanism_audit import audit as audit_mechanism
from ride.carla.response_calibration import sustained_overspeed


def audit_one(path: Path):
    rows=read_jsonl(path)
    meta=next(r for r in rows if r.get('type')=='manifest')
    trials=[x for x in rows if x.get('type')=='trial']
    ranking=audit_rankings(rows)
    mech=audit_mechanism(rows)
    fam=collections.defaultdict(lambda:collections.Counter())
    problems=collections.Counter(); collision_profile=collections.Counter()
    for r in trials:
        m=r.get('metrics') or {}; f=r['family']; c=fam[f]
        c['trials']+=1
        c['valid']+=bool(m.get('valid'))
        c['mechanism_executed']+=bool(m.get('mechanism_executed'))
        c['valid_and_mechanism']+=bool(m.get('valid') and m.get('mechanism_executed'))
        c['thirdparty_collision']+=bool(m.get('non_ego_pair_collision'))
        c['valid_thirdparty_collision']+=bool(m.get('valid') and m.get('non_ego_pair_collision'))
        c['ego_collision']+=bool(m.get('ego_collision'))
        c['cruise_collision']+=bool(r.get('profile')=='cruise' and m.get('non_ego_pair_collision'))
        if r.get('trace'):
            target=float(meta['cruise_mps'])+(float(meta.get('rear_speed_delta_mps',0.)) if r.get('archetype')=='cut_in' else 0.)
            for role,nominal in [('follower_a',float(meta['cruise_mps'])),('follower_b',target)]:
                speed=max(float(x[f'{role}_speed_mps']) for x in r['trace'])
                c[f'{role}_peak_above_1p5x_nominal']+=bool(speed > nominal*1.5)
                c[f'{role}_peak_above_2x_nominal']+=bool(speed > nominal*2.)
                if r.get('profile')=='cruise':
                    c[f'{role}_cruise_sustained_above_1p5x_nominal'] += int(sustained_overspeed(r['trace'],role,nominal))
        if r.get('simulation_exception'): c['exceptions']+=1;problems['exception:'+str(r['simulation_exception']).split(':')[0]]+=1
        for reason in m.get('reasons',[]):problems[reason]+=1
        if m.get('non_ego_pair_collision'):
            collision_profile[(f,str(r['profile']),bool(m.get('valid')))]+=1
    ranking_reasons=collections.Counter(x for pair in ranking['all_pairs'] for x in pair.get('reasons',[]))
    keys=[{'anchor':a.get('anchor'),'key':a['key']} for a in meta.get('anchors',[])]
    return {
      'path':str(path), 'complete':mech['complete'],
      'manifest':{x:meta.get(x) for x in ('map','seed','anchor_protocol_version','cutin_late_switch_s','cutin_lateral_lookahead_m','cutin_rear_gap_m','rear_speed_delta_mps','cruise_mps','world_reset_policy','tm_speed_protocol')},
      'total_trials':len(trials), 'num_anchors':len(keys), 'anchor_keys':keys,
      'by_family':{k:dict(v) for k,v in fam.items()},
      'invalid_reason_counts':dict(problems),
      'collision_profile_counts':[{"family":k[0],"profile":k[1],"valid":k[2],"n":v} for k,v in sorted(collision_profile.items())],
      'paired_summary':ranking['summary'], 'pair_invalid_reasons':dict(ranking_reasons),
      'within_family_task_matched':mech['summary']['valid_task_matched_mechanism_pairs'],
      'exploratory_burden_valid_pairs':sum(1 for q in mech['task_matched_continuous_diagnostics'] if q.get('eligible',False) and abs(q.get('delta_burden_b_mps_a_minus_b',0) or 0.)>=0.5),
      'status': 'DESCRIPTIVE_ONLY_NOT_HUMAN_CAUSAL_EVIDENCE',
      'ttc_missing_or_inapplicable_trial_count':sum(1 for r in trials if (r.get('metrics') or {}).get('ttc_applicable_steps',0)==0),
      'ttc_sparse_trial_count_less_than_10_steps':sum(1 for r in trials if (r.get('metrics') or {}).get('ttc_applicable_steps',0)<10)
    }


def compare_panels(panels:list[dict], spatial_radius_m: float=90.):
    dev=panels[0]; out=[]
    for panel in panels[1:]:
        drift={k:[dev['manifest'].get(k),panel['manifest'].get(k)] for k in (
            'anchor_protocol_version','cutin_late_switch_s','cutin_lateral_lookahead_m','cutin_rear_gap_m','rear_speed_delta_mps','cruise_mps','world_reset_policy')
               if dev['manifest'].get(k)!=panel['manifest'].get(k)}
        common=[]
        if dev['manifest']['map']==panel['manifest']['map']:
            for a in panel['anchor_keys']:
                dists=[math.dist(a['key'][:2],b['key'][:2]) for b in dev['anchor_keys']]
                if dists and min(dists)<spatial_radius_m:
                    common.append({'confirm_anchor':a['anchor'],'min_dev_distance_m':round(min(dists),1)})
        out.append({'path':panel['path'],'protocol_drift':drift,'not_spatially_independent_of_dev':common,
                    'independent_confirmation':not drift and not common and panel['complete']})
    return out


def audit(paths,coverage_path=None):
    panels=[audit_one(Path(path)) for path in paths]
    comparisons=compare_panels(panels)
    rq1_valid=sum(p['paired_summary']['task_matched_cross_model_pairs'] for p in panels)
    rq1_reversals=sum(p['paired_summary']['strict_risk_order_reversals'] for p in panels)
    # Same physical corridor under another seed does not add a new independent cluster.
    cross_confirm=[]
    for i in range(1,len(panels)):
        for j in range(i+1,len(panels)):
            if panels[i]['manifest']['map']!=panels[j]['manifest']['map']:continue
            overlaps=[]
            for a in panels[i]['anchor_keys']:
                dists=[math.dist(a['key'][:2],b['key'][:2]) for b in panels[j]['anchor_keys']]
                if dists and min(dists)<90.: overlaps.append({'left_anchor':a['anchor'],'near_right_distance_m':round(min(dists),1)})
            if overlaps:cross_confirm.append({'left':i,'right':j,'overlapping_corridors':overlaps})
    rq1_label=('DESCRIPTIVE_RANKING_REVERSAL_OBSERVED_NOT_EX_ANTE_PLANNER_ERROR' if rq1_reversals else
       'NOT_IDENTIFIED: no strict reversal among %d eligible cross-family pairs' % rq1_valid)
    result={'panels':panels,'confirmation_checks':comparisons,
       'cross_confirmation_physical_corridor_overlap':cross_confirm,
       'rq1':rq1_label,
       'rq2':'NOT_ESTABLISHED: requires valid same-start task-matched pair with independent response evaluation; collisions alone insufficient',
       'rq3':'NOT_TESTED_OFFICIAL_PLANNERS',
       'science_gate':'NO: frozen matching protocol + speed-calibrated TM + learned independent evaluator + ex-ante choices required',
       'disclaimer':'CARLA model-response differences are not observed human counterfactuals'}
    if coverage_path:
        coverage=json.loads(Path(coverage_path).read_text())
        groups=coverage['groups']
        missing=[x for x in groups if x['status']=='MISSING_IN_ALL_LOCAL_SPLITS']
        result['db_coverage']={'scenario_groups':len(groups),'missing_logs':len(missing),
            'missing_scenarios':sum(x['scenario_count'] for x in missing),
            'missing_by_month':dict(sorted(collections.Counter(x['log_name'][:7] for x in missing).items())),
            'available_scenarios':sum(x['scenario_count'] for x in groups if x['status']!='MISSING_IN_ALL_LOCAL_SPLITS'),
            'status_counts':coverage['status_counts']}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panels',nargs='+',required=True,type=Path)
    p.add_argument('--coverage',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    report=audit(a.panels,a.coverage)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'panels':len(report['panels']),'confirm':report['confirmation_checks'],
        'science_gate':report['science_gate']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
