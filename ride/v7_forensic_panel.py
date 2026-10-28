"""Review pre-v8 CARLA evidence without importing CARLA or re-labeling invalid outcomes.

Read-only: consumes historical JSONL, preserves the exact protocol recorded in
manifests, never recomputes historical risk labels with newer definitions.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
from ride.ranking import audit_rankings
from ride.carla.mechanism_audit import audit as audit_mechanism
from ride.evidence import read_jsonl


def _source_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit_panel(path):
    path=Path(path)
    data=read_jsonl(path)
    if not data or data[0].get('type')!='manifest':
        raise ValueError('Missing CARLA manifest: '+str(path))
    man=data[0]; trials=[r for r in data if r.get('type')=='trial' and not r.get('joint_targets') and not r.get('recovery')]
    summary=audit_rankings(data)
    mechanism=audit_mechanism(data)
    stages=collections.Counter()
    reasons=collections.Counter()
    events=collections.Counter()
    pre_event=collections.Counter()
    for r in trials:
        m=r.get('metrics') or {}
        stages['all_trials']+=1
        stages['valid']+=int(m.get('valid') is True)
        stages['mechanism_raw']+=int(m.get('mechanism_executed') is True)
        stages['valid_mechanism']+=int(m.get('valid') is True and m.get('mechanism_executed') is True)
        stages['ego_collision_raw']+=int(bool(m.get('ego_collision')))
        stages['third_pair_collision_raw']+=int(bool(m.get('non_ego_pair_collision')))
        stages['third_pair_collision_valid']+=int(bool(m.get('valid')) and bool(m.get('non_ego_pair_collision')))
        stages['cruise_collision_raw']+=int(r.get('profile')=='cruise' and bool(m.get('non_ego_pair_collision')))
        stages['valid_ego_safe']+=int(bool(m.get('valid')) and not m.get('ego_collision'))
        for reason in m.get('reasons') or []:reasons[reason]+=1
        seen=set()
        for ev in r.get('collision_events') or []:
            k=tuple(sorted((str(ev.get('role_a')),str(ev.get('role_b')))))
            seen.add(k)
        for e in seen:events['--'.join(e)]+=1
        for ev in r.get('pre_action_collision_events') or []:
            pre_event['--'.join(sorted((str(ev.get('role_a')),str(ev.get('role_b')))))]+=1
    anchors=tuple(sorted((round(float(a['coords']['ego'][0]),1),round(float(a['coords']['ego'][1]),1))
                        for a in man.get('anchors',[])))
    pair_reasons=collections.Counter(z for r in summary['all_pairs'] for z in r.get('reasons') or [])
    valid_within=[r for r in mechanism['task_matched_continuous_diagnostics'] if r['eligible']]
    return {'source':str(path),'source_sha256':_source_sha(path),
            'complete':bool(data[-1].get('type')=='run_end' and data[-1].get('completed')),
            'protocol':{k:man.get(k) for k in ('seed','anchor_protocol_version','world_reset_protocol',
              'cutin_late_switch_s','cutin_lateral_lookahead_m','cutin_ego_gap_m',
              'cutin_rear_gap_m','rear_speed_delta_mps','action_protocol_v5','map')},
            'anchor_coords_rounded_1m':list(anchors),
            'physical_anchor_unique':len(set(anchors)),
            'counts':dict(stages), 'invalid_reason_counts':dict(reasons),
            'physical_collision_event_pair_trials':dict(events),
            'pre_action_event_counts':dict(pre_event),
            'ranking_summary':summary['summary'],
            'risk_pair_rejection_reasons':dict(pair_reasons),
            'eligible_within_family_pairs':len(valid_within),
            'valid_within_family_pairs_with_nonzero_continuous_burden':sum(
                abs(float(r['delta_burden_b_mps_a_minus_b']))>0.1 for r in valid_within),
            'valid_within_family_pairs_nonzero_binary_brake':sum(
                x['binary_brake_pair'][0] != x['binary_brake_pair'][1] for x in valid_within),
            'claim':'Historical CARLA rollout-only exploratory evidence; not logged planner regret or human causality'}


def summarize(paths):
    panels=[audit_panel(p) for p in paths]
    if not panels:raise ValueError('No panels supplied')
    protocols={tuple((key,str(p['protocol'].get(key))) for key in
                 ('cutin_late_switch_s','cutin_lateral_lookahead_m','cutin_rear_gap_m','rear_speed_delta_mps'))
               for p in panels}
    allcounts=collections.Counter()
    total_matched=total_reversals=0
    coord_groups=collections.defaultdict(list)
    for p in panels:
        allcounts.update(p['counts'])
        total_matched+=p['ranking_summary']['task_matched_cross_model_pairs']
        total_reversals+=p['ranking_summary']['strict_risk_order_reversals']
        for xy in p['anchor_coords_rounded_1m']:
            coord_groups[tuple(xy)].append(p['protocol']['seed'])
    return {'panels':panels,'overall':{'trials':allcounts['all_trials'],
            'physical_corridors_unique_across_panels':len(coord_groups),
            'physical_corridors_reused_across_runs':sum(len(v)>1 for v in coord_groups.values()),
            'raw_third_party_pair_collision_trials':allcounts['third_pair_collision_raw'],
            'valid_third_party_pair_collision_trials':allcounts['third_pair_collision_valid'],
            'valid_mechanism_trials':allcounts['valid_mechanism'],
            'task_matched_cross_model_pairs':total_matched,
            'strict_ranking_reversals':total_reversals,
            'protocols_identical':len(protocols)==1,
            'protocols_detected':len(protocols),
            'rq1':'NOT_SUPPORTED' if total_reversals==0 else 'EXPLORATORY_SIGNAL_ONLY',
            'rq2':'NOT_SUPPORTED_FROM_VALID_PAIR_COLLISIONS' if allcounts['third_pair_collision_valid']==0 else 'REQUIRES_PAIRED_EFFECT_AUDIT',
            'rq3':'NOT_TESTED_NO_EX_ANTE_STRONG_PLANNER_CHOICE'},
            'warnings':['No valid trial may be converted into a negative claim about models across changed protocols.',
                        'Different seeds can resample physical corridors; they are not independent by name alone.',
                        'Raw third-party collisions on invalid TM trajectories cannot be claimed as preventable RQ2 cases.',
                        'Collision-free trials are not proof of safe response probability under human drivers.']}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',action='append',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args(argv)
    result=summarize(a.panel)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result['overall'],indent=2,ensure_ascii=False))
if __name__=='__main__':main()
