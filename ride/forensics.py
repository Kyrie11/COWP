"""Stream CARLA RIDE JSONL, including directly from zip, without assuming completion.

This is a descriptive forensic diagnostic. A single score or count is never an
RQ pass. It records denominators, censoring, and physically observed action pairs.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import json
import math
import zipfile
from pathlib import Path


def _stream(path, member=None):
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            matches=[n for n in z.namelist() if n.endswith('.jsonl') and not n.startswith('__MACOSX')]
            if member is None:
                if len(matches)!=1:
                    raise ValueError('Zip has %d JSONLs: specify --member (examples: %s)' %
                                     (len(matches), matches[:6]))
                member=matches[0]
            if member not in z.namelist():
                raise ValueError('Zip member not found: '+str(member))
            with z.open(member) as f:
                for line in f:
                    if line.strip():yield json.loads(line)
    else:
        with open(path,'rb') as f:
            for line in f:
                if line.strip():yield json.loads(line)


def _measured(r):
    m=r.get('metrics') or {}
    return (int(bool(m.get('non_ego_pair_collision'))),
            int(bool(m.get('hard_brake'))),
            int(m.get('min_rear_pair_ttc_s') is not None and
                float(m['min_rear_pair_ttc_s']) < 1.5))


def summarize(rows):
    counts=collections.Counter(); fam=collections.defaultdict(collections.Counter)
    pairs=collections.defaultdict(dict); run_ends=[]; manifest=None; last_type=None
    for row in rows:
        typ=row.get('type');last_type=typ
        counts['records']+=1
        counts['type_'+str(typ)]+=1
        if typ=='manifest':
            if manifest is not None:counts['duplicate_manifests']+=1
            manifest={k:v for k,v in row.items() if k not in ('anchors','rejections','rejection_counts')}
            counts['declared_anchor_count']=len(row.get('anchors') or [])
            continue
        if typ=='run_end':run_ends.append(row);continue
        if typ!='trial':continue
        m=row.get('metrics') or {};ts=row.get('trace') or []
        ff=row.get('family','unknown'); fam[ff]['trials']+=1
        for name, predicate in (
            ('valid',bool(m.get('valid'))),('mechanism_executed',bool(m.get('mechanism_executed'))),
            ('third_party_pair_collision',bool(m.get('non_ego_pair_collision'))),
            ('ego_collision',bool(m.get('ego_collision'))),
            ('hard_brake_original_proxy',bool(m.get('hard_brake'))),
            ('target_lane_touched',any(x.get('ego_in_target_lane') for x in ts)),
            ('ego_junction_touched',any(x.get('ego_in_junction') for x in ts)),
            ('simulation_exception',bool(row.get('simulation_exception'))),
            ('incomplete_trial_trace',bool(ts and len(ts)!=round(
                float((manifest or {}).get('duration_s',8))/float((manifest or {}).get('dt',0.05)))))
        ):
            if predicate:counts[name]+=1;fam[ff][name]+=1
        for reason in m.get('reasons',[]):counts['invalid_reason_'+str(reason)]+=1
        if row.get('simulation_exception'):
            counts['exception_'+str(row['simulation_exception']).split(':')[0]]+=1
        if ts:
            # Post-hoc sensitivity diagnostic only. Metrics in the saved
            # experiment are not overwritten or retrospectively preregistered.
            late=[x for x in ts if float(x['t'])>=1.0]
            if late:
                counts['available_post_1s']+=1
                if any(min(float(x['follower_a_accel']),float(x['follower_b_accel'])) < -4
                       for x in late):counts['post_1s_instantaneous_brake_proxy']+=1
                for k in ('follower_a_accel','follower_b_accel'):
                    run=0
                    for t in late:
                        run=run+1 if float(t[k]) < -4.0 else 0
                        if run>=4:
                            counts['post_1s_sustained_brake_proxy']+=1
                            break
                    else:continue
                    break
        if ff in ('tm','behavior') and not row.get('joint_targets') and not row.get('recovery'):
            key=(str(row.get('archetype')),str(row.get('anchor')),str(row.get('seed')),ff)
            action=row.get('profile')
            if action in pairs[key]:counts['duplicate_action_arm']+=1
            pairs[key][action]={
                'valid':bool(m.get('valid')),
                'mechanism':bool(m.get('mechanism_executed')),
                'progress':m.get('ego_progress_m'),
                'risk':_measured(row),
                'trace_len':len(ts),
            }
    cohort=collections.Counter()
    contexts=set()
    for (scenario,anchor,seed,ff),arms in pairs.items():
        contexts.add((scenario,anchor,seed))
        for action_a,action_b in (('early_slow','late_slow'),('mild_early','mild_late')):
            if action_a not in arms or action_b not in arms:
                cohort['missing_preregistered_arm']+=1;continue
            cohort['complete_preregistered_arm_pairs']+=1
            a,b=arms[action_a],arms[action_b]
            if not a['valid'] or not b['valid']:
                cohort['invalid_pair']+=1;continue
            if not a['mechanism'] or not b['mechanism']:
                cohort['maneuver_not_executed']+=1;continue
            if a['progress'] is None or b['progress'] is None or not all(
                math.isfinite(float(v)) for v in (a['progress'],b['progress'])):
                cohort['missing_or_nonfinite_progress']+=1;continue
            if abs(float(a['progress'])-float(b['progress']))>3:
                cohort['progress_not_matched']+=1;continue
            # NOT the official gate: requires physical distinctness,
            # identical initial states and cross-family parity separately.
            cohort['potential_pair_before_physical_check']+=1
            if a['risk']!=b['risk']:cohort['risk_change_before_physical_check']+=1
    done=(len(run_ends)==1 and bool(run_ends[0].get('completed')) and
          int(run_ends[0].get('trial_count',-1))==counts['type_trial'] and
          last_type=='run_end')
    return {
        'protocol':'descriptive_forensics_v1_not_science_gate',
        'manifest':manifest, 'run_completed_and_count_verified':done,
        'last_record_type':last_type,
        'run_end_records':run_ends,
        'counts':dict(counts),
        'family_counts':{k:dict(v) for k,v in fam.items()},
        'rough_action_pairs_NOT_official_RQ1_or_RQ2':dict(cohort),
        'distinct_anchor_seed_contexts':len(contexts),
        'scientific_claim_allowed':False,
        'limitations':[
            'Only basic within-family action pairs are checked; use ride.evidence/ride.ranking for full validity',
            'Post-1s braking counts are post-hoc sensitivity diagnostics, not pre-registered endpoints',
            'Joint-witness externally controlled agents are not independent human responses',
            'CARLA TM and BehaviorAgent share CARLA dynamics and map',
        ]}


def main(argv=None):
    pa=argparse.ArgumentParser(description='Stream incomplete CARLA files without losing censoring')
    pa.add_argument('--input',required=True,help='JSONL or ZIP')
    pa.add_argument('--member',help='Exact member path when ZIP contains multiple JSONL files')
    pa.add_argument('--output',required=True)
    a=pa.parse_args(argv)
    result=summarize(_stream(a.input,a.member))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in (
        'run_completed_and_count_verified','counts',
        'rough_action_pairs_NOT_official_RQ1_or_RQ2','distinct_anchor_seed_contexts')},
        ensure_ascii=False,indent=2))

if __name__=='__main__':main()
