"""RIDE v13: prospective model-to-independent-response *diagnostic*.

This is not an official planner or evidence of human causal effects. Source
rollouts are an unusually informative simulator-model lookahead, executed and
hashed BEFORE any target-family rollout in the supported prospective workflow.
A choice error is defined only when a task-matched counterfactual in the
independent evaluator has a preregistered, meaningful third-party advantage.

CARLA TM/BehaviorAgent share an engine: cross-family validation is useful
screening, not two independent human behavior data sources. Retrospective
analysis is explicitly distinct from prospectively locked selection.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from .evidence import read_jsonl
from .science import Protocol, same_ego_execution, same_initial, validate_pair

VERSION = 'ride_v13_source_before_target_20261029'
PAIRS = (('early_slow', 'late_slow'),
         ('mild_early', 'mild_late'),
         ('speed_early', 'speed_late'))
POLICIES = ('SOURCE_THIRD_PARTY', 'PROGRESS_ONLY')
ROLE = 'follower_c'
DEFAULT_COST_THRESHOLD = 2.0  # m/s^2, threshold fixed before the next run
DEFAULT_MEANINGFUL_DELTA = 0.25  # m/s integrated acceleration exceedance
DEFAULT_TIE_RESOLUTION = 0.05  # m/s integrated acceleration exceedance


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    return path


def _finite(x):
    v = float(x)
    if not math.isfinite(v):
        raise ValueError('nonfinite measurement')
    return v


def extract_trial_cost(row, *, comfort_limit=DEFAULT_COST_THRESHOLD, ttc_limit=1.5):
    """C-specific, post-startup burden (m/s) and original primary hazard.

    Does NOT subtract cruise: cruise is a non-task-matched *negative control*.
    Requires all frames and C's acceleration. Missing is UNKNOWN, not 0.
    """
    m = row.get('metrics') or {}
    if not m.get('valid') or m.get('ego_collision') or not m.get('mechanism_executed'):
        raise ValueError('trial_invalid_or_ego_unsafe_or_mechanism_missing')
    if row.get('archetype') != 'chain_cut_in':
        raise ValueError('not_chain_cut_in')
    trace = row.get('trace') or []
    if len(trace) < 20:
        raise ValueError('short_trace')
    dt = None
    total = 0.0
    covered = 0
    for i, f in enumerate(trace):
        t = _finite(f['t'])
        if i:
            d = t - _finite(trace[i-1]['t'])
            if d <= 0 or d > 0.2:
                raise ValueError('bad_timebase')
            if dt is None:
                dt = d
            if abs(d-dt)>0.002:
                raise ValueError('nonuniform_timebase')
        if t < 1.0:
            continue
        acc = _finite(f['follower_c_accel'])
        total += max(0.0, -acc-comfort_limit) * (dt if dt is not None else 0.05)
        covered += 1
    if covered < 20:
        raise ValueError('insufficient_poststartup_frames')
    if int(m.get('ttc_applicable_steps') or 0) < 10:
        raise ValueError('insufficient_BC_TTC_applicability')
    ttc = m.get('min_rear_pair_ttc_s')
    if ttc is None:
        raise ValueError('missing_pair_TTC')
    ttc = _finite(ttc)
    # C's own braking, not any-agent braking.
    c_brake = _finite(m['follower_c_max_sustained_brake_s']) >= 0.2 - 1e-8
    return {
        'primary': [int(bool(m.get('non_ego_pair_collision'))), int(c_brake), int(ttc < ttc_limit)],
        'c_burden_mps': round(total, 6),
        'ego_progress_m': _finite(m['ego_progress_m']),
        'min_BC_ttc_s': ttc,
        'c_brake_episode': c_brake,
        'label': 'SIMULATED_CARLA_OUTCOME_NOT_HUMAN_GROUND_TRUTH',
    }


def _index(rows, family):
    manifest = next((r for r in rows if r.get('type')=='manifest'), None)
    if manifest is None or manifest.get('archetype') != 'chain_cut_in':
        raise ValueError('Expected a complete chain_cut_in CARLA panel')
    out = {}
    for r in rows:
        if r.get('type') != 'trial':
            continue
        if r.get('joint_targets') or r.get('recovery'):
            raise ValueError('Joint/witness rollouts cannot feed this planner diagnostic')
        if r.get('family') != family:
            raise ValueError('Unexpected response family in exclusive source/target panel')
        k = (str(r['anchor']), str(r['seed']), r['profile'])
        if k in out:
            raise ValueError('Duplicate trial identity: ' + str(k))
        out[k] = r
    if not out:
        raise ValueError('No trial rows')
    return manifest, out


def _manifest_parity(src, target):
    keys = ('simulator','archetype','map','anchor_protocol_version','seed',
            'cruise_mps','duration_s','warmup_ticks','rear_speed_delta_mps',
            'cutin_ego_gap_m','cutin_rear_gap_m','chain_bc_gap_m',
            'chain_c_speed_delta_mps','cutin_late_switch_s',
            'cutin_lateral_lookahead_m','cutin_paired_amplitude_mps',
            'behavior_speed_protocol','behavior_route_protocol','anchors')
    for k in keys:
        if src.get(k) != target.get(k):
            return False, 'source_target_manifest_mismatch:' + k
    return True, ''


def _risk_order(a, b, resolution=DEFAULT_TIE_RESOLUTION):
    pa, pb = tuple(a['primary']), tuple(b['primary'])
    if pa != pb:
        return (-1 if pa < pb else 1), 'primary'
    diff = a['c_burden_mps'] - b['c_burden_mps']
    if abs(diff) <= resolution:
        return 0, 'burden_tie'
    return (-1 if diff < 0 else 1), 'burden'


def _select(a, b, policy, tie_resolution):
    if policy not in POLICIES:
        raise ValueError('Unknown reference selector')
    if policy == 'PROGRESS_ONLY':
        order, why = 0, 'ignores_other_response_evidence'
    else:
        order, why = _risk_order(a,b,tie_resolution)
    if order == -1:
        return 'a', 'MODEL_PREFERS_A', why
    if order == 1:
        return 'b', 'MODEL_PREFERS_B', why
    # Explicit deterministic tie policy, including productivity selection;
    # never silently privilege the first candidate.
    if a['ego_progress_m'] > b['ego_progress_m'] + 1e-9:
        return 'a','MODEL_TIE','progress_tiebreak'
    if b['ego_progress_m'] > a['ego_progress_m'] + 1e-9:
        return 'b','MODEL_TIE','progress_tiebreak'
    return 'a','MODEL_TIE','lexical_tiebreak'


def lock_choices(source_rows, *, source_family, target_family, frozen_sha,
                 source_sha, expected_target_path,
                 tie_resolution=DEFAULT_TIE_RESOLUTION,
                 comfort_limit=DEFAULT_COST_THRESHOLD,
                 meaningful_delta=DEFAULT_MEANINGFUL_DELTA, placebo_same_family=False):
    if source_family not in ('tm','behavior') or target_family not in ('tm','behavior'):
        raise ValueError('Unsupported response family')
    if (source_family==target_family) != bool(placebo_same_family):
        raise ValueError('Use distinct response families for model transfer, same family only for explicit placebo')
    m, index = _index(source_rows, source_family)
    contexts = sorted({(k[0], k[1]) for k in index})
    records = []
    protocol = Protocol()
    for anchor, seed in contexts:
        for a, b in PAIRS:
            x = index.get((anchor,seed,a))
            y = index.get((anchor,seed,b))
            for policy in POLICIES:
                result = dict(anchor=anchor, seed=seed, pair=[a,b], policy=policy,
                              status='UNKNOWN', reason='', selected_action=None)
                if x is None or y is None:
                    result['reason']='missing_source_action'
                else:
                    ok, reason, difference = validate_pair(x,y,protocol)
                    result['source_progress_delta_m']=difference
                    if not ok:
                        result['reason']='invalid_source_pair:'+reason
                    else:
                        try:
                            score_a = extract_trial_cost(x, comfort_limit=comfort_limit)
                            score_b = extract_trial_cost(y, comfort_limit=comfort_limit)
                            label, state, why = _select(score_a,score_b,policy,tie_resolution)
                            result.update(status='SELECTED', reason='',
                                          selected_action=a if label=='a' else b,
                                          source_preference=state,
                                          source_reason=why,
                                          source_scores={a:score_a,b:score_b})
                        except (ValueError,KeyError,TypeError) as e:
                            result['reason']='incomplete_source_cost:'+str(e)
                records.append(result)
    return dict(
        protocol=VERSION, mode=('PROSPECTIVE_SAME_MODEL_PLACEBO' if placebo_same_family else
                                'PROSPECTIVE_SOURCE_SIMULATOR_LOOKAHEAD'),
        placebo_same_family=bool(placebo_same_family),
        warning='Source uses complete hypothetical model rollouts, NOT a deployed predictive planner; TM/BA share CARLA physics.',
        source_family=source_family, target_family=target_family,
        source_sha256=source_sha, frozen_sha256=frozen_sha,
        target_path_expected=str(Path(expected_target_path).resolve()),
        frozen_context={'map':m['map'],'seed':m['seed'],
                        'anchor_protocol_version':m['anchor_protocol_version'],
                        'anchors':m['anchors']},
        locked_params={'pairs':PAIRS,'policies':POLICIES,
                       'tie_resolution_mps':tie_resolution,
                       'comfort_threshold_mps2':comfort_limit,
                       'meaningful_C_burden_delta_mps':meaningful_delta,
                       'progress_tolerance_m':protocol.progress_m,
                       'ego_cross_family_pos_tol_m':0.5,
                       'ego_cross_family_speed_tol_mps':0.75,
                       'short_TTC_threshold_s':1.5},
        records=records)


def evaluate_lock(choices, source_rows, target_rows):
    source_family=choices['source_family']; target_family=choices['target_family']
    source_m, src = _index(source_rows, source_family)
    target_m, tgt = _index(target_rows, target_family)
    good, why = _manifest_parity(source_m, target_m)
    if not good: raise ValueError(why)
    if choices['frozen_context'] != {k:source_m[k] for k in ('map','seed','anchor_protocol_version','anchors')}:
        raise ValueError('Frozen context changed after source selection')
    params=choices['locked_params']; protocol=Protocol(progress_m=params['progress_tolerance_m'])
    seen=set(); outcomes=[]
    for ch in choices['records']:
        ident=(ch['anchor'],ch['seed'],tuple(ch['pair']),ch['policy'])
        if ident in seen: raise ValueError('Duplicate committed choice')
        seen.add(ident)
        result={k:ch[k] for k in ('anchor','seed','pair','policy','status','selected_action')}
        result['scientific_status']='UNKNOWN'
        if ch['status']!='SELECTED':
            result['reason']='source_evidence_unknown:'+ch['reason'];outcomes.append(result);continue
        a,b=ch['pair']
        s1,s2 = src.get((ch['anchor'],ch['seed'],a)),src.get((ch['anchor'],ch['seed'],b))
        t1,t2 = tgt.get((ch['anchor'],ch['seed'],a)),tgt.get((ch['anchor'],ch['seed'],b))
        if None in (s1,s2,t1,t2):
            result['reason']='missing_source_or_target_action';outcomes.append(result);continue
        ok, why, delta=validate_pair(t1,t2,protocol)
        result['target_progress_delta_m']=delta
        if not ok:
            result['reason']='invalid_target_pair:'+why;outcomes.append(result);continue
        cross_reasons=[]
        for s,t,action in ((s1,t1,a),(s2,t2,b)):
            ini, ini_reason = same_initial(s,t,protocol)
            if not ini:
                cross_reasons.append(action+':'+ini_reason)
            fixed, fixed_reason=same_ego_execution(s,t,
                pos_tol_m=params['ego_cross_family_pos_tol_m'],
                speed_tol_mps=params['ego_cross_family_speed_tol_mps'])
            if not fixed: cross_reasons.append(action+':'+fixed_reason)
        if cross_reasons:
            result['reason']='cross_response_mismatch:'+','.join(cross_reasons)
            outcomes.append(result);continue
        try:
            cost_a=extract_trial_cost(t1,comfort_limit=params['comfort_threshold_mps2'])
            cost_b=extract_trial_cost(t2,comfort_limit=params['comfort_threshold_mps2'])
        except (ValueError,KeyError,TypeError) as e:
            result['reason']='target_cost_unknown:'+str(e);outcomes.append(result);continue
        chosen=a if ch['selected_action']==a else b
        other=b if chosen==a else a
        if ch['selected_action'] not in (a,b):
            raise ValueError('Committed action not in action pair')
        score={a:cost_a,b:cost_b}
        selected=score[chosen];alternate=score[other]
        primary_selected=tuple(selected['primary']);primary_alternate=tuple(alternate['primary'])
        burden_diff=selected['c_burden_mps']-alternate['c_burden_mps']
        if primary_selected>primary_alternate:
            loss='AVOIDABLE_PRIMARY_THIRD_PARTY_LOSS'
        elif primary_selected<primary_alternate:
            loss='SELECTED_PRIMARY_SAFER'
        elif burden_diff>params['meaningful_C_burden_delta_mps']:
            loss='AVOIDABLE_C_BURDEN_LOSS'
        elif burden_diff < -params['meaningful_C_burden_delta_mps']:
            loss='SELECTED_C_BURDEN_BETTER'
        else:
            loss='NO_MEANINGFUL_TARGET_DIFFERENCE'
        src_order,_ = _risk_order(ch['source_scores'][a],ch['source_scores'][b],params['tie_resolution_mps'])
        target_order,_ = _risk_order(cost_a,cost_b,params['meaningful_C_burden_delta_mps'])
        blind=(src_order==0 and target_order!=0)
        wrong_rank=(src_order!=0 and target_order!=0 and src_order!=target_order)
        result.update(scientific_status=loss, reason='',
                      target_scores=score,
                      source_order=src_order, target_order=target_order,
                      model_blind_distinction=blind,
                      strict_wrong_model_order=wrong_rank,
                      selected_C_burden_excess_mps=round(burden_diff,6),
                      target_avoidable=(loss in ('AVOIDABLE_PRIMARY_THIRD_PARTY_LOSS','AVOIDABLE_C_BURDEN_LOSS')),
                      tie_induced_error=blind and loss.startswith('AVOIDABLE_'),
                      paired_model_induced_selection_error=(not choices.get('placebo_same_family',False) and
                           loss.startswith('AVOIDABLE_') and ch['policy']=='SOURCE_THIRD_PARTY' and
                           (blind or wrong_rank)),
                      evaluation_semantics='model-conditional simulator effect, NOT strong real-world planner failure')
        outcomes.append(result)
    valid=[r for r in outcomes if r['scientific_status']!='UNKNOWN']
    eligible_anchor_contexts={(r['anchor'],r['seed']) for r in valid}
    summary=dict(
        committed_records=len(outcomes), valid_comparisons=len(valid),
        source_unknown=sum(c['status']!='SELECTED' for c in choices['records']),
        target_or_parity_unknown=sum(r['scientific_status']=='UNKNOWN' and r['status']=='SELECTED' for r in outcomes),
        model_blind_distinctions=sum(r.get('model_blind_distinction',False) for r in valid),
        strict_wrong_model_orders=sum(r.get('strict_wrong_model_order',False) for r in valid),
        target_avoidable_choice_losses=sum(r.get('target_avoidable',False) for r in valid),
        diagnostic_model_induced_choice_errors=sum(r.get('paired_model_induced_selection_error',False) for r in valid),
        physical_anchor_contexts_with_valid_comparisons=len(eligible_anchor_contexts),
        physical_anchor_contexts_with_diagnostic_errors=len({(r['anchor'],r['seed']) for r in valid if r.get('paired_model_induced_selection_error')}),
        # Each row is a candidate *comparison*, NOT an IID real-world scene.
        independent_human_response_validation=False,
        official_strong_baselines_run=False,
        same_model_placebo=bool(choices.get('placebo_same_family',False)),
        decision_error_verified_only_for_transparent_model_based_simulation_selector=True)
    return dict(protocol=VERSION, mode=choices['mode'],
                source_family=source_family,target_family=target_family,
                frozen_context=choices['frozen_context'], locked_params=params,
                summary=summary, results=outcomes,
                important_limitations=[
                    'Source lookahead runs hypothetical CARLA futures before target outcome; not a realtime predictor or strong planner.',
                    'Target and source use distinct CARLA responder policies but common engine; no human counterfactual truth.',
                    'Each action pair must pass initial-state, maneuver, matched-progress and cross-family ego-execution checks.',
                    'No measurable event may be inferred from an UNKNOWN, missing or invalid trial.',
                    'Cluster by physical corridor; do not treat choices/seeds/trials as independent road scenes.',
                    'Continuous burden is a newly preregistered diagnostic endpoint, not a retroactive pass of old RQ1.',
                ])


def _read_source(path):
    return read_jsonl(path)


def _do_lock(args):
    if Path(args.target).exists():
        raise RuntimeError('TARGET_ALREADY_EXISTS: lock BEFORE any target-family rollout')
    if Path(args.choices).exists() or Path(args.commit).exists():
        raise RuntimeError('Refuse overwriting an existing decision or commitment')
    source=_read_source(args.source)
    frozen=Path(args.frozen)
    frozen_row=json.loads(frozen.read_text(encoding='utf8').splitlines()[0])
    source_manifest=next(r for r in source if r['type']=='manifest')
    if frozen_row.get('anchors') != source_manifest.get('anchors') or frozen_row.get('map')!=source_manifest.get('map'):
        raise RuntimeError('Source and frozen manifest identify different physical scenes')
    decisions=lock_choices(source,source_family=args.source_family,target_family=args.target_family,
                           source_sha=sha256(args.source),frozen_sha=sha256(args.frozen),
                           expected_target_path=args.target,
                           tie_resolution=args.tie_resolution,
                           comfort_limit=args.comfort_limit,
                           meaningful_delta=args.meaningful_delta,
                           placebo_same_family=bool(getattr(args,'placebo',False)))
    if args.min_decisions>sum(r['status']=='SELECTED' and r['policy']=='SOURCE_THIRD_PARTY' for r in decisions['records']):
        raise RuntimeError('INSUFFICIENT_SOURCE_ACTION_PAIRS: no target rollout authorized')
    write_new(args.choices, decisions)
    write_new(args.commit, {
        'version':VERSION, 'decision_sha256':sha256(args.choices),
        'source_sha256':sha256(args.source),'frozen_sha256':sha256(args.frozen),
        'target_expected':str(Path(args.target).resolve()),
        'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'timestamp_caveat':'Filesystem timestamp and digest are auditable commitments, not trusted external notarization.'})
    return {'source_candidates_locked':len(decisions['records']),
            'valid_source_choices':sum(r['status']=='SELECTED' for r in decisions['records']),
            'target_did_not_exist_at_lock':True}


def _do_evaluate(args):
    committed=json.loads(Path(args.commit).read_text(encoding='utf8'))
    if committed['decision_sha256']!=sha256(args.choices) or committed['source_sha256']!=sha256(args.source) or committed['frozen_sha256']!=sha256(args.frozen):
        raise RuntimeError('COMMITMENT_HASH_MISMATCH: source, frozen or decisions changed')
    if str(Path(args.target).resolve())!=committed['target_expected']:
        raise RuntimeError('EVALUATOR_TARGET_PATH_MISMATCH')
    # Only a weak local anti-leakage check. The prospective script creates the
    # target file after a successful lock; publish raw logs for independent audit.
    if Path(args.target).stat().st_mtime + 2 < Path(args.commit).stat().st_mtime:
        raise RuntimeError('TARGET_FILE_PREDATES_LOCK: not prospective evidence')
    choices=json.loads(Path(args.choices).read_text(encoding='utf8'))
    if choices['source_sha256']!=committed['source_sha256'] or choices['frozen_sha256']!=committed['frozen_sha256']:
        raise RuntimeError('CHOICE_PROVENANCE_MISMATCH')
    result=evaluate_lock(choices,_read_source(args.source),_read_source(args.target))
    result['commit_sha256']=sha256(args.commit)
    write_new(args.output,result)
    return result['summary']


def _do_retrospective(args):
    rows=read_jsonl(args.panel)
    # Deliberately runs on already-known source & target outcomes. Never use
    # the output as a prospectively locked proof or train/holdout validation.
    manifest=next(r for r in rows if r['type']=='manifest')
    source=[r for r in rows if r.get('type') in ('manifest','run_end') or
            (r.get('type')=='trial' and r.get('family')==args.source_family)]
    target=[r for r in rows if r.get('type') in ('manifest','run_end') or
            (r.get('type')=='trial' and r.get('family')==args.target_family)]
    # The two subset run_end counts must reflect their distinct rows.
    for subset in (source,target):
        for r in subset:
            if r.get('type')=='run_end':
                r=r.copy()  # below replace only the subset reference
        for i,r in enumerate(subset):
            if r.get('type')=='run_end':
                subset[i]={**r, 'trial_count':sum(x.get('type')=='trial' for x in subset)}
    choices=lock_choices(source,source_family=args.source_family,target_family=args.target_family,
                         source_sha='RETROSPECTIVE',frozen_sha='RETROSPECTIVE',
                         expected_target_path=args.panel,
                         meaningful_delta=args.meaningful_delta,
                         tie_resolution=args.tie_resolution,
                         comfort_limit=args.comfort_limit,
                         placebo_same_family=(args.source_family==args.target_family))
    result=evaluate_lock(choices,source,target)
    result['mode']='POSTHOC_RETROSPECTIVE_DIAGNOSTIC_NOT_PRECOMMITTED'
    result['summary']['prospective_evidence']=False
    write_new(args.output,result)
    return result['summary']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    lock=sub.add_parser('lock',help='Source rollouts must exist; target outcome MUST NOT exist')
    ev=sub.add_parser('evaluate',help='Verify locked decisions then independently evaluate target rollouts')
    rt=sub.add_parser('retrospective',help='Analyze old combined panel for hypotheses ONLY')
    for child in (lock,ev):
        for k in ('source','frozen','choices','commit','target'):
            child.add_argument('--'+k,required=True)
    lock.add_argument('--source-family',choices=('tm','behavior','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'),required=True)
    lock.add_argument('--target-family',choices=('tm','behavior','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'),required=True)
    lock.add_argument('--min-decisions',type=int,default=1)
    lock.add_argument('--placebo',action='store_true',
                      help='SAME family source and target to measure CARLA reset/seed instability; never model errors')
    ev.add_argument('--output',required=True)
    rt.add_argument('--panel',required=True)
    rt.add_argument('--source-family',choices=('tm','behavior','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'),default='behavior')
    rt.add_argument('--target-family',choices=('tm','behavior','mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'),default='tm')
    rt.add_argument('--output',required=True)
    for child in (lock,rt):
        child.add_argument('--tie-resolution',type=float,default=DEFAULT_TIE_RESOLUTION)
        child.add_argument('--meaningful-delta',type=float,default=DEFAULT_MEANINGFUL_DELTA)
        child.add_argument('--comfort-limit',type=float,default=DEFAULT_COST_THRESHOLD)
    args=p.parse_args(argv)
    for attr in ('tie_resolution','meaningful_delta','comfort_limit'):
        if hasattr(args,attr) and (not math.isfinite(getattr(args,attr)) or getattr(args,attr)<0):
            p.error('--'+attr.replace('_','-')+' must be finite and nonnegative')
    if args.cmd=='lock': out=_do_lock(args)
    elif args.cmd=='evaluate': out=_do_evaluate(args)
    else: out=_do_retrospective(args)
    print(json.dumps(out,indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
