"""Read-only independent RIDE v13/v14 same-geometry reproducibility audit.

No simulator calls, no inferred causal proof, no independent-scene assumption.
Model-based source choice errors require an *eligible* original model-audit row.
Raw arm burdens are auxiliary evidence even when action-pair matching fails.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from .evidence import read_jsonl
from .science import Protocol,validate_pair
from .v14_counterfactual_audit import burdens,trace_integrity

PAIRS=(('early_slow','late_slow'),('mild_early','mild_late'),('speed_early','speed_late'))


def _load_panel(path):
    path=Path(path)
    items=read_jsonl(path)
    manifest=[x for x in items if x.get('type')=='manifest']
    ends=[x for x in items if x.get('type')=='run_end']
    if len(manifest)!=1 or len(ends)!=1 or ends[0].get('completed') is not True:
        raise ValueError('INCOMPLETE_PANEL:'+str(path))
    rows={}
    for x in items:
        if x.get('type')!='trial':continue
        k=(str(x['anchor']),str(x['seed']),str(x['profile']))
        if k in rows:raise ValueError('DUPLICATE_ARMS:'+repr(k))
        rows[k]=x
    if len(rows)!=ends[0]['trial_count']:
        raise ValueError('TRIAL_COUNT_MISMATCH:'+str(path))
    return manifest[0],rows


def _arm(r):
    try:
        c=burdens(r['trace'])
        capped=burdens(r['trace'],cap=6)
        med=burdens(r['trace'],median_window=5)
        info=trace_integrity(r['trace'])
    except (KeyError,ValueError,TypeError) as exc:
        return {'eligible_cost':False,'reason':str(exc)}
    m=r.get('metrics') or {}
    return {'eligible_cost':bool(m.get('valid') and m.get('mechanism_executed') and
                                  not m.get('ego_collision')),
            'burden_mps':c,'burden_cap6_mps':capped,'burden_median5_mps':med,
            'c_sustained_brake_s':m.get('follower_c_max_sustained_brake_s'),
            'BC_ttc_s':m.get('min_rear_pair_ttc_s'),
            'BC_pair_collision':m.get('non_ego_pair_collision'),
            'trace_integrity':info}


def one_root(root):
    root=Path(root)
    decision=json.loads((root/'v13_target_decision_audit.json').read_text())
    frozen=(root/'freeze/frozen/chain_cut_in_anchors.jsonl')
    f=json.loads(frozen.read_text().splitlines()[0])
    src= root/'source/carla/chain_cut_in.jsonl'
    tgt= root/'target/carla/chain_cut_in.jsonl'
    sm,source=_load_panel(src)
    tm,target=_load_panel(tgt)
    commit=json.loads((root/'locks/source_model_choices.commit.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes_verified=(digest(src)==commit['source_sha256'] and
                     digest(frozen)==commit['frozen_sha256'])
    if not hashes_verified:raise ValueError('PROSPECTIVE_COMMIT_BROKEN:'+str(root))
    original_count={}
    for row in decision.get('results',[]):
        if row.get('policy')!='SOURCE_THIRD_PARTY':continue
        k=(str(row['anchor']),str(row['seed']),tuple(row['pair']))
        original_count[k]=row
    matched=[]
    for anchor,seed in sorted({(k[0],k[1]) for k in source|target}):
        for a,b in PAIRS:
            ss1,ss2=source.get((anchor,seed,a)),source.get((anchor,seed,b))
            tt1,tt2=target.get((anchor,seed,a)),target.get((anchor,seed,b))
            c=original_count.get((anchor,seed,(a,b)),{})
            item={'anchor':anchor,'seed':seed,'pair':[a,b],
                  'prelocked_selected_action':c.get('selected_action'),
                  'prelocked_scientific_status':c.get('scientific_status'),
                  'model_induced_error_eligible':bool(c.get('paired_model_induced_selection_error')),
                  'missing_source_or_target_arm':not all((ss1,ss2,tt1,tt2))}
            if ss1 and ss2 and tt1 and tt2:
                sv,swhy,_=validate_pair(ss1,ss2,Protocol())
                tv,twhy,_=validate_pair(tt1,tt2,Protocol())
                item['source_pair_valid']=sv
                item['source_pair_reason']=swhy
                item['target_pair_valid']=tv
                item['target_pair_reason']=twhy
                for stage,arr in (('source',(ss1,ss2)),('target',(tt1,tt2))):
                    aa,bb=_arm(arr[0]),_arm(arr[1]); item[stage+'_raw_burden']={a:aa,b:bb}
                    item[stage+'_signed_burden_delta_a_minus_b_mps']=(
                        round(aa['burden_mps']-bb['burden_mps'],6)
                        if 'burden_mps' in aa and 'burden_mps' in bb else None)
            matched.append(item)
    event_count=sum(bool((r.get('metrics') or {}).get('non_ego_pair_collision')) for r in target.values())
    return {'root':str(root),'map':f.get('map'),'seed':f.get('seed'),
            'frozen_sha256':digest(frozen),
            'anchors':[{k:a.get(k) for k in ('anchor','key','coords')} for a in f.get('anchors',[])],
            'frozen_parent_sha256':f.get('v14_frozen_replay_parent_sha256'),
            'source_family':decision.get('source_family'),'target_family':decision.get('target_family'),
            'same_model_placebo':decision.get('summary',{}).get('same_model_placebo'),
            'commit_hashes_verified':hashes_verified,
            'summary_from_locked_decision_audit':decision.get('summary'),
            'source_trial_count':len(source),'target_trial_count':len(target),
            'target_simulator_BC_collision_trial_count':event_count,
            'paired_contrasts':matched}


def _identity(report):
    return {(a['anchor'],tuple(a.get('key') or [])) for a in report['anchors']}


def summarize(roots,stress_root=None,blocked_roots=()):
    results=[one_root(p) for p in roots]
    original_sha={rep['frozen_sha256'] for rep in results if Path(rep['root']).name.startswith('v13_')}
    for rep in results:
        parent=rep.get('frozen_parent_sha256')
        rep['frozen_parent_matches_supplied_original_v13']=(parent in original_sha if parent else None)
    by_scene={}
    for rep in results:
        for anchor,key in _identity(rep):
            by_scene.setdefault((rep['map'],anchor,key),[]).append(Path(rep['root']).name)
    comparisons=[]
    # Exact same road, same seed, same target response family, same pair.
    for i,a in enumerate(results):
        for b in results[i+1:]:
            if a['seed']!=b['seed'] or a['target_family']!=b['target_family'] or a['map']!=b['map']:
                continue
            match_keys=_identity(a)&_identity(b)
            if not match_keys:continue
            A={(p['anchor'],tuple(p['pair'])):p for p in a['paired_contrasts']}
            B={(p['anchor'],tuple(p['pair'])):p for p in b['paired_contrasts']}
            for anchor,key in match_keys:
                for pa in PAIRS:
                    aa=A.get((str(anchor),pa));bb=B.get((str(anchor),pa))
                    if not aa or not bb:continue
                    x,y=aa.get('target_signed_burden_delta_a_minus_b_mps'),bb.get('target_signed_burden_delta_a_minus_b_mps')
                    comparisons.append({'anchor':anchor,'pair':list(pa),
                      'source_a':Path(a['root']).name,'source_b':Path(b['root']).name,
                      'target_family':a['target_family'],
                      'raw_difference_in_target_action_effect_mps':(
                        round(abs(x-y),6) if x is not None and y is not None else None),
                      'both_target_action_pairs_valid':bool(aa.get('target_pair_valid') and bb.get('target_pair_valid')),
                      'effect_A_mps':x,'effect_B_mps':y})
    stress={}
    if stress_root:
        root=Path(stress_root)
        s=json.loads((root/'carla/v14_physical_stress_audit.json').read_text())
        analytic=json.loads((root/'carla/v14_hypothetical_stop_margin.json').read_text())
        panel=read_jsonl(root/'carla/chain_cut_in.jsonl')
        raw_manifest=next(x for x in panel if x.get('type')=='manifest')
        assert panel[-1].get('type')=='run_end' and panel[-1].get('completed') is True
        stress={'summary':s['summary'], 'margin_summary':analytic['summary'],
                'normal_stress_parameters':{'delay_s':raw_manifest.get('chain_c_control_delay_s'),
                                             'brake_command_cap':raw_manifest.get('chain_c_brake_command_cap')},
                'same_controller_identity_checks':{
                  'compared':len(s['same_controller_C_vs_lagged_C_parity']),
                  'strict_initial_ego_B_parity':sum(
                   x['controlled_C_policy_effect_identifiable_in_CARLA']
                   for x in s['same_controller_C_vs_lagged_C_parity'])},
                'interpretation':'Zero-delay cap-one arm is identity/no-perturbation, not a tested delayed response.'}
    missing=[]
    for root in blocked_roots:
        p=Path(root)
        complete=(p/'carla/chain_cut_in.jsonl').exists()
        missing.append({'root':str(root),'completed_panel':complete,
                        'status':'NOT_RUN_OR_BLOCKED' if not complete else 'PANEL_PRESENT_REAUDIT_REQUIRED'})
    return {'protocol':'V15_INDEPENDENT_READ_ONLY_V13_V14_REPRO_AUDIT',
      'summary':{'prospective_decision_roots':len(results),
                 'physical_road_identifiers_in_chosen_files':len(by_scene),
                 'do_not_use_roots_as_independent_scenes':True,
                 'model_induced_error_rows_historical_and_replay':sum(r['summary_from_locked_decision_audit'].get('diagnostic_model_induced_choice_errors',0) for r in results),
                 'model_induced_error_rows_v14_replays':sum(r['summary_from_locked_decision_audit'].get('diagnostic_model_induced_choice_errors',0) for r in results if Path(r['root']).name.startswith('v14_')),
                 'valid_comparison_rows_double_selector_counted':sum(r['summary_from_locked_decision_audit'].get('valid_comparisons',0) for r in results),
                 'target_BC_physical_collision_trial_count':sum(r['target_simulator_BC_collision_trial_count'] for r in results)},
      'roots':results,'same_road_same_seed_target_family_comparisons':comparisons,
      'normal_C_stress_baseline':stress,'blocked_stress_roots':missing,
      'scientific_limitations':[
        'A source or target raw cost does not establish eligible paired decision regret.',
        'Repeated identical road geometry is not an independent road sample.',
        'Failed matching action pairs are UNKNOWN, never zero burden.',
        'CARLA BehaviorAgent and Traffic Manager are not human counterfactual truth.',
        'A hypothetical stopping margin is not a measured collision.',
        'All model transfer choices refer to transparent simulated-lookahead selection, not GameFormer/RACP.']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--stress-root');p.add_argument('--blocked-roots',nargs='*',default=[])
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    report=summarize(a.roots,a.stress_root,a.blocked_roots)
    dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
