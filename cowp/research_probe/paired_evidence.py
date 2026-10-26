"""Evidence-preserving audit of native WOMD observational and proxy-reactive JSON.

This is not a counterfactual causal estimator: no NPZ, trained simulator or
intervention observations are required. It only checks what the EXISTING rows
actually establish, and it explicitly distinguishes ego collisions from
non-ego/non-ego collisions and fixed-pair action dependence.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _pairs(action):
    return {tuple(sorted(map(int, p))) for p in action.get('non_ego_collision_pairs', [])}


def summarize(screen, manifest=None, *, tolerance=3.0):
    rows = screen['rows']
    eligible = [r for r in rows if r.get('eligible')]
    manifest_entries = {e['scenario_id']: e for e in (manifest or {}).get('entries', [])}
    regimes = sorted(set.intersection(*(set(r.get('regimes', {})) for r in eligible))) if eligible else []
    result = {
        'kind': 'descriptive_audit_not_real_world_incidence',
        'n_checked': len(rows), 'n_eligible': len(eligible),
        'n_errors': len(screen.get('errors', [])),
        'original_stable_proxy_scenes': sum(bool(r.get('stable_intervention_proxy_actions')) for r in eligible),
        'eligibility_exclusions': dict(Counter(r.get('reason', 'unspecified') for r in rows if not r.get('eligible'))),
        'cohorts': dict(Counter(c for r in eligible for c in manifest_entries.get(r['scenario_id'], {}).get('cohorts', []))),
        'model_regimes_share_structure': True,
        'regimes': {}, 'joint_evidence': {}, 'flags': [], 'review_rows': []
    }
    candidate_ids = defaultdict(list)
    all_variation_ids = defaultdict(list)
    for regime in regimes:
        totals = Counter()
        for row in eligible:
            acts = row['regimes'][regime]['actions']
            if not acts: continue
            totals['n_scenes'] += 1
            collision = [bool(a['collision']) for a in acts]
            ego = [bool(a['ego_collision']) for a in acts]
            non = [bool(a['non_ego_collision']) for a in acts]
            psets = [_pairs(a) for a in acts]
            any_non = any(non)
            totals['scenes_with_any_collision'] += any(collision)
            totals['scenes_with_non_ego_collision'] += any_non
            totals['scenes_any_collision_action_variant'] += len(set(collision)) > 1
            totals['scenes_ego_collision_action_variant'] += len(set(ego)) > 1
            totals['scenes_non_ego_collision_action_variant'] += len(set(non)) > 1
            totals['scenes_non_ego_pair_set_action_variant'] += len(set(map(frozenset,psets))) > 1
            if len(set(map(frozenset,psets))) > 1: all_variation_ids[regime].append(row['scenario_id'])
            # Same-pair contrasts; do not count unrelated hazards elsewhere as changes.
            union=set().union(*psets)
            varying_pairs = [p for p in union if 0 < sum(p in q for q in psets) < len(acts)]
            totals['scenes_with_specific_pair_variation'] += bool(varying_pairs)
            # Matched alternative must remove the exact same non-ego pair collision,
            # be free from ALL collision, and not sacrifice > tolerance of progress.
            good=[]
            matched=[]
            for i,a in enumerate(acts):
                if not a.get('near_term_ego_clear', False) or not psets[i]: continue
                for j,b in enumerate(acts):
                    if i == j or b['collision'] or float(b['ego_progress_m']) + tolerance < float(a['ego_progress_m']):continue
                    if psets[i].intersection(varying_pairs):
                        case={'bad_accel':a['ego_acceleration_mps2'],
                                     'safe_accel':b['ego_acceleration_mps2'],
                                     'progress_bad_m':a['ego_progress_m'],
                                     'progress_safe_m':b['ego_progress_m'],
                                     'changed_non_ego_pairs':[list(p) for p in sorted(psets[i].intersection(varying_pairs))]}
                        good.append(case)
                        if abs(float(b['ego_progress_m'])-float(a['ego_progress_m']))<=tolerance:
                            matched.append(case)
            totals['scenes_with_progress_noninferior_pair_sensitive_preventability'] += bool(good)
            totals['scenes_with_symmetric_progress_matched_pair_sensitive_preventability'] += bool(matched)
            if matched: candidate_ids[regime].append(row['scenario_id'])
            if varying_pairs or row['regimes'][regime].get('preventable_proxy_candidates'):
                result['review_rows'].append({'scenario_id':row['scenario_id'], 'regime':regime,
                         'varying_pairs':[list(x) for x in sorted(varying_pairs)],
                         'noninferior_cases':good,'strict_matched_cases':matched,
                         'original_proxy_cases':row['regimes'][regime].get('preventable_proxy_candidates',[])})
        result['regimes'][regime]=dict(totals)
        result['regimes'][regime]['pair_variant_scene_ids']=sorted(all_variation_ids[regime])
        result['regimes'][regime]['strict_matched_scene_ids']=sorted(candidate_ids[regime])
    intersections=[set(candidate_ids[k]) for k in regimes]
    stable=set.intersection(*intersections) if intersections else set()
    result['joint_evidence']['task_matched_pair_sensitive_all_regimes_scene_ids']=sorted(stable)
    result['joint_evidence']['n_task_matched_pair_sensitive_all_regimes']=len(stable)
    result['joint_evidence']['independent_model_validation_passed']=False
    result['joint_evidence']['Waymax_closed_loop_tested']=False
    result['flags']=[
        'Observed logged-future ADE is NOT validation of do(ego_action) responses',
        'Three regimes share a single longitudinal IDM-like model family',
        'Non-ego collision cannot be attributed to ego without a same-root no-ego/control counterfactual',
        'Progress is net displacement in simplified simulation, not longitudinal route progress',
        'First 80 rows are scenario-ID ordered from a four-shard cohort; no population frequency inference',
        'Progress-matched means ABSOLUTE displacement difference <= tolerance; non-inferior progress is reported separately',
        'This audit cannot test pairwise/joint existence H1 or closed-loop recovery H3',
    ]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--progress-tolerance',type=float,default=3.0)
    a=p.parse_args()
    screen=json.loads(a.screen.read_text())
    manifest=json.loads(a.manifest.read_text()) if a.manifest else None
    r=summarize(screen,manifest,tolerance=a.progress_tolerance)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('review_rows',)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
