"""Strict machine-readable closure gate for the three consecutive H2 questions.

Absence of evidence is UNTESTED, never FALSIFIED. Requires actual export from
model-specific Waymax pair simulation, independent second family, planner trials.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def evaluate(gate,fit,deep,physical=None,cross_model=None,planner=None,h1=None):
    g=gate['strict_h2'];f=fit['summary'];d=deep['summary']
    fit_metrics=f['metrics']; cv=fit_metrics['constant_velocity']
    comparisons={}
    for x in ('cautious','assertive','heterogeneous'):
        comparisons[x]={'one_sec_ade_m':fit_metrics[x]['ade1']['mean_m'],
                        'cv_one_sec_ade_m':cv['ade1']['mean_m'],
                        'relative_to_cv':fit_metrics[x]['ade1']['mean_m']/cv['ade1']['mean_m'],
                        'cv_better_scenes':f['cv_better_ade1_scene_count'][x]}
    q1={'status':'UNTESTED' if physical is None else 'EXTERNAL_DATA_REQUIRES_SAME_ACTION_GEOMETRY_AUDIT',
        'reason':'Supplied 8s proxy uses longitudinal fixed headings, not map-valid Waymax executed ego action' if physical is None else
        'Waymax IDMRoutePolicy speed-target conditions are not identical to pre-registered cosine acceleration profiles unless action parity is explicitly demonstrated'}
    q2={'status':'UNTESTED' if cross_model is None else ('STRUCTURAL_SCREEN_PRESENT_NOT_HUMAN_CAUSAL_PROOF' if cross_model.get('n_two_family_screens',0)>0 else 'CROSS_MODEL_SCREEN_NO_QUALIFIED_CASE'),
        'reason':'Three IDM parameter regimes are one architecture and observed CV is a nonreactive null model'}
    q3={'status':'UNTESTED' if planner is None else 'EXTERNAL_BASELINE_CHOICE_SCREEN_ONLY',
        'reason':'No actual GameFormer, RA-CBF or common-interface strong planner preferences in current uploaded results'}
    return {'status':'NOT_FULLY_CLOSED_H2_NOT_FALSIFIED',
        'Q1_same_actions_real_geometry':q1,'Q2_structurally_independent_reactions':q2,
        'Q3_actual_baseline_selection':q3,
        'H1_joint_feasibility':{'status':'NOT_TESTED' if h1 is None else 'MODEL_CONDITIONAL_POSITIVE_WITNESS_SCREEN',
                                'note':'A finite sampled proposal failure is UNKNOWN, not global UNSAT'},
        'H2':{'eligible_development_scenes':gate['coverage']['matched_eligible'],
               'strict_any_IDM_config_scenes':g['scene_any_regime'],
               'exact_same_pair_same_actions_IDM_configs':g['same_pair_same_action_repeated_IDM_family_scenes'],
               'all_three_configs':g['all_three_regimes_exact'],
               'independent_calibrated_models_proven':False,
               'real_human_do_effect_proven':False},
        'observational_fit':{'eligible_scenes':f['eligible'],'compared_models':comparisons,
                             'important':'CV superiority applies to observed single-future fit and cannot falsify behavioral intervention sensitivity'},
        'prioritized_8s_simulator':{'attempted':d['attempted'],'eligible':d['eligible'],
            'per_regime_positive_scenes':d['n_regime_matched'],
            'all_three':d['n_shared_three_regime'],
            'old_8s_profiles_same_initial_4s_as_4s_profiles':False,
            'reason':'Old 8s cosines were recomputed with steps=40; use fixed-prefix repair.'},
        'decision':'RUN interaction-stratified fit + true fixed-prefix 4s-to-8s + Waymax actual parity -> independent response -> actual planner preference; no sealed holdout yet',
        'scientific_claim_allowed':False}

def run(args):
    rd=lambda p:json.loads(Path(p).read_text()) if p else None
    result=evaluate(rd(args.gate),rd(args.fit),rd(args.deep),rd(args.physical),rd(args.cross_model),rd(args.planner),rd(args.h1))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--fit',required=True)
    p.add_argument('--deep',required=True);p.add_argument('--output',required=True)
    for name in ('physical','cross_model','planner','h1'):p.add_argument('--'+name)
    a=p.parse_args();print(json.dumps(run(a),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
