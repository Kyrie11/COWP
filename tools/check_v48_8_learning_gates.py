#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(p: Path) -> dict:
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--preference-top1-min',type=float,default=0.10)
    ap.add_argument('--preference-accuracy-min',type=float,default=0.60)
    ap.add_argument('--near-auc-min',type=float,default=0.75)
    ap.add_argument('--contact-auc-min',type=float,default=0.80)
    args=ap.parse_args(); variants={}
    for variant in ('balanced','precision'):
        base=args.run/'candidates'/variant; row={'preference':{},'gain':{},'certificate':{}}
        for regime in ('near','contact'):
            final=load(base/'calibration'/f'direct_value_risk_{regime}_v48.json')
            pref=load(base/'stages'/'preference'/'preference_audit'/f'preference_{regime}.json')
            top1=pref.get('unconstrained_group_top1_correlation')
            acc=pref.get('positive_group_top1_accuracy')
            row['preference'][regime]={
                'top1_correlation':top1,'acceptable_top1_accuracy':acc,
                'passed':bool(top1 is not None and top1>=args.preference_top1_min and acc is not None and acc>=args.preference_accuracy_min),
            }
            auc=final.get('candidate_positive_auc'); auc_min=args.near_auc_min if regime=='near' else args.contact_auc_min
            row['gain'][regime]={
                'candidate_positive_auc':auc,'candidate_harm_auc':final.get('candidate_risk_harm_auc'),
                'mean_regret':final.get('positive_group_top1_regret_mean'),
                'passed':bool(auc is not None and auc>=auc_min),
            }
            verify=final.get('verify') or {}
            row['certificate'][regime]={
                'valid_for_deployment':bool(final.get('valid_for_deployment',False)),
                'verify_selected':verify.get('num_selected'),
                'precision_lcb90':verify.get('precision_wilson_lcb90'),
                'harmful_selected_ucb90':verify.get('harmful_selected_ucb90'),
                'positive_recall':verify.get('positive_recall'),
                'passed':bool(final.get('valid_for_deployment',False)),
                'near_miss_verify_frontier':final.get('near_miss_verify_frontier',[])[:5],
            }
        row['stage_p_passed']=all(x['passed'] for x in row['preference'].values())
        row['stage_c_discrimination_passed']=all(x['passed'] for x in row['gain'].values())
        row['natural_gate_passed']=all(x['passed'] for x in row['certificate'].values())
        variants[variant]=row
    doc={
        'version':'v48.8','run':str(args.run),'variants':variants,
        'decision':{
            'continue_to_multiseed':any(v['stage_p_passed'] and v['stage_c_discrimination_passed'] for v in variants.values()),
            'continue_to_stress_closed_loop':any(v['natural_gate_passed'] for v in variants.values()),
        },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(doc,ensure_ascii=False,indent=2))
    return 0 if doc['decision']['continue_to_multiseed'] else 10
if __name__=='__main__': raise SystemExit(main())
