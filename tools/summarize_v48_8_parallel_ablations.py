#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

GROUPS = [
    "A_engineering_fixed_v487",
    "B_conflict_free_preference",
    "C_robust_conformal_certificate",
    "D_full_scope",
]

def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); rows=[]
    for group in GROUPS:
        for variant in ('balanced','precision'):
            task=args.root/'tasks'/f'{group}_{variant}'
            base=task/'candidates'/variant
            row={'group':group,'variant':variant,'task_root':str(task)}
            row['controller_exit_code']=(task/'controller.exit_code').read_text().strip() if (task/'controller.exit_code').is_file() else None
            for regime in ('near','contact'):
                d=load(base/'calibration'/f'direct_value_risk_{regime}_v48.json') or {}
                row[regime]={k:d.get(k) for k in (
                    'valid_for_deployment','candidate_positive_auc','candidate_risk_harm_auc',
                    'candidate_rank_teacher_correlation','unconstrained_group_top1_correlation',
                    'positive_group_top1_accuracy','positive_group_strict_top1_accuracy',
                    'positive_group_top1_regret_mean','top1_correctness_rank_margin_auc',
                    'strict_top1_correctness_rank_margin_auc','risk_source','warnings')}
                row[regime]['fit']=(d.get('fit') or {})
                row[regime]['verify']=(d.get('verify') or {})
                row[regime]['conformal']=d.get('conformal')
            pref=base/'stages'/'preference'/'preference_audit'
            row['stage_p_audit']={}
            for regime in ('near','contact'):
                pd=load(pref/f'preference_{regime}.json') or {}
                row['stage_p_audit'][regime]={k:pd.get(k) for k in (
                    'candidate_rank_teacher_correlation','unconstrained_group_top1_correlation',
                    'positive_group_top1_accuracy','positive_group_top1_regret_mean')}
            rows.append(row)
    doc={'version':'v48.8','groups':GROUPS,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(doc,ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
