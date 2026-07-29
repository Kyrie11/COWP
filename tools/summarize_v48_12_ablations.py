#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    tasks=args.root/'tasks'
    groups=sorted({re.sub(r'_(balanced|precision)$','',p.name) for p in tasks.iterdir() if p.is_dir()}) if tasks.is_dir() else []
    rows=[]
    for group in groups:
        for variant in ('balanced','precision'):
            task=tasks/f'{group}_{variant}'; base=task/'candidates'/variant
            if not task.is_dir(): continue
            row={'group':group,'variant':variant,'task_root':str(task),'complete':(task/'TASK_COMPLETE.json').is_file()}
            row['controller_exit_code']=(task/'controller.exit_code').read_text().strip() if (task/'controller.exit_code').is_file() else None
            for regime in ('near','contact'):
                d=load(base/'calibration'/f'direct_value_risk_{regime}_v48.json') or {}
                row[regime]={k:d.get(k) for k in (
                    'valid_for_deployment','candidate_positive_auc','candidate_risk_harm_auc',
                    'candidate_rank_teacher_correlation','unconstrained_group_top1_correlation',
                    'positive_group_top1_accuracy','positive_group_strict_top1_accuracy',
                    'positive_group_top1_regret_mean','top1_correctness_rank_margin_auc',
                    'strict_top1_correctness_rank_margin_auc','policy_top1_positive_auc','policy_top1_harm_auc','policy_top1_gain_mae','risk_source','warnings')}
                row[regime]['fit']=d.get('fit') or {}
                row[regime]['verify']=d.get('verify') or {}
            for stage_name in ('set_tournament','conditional_preference','preference'):
                pref=base/'stages'/stage_name/'preference_audit'
                if pref.is_dir():
                    row['stage_p_audit']={}
                    for regime in ('near','contact'):
                        pd=load(pref/f'preference_{regime}.json') or {}
                        row['stage_p_audit'][regime]={k:pd.get(k) for k in (
                            'candidate_rank_teacher_correlation','unconstrained_group_top1_correlation',
                            'positive_group_top1_accuracy','positive_group_top1_regret_mean')}
                    break
            rows.append(row)
    doc={'version':'v48.12','groups':groups,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'groups':groups,'rows':len(rows)},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
