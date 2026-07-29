#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def read(p):
    try:return json.loads(p.read_text())
    except:return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=[]
    for exp in sorted(x for x in a.root.iterdir() if x.is_dir()):
      for variant in ('balanced','precision'):
        base=exp/'candidates'/variant; tr=read(base/'model_v48_trac_sr'/'train_summary.json') or {}
        row={'experiment':exp.name,'variant':variant,'training_complete':(base/'TRAINING_COMPLETE.json').is_file(),'best_epoch':tr.get('best_epoch'),'epochs_completed':tr.get('epochs_completed'),'best_metric':tr.get('best_metric'),'best_val_loss':tr.get('best_val_loss')}
        for b in ('near','contact'):
          d=read(base/'calibration'/f'direct_value_risk_{b}_v48.json') or {}
          row[f'{b}_auc']=d.get('candidate_positive_auc'); row[f'{b}_rank_corr']=d.get('candidate_rank_teacher_correlation'); row[f'{b}_top1_corr']=d.get('unconstrained_group_top1_correlation'); row[f'{b}_top1_acc']=d.get('positive_group_top1_accuracy'); row[f'{b}_top1_regret']=d.get('positive_group_top1_regret_mean'); row[f'{b}_valid']=d.get('valid_for_deployment'); row[f'{b}_selected']=(d.get('verify') or {}).get('num_selected')
        rows.append(row)
    out={'root':str(a.root),'rows':rows}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
