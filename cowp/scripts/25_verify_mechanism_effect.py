from __future__ import annotations
import argparse, json, math
from pathlib import Path

SELECTION_KEYS=("EP","FallbackRate","SelectedFalseSafeRate","LearnedAcceptedCandidateRate","LearnedAcceptNCFRecall")

def main():
    ap=argparse.ArgumentParser(description='Fail fast when witness threshold is disconnected from selection.')
    ap.add_argument('--input',required=True); ap.add_argument('--method',default='cowp'); ap.add_argument('--output',required=True)
    ap.add_argument('--min-unique-selection-points',type=int,default=2)
    ap.add_argument('--min-ncf-recall',type=float,default=0.20)
    ap.add_argument('--min-witness-auprc',type=float,default=0.50)
    ap.add_argument('--min-accepted-rate',type=float,default=0.08)
    ap.add_argument('--max-fallback',type=float,default=0.30)
    ap.add_argument('--min-false-safe-improvement',type=float,default=0.03)
    args=ap.parse_args(); d=json.load(open(args.input))
    rows=d.get('witness_threshold_sweep',{}).get(args.method,[])
    signatures=[]
    for row in rows:
        metrics=row.get('metrics',row)
        signatures.append(tuple(round(float(metrics.get(k,float('nan'))),8) for k in SELECTION_KEYS))
    unique=len(set(signatures)) if signatures else 0
    main=d.get(args.method,{})
    recall=float(main.get('LearnedAcceptNCFRecall',0.0))
    auprc=float(main.get('WitnessQuality/AUPRC',0.0) or 0.0)
    accepted=float(main.get('LearnedAcceptedCandidateRate',0.0) or 0.0)
    fallback=float(main.get('FallbackRate',1.0) or 1.0)
    fs=float(main.get('SelectedFalseSafeRate',1.0) or 1.0)
    conv=d.get('conventional_safety',{})
    conv_fs=float(conv.get('SelectedFalseSafeRate',1.0) or 1.0)
    fs_gain=conv_fs-fs
    report={
        'threshold_points':len(rows),'unique_selection_points':unique,
        'threshold_connected_to_selection': unique>=args.min_unique_selection_points,
        'learned_accept_ncf_recall':recall,'ncf_recall_pass':recall>=args.min_ncf_recall,
        'witness_auprc':auprc,'witness_auprc_pass':auprc>=args.min_witness_auprc,
        'learned_accepted_candidate_rate':accepted,'accepted_rate_pass':accepted>=args.min_accepted_rate,
        'fallback_rate':fallback,'fallback_pass':fallback<=args.max_fallback,
        'selected_false_safe_rate':fs,'conventional_selected_false_safe_rate':conv_fs,
        'selected_false_safe_improvement':fs_gain,
        'false_safe_improvement_pass':fs_gain>=args.min_false_safe_improvement,
    }
    report['pass']=bool(
        report['threshold_connected_to_selection'] and report['ncf_recall_pass']
        and report['witness_auprc_pass'] and report['accepted_rate_pass']
        and report['fallback_pass'] and report['false_safe_improvement_pass']
    )
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['pass']: raise SystemExit(2)
if __name__=='__main__': main()
