from __future__ import annotations
import argparse, json, math
from pathlib import Path

SELECTION_KEYS=("EP","FallbackRate","SelectedFalseSafeRate","LearnedAcceptedCandidateRate","LearnedAcceptNCFRecall")

def main():
    ap=argparse.ArgumentParser(description='Fail fast when witness threshold is disconnected from selection.')
    ap.add_argument('--input',required=True); ap.add_argument('--method',default='cowp'); ap.add_argument('--output',required=True)
    ap.add_argument('--min-unique-selection-points',type=int,default=2)
    ap.add_argument('--min-ncf-recall',type=float,default=0.20)
    args=ap.parse_args(); d=json.load(open(args.input))
    rows=d.get('witness_threshold_sweep',{}).get(args.method,[])
    signatures=[]
    for row in rows:
        metrics=row.get('metrics',row)
        signatures.append(tuple(round(float(metrics.get(k,float('nan'))),8) for k in SELECTION_KEYS))
    unique=len(set(signatures)) if signatures else 0
    main=d.get(args.method,{})
    recall=float(main.get('LearnedAcceptNCFRecall',0.0))
    report={
        'threshold_points':len(rows),'unique_selection_points':unique,
        'threshold_connected_to_selection': unique>=args.min_unique_selection_points,
        'learned_accept_ncf_recall':recall,'ncf_recall_pass':recall>=args.min_ncf_recall,
        'witness_auprc':main.get('WitnessQuality/AUPRC'),
        'selected_false_safe_rate':main.get('SelectedFalseSafeRate'),
    }
    report['pass']=bool(report['threshold_connected_to_selection'] and report['ncf_recall_pass'])
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['pass']: raise SystemExit(2)
if __name__=='__main__': main()
