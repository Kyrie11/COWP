from __future__ import annotations
import argparse, json
from pathlib import Path

SELECTION_KEYS=("EP","FallbackRate","SelectedFalseSafeRate","SemanticFeasibleCandidateRate","SemanticAcceptNCFRecall")

def _metric(row: dict, key: str) -> float:
    if key == "SemanticFeasibleCandidateRate":
        return float(row.get(key, row.get("LearnedAcceptedCandidateRate", float("nan"))))
    if key == "SemanticAcceptNCFRecall":
        return float(row.get(key, row.get("LearnedAcceptNCFRecall", 0.0)))
    return float(row.get(key, float("nan")))

def main():
    ap=argparse.ArgumentParser(description="Fail fast when the semantic witness gate is disconnected from feasibility.")
    ap.add_argument("--input",required=True); ap.add_argument("--method",default="cowp"); ap.add_argument("--output",required=True)
    ap.add_argument("--min-unique-selection-points",type=int,default=2)
    ap.add_argument("--min-ncf-recall",type=float,default=0.70)
    ap.add_argument("--max-semantic-false-safe-accept",type=float,default=0.35)
    args=ap.parse_args(); d=json.load(open(args.input))
    rows=d.get("witness_threshold_sweep",{}).get(args.method,[])
    signatures=[]
    for row in rows:
        metrics=row.get("metrics",row)
        signatures.append(tuple(round(_metric(metrics,k),8) for k in SELECTION_KEYS))
    unique=len(set(signatures)) if signatures else 0
    main=d.get(args.method,{})
    recall=_metric(main,"SemanticAcceptNCFRecall")
    fs_accept=float(main.get("SemanticAcceptFalseSafeRate", main.get("LearnedAcceptFalseSafeRate",1.0)))
    report={
        "threshold_points":len(rows),"unique_semantic_points":unique,
        "threshold_connected_to_semantic_gate": unique>=args.min_unique_selection_points,
        "semantic_accept_ncf_recall":recall,"ncf_recall_pass":recall>=args.min_ncf_recall,
        "semantic_accept_false_safe_rate":fs_accept,
        "false_safe_accept_pass":fs_accept<=args.max_semantic_false_safe_accept,
        "witness_auprc":main.get("WitnessQuality/AUPRC"),
        "selected_false_safe_rate":main.get("SelectedFalseSafeRate"),
        "frontier_ncf_recall":main.get("LearnedAcceptNCFRecall"),
    }
    report["pass"]=bool(report["threshold_connected_to_semantic_gate"] and report["ncf_recall_pass"] and report["false_safe_accept_pass"])
    Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["pass"]: raise SystemExit(2)
if __name__=="__main__": main()
