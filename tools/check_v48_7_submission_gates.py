#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--near-auc',type=float,default=0.78)
    ap.add_argument('--contact-auc',type=float,default=0.82)
    ap.add_argument('--top1-mean',type=float,default=0.10)
    ap.add_argument('--rank-margin-auc',type=float,default=0.65)
    ap.add_argument('--precision-lcb',type=float,default=0.60)
    ap.add_argument('--positive-recall',type=float,default=0.35)
    ap.add_argument('--harm-ucb',type=float,default=0.10)
    ap.add_argument('--macro-share',type=float,default=0.70)
    args=ap.parse_args()
    d=json.loads(args.summary.read_text())
    rows=[r for r in d.get('rows',[]) if not r.get('missing')]
    grouped=defaultdict(list)
    for r in rows: grouped[(r['variant'],r['bucket'])].append(r)
    report={'source':str(args.summary),'groups':{},'all_passed':True}
    for (variant,bucket),xs in sorted(grouped.items()):
        top=[float(x.get('top1_correlation') or 0.0) for x in xs]
        auc=[float(x.get('candidate_positive_auc') or 0.0) for x in xs]
        rank_auc=[float(x.get('rank_margin_correctness_auc') or 0.0) for x in xs]
        selected=[int(x.get('verify_selected') or 0) for x in xs]
        lcb=[x.get('verify_precision_lcb90') for x in xs]
        recall=[float(x.get('verify_positive_recall') or 0.0) for x in xs]
        harm=[float(x.get('verify_harmful_selected_ucb90') or 1.0) for x in xs]
        macro=[float(x.get('verify_macro_share') or 0.0) for x in xs]
        auc_thr=args.near_auc if bucket=='near' else args.contact_auc
        checks={
          'candidate_auc_mean': sum(auc)/len(auc) >= auc_thr,
          'top1_all_positive': all(v>0 for v in top),
          'top1_mean': sum(top)/len(top) >= args.top1_mean,
          'rank_margin_auc_mean': sum(rank_auc)/len(rank_auc) >= args.rank_margin_auc,
          'nonzero_verify_each_seed': all(v>0 for v in selected),
          'precision_lcb_each_seed': all(v is not None and float(v)>=args.precision_lcb for v in lcb),
          'positive_recall_each_seed': all(v>=args.positive_recall for v in recall),
          'harm_ucb_each_seed': all(v<=args.harm_ucb for v in harm),
          'macro_share_each_seed': all(v<=args.macro_share for v in macro),
        }
        passed=all(checks.values())
        report['groups'][f'{variant}:{bucket}']={
          'passed':passed,'checks':checks,
          'values':{'candidate_auc':auc,'top1_correlation':top,'rank_margin_auc':rank_auc,
                    'verify_selected':selected,'precision_lcb90':lcb,'positive_recall':recall,
                    'harmful_selected_ucb90':harm,'macro_share':macro},
        }
        report['all_passed'] &= passed
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['all_passed'] else 20

if __name__=='__main__':
    raise SystemExit(main())
