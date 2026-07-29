from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

FIELDS = {
    'candidate_positive_auc': 'candidate_positive_auc',
    'group_top1_correlation': 'unconstrained_group_top1_correlation',
}

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); root=Path(args.root); rows=[]
    for sd in sorted(root.glob('seed_*')):
        seed=sd.name.removeprefix('seed_')
        for variant in ('balanced','precision'):
            for bucket in ('near','contact'):
                p=sd/'candidates'/variant/'calibration'/f'direct_value_risk_{bucket}_v48.json'
                if not p.exists():
                    rows.append({'seed':seed,'variant':variant,'bucket':bucket,'missing':True}); continue
                d=json.loads(p.read_text()); verify=d.get('verify') or {}
                rows.append({
                    'seed':int(seed), 'variant':variant, 'bucket':bucket,
                    'valid':bool(d.get('valid_for_deployment',False)),
                    'candidate_positive_auc':d.get('candidate_positive_auc'),
                    'group_top1_correlation':d.get('unconstrained_group_top1_correlation'),
                    'verify_selected':verify.get('selected'),
                    'verify_precision':verify.get('precision_selected'),
                    'verify_harmful_rate':verify.get('harmful_rate_selected'),
                    'verify_positive_recall':verify.get('positive_recall'),
                })
    aggregate=[]
    for variant in ('balanced','precision'):
        for bucket in ('near','contact'):
            subset=[r for r in rows if r.get('variant')==variant and r.get('bucket')==bucket and not r.get('missing')]
            item={'variant':variant,'bucket':bucket,'num_seeds':len(subset),'valid_seed_count':sum(bool(r.get('valid')) for r in subset)}
            for field in ('candidate_positive_auc','group_top1_correlation','verify_selected','verify_precision','verify_harmful_rate','verify_positive_recall'):
                vals=[float(r[field]) for r in subset if r.get(field) is not None]
                if vals:
                    item[field+'_mean']=statistics.fmean(vals); item[field+'_min']=min(vals); item[field+'_max']=max(vals)
                    item[field+'_std']=statistics.pstdev(vals) if len(vals)>1 else 0.0
            aggregate.append(item)
    payload={'root':str(root),'rows':rows,'aggregate':aggregate}
    Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
