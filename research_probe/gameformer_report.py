"""Sanity-check author's GameFormer open-loop CSV on matched cohort IDs.

This is not evidence for a closed-loop or action-intervention safety baseline.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path


def score(cohort,gameformer_csv):
    manifest=json.loads((Path(cohort)/'cohort_manifest.json').read_text())
    wanted={r['scenario_id'] for r in manifest['entries']}
    rows=list(csv.DictReader(Path(gameformer_csv).open(newline='')))
    matched=[];other=[]
    for r in rows:
        key=r.get('scenarios',''); sid=key.rsplit('_',1)[0]
        (matched if sid in wanted else other).append(r)
    scores={}
    for key in ('collision','miss','Prediction_ADE','Prediction_FDE','Human_L2_1s','Human_L2_3s','Human_L2_5s'):
        vals=[]
        for r in matched:
            try:vals.append(float(r[key]))
            except (ValueError,TypeError,KeyError):pass
        scores[key]={'n_numeric':len(vals),'mean':statistics.mean(vals) if vals else None}
    return {'scientific_claim_allowed':False,'baseline':'GameFormer open-loop (author code, matched cohort IDs)',
        'matched_rows':len(matched),'unmatched_rows':len(other),
        'matched_unique_scenarios':len({r['scenarios'].rsplit('_',1)[0] for r in matched}),
        'requested_unique_scenarios':len(wanted),'statistics':scores,
        'limitations':['Not reactive closed-loop','No independent behavior intervention','No H3 successor safety certification',
                       'Author evaluates several timesteps per scenario','Must document model checkpoint provenance and training split']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--gameformer-csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=score(a.cohort,a.gameformer_csv)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__': main()
