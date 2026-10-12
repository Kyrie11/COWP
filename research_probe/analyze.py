"""Conservative comparison of independent development recourse-probe sweeps.

Does NOT convert proxy rates into claims about physical road safety.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

LABELS={
    'pairwise_joint_discordance':'Individual-safe but sample-bank joint-incompatible',
    'joint_but_local_conflict':'Joint exists but a specified independent local policy conflicts',
    'temporal_collapse_proxy':'Current joint feasible, local responses compatible, successor bank empty',
    'avoidable_collapse_proxy':'Successor collapse with a task-comparable sampled alternative',
}

def wilson(k,n,z=1.96):
    if n<=0:return [None,None]
    p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=(z/den)*math.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))
    return [max(0,mid-half),min(1,mid+half)]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',required=True,nargs='+',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    runs=[json.loads(f.read_text()) for f in args.inputs]
    if len(runs)<2:ap.error('Use at least two independently specified response banks')
    out={'status':'INCONCLUSIVE_SCREENING','scientific_claim_allowed':False,
         'limitations':'Fixed historically selected candidate bank; simplified dynamics; no independent reactive calibration or strong baseline reproduction',
         'protocols':[], 'rates':{}, 'sensitivity':{}}
    by_run=[]
    for p,x in zip(args.inputs,runs):
        rows={str(y['scenario_id']):y for y in x['scenarios'] if y.get('eligible')}
        by_run.append(rows)
        out['protocols'].append({'file':str(p),'scenes_scanned':x['summary']['scenes_scanned'],
                                  'scenes_eligible':len(rows),'response_accels':x['summary'].get('response_accels'),
                                  'backup_accels':x['summary'].get('backup_accels'),
                                  'errors':x['summary']['errors']})
    common=set.intersection(*(set(r) for r in by_run))
    out['common_scenario_ids']=len(common)
    out['sensitivity']['reliable_to_compare']=bool(common) and all(r['summary'].get('errors')==0 for r in runs)
    for key,label in LABELS.items():
        info=[]
        for r in by_run:
            k=sum(r[s][key]>0 for s in common)
            info.append({'positive_scenes':k,'total_scenes':len(common),'fraction':k/max(len(common),1),
                         'wilson_interval_95':wilson(k,len(common))})
        stable=sum(all(r[s][key]>0 for r in by_run) for s in common)
        out['rates'][key]={'description':label,'by_run':info,'consistent_positive_scenes':stable,
                           'status':'sampled-model-only; NOT real failure ground truth'}
    out['decision_guidance']={
        'if_zero_in_all':'No evidence under this simplified bank; widen candidate/response model or pivot before adding an algorithm.',
        'if_positive_only_in_one':'Model-dependent artefact risk; DO NOT promote the research claim.',
        'if_consistently_positive':'Select cases for independent geometric verifier, Waymax real executed-action reconstruction, and calibrated reactive agents; no GO yet.',
        'strong_prior_art':'Studt & Schildbach (2026) decentralized contingency MPC and R3R (ACC 2026) must be faithfully compared before novelty claim.',
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'status':out['status'],'common_scenes':len(common),
                      'consistent_positive_scenes':{k:v['consistent_positive_scenes'] for k,v in out['rates'].items()}},indent=2))

if __name__=='__main__':main()
