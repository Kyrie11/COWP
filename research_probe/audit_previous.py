"""Recomputes screening-evidence limitations from saved JSON without raw WOMD access."""
import argparse,json,hashlib
from pathlib import Path


def audit(main,alternate):
    m={s['scenario_id']:s for s in main['scenarios'] if s.get('eligible')}
    a={s['scenario_id']:s for s in alternate['scenarios'] if s.get('eligible')}
    ids=sorted(m.keys() & a.keys())
    def stats(x):
        r=[x[i] for i in ids];allc=[c for s in r for c in s['candidates']]
        positive=[s for s in r if s['pairwise_joint_discordance']>0]
        return {'eligible':len(r),'candidates_total':len(allc),'pairwise_safe':sum(c['pairwise'] for c in allc),
                'joint_feasible':sum(c['joint'] for c in allc),
                'joint_feasible_independent_conflict':sum(c['joint'] and not c['local_compatible'] for c in allc),
                'successor_tested':sum(c['successor_viable'] is not None for c in allc),
                'successor_failed':sum(c['successor_viable'] is False for c in allc),
                'discordance_positive_scenes':len(positive),
                'discordance_positive_candidates':sum(s['pairwise_joint_discordance'] for s in positive),
                'discordance_with_joint_candidate_alternative_scenes':sum(any(c['joint'] for c in s['candidates']) for s in positive),
                'discordance_all_candidates_positive_scenes':sum(s['pairwise_joint_discordance']==len(s['candidates']) for s in positive),
                'independent_conflict_scenes':sum(s['joint_but_local_conflict']>0 for s in r),
                'temporal_collapse_scenes':sum(s['temporal_collapse_proxy']>0 for s in r),
                'avoidable_collapse_scenes':sum(s['avoidable_collapse_proxy']>0 for s in r)}
    both=[i for i in ids if m[i]['pairwise_joint_discordance'] and a[i]['pairwise_joint_discordance']]
    failures=[i for i in ids if m[i]['pairwise_joint_discordance'] != a[i]['pairwise_joint_discordance']]
    return {'status':'NO_SCIENTIFIC_GO','shared_scenarios':len(ids),'scenario_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'base':stats(m),'alternate':stats(a),'stable_discordance_ids':both,
            'discordance_count_changes':failures,
            'critical_caveats':['All 14 positive scenes lack an in-bank joint-feasible alternative',
              'Need initial-time overlap and oriented-rectangle audit; circle footprint may create candidate-invariant artifacts',
              'Successor test only runs on joint-feasible+locally-compatible candidates and uses permissive short longitudinal backups',
              'No Waymax actual execution, independent reactive validation, or Studt/R3R baselines',
              'The two response banks are both small constant longitudinal acceleration families'],
            'conditional_status':'Geometry audit P0; independent behavior validation P1; temporal research P2 only if new evidence.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True);p.add_argument('--alternative',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=audit(json.loads(a.base.read_text()),json.loads(a.alternative.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in d.items() if k!='stable_discordance_ids'},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
