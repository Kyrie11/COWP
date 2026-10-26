"""Strict JSON-only evidence report; explicitly refuses human-causal evidence claims."""
from __future__ import annotations
import argparse, json
from pathlib import Path

REGIMES=('cautious','assertive','heterogeneous')


def _compact(v):
    valid=[r for r in v.get('rows',[]) if r.get('eligible')]
    return {'checked':len(v.get('rows',[])),'eligible':len(valid),'errors':len(v.get('errors',[]))}


def report(matched,ghost,identity):
    m=json.loads(Path(matched).read_text());g=json.loads(Path(ghost).read_text())
    i=json.loads(Path(identity).read_text()) if identity else None
    mm=_compact(m);gg=_compact(g)
    me=[r for r in m['rows'] if r.get('eligible')]
    ge=[r for r in g['rows'] if r.get('eligible')]
    cases={r:{'strict_scenes':sum(bool(x['regimes'][r].get('strict_matches')) for x in me),
              'unmatched_scenes':sum(bool(x['regimes'][r].get('unmatched_pairs')) for x in me),
              'burden_proxy_scenes':sum(bool(x['regimes'][r].get('matched_burden_proxy_cases')) for x in me),
              'near_pair_proxy_scenes':(sum(bool(x['regimes'][r].get('near_pair_proxy_cases')) for x in me)
                 if me and all('near_pair_proxy_cases' in x['regimes'][r] for x in me) else None)}
           for r in REGIMES}
    ghosts={r:{'strict_scenes':sum(bool(x['regimes'][r].get('strict_proxy_cases')) for x in ge),
               'noninferior_scenes':sum(bool(x['regimes'][r].get('progress_noninferior_proxy_cases')) for x in ge)}
            for r in REGIMES}
    return {'scientific_claim_supported':False,
        'reason':['IDM-family parametric regimes are not independent behavioral families',
                  'A real WOMD observed future does not expose do(ego_action) potential outcomes',
                  'Proxy dynamics omit turning/map/lights and human driver policy',
                  'Strict H2 requires stable same-pair progress-matched intervenability and independent validation'],
        'matched':mm,'matched_regimes':cases,
        'zero_event_one_sided_95_upper_bound_selected_matched':(1-0.05**(1/max(len(me),1))) if me else None,
        'matched_shared_three_regime':sum(bool(x.get('all_three_regimes_shared_matched_pair_keys')) for x in me),
        'matched_shared_burden_proxy':sum(bool(x.get('all_three_regimes_shared_burden_proxy_keys')) for x in me),
        'paired_ghost':gg,'paired_ghost_regimes':ghosts,
        'paired_ghost_shared_three_regime':sum(bool(x.get('three_regime_shared_action_pairs')) for x in ge),
        'raw_identity':None if i is None else {k:i.get(k) for k in ('pass','attempted','verified','raw_records_verified')},
        'hypothesis_gates':{
            'joint_feasibility':'NOT TESTED by these two JSON files: finite H1 CSP is distinct',
            'decentralized_realizability':'NOT PROVEN: one longitudinal IDM structural family only',
            'ego_conditional_preventability':'NOT PROVEN: strict shared pair/progress-match is zero',
            'H1':'NOT PROVEN','H2':'NOT PROVEN','H3':'NOT PROVEN','H4':'NOT RUN'}}


def main():
    p=argparse.ArgumentParser()
    for arg in ('matched','ghost'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--raw-identity',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=report(a.matched,a.ghost,a.raw_identity)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in r.items() if k not in ('reason','hypothesis_gates')},indent=2))
if __name__=='__main__':main()
