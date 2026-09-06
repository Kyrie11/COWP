from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

TOL=1e-12

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def close(a,b,tol=TOL):
    return math.isclose(float(a),float(b),rel_tol=0.0,abs_tol=tol)

def main():
    ap=argparse.ArgumentParser(description='Independent V45R3 Stage-0 raw-count reliability audit')
    ap.add_argument('--result-dir',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--selected-checkpoint',default='')
    a=ap.parse_args()
    root=Path(a.result_dir)
    audit_p=root/'stage0_val_support_audit.json'
    partials=[root/'stage0_partials'/f'stage0_val_s{i}.json' for i in range(2)]
    checks=[]
    def chk(name,ok,detail=None):
        checks.append({'name':name,'pass':bool(ok),'detail':detail})
        return bool(ok)
    chk('audit_exists',audit_p.is_file(),str(audit_p))
    for i,p in enumerate(partials): chk(f'partial_{i}_exists',p.is_file(),str(p))
    if not all(c['pass'] for c in checks):
        out={'overall_reliability':'FAIL','algorithm_attribution':'NOT_ALLOWED','checks':checks}
        Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); raise SystemExit(4)
    audit=json.loads(audit_p.read_text())
    ps=[json.loads(p.read_text()) for p in partials]
    chk('shard_indices',[x['shard_index'] for x in ps]==[0,1],[x['shard_index'] for x in ps])
    chk('num_shards',[x['num_shards'] for x in ps]==[2,2])
    chk('all_examples_completed',all(int(x['processed_examples'])==int(x['assigned_examples']) for x in ps),[(x['processed_examples'],x['assigned_examples']) for x in ps])
    s0=set(map(int,ps[0]['scenario_hashes'])); s1=set(map(int,ps[1]['scenario_hashes']))
    chk('scenario_disjoint',not (s0&s1),{'s0':len(s0),'s1':len(s1),'overlap':len(s0&s1),'union':len(s0|s1)})
    ck={x['checkpoint_sha256'] for x in ps}; sh={x['sidecar_summary_sha256'] for x in ps}
    chk('checkpoint_hash_agreement',len(ck)==1,list(ck)); chk('sidecar_hash_agreement',len(sh)==1,list(sh))
    prov=audit.get('provenance',{})
    chk('audit_checkpoint_hash_matches',len(ck)==1 and prov.get('checkpoint_sha256') in ck,prov.get('checkpoint_sha256'))
    chk('audit_sidecar_hash_matches',len(sh)==1 and prov.get('sidecar_summary_sha256') in sh,prov.get('sidecar_summary_sha256'))
    # Merge raw integer counts independently.
    examples=sum(int(x['stats']['examples']) for x in ps); eligible=sum(int(x['stats']['eligible_examples']) for x in ps)
    groups=sum(int(x['stats']['hypothesis_groups']) for x in ps); oracle=sum(int(x['stats']['oracle_positive_roots']) for x in ps)
    verifier=sum(int(x['stats']['verifier_calls']) for x in ps)
    chk('examples_recompute',examples==int(audit['examples']),{'recomputed':examples,'audit':audit['examples']})
    chk('eligible_recompute',eligible==int(audit['eligible_examples']))
    chk('groups_recompute',groups==int(audit['hypothesis_groups']),{'recomputed':groups,'audit':audit['hypothesis_groups']})
    chk('oracle_recompute',oracle==int(audit['oracle_positive_roots']),{'recomputed':oracle,'audit':audit['oracle_positive_roots']})
    chk('verifier_calls_recompute',verifier==int(audit['verifier_calls']))
    merged={'baseline':{},'rcrso':{}}
    for b in ['fixed_bank','v44_analytic_extension']:
        merged['baseline'][b]={k:sum(int(x['stats']['baseline'][b][k]) for x in ps) for k in ['root_num','full_num','csp_num']}
    kvals=[2,4,8,16]
    for K in kvals:
        key=str(K)
        merged['rcrso'][K]={k:sum(int(x['stats']['rcrso'][key][k]) for x in ps) for k in ['root_num','full_num','csp_num','learned_root_nonempty_num','learned_verified_profile_count','burden_count']}
        merged['rcrso'][K]['burden_sum']=sum(float(x['stats']['rcrso'][key]['burden_sum']) for x in ps)
    metrics={}
    for b,c in merged['baseline'].items():
        metrics[b]={'VerifiedRootRecall':c['root_num']/oracle,'FullHypothesisRootCoverage':c['full_num']/groups,'ExactCSPCompletionRate':c['csp_num']/groups}
        for m,v in metrics[b].items(): chk(f'baseline_{b}_{m}',close(v,audit['baseline'][b][m]),{'recomputed':v,'audit':audit['baseline'][b][m]})
    curve=[]
    for K in kvals:
        c=merged['rcrso'][K]
        row={'K':K,'VerifiedRootRecall':c['root_num']/oracle,'FullHypothesisRootCoverage':c['full_num']/groups,'ExactCSPCompletionRate':c['csp_num']/groups,'learned_only_root_nonempty_rate':c['learned_root_nonempty_num']/examples,'learned_verified_profile_count':c['learned_verified_profile_count'],'learned_verified_response_burden_mean':c['burden_sum']/max(1,c['burden_count'])}
        curve.append(row)
        ref=next(r for r in audit['rcrso_curve'] if int(r['K'])==K)
        for m in ['VerifiedRootRecall','FullHypothesisRootCoverage','ExactCSPCompletionRate','learned_only_root_nonempty_rate','learned_verified_response_burden_mean']:
            chk(f'K{K}_{m}',close(row[m],ref[m]),{'recomputed':row[m],'audit':ref[m]})
        chk(f'K{K}_profile_count',row['learned_verified_profile_count']==int(ref['learned_verified_profile_count']))
    plateau=max(r['FullHypothesisRootCoverage'] for r in curve); target=0.95*plateau
    selected=next(r for r in curve if r['FullHypothesisRootCoverage']+TOL>=target)
    chk('selected_k_rule',int(audit['selected_k'])==int(selected['K']),{'recomputed':selected['K'],'audit':audit['selected_k'],'target':target})
    best=max(metrics['fixed_bank']['FullHypothesisRootCoverage'],metrics['v44_analytic_extension']['FullHypothesisRootCoverage'])
    lift=selected['FullHypothesisRootCoverage']-best
    min_lift=float(audit['minimum_required_lift_pp'])/100.0
    gate=(lift+TOL>=min_lift and selected['VerifiedRootRecall']>0.0)
    chk('coverage_lift',close(lift,audit['coverage_lift_over_best_frozen_baseline']),{'recomputed':lift,'audit':audit['coverage_lift_over_best_frozen_baseline']})
    chk('stage0_gate_recompute',gate is bool(audit['stage0_support_gate']['pass']),{'recomputed':gate,'audit':audit['stage0_support_gate']['pass']})
    ckpt_check={'provided':False}
    if a.selected_checkpoint:
        p=Path(a.selected_checkpoint); ckpt_check={'provided':True,'path':str(p),'exists':p.is_file()}
        if p.is_file():
            ckpt_check['sha256']=sha256(p); ckpt_check['matches']=ckpt_check['sha256'] in ck
            chk('selected_checkpoint_hash_matches',ckpt_check['matches'],ckpt_check)
        else: chk('selected_checkpoint_exists',False,ckpt_check)
    allpass=all(c['pass'] for c in checks)
    full=selected['FullHypothesisRootCoverage']; csp=selected['ExactCSPCompletionRate']
    out={
      'version':'V16.8.45R4 Stage-0 independent result audit',
      'overall_reliability':'PASS' if allpass else 'FAIL',
      'algorithm_attribution':'ALLOWED' if allpass else 'NOT_ALLOWED',
      'scientific_verdict': 'STAGE0_GO_FULL_POLICY_PENDING' if allpass and gate else ('STAGE0_STOP' if allpass else 'UNRESOLVED'),
      'raw_counts':{'examples':examples,'hypothesis_groups':groups,'oracle_positive_roots':oracle,'scenario_union':len(s0|s1),'scenario_overlap':len(s0&s1),'baseline':merged['baseline'],'rcrso':merged['rcrso']},
      'metrics':{'baseline':metrics,'rcrso_curve':curve,'selected_k':selected['K'],'coverage_lift_pp':100*lift,'full_to_csp_drop_groups':merged['rcrso'][selected['K']]['full_num']-merged['rcrso'][selected['K']]['csp_num'],'selected_full_groups':merged['rcrso'][selected['K']]['full_num'],'selected_csp_groups':merged['rcrso'][selected['K']]['csp_num'],'absolute_uncovered_group_rate':1-full,'query_budget_at_max_tested':selected['K']==max(kvals)},
      'checkpoint_boundary':ckpt_check,
      'checks':checks,
      'decision':{'stage0_go':bool(allpass and gate),'full_policy_go':False,'next_required_gate':'equivalence16 then progressive lost7 2+2+3','do_not_design_v46_yet':True}
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
    if not allpass: raise SystemExit(4)

if __name__=='__main__': main()
