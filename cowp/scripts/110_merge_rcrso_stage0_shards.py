from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch


def _sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser(description='Exact merge/finalize for scenario-disjoint V16.8.45R3 Stage-0 shards.')
    ap.add_argument('--inputs',nargs='+',required=True)
    ap.add_argument('--checkpoint',required=True,help='The same RCRSO unselected checkpoint used by every shard.')
    ap.add_argument('--output',required=True)
    ap.add_argument('--selected-checkpoint',required=True)
    ap.add_argument('--minimum-full-hypothesis-coverage-lift-pp',type=float,default=3.0)
    args=ap.parse_args()

    partials=[json.load(open(p,encoding='utf-8')) for p in args.inputs]
    if not partials:
        raise SystemExit('ERROR: no Stage-0 shard inputs')
    expected_n=int(partials[0].get('num_shards',len(partials)))
    shard_ids=[int(p.get('shard_index',-1)) for p in partials]
    if len(partials)!=expected_n or sorted(shard_ids)!=list(range(expected_n)):
        raise SystemExit(f'ERROR: incomplete Stage-0 shard set: expected {expected_n}, got indices={sorted(shard_ids)}')
    if any(int(p.get('num_shards',-1))!=expected_n for p in partials):
        raise SystemExit('ERROR: num_shards mismatch')
    if any(p.get('split')!=partials[0].get('split') for p in partials):
        raise SystemExit('ERROR: split mismatch')
    if any(p.get('k_values')!=partials[0].get('k_values') for p in partials):
        raise SystemExit('ERROR: K mismatch')
    if any(int(p.get('processed_examples',-1))!=int(p.get('assigned_examples',-2)) for p in partials):
        raise SystemExit('ERROR: at least one Stage-0 shard did not finish its assigned examples')

    hashes={str(p.get('checkpoint_sha256')) for p in partials}
    if len(hashes)!=1:
        raise SystemExit(f'ERROR: checkpoint hash mismatch across shards: {hashes}')
    ckpt_hash=_sha256_file(args.checkpoint)
    if hashes!={ckpt_hash}:
        raise SystemExit(f'ERROR: merge checkpoint hash {ckpt_hash} does not match shard hash {next(iter(hashes))}')
    sidecar_hashes={p.get('sidecar_summary_sha256') for p in partials}
    if len(sidecar_hashes)!=1:
        raise SystemExit(f'ERROR: sidecar summary hash mismatch across shards: {sidecar_hashes}')

    seen=set()
    for p in partials:
        hs=set(int(x) for x in p.get('scenario_hashes',[]))
        overlap=seen & hs
        if overlap:
            raise SystemExit(f'ERROR: Stage-0 scenario overlap across shards, sample={sorted(overlap)[:8]}')
        seen |= hs

    mod=importlib.import_module('cowp.scripts.106_eval_rcrso_support')
    summary=mod._partial_to_summary(partials,float(args.minimum_full_hypothesis_coverage_lift_pp))
    summary['runtime_observability']['scenario_count']=len(seen)
    summary['runtime_observability']['shard_indices']=sorted(shard_ids)
    summary['provenance']['partial_files']=[str(Path(p).resolve()) for p in args.inputs]
    summary['provenance']['merge_checkpoint_sha256']=ckpt_hash
    summary['contract']['merge_validated_disjoint_scenarios']=True
    summary['contract']['all_assigned_examples_completed']=True

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')

    ckpt=torch.load(args.checkpoint,map_location='cpu')
    selected_ckpt=dict(ckpt)
    selected_ckpt['selected_k']=int(summary['selected_k'])
    selected_ckpt['stage0_support_audit']=summary
    selected_ckpt['contract']={
        **dict(selected_ckpt.get('contract',{})),
        'selected_k_status':'frozen_validation_selected',
        'stage0_support_gate_pass':bool(summary['stage0_support_gate']['pass']),
        'stage0_parallel_merge_semantics':'scenario-disjoint exact raw-count merge',
    }
    sel=Path(args.selected_checkpoint); sel.parent.mkdir(parents=True,exist_ok=True)
    torch.save(selected_ckpt,sel)
    print(json.dumps(summary,indent=2,sort_keys=True))
    if not summary['stage0_support_gate']['pass']:
        raise SystemExit(4)


if __name__=='__main__':
    main()
