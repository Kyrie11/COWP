#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def load(p: Path) -> dict[str, Any]:
    with p.open() as f: return json.load(f)

def fmt(v: Any) -> str:
    return "None" if v is None else f"{float(v):.6f}"

def check(name: str, v: Any, op: str, t: float, eps: float = 1e-9) -> bool:
    ok = v is not None and ((float(v) <= t + eps) if op == "<=" else (float(v) + eps >= t))
    print(f"{name:38s} {fmt(v):>10s} {op} {t:.6f}  {'PASS' if ok else 'FAIL'}")
    return bool(ok)

def main() -> int:
    ap=argparse.ArgumentParser(description='Check buffered OC-RAP v39 regime targets.')
    ap.add_argument('run_dir',type=Path)
    ap.add_argument('--near-miss-max',type=float,default=0.0334)
    ap.add_argument('--contact-miss-max',type=float,default=0.0334)
    args=ap.parse_args(); r=args.run_dir
    paths={
      'safe':r/'closed_loop_safe_fast_v39.json',
      'near':r/'audit_near_contact_selected_topk_v39_v39.json',
      'contact':r/'audit_contact_selected_topk_v39_v39.json',
      'offline':r/'eval_contact_v39_v39.json',
    }
    miss=[str(p) for p in paths.values() if not p.exists()]
    if miss: print('missing files:',*miss,sep='\n  '); return 2
    safe,near,contact,offline=(load(paths[k]) for k in ('safe','near','contact','offline'))
    ok=True
    print('===== safe: nominal preservation =====')
    ok &= check('safe intervention',safe.get('intervention_rate'),'<=',0.0)
    ok &= check('safe bounded NUP',safe.get('closed_loop_bounded_NUP'),'>=',0.999)
    print('===== near-contact: active but low-disturbance margin recovery =====')
    ok &= check('near paper-PCD miss',near.get('closed_loop_audit_paper_pcd_selector_miss_rate'),'<=',args.near_miss_max)
    ok &= check('near PCD',near.get('closed_loop_post_contact_deployability'),'>=',0.558)
    ok &= check('near FRA',near.get('closed_loop_FRA_exec'),'<=',0.10)
    ok &= check('near DRS',near.get('closed_loop_DRS'),'>=',0.90)
    ok &= check('near bounded NUP',near.get('closed_loop_bounded_NUP'),'>=',0.995)
    ok &= check('near intervention',near.get('intervention_rate'),'<=',0.02)
    print('near macro counts:',near.get('macro_counts'))
    print('near physical metrics:',{k:v for k,v in (near.get('waymax_metrics') or {}).items() if any(x in k for x in ('clearance','ttc','contact_exposure','stable_stop','secondary_overlap'))})
    print('===== contact: buffered recovery, no repeated-brake exploit =====')
    ok &= check('contact paper-PCD miss',contact.get('closed_loop_audit_paper_pcd_selector_miss_rate'),'<=',args.contact_miss_max)
    ok &= check('contact PCD',contact.get('closed_loop_post_contact_deployability'),'>=',0.50)
    ok &= check('contact FRA',contact.get('closed_loop_FRA_exec'),'<=',0.185)
    ok &= check('contact DRS',contact.get('closed_loop_DRS'),'>=',0.815)
    ok &= check('contact bounded NUP',contact.get('closed_loop_bounded_NUP'),'>=',0.985)
    ok &= check('contact intervention',contact.get('intervention_rate'),'<=',0.04)
    ok &= check('contact episode rate',contact.get('intervention_episode_rate'),'<=',0.02)
    ok &= check('contact max intervention run',contact.get('max_intervention_run_length'),'<=',2.0)
    print('contact macro counts:',contact.get('macro_counts'))
    print('contact reason counts:',contact.get('selection_reason_counts'))
    print('contact physical metrics:',{k:v for k,v in (contact.get('waymax_metrics') or {}).items() if any(x in k for x in ('clearance','ttc','contact_exposure','stable_stop','secondary_overlap'))})
    print('===== offline contact sanity =====')
    o=offline.get('methods',{}).get('ocrap',{})
    ok &= check('offline contact PCD',o.get('post_contact_deployability'),'>=',0.56)
    ok &= check('offline contact FRA',o.get('FRA_exec'),'<=',0.09)
    ok &= check('offline contact DRS',o.get('DRS'),'>=',0.91)
    ok &= check('offline contact NUP',o.get('bounded_NUP'),'>=',0.98)
    ok &= check('offline contact intervention',o.get('intervention_rate'),'<=',0.03)
    return 0 if ok else 1

if __name__=='__main__': raise SystemExit(main())
