#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ['FRA_exec', 'DRS', 'bounded_NUP', 'ODG', 'artifact_selection_rate', 'post_contact_deployability', 'intervention_rate', 'selected_admitted_rate']
AUDIT_KEYS = ['closed_loop_bounded_NUP', 'closed_loop_FRA_exec', 'closed_loop_DRS', 'closed_loop_ODG', 'closed_loop_post_contact_deployability', 'closed_loop_audit_paper_pcd_selector_miss_rate', 'closed_loop_audit_paper_selected_PCD_regret']

def fmt(x):
    if isinstance(x, float):
        return f'{x:.6g}'
    return str(x)

def load(path: Path):
    with path.open() as f:
        return json.load(f)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--tag', default='v29')
    args = ap.parse_args()
    root = args.run_dir
    tag = args.tag
    print(f'# Regime summary: {root}')
    missing_audit = []
    for reg in ['safe', 'near_contact', 'contact']:
        p = root / f'eval_{reg}_{tag}_{tag}.json'
        if not p.exists():
            alt = sorted(root.glob(f'eval_{reg}_*_{tag}.json')) or sorted(root.glob(f'eval_{reg}_*_v27.json'))
            p = alt[0] if alt else p
        print(f'\n## {reg}')
        if not p.exists():
            print(f'- offline eval: MISSING ({p.name})')
        else:
            d = load(p)
            r = d.get('methods', {}).get('ocrap', {})
            print('|metric|value|')
            print('|---|---:|')
            for k in KEYS:
                print(f'|{k}|{fmt(r.get(k))}|')
            print(f"|selection_reason_counts|{r.get('selection_reason_counts')}|")
        if reg == 'safe':
            audit_name = f'closed_loop_safe_fast_{tag}.json'
        else:
            audit_name = f'audit_{reg}_selected_topk_{tag}_{tag}.json'
        a = root / audit_name
        if not a.exists() and tag == 'v29':
            legacy = root / audit_name.replace('v29', 'v27')
            a = legacy if legacy.exists() else a
        if not a.exists():
            missing_audit.append(str(a.name))
            print(f'- audit/closed-loop JSON: MISSING ({a.name})')
        else:
            d = load(a)
            print('- audit/closed-loop:')
            for k in AUDIT_KEYS:
                if k in d:
                    print(f'  - {k}: {fmt(d.get(k))}')
            for k in ['macro_counts', 'audit_paper_pcd_miss_best_macro_counts', 'audit_miss_selected_macro_counts', 'selection_reason_counts']:
                if k in d:
                    print(f'  - {k}: {d.get(k)}')
    if missing_audit:
        print('\nWARNING: Missing audit outputs: ' + ', '.join(missing_audit))
        print('paper_PCD_miss_rate cannot be judged for missing regimes.')
        return 3
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
