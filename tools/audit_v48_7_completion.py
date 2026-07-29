#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--require-calibration', action='store_true')
    args=ap.parse_args()
    root=args.root
    issues=[]; variants={}
    marker=root/'TRAINING_COMPLETE.json'
    marker_doc={}
    if marker.is_file():
        try: marker_doc=json.loads(marker.read_text())
        except Exception as e: issues.append(f'invalid TRAINING_COMPLETE.json: {e}')
    else: issues.append('missing TRAINING_COMPLETE.json')
    expected=(marker_doc.get('variants') or {}) if isinstance(marker_doc,dict) else {}
    for name in ('balanced','precision'):
        run=root/'candidates'/name
        ckpt=run/'model_v48_trac_sr'/'best.pt'
        summ=run/'model_v48_trac_sr'/'train_summary.json'
        v={'checkpoint_exists':ckpt.is_file(),'summary_exists':summ.is_file()}
        if ckpt.is_file():
            v['checkpoint_sha256']=sha256(ckpt); v['checkpoint_size']=ckpt.stat().st_size
            exp=(expected.get(name) or {}).get('sha256')
            if exp and exp != v['checkpoint_sha256']: issues.append(f'{name}: checkpoint hash changed after completion marker')
        if summ.is_file():
            try:
                d=json.loads(summ.read_text()); v.update(best_epoch=d.get('best_epoch'),epochs_completed=d.get('epochs_completed'),best_metric=d.get('best_metric'))
            except Exception as e: issues.append(f'{name}: invalid train_summary: {e}')
        for bucket in ('near','contact'):
            p=run/'calibration'/f'direct_value_risk_{bucket}_v48.json'
            v[f'{bucket}_calibration_exists']=p.is_file()
            if p.is_file():
                try:
                    d=json.loads(p.read_text()); v[f'{bucket}_valid']=d.get('valid_for_deployment'); v[f'{bucket}_warnings']=d.get('warnings',[])
                except Exception as e: issues.append(f'{name}/{bucket}: invalid calibration json: {e}')
            elif args.require_calibration and ckpt.is_file(): issues.append(f'{name}/{bucket}: missing calibration result')
        if ckpt.is_file() and not summ.is_file(): issues.append(f'{name}: checkpoint without train_summary')
        variants[name]=v
    comparable=not issues and any(v.get('checkpoint_exists') for v in variants.values())
    doc={'root':str(root),'comparable':comparable,'issues':issues,'variants':variants}
    out=args.output or root/'completion_audit_v48_7.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(doc,ensure_ascii=False,indent=2))
    return 0 if comparable else 4
if __name__=='__main__': raise SystemExit(main())
