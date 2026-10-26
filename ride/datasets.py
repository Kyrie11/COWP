"""Offline dataset inventory only; never silently downloads or reads outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def inventory(raw_root, existing_ids=None):
    root=Path(raw_root)
    if not root.exists():return {'root':str(root),'exists':False,'needs':'WOMD v1.3.1 TFExample files'}
    splits={}
    for split in ('training','validation','validation_interactive','testing'):
        d=root/split
        tfrecs=sorted(d.glob('*.tfrecord*')) if d.is_dir() else sorted(root.glob(split+'*.tfrecord*'))
        splits[split]={'directory':str(d),'shards_found':len(tfrecs),
                       'total_bytes':sum(p.stat().st_size for p in tfrecs),
                       'example_paths':[str(p) for p in tfrecs[:3]],
                       'decoded_records':False}
    ids=Path(existing_ids) if existing_ids else None
    return {'root':str(root),'exists':True,'splits':splits,
            'prior_development_exclusion_ids':sum(1 for x in ids.read_text().splitlines() if x.strip()) if ids and ids.exists() else None,
            'note':'File inventory only. Not a decoded, independent scenario cohort.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--existing-ids')
    p.add_argument('--output',required=True);a=p.parse_args()
    r=inventory(a.raw_root,a.existing_ids)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
