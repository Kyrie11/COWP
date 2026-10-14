"""Zero-TensorFlow filesystem audit of WOMD numbered TFRecord shards.

Presence and naming do NOT prove record integrity; the streamed extractor
records successful per-shard reads and per-record parsing separately.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from .native_womd import locate_shards

COUNTS={'validation':150,'training':1000,'testing':150}


def inventory(root,split='validation',expected=None):
    root=Path(root)
    if expected is None: expected=COUNTS[split]
    pattern=re.compile(r'tfrecord-(\d+)-of-(\d+)$')
    try:paths,fmt=locate_shards(root,split,0,'tfexample')
    except FileNotFoundError:
        return {'root':str(root),'split':split,'expected':expected,'found':0,
                'present_indices':[],'missing_indices':list(range(expected)),
                'duplicate_indices':[],'wrong_total_files':[],
                'complete_names':False,'file_bytes_total':0}
    indices=[];wrong=[];size=0
    for p in paths:
        mat=pattern.search(p.name)
        if mat is None:wrong.append(p.name);continue
        i,total=map(int,mat.groups())
        if total!=expected or not 0<=i<expected:wrong.append(p.name);continue
        indices.append(i);size+=p.stat().st_size
    missing=sorted(set(range(expected))-set(indices))
    duplicate=sorted(i for i in set(indices) if indices.count(i)>1)
    return {'root':str(root),'split':split,'expected':expected,'found':len(paths),
            'present_indices':sorted(set(indices)),'missing_indices':missing,
            'duplicate_indices':duplicate,'wrong_total_files':wrong,
            'complete_names':not(missing or duplicate or wrong) and len(paths)==expected,
            'file_bytes_total':size,'valid_shard_count':len(indices),
            'warning':'Complete filenames do not establish decoded-record integrity'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',required=True,type=Path)
    p.add_argument('--split',choices=tuple(COUNTS),default='validation')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=inventory(args.raw_root,args.split)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v if k not in ('present_indices','missing_indices') else (v if len(v)<25 else v[:25]+['...'])
         for k,v in result.items()},indent=2))
    if not result['complete_names']:raise SystemExit(2)
if __name__=='__main__':main()
