"""Strictly verify whether a native WOMD manifest supports a full-shard claim.

A manifest covering 150 shards with max_records_per_shard=0 and completion
counts is necessary, NOT sufficient to prove that raw corpus hasn't changed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .native_womd import verify_shard_inventory

def verify(manifest,expected=150,require_full_records=True):
    m=json.loads(Path(manifest).read_text())
    shards=[Path(x) for x in m.get('shards_scanned',[])]
    audit=verify_shard_inventory(shards,m.get('split','validation'),expected)
    if require_full_records and m.get('max_records_per_shard') != 0:
        raise RuntimeError('Full shard inventory, but max_records_per_shard!=0: incomplete record scan')
    c=m.get('per_shard_counts')
    if not isinstance(c,list) or len(c)!=expected or any(not x.get('records_processed',0)>0 for x in c):
        raise RuntimeError('Manifest lacks nonzero per-shard processing records for all shards')
    if sum(x['records_processed'] for x in c)!=m.get('counts',{}).get('examined'):
        raise RuntimeError('Per-shard count != examined count')
    return {'status':'COMPLETE_MANIFEST_SELF_CONSISTENT','manifest':str(manifest),
            'n_shards':expected,'records_examined':m['counts']['examined'],
            'n_selected':len(m.get('entries',[])), 'n_bad_records':m['counts']['bad_records'],
            'validation_limit':'Does not independently re-hash TFRecord contents',**audit}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--expect-shards',type=int,default=150)
    p.add_argument('--require-full-records',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.cohort/'cohort_manifest.json',a.expect_shards,a.require_full_records),indent=2))
if __name__=='__main__':main()
