"""Native cohort provenance checks; refuse claims when ego identity is ambiguous.

Offline checks require NPZ files. --raw-root additionally verifies that each
source record's actual SDC ID and pose match the local ego (index zero).
Raw verification requires TensorFlow and original TFExample files.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np


def verify(cohort,*,raw_root=None,exclusion_ids=None,limit=0):
    cohort=Path(cohort);m=json.loads((cohort/'cohort_manifest.json').read_text())
    rows=[];errors=[];seen=set();shards={}
    entries=m['entries'][:limit or None]
    for e in entries:
        sid=e['scenario_id']
        try:
            if sid in seen:raise ValueError('duplicate scenario id')
            seen.add(sid)
            if exclusion_ids is not None and sid in exclusion_ids:raise ValueError('historical train/validation/test overlap')
            if 'source_record_index' not in e:raise ValueError('MISSING source_record_index; re-extract native WOMD with patched writer')
            if 'selected_source_indices' not in e:raise ValueError('MISSING selected_source_indices; re-extract with patched writer')
            with np.load(cohort/e['file'],allow_pickle=False) as d:
                ids=np.asarray(d['agent_ids']).reshape(-1);ego=int(d['ego_index'])
                if ego!=0:raise ValueError('local ego index must be 0')
                if not len(ids) or len(set(map(int,ids)))!=len(ids):raise ValueError('bad/non-unique agent IDs')
                if len(ids)!=len(e['selected_source_indices']):raise ValueError('NPZ/selected source indices length mismatch')
                if len(ids)!=len(e['features_from_current']['neighbor_indices'])+1:
                    raise ValueError('NPZ should have only ego plus selected neighbors')
                xy=np.array([float(d['current_x'][ego,0]),float(d['current_y'][ego,0])])
                if not np.isfinite(xy).all():raise ValueError('nonfinite SDC XY')
                target_ids=ids.astype(np.int64).tolist()
            if raw_root is not None:
                if m.get('source_format')!='tfexample':raise ValueError('raw source verifier requires tfexample')
                import tensorflow as tf
                shname=e['source_shard']
                if shname not in shards:
                    found=list(Path(raw_root).rglob(shname))
                    if len(found)!=1:raise ValueError(f'expect one raw shard {shname}; got {len(found)}')
                    shards[shname]=found[0]
                idx=int(e['source_record_index'])
                raws=list(tf.data.TFRecordDataset(str(shards[shname])).skip(idx).take(1).as_numpy_iterator())
                if len(raws)!=1:raise ValueError('missing raw record')
                ex=tf.train.Example.FromString(bytes(raws[0]))
                features=ex.features.feature
                orig_id=features['scenario/id'].bytes_list.value[0].decode()
                if orig_id!=sid:raise ValueError('source scenario ID mismatch')
                raw_ids=np.asarray(features['state/id'].float_list.value if features['state/id'].float_list.value else features['state/id'].int64_list.value,dtype=np.int64)
                sdc_mask=np.asarray(features['state/is_sdc'].int64_list.value,dtype=int)
                if not len(sdc_mask):sdc_mask=np.asarray(features['state/is_sdc'].float_list.value,dtype=int)
                if sdc_mask.sum()!=1:raise ValueError('no unique source SDC')
                true_idx=int(np.flatnonzero(sdc_mask)[0])
                src_idx=e['selected_source_indices']
                if true_idx!=src_idx[0]:raise ValueError('wrong SDC source index')
                if raw_ids[src_idx].tolist()!=target_ids:raise ValueError('NPZ agent IDs disagree with source')
                n=len(raw_ids)
                x=np.asarray(features['state/current/x'].float_list.value,dtype=float).reshape(n,1)[true_idx,0]
                y=np.asarray(features['state/current/y'].float_list.value,dtype=float).reshape(n,1)[true_idx,0]
                if np.linalg.norm(xy-np.array([x,y]))>0.05:raise ValueError('SDC source pose error > 5cm')
            rows.append({'scenario_id':sid,'agent_count':len(target_ids),'ego_agent_id':int(target_ids[0]),
                         'source_record_index':int(e['source_record_index']),'identity_verified':True})
        except Exception as exc:errors.append({'scenario_id':sid,'error':str(exc)})
    return {'pass':len(errors)==0 and len(rows)==len(entries),'attempted':len(entries),
            'verified':len(rows),'errors':errors,
            'raw_records_verified':raw_root is not None,
            'claim':'Cohort identity/provenance only; no causal behavioral verification'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--raw-root',type=Path)
    p.add_argument('--historical-ids',type=Path)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    excluded=None
    if a.historical_ids:
        excluded={ln.strip().split()[0] for ln in a.historical_ids.read_text().splitlines() if ln.strip() and not ln.startswith('#')}
    out=verify(a.cohort,raw_root=a.raw_root,exclusion_ids=excluded,limit=a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='errors'},indent=2))
    if out['errors']:print('FIRST ERRORS',out['errors'][:5]);raise SystemExit(2)
if __name__=='__main__':main()
