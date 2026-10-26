"""Verify saved NPZ ego identity against raw WOMD, without self-consistency shortcuts.

All raw records in one shard are read ONCE, in index order. The exact SAME
TFExample decoder is used during writing and raw verification. The report
preserves failure reasons and refuses a PASS when any provenance check fails.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .native_womd import decode_example


def check_raw_state(e, expected_ids, expected_xy, raw):
    """Compare decoded source *by index*; safe to unit test without TensorFlow."""
    sid=e['scenario_id']
    if raw['scenario_id']!=sid:
        raise ValueError(f'source scenario ID mismatch: {raw["scenario_id"]} != {sid}')
    idx=int(raw['ego_index'])
    src_idx=[int(x) for x in e['selected_source_indices']]
    if idx!=src_idx[0]:
        raise ValueError(f'raw SDC index {idx} != selected index {src_idx[0]}')
    if idx!=int(e['source_ego_index']):
        raise ValueError(f'raw SDC index {idx} != manifest index {e["source_ego_index"]}')
    source_ids=np.asarray(raw['agent_ids'],dtype=np.int64)
    if source_ids[src_idx].tolist()!=list(expected_ids):
        raise ValueError('NPZ agent IDs disagree with raw TFExample')
    source_xy=np.array([raw['current_x'][idx,0],raw['current_y'][idx,0]],dtype=float)
    delta=float(np.linalg.norm(np.asarray(expected_xy,dtype=float)-source_xy))
    if delta>0.05:
        raise ValueError(f'SDC source pose error {delta:.3f}m > 5cm')
    return {'source_sdc_index':idx,'source_sdc_id':int(source_ids[idx]),'ego_pose_delta_m':delta}


def verify(cohort,*,raw_root=None,exclusion_ids=None,limit=0):
    cohort=Path(cohort)
    manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    entries=manifest['entries'][:limit or None]
    failures=[];valid={};seen=set()
    for e in entries:
        sid=e['scenario_id']
        try:
            if sid in seen:raise ValueError('duplicate scenario id')
            seen.add(sid)
            if exclusion_ids is not None and sid in exclusion_ids:
                raise ValueError('historical overlap')
            for field in ('source_record_index','selected_source_indices'):
                if field not in e:raise ValueError(f'MISSING {field}; re-extract raw WOMD')
            # Older provenance manifests may omit redundant source_ego_index.
            # The first selected source index is the stored SDC by contract.
            if 'source_ego_index' not in e: e['source_ego_index']=int(e['selected_source_indices'][0])
            with np.load(cohort/e['file'],allow_pickle=False) as d:
                ids=np.asarray(d['agent_ids']).reshape(-1).astype(np.int64).tolist()
                idx=int(d['ego_index'])
                if idx!=0:raise ValueError('local ego must be zero')
                if len(ids)!=len(set(ids)):raise ValueError('duplicate IDs')
                if len(ids)!=len(e['selected_source_indices']):raise ValueError('NPZ/source index length differs')
                if len(ids)!=len(e['features_from_current']['neighbor_indices'])+1:
                    raise ValueError('Unexpected NPZ agent count')
                xy=[float(d['current_x'][idx,0]),float(d['current_y'][idx,0])]
                if not np.isfinite(xy).all():raise ValueError('nonfinite ego XY')
            valid[sid]={'scenario_id':sid,'agent_count':len(ids),'ego_agent_id':ids[0],
                        'source_record_index':int(e['source_record_index']),
                        'identity_verified':raw_root is None,'_ids':ids,'_xy':xy}
        except Exception as exc:failures.append({'scenario_id':sid,'phase':'npz','error':str(exc)})

    if raw_root is not None:
        if manifest.get('source_format')!='tfexample':
            raise ValueError('Raw verifier requires TFExample source')
        try:import tensorflow as tf
        except ImportError as exc:raise RuntimeError('Raw identity verification needs tensorflow-cpu') from exc
        # The same raw record may be requested several times: index once, not O(n)
        # TFRecordDataset.skip(idx) per scene, which is slow at full 150 shards.
        pending=defaultdict(dict)
        for e in entries:
            if e['scenario_id'] in valid:
                pending[e['source_shard']][int(e['source_record_index'])]=e
        raw_root=Path(raw_root)
        for shard_name,requests in sorted(pending.items()):
            candidates=list(raw_root.rglob(shard_name))
            if len(candidates)!=1:
                for e in requests.values():
                    failures.append({'scenario_id':e['scenario_id'],'phase':'raw',
                                     'error':f'expected one shard {shard_name}; found {len(candidates)}'})
                    valid.pop(e['scenario_id'],None)
                continue
            remaining=set(requests)
            try:
                # Streaming sequential scan; no future label used for cohort selection.
                for j,record in enumerate(tf.data.TFRecordDataset(str(candidates[0])).as_numpy_iterator()):
                    if j not in remaining:continue
                    e=requests[j];sid=e['scenario_id'];remaining.remove(j)
                    try:
                        raw=decode_example(tf.train.Example.FromString(bytes(record)))
                        extra=check_raw_state(e,valid[sid]['_ids'],valid[sid]['_xy'],raw)
                        valid[sid].update(extra);valid[sid]['identity_verified']=True
                    except Exception as exc:
                        # Include source index. 'no unique SDC' must be traced
                        # rather than reinterpreted as a behavioral result.
                        failures.append({'scenario_id':sid,'phase':'raw','shard':shard_name,
                                         'record_index':j,'error':str(exc)})
                        valid.pop(sid,None)
                    if not remaining:break
                for j in remaining:
                    e=requests[j];sid=e['scenario_id']
                    failures.append({'scenario_id':sid,'phase':'raw','shard':shard_name,
                                     'record_index':j,'error':'record index unavailable'})
                    valid.pop(sid,None)
            except Exception as exc:
                for j in remaining:
                    e=requests[j];sid=e['scenario_id']
                    failures.append({'scenario_id':sid,'phase':'raw','shard':shard_name,
                                     'record_index':j,'error':f'shard read failure: {exc}'})
                    valid.pop(sid,None)
    rows=[]
    for v in valid.values():
        rows.append({k:val for k,val in v.items() if not k.startswith('_')})
    return {'pass':len(failures)==0 and len(rows)==len(entries),
            'attempted':len(entries),'verified':len(rows),'errors':failures,
            'raw_records_verified':raw_root is not None,
            'source_decode_parity':raw_root is not None,
            'claim':'Ego/source identity only; does NOT validate response causality',
            'rows':rows}


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
        excluded={line.split()[0] for line in a.historical_ids.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')}
    out=verify(a.cohort,raw_root=a.raw_root,exclusion_ids=excluded,limit=a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ('errors','rows')},indent=2))
    if out['errors']:
        print('FIRST RAW/NPZ ERRORS',out['errors'][:5]);raise SystemExit(2)
if __name__=='__main__':main()
