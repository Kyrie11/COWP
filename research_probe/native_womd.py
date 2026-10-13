"""Selection-blind WOMD TFExample audit cohort, without COWP pseudo-labels.

Streams raw TFExample TFRecords; only past/current state fields determine scene
admission and interactive strata. Future ground truth is saved for *observational*
validation, not for online ego action choice or causal intervention labels.

WOMD TFExample format is required. Native Scenario proto TFRecords are intentionally
rejected rather than parsed with an incorrect schema.
"""
from __future__ import annotations
import argparse, hashlib, heapq, json, math, os
from pathlib import Path
import numpy as np

PAST=10
FUTURE=80
STATE_ATTRS=('x','y','velocity_x','velocity_y','bbox_yaw','length','width','valid')


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(),16)


def _feature(ex, name, required=False):
    f=ex.features.feature.get(name)
    if f is None:
        if required:raise ValueError('Missing TFExample feature '+name+'; is input Scenario proto rather than tf_example?')
        return None
    if f.float_list.value:return np.asarray(f.float_list.value,dtype=np.float64)
    if f.int64_list.value:return np.asarray(f.int64_list.value,dtype=np.int64)
    if f.bytes_list.value:return list(f.bytes_list.value)
    if required: raise ValueError('Empty required feature '+name)
    return None


def decode_example(ex):
    sid=_feature(ex,'scenario/id',True)
    if not isinstance(sid,list) or not sid:raise ValueError('Invalid scenario/id')
    sid=sid[0].decode('utf-8',errors='strict')
    ego=_feature(ex,'state/is_sdc',True)
    n=len(ego)
    if sum(np.asarray(ego)>0.5)!=1:raise ValueError('Need exactly one SDC')
    obj=_feature(ex,'state/type')
    obj=np.asarray(obj if obj is not None else np.ones(n),dtype=int)
    if len(obj)!=n:raise ValueError('Bad state/type size')
    state={}
    for period, steps in [('past',PAST),('current',1),('future',FUTURE)]:
        for key in STATE_ATTRS:
            a=_feature(ex,f'state/{period}/{key}',required=key in ('x','y','valid'))
            if a is None:
                if key in ('length','width'): a=np.full(n*steps,4.5 if key=='length' else 1.9)
                else: a=np.zeros(n*steps)
            a=np.asarray(a,dtype=float)
            if a.size!=n*steps:raise ValueError(f'{period}/{key} size {a.size} != {n}x{steps}')
            state[f'{period}_{key}']=a.reshape(n,steps)
    ids=_feature(ex,'state/id')
    if ids is None:ids=np.arange(n)
    ids=np.asarray(ids).reshape(-1)
    if len(ids)!=n: raise ValueError('Malformed object IDs')
    state['agent_ids']=ids
    state['types']=obj
    state['ego_index']=int(np.flatnonzero(np.asarray(ego)>.5)[0])
    state['scenario_id']=sid
    return state



def decode_scenario_proto(sc):
    """Decode official waymo_open_dataset.protos.scenario_pb2.Scenario.

    Uses native current_time_index rather than assuming a particular fixed split.
    Pads missing trajectory states with invalid flags; no future values enter
    the selection stratum.
    """
    n=len(sc.tracks)
    if n==0:raise ValueError('Empty Scenario proto')
    ego=int(sc.sdc_track_index)
    if ego<0 or ego>=n:raise ValueError('Invalid sdc_track_index')
    t=int(sc.current_time_index)
    if t<PAST:raise ValueError(f'Insufficient past frames: current_time_index={t}')
    state={'agent_ids':np.asarray([track.id for track in sc.tracks],dtype=np.int64),
           'types':np.asarray([track.object_type for track in sc.tracks],dtype=np.int16),
           'ego_index':ego,'scenario_id':sc.scenario_id}
    attrs={'x':'center_x','y':'center_y','velocity_x':'velocity_x','velocity_y':'velocity_y',
           'bbox_yaw':'heading','length':'length','width':'width','valid':'valid'}
    for phase,indices in [('past',range(t-PAST,t)),('current',[t]),('future',range(t+1,t+1+FUTURE))]:
        for name,proto_attr in attrs.items():
            a=np.zeros((n,len(indices)),dtype=float)
            for i,track in enumerate(sc.tracks):
                for j,k in enumerate(indices):
                    if k<len(track.states):
                        val=getattr(track.states[k],proto_attr)
                        a[i,j]=float(val)
            state[f'{phase}_{name}']=a
    return state

def min_nearby_clearance(state, near, ego):
    p=np.c_[state['current_x'][:,0],state['current_y'][:,0]]
    d=np.linalg.norm(p[near]-p[ego],axis=1)
    return float(d.min()) if len(d) else float('inf')


def scene_features(s, radius=50.0, max_neigh=10):
    ego=s['ego_index'];v=s['current_valid'][:,0]>.5;type_vehicle=s['types']==1
    p=np.c_[s['current_x'][:,0],s['current_y'][:,0]]
    if not v[ego] or not np.isfinite(p[ego]).all():return None
    dist=np.linalg.norm(p-p[ego],axis=1)
    neigh=np.flatnonzero(v & type_vehicle & (dist<radius) & (dist>0) & (np.arange(len(v))!=ego))
    neigh=neigh[np.argsort(dist[neigh],kind='stable')][:max_neigh]
    if len(neigh)<2:return None
    vel=np.c_[s['current_velocity_x'][:,0],s['current_velocity_y'][:,0]]
    relative=p[neigh]-p[ego]
    rv=vel[neigh]-vel[ego]
    approach=-np.sum(relative*rv,axis=1)/np.maximum(dist[neigh],1e-3)
    ttc=np.where(approach>0.1,dist[neigh]/np.maximum(approach,.1),np.inf)
    return {'neighbor_indices':neigh.tolist(),'nearby_count':len(neigh),
        'nearest_m':float(np.min(dist[neigh])),
        'radial_ttc_s':float(np.min(ttc)) if np.isfinite(ttc).any() else None,
        'past_only_interactive': bool(np.min(dist[neigh])<16 or (np.min(ttc) if np.isfinite(ttc).any() else np.inf)<4.5),
        'ego_speed_mps':float(np.linalg.norm(vel[ego]))}


def observational_metrics(s, neighbors):
    """Observed future co-occurrence ONLY: no hypothetical ego intervention and no causal claim."""
    ix=[s['ego_index']]+list(neighbors)
    x=s['future_x'][ix];y=s['future_y'][ix]; valid=s['future_valid'][ix]>.5
    vel=np.stack([s['future_velocity_x'][ix],s['future_velocity_y'][ix]],axis=-1)
    speed=np.linalg.norm(vel,axis=-1)
    dv=np.diff(speed,axis=1)/0.1
    ok=valid[:,1:] & valid[:,:-1]
    mins=[]
    for a in range(len(ix)):
        for b in range(a+1,len(ix)):
            mask=valid[a]&valid[b]
            if mask.any():mins.append(float(np.linalg.norm(np.stack([x[a]-x[b],y[a]-y[b]],axis=1)[mask],axis=1).min()))
    brakes=[]
    for j in range(len(ix)):
        brakes.append(float(dv[j][ok[j]].min()) if ok[j].any() else None)
    return {'observed_min_center_distance_m':min(mins) if mins else None,
            'observed_min_longitudinal_accel_mps2':brakes,
            'observed_non_ego_hard_braking':any(a is not None and a<-4.0 for a in brakes[1:]),
            'future_label_semantics':'Observed logged co-occurrence, NOT ego intervention or causal recourse failure'}


def locate_shards(root,split,max_shards,data_format="auto"):
    root=Path(root)
    if not root.exists():raise FileNotFoundError(root)
    # WOMD can be stored as .../tf_example/validation/*.tfrecord-*.
    files=sorted(p for p in root.rglob('*') if p.is_file() and
        ('tfrecord' in p.name.lower() or p.suffix.lower() in ('.tfrecord','.record')) and
        any(part.lower()==split.lower() or part.lower().startswith((split.lower()+'_',split.lower()+'.',split.lower()+'-')) for part in p.parts))
    if not files:raise FileNotFoundError(f'No {split} TFRecord shards in {root}; ensure tf_example (not scenario proto).')
    examples=[p for p in files if any('tf_example' in part.lower() or 'tfexample' in part.lower() for part in p.parts)]
    proto=[p for p in files if any(part.lower() in ('scenario','scenarios','scenario_proto') for part in p.parts)]
    if data_format=='tfexample':mode='tfexample'; files=examples
    elif data_format=='scenario':mode='scenario';files=proto
    elif examples:mode='tfexample';files=examples
    elif proto:mode='scenario';files=proto
    else:raise FileNotFoundError('TFRecords found but cannot determine TFExample vs Scenario proto. Put files under tf_example/<split> or scenario/<split>, or select a known format.')
    if not files:raise FileNotFoundError(f'No {mode} files in {root} for split {split}')
    return (files[:max_shards] if max_shards else files),mode


def build_cohort(root,output,split='validation',max_shards=4,max_records_per_shard=350,keep_random=120,keep_interactive=120,salt='womd-blind-20261012',data_format='auto',exclude_ids=None):
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError('Install tensorflow-cpu in a separate data extraction environment: pip install tensorflow-cpu') from e
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    shards,record_format=locate_shards(root,split,max_shards,data_format)
    if record_format=='scenario':
        try:
            from waymo_open_dataset.protos import scenario_pb2
        except ImportError as e:
            raise RuntimeError('Scenario proto shards selected, but waymo_open_dataset.protos.scenario_pb2 unavailable. Install an officially compatible Waymo Open Dataset package, or use tf_example shards.') from e
    excluded=set(exclude_ids or ())
    selected={'blind':[],'interaction':[]}
    counts={'examined':0,'usable':0,'bad_records':0,'past_only_interactive':0,'excluded_overlap':0}
    errors=[]
    # Use eager TFRecordDataset: the v1 iterator is deprecated and can be brittle
    # across TensorFlow versions. Do not load every record into memory.
    if not tf.executing_eagerly():
        raise RuntimeError('Native WOMD extraction requires TensorFlow eager execution; run in a fresh Python process.')
    for shard in shards:
        records = tf.data.TFRecordDataset(str(shard), num_parallel_reads=1)
        for j,raw_record in enumerate(records.as_numpy_iterator()):
            if max_records_per_shard and j>=max_records_per_shard:break
            record=bytes(raw_record)
            counts['examined']+=1
            try:
                if record_format=='tfexample':
                    ex=tf.train.Example.FromString(record)
                    s=decode_example(ex)
                else:
                    ex=scenario_pb2.Scenario.FromString(record)
                    s=decode_scenario_proto(ex)
                if s['scenario_id'] in excluded:
                    counts['excluded_overlap']+=1
                    continue
                f=scene_features(s)
                if f is None:continue
                counts['usable']+=1
                if f['past_only_interactive']:counts['past_only_interactive']+=1
                item=(s,f,str(shard.name),j)
                for cohort_name,k in [('blind',keep_random),('interaction',keep_interactive if f['past_only_interactive'] else 0)]:
                    if k<=0:continue
                    h=_stable_hash(salt+'|'+cohort_name+'|'+s['scenario_id'])
                    arr=selected[cohort_name];tup=(-h,s['scenario_id'],item)
                    if len(arr)<k:heapq.heappush(arr,tup)
                    elif h < -arr[0][0]:heapq.heapreplace(arr,tup)
            except Exception as e:
                counts['bad_records']+=1
                if len(errors)<10: errors.append({'shard':shard.name,'index':j,'error':str(e)})
    if counts['examined'] and counts['bad_records'] > 0.05*counts['examined']:
        raise ValueError(f'{record_format} decoding failures {counts["bad_records"]}/{counts["examined"]}; inspect schema: {errors[:3]}')
    if not counts['usable']:
        raise ValueError('No usable scenes from raw TFExample records. Check WOMD version and feature schema.')
    merged={}
    for mode,arr in selected.items():
        for _,sid,item in arr:
            if sid not in merged:merged[sid]={'item':item,'cohorts':[]}
            merged[sid]['cohorts'].append(mode)
    entries=[]
    for sid,rec in sorted(merged.items()):
        s,f,source,record_index=rec['item']; ix=[s['ego_index']]+f['neighbor_indices']
        arrays={f'{period}_{field}':s[f'{period}_{field}'][ix].astype(np.float32)
              for period in ('past','current','future') for field in STATE_ATTRS}
        arrays.update({'agent_ids':np.asarray(s['agent_ids'][ix],dtype=np.int64),
                       'types':np.asarray(s['types'][ix],dtype=np.int16),
                       'ego_index':np.asarray(0,dtype=np.int16)})
        np.savez_compressed(output/f'{sid}.npz',**arrays)
        entries.append({'scenario_id':sid,'file':f'{sid}.npz','source_shard':source,
            'cohorts':rec['cohorts'],'source_record_index':int(record_index),'features_from_current':f,
            'observed_future_diagnostic':observational_metrics(s,f['neighbor_indices'])})
    result={'status':'OBSERVATIONAL_SELECTION_BLIND_NOT_COUNTERFACTUAL',
        'split':split,'source_format':record_format,'raw_womd_root':str(root),'data_root':str(output),'salt':salt,
        'shards_scanned':[str(s) for s in shards],'max_records_per_shard':max_records_per_shard,
        'excluded_scenario_id_count':len(excluded),
        'counts':counts,'selection':'SHA256 minimum-hash reservoir within visited records; interaction stratum uses CURRENT only; cohort is not representative of entire WOMD unless full shards scanned',
        'entries':entries,'errors':errors}
    (output/'cohort_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def load_exclusion_file(path):
    """Load source IDs without ever inspecting future trajectories or recourse labels."""
    if path is None:return set()
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    if path.suffix.lower()=='.json':
        data=json.loads(path.read_text())
        if isinstance(data,list):return set(str(s) for s in data)
        if 'entries' in data:return {str(r['scenario_id']) for r in data['entries']}
        for key in ('scenario_ids','ids','heldout_ids','development_ids'):
            if key in data:return {str(s) for s in data[key]}
        raise ValueError(f'Unknown JSON exclusion format: {path}')
    return {x.strip().split()[0] for x in path.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith('#')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['training','validation'],default='validation')
    p.add_argument('--format',choices=['auto','tfexample','scenario'],default='auto')
    p.add_argument('--max-shards',type=int,default=4)
    p.add_argument('--max-records-per-shard',type=int,default=350)
    p.add_argument('--keep-random',type=int,default=120)
    p.add_argument('--keep-interactive',type=int,default=120)
    p.add_argument('--exclude-ids',type=Path,help='Optional previous dev/Compact IDs, for anti-leakage selection')
    args=p.parse_args()
    r=build_cohort(args.raw_root,args.output,args.split,args.max_shards,args.max_records_per_shard,args.keep_random,args.keep_interactive,data_format=args.format,exclude_ids=load_exclusion_file(args.exclude_ids))
    print(json.dumps({'counts':r['counts'],'selected':len(r['entries']),'manifest':str(args.output/'cohort_manifest.json')},indent=2))

if __name__=='__main__':main()
