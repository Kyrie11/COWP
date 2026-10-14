"""Selection-blind WOMD TFExample audit cohort, without COWP pseudo-labels.

Streams raw TFExample TFRecords; only past/current state fields determine scene
admission and interactive strata. Future ground truth is saved for *observational*
validation, not for online ego action choice or causal intervention labels.

WOMD TFExample format is required. Native Scenario proto TFRecords are intentionally
rejected rather than parsed with an incorrect schema.
"""
from __future__ import annotations
import argparse, hashlib, heapq, json, math, os, re
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
    # Past/current-only scenario semantics. These are candidate strata,
    # not ground-truth outcomes, and MUST NOT use future paths.
    angles=(s['current_bbox_yaw'][:,0] if 'current_bbox_yaw' in s else
            np.arctan2(vel[:,1],vel[:,0]))
    h=np.array([math.cos(float(angles[ego])),math.sin(float(angles[ego]))])
    side=np.array([-h[1],h[0]])
    rel=p[neigh]-p[ego]
    longitudinal=rel@h
    lateral=np.abs(rel@side)
    widths=np.asarray(s['current_width'][:,0],dtype=float) if 'current_width' in s else np.full(len(p),1.9)
    close_lane=lateral<np.maximum(2.5,0.5*(widths[ego]+widths[neigh])+1.)
    sp_ego=float(vel[ego]@h)
    sp_ng=vel[neigh]@h
    front=(longitudinal>0)&(longitudinal<38)&close_lane
    rear=(longitudinal<0)&(longitudinal>-38)&close_lane
    closing_rear=rear&(sp_ng>sp_ego+1.5)
    # Crossing: angular heading difference large and objects moderately near.
    headings=np.asarray(angles[neigh])
    ego_yaw=float(angles[ego])
    angle=np.abs(np.arctan2(np.sin(headings-ego_yaw),np.cos(headings-ego_yaw)))
    crossing=bool(np.any((angle>0.55)&(angle<2.6)&(dist[neigh]<32)))
    chain=bool(np.any(front)&np.any(closing_rear))
    # Current-state ONLY deterministic four-second linear closest approach.
    # A screening STRATUM (not a safety label, future outcome or intervention).
    best=float('inf');tbest=None
    for a, ia in enumerate(neigh):
        for ib in neigh[a+1:]:
            r=p[ia]-p[ib]; u=vel[ia]-vel[ib]
            u2=float(u@u)
            t=float(np.clip(-float(r@u)/u2,0.,4.)) if u2>1e-6 else 0.
            miss=float(np.linalg.norm(r+t*u))
            if miss<best: best=miss;tbest=t
    near_conflict=bool(best<6.5 and tbest is not None and .15<tbest<=4.)
    return {'scenario_strata':{'braking_chain':chain,'crossing':crossing,
        'non_ego_linear_cpa':near_conflict},
        'non_ego_linear_cpa_min_m':best if np.isfinite(best) else None,
        'non_ego_linear_cpa_time_s':tbest,
        'neighbor_indices':neigh.tolist(),'nearby_count':len(neigh),
        'nearest_m':float(np.min(dist[neigh])),
        'radial_ttc_s':float(np.min(ttc)) if np.isfinite(ttc).any() else None,
        'past_only_interactive': bool(np.min(dist[neigh])<16 or (np.min(ttc) if np.isfinite(ttc).any() else np.inf)<4.5),
        'ego_speed_mps':float(np.linalg.norm(vel[ego]))}


def compact_selected_state(s, features):
    """Save only ego + selected neighbors in reservoir (bounded RAM across 150 shards).

    Original source indices remain in `features.neighbor_indices` for audit;
    stored NPZ order is ego=0 and local neighbors 1..N. This does not change
    selection scores or use any future information for admission.
    """
    ids=[int(s['ego_index'])]+list(features['neighbor_indices'])
    d={f'{period}_{attr}':np.asarray(s[f'{period}_{attr}'][ids],dtype=np.float32)
       for period in ('past','current','future') for attr in STATE_ATTRS}
    d.update({'agent_ids':np.asarray(s['agent_ids'][ids],dtype=np.int64),
              'types':np.asarray(s['types'][ids],dtype=np.int16),
              'ego_index':0,'scenario_id':s['scenario_id']})
    return d


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


def verify_shard_inventory(shards, split='validation', expected=150):
    """Demand full numbered, nonduplicated 00000-of-00150 TFExample inventory."""
    rgx=re.compile(r'tfrecord-(\d+)-of-(\d+)$')
    numbers=[]
    for p in shards:
        match=rgx.search(p.name)
        if not match:
            raise ValueError(f'Unrecognized shard name for completeness audit: {p.name}')
        idx,total=map(int,match.groups())
        if total!=expected or idx<0 or idx>=expected:
            raise ValueError(f'Invalid shard numbering {p.name}; expected {expected} total')
        numbers.append(idx)
    missing=sorted(set(range(expected))-set(numbers))
    if missing or len(shards)!=expected or len(set(numbers))!=len(numbers):
        raise ValueError(f'INCOMPLETE {split} split: got {len(shards)}/{expected}, missing={missing[:30]}; refusing full scan claim')
    return {'complete':True,'expected_shards':expected,'actual_shards':len(shards),
            'shard_indices_sha256':hashlib.sha256(','.join(map(str,sorted(numbers))).encode()).hexdigest()}



def build_cohort(root,output,split='validation',max_shards=4,max_records_per_shard=350,keep_random=120,keep_interactive=120,salt='womd-blind-20261012',data_format='auto',exclude_ids=None,require_complete_split=False,keep_chain=0,keep_crossing=0,keep_cpa=0,max_neighbors=10,neighbor_radius=50.0,holdout_fraction=0.0,expected_shards=None):
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError('Install tensorflow-cpu in a separate data extraction environment: pip install tensorflow-cpu') from e
    if split=='testing':
        raise ValueError('WOMD testing hides the 8-second observed future; use validation/training for response calibration. Do not fabricate future labels.')
    if not 0.0<=holdout_fraction<1.0:raise ValueError('holdout_fraction must be in [0,1)')
    if max_neighbors<2 or neighbor_radius<=0:raise ValueError('Invalid neighbor selection')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('*.npz')):
        raise FileExistsError('Found stale NPZ output; use a new EICDRR_FULL_ROOT for a clean scan')
    if (output/'cohort_manifest.json').exists():
        raise FileExistsError('Refuse to overwrite existing cohort_manifest.json: '+str(output))
    shards,record_format=locate_shards(root,split,max_shards,data_format)
    expected=(expected_shards if expected_shards is not None else (150 if split=='validation' else 1000))
    coverage={'complete':False,'expected_shards':expected,
              'actual_shards':len(shards),'reason':'partial_or_not_audited'}
    if require_complete_split:
        if max_records_per_shard:raise ValueError('Complete scan requires max_records_per_shard=0')
        if max_shards:raise ValueError('Complete scan requires max_shards=0')
        if record_format!='tfexample':raise ValueError('Complete scan requested for tfexample only')
        coverage=verify_shard_inventory(shards,split,expected)
    if record_format=='scenario':
        try:
            from waymo_open_dataset.protos import scenario_pb2
        except ImportError as e:
            raise RuntimeError('Scenario proto shards selected, but waymo_open_dataset.protos.scenario_pb2 unavailable. Install an officially compatible Waymo Open Dataset package, or use tf_example shards.') from e
    excluded=set(exclude_ids or ())
    selected={'blind':[],'interaction':[],'braking_chain':[],'crossing':[],'non_ego_linear_cpa':[]}
    if holdout_fraction: selected['blind_holdout']=[]
    counts={'examined':0,'usable':0,'bad_records':0,'past_only_interactive':0,'excluded_overlap':0}
    errors=[]
    per_shard_counts=[]
    # Use eager TFRecordDataset: the v1 iterator is deprecated and can be brittle
    # across TensorFlow versions. Do not load every record into memory.
    if not tf.executing_eagerly():
        raise RuntimeError('Native WOMD extraction requires TensorFlow eager execution; run in a fresh Python process.')
    for shard in shards:
        before=counts['examined']
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
                f=scene_features(s,radius=neighbor_radius,max_neigh=max_neighbors)
                if f is None:continue
                counts['usable']+=1
                if f['past_only_interactive']:counts['past_only_interactive']+=1
                # Freeze an ID-level holdout BEFORE seeing future outcomes.  Holdout
                # IDs cannot enter interaction/crossing/chain development strata.
                in_holdout=(holdout_fraction>0.0 and
                    (_stable_hash('split-v2|'+salt+'|'+s['scenario_id'])%10**12)/10**12<holdout_fraction)
                choices=[('blind_holdout',keep_random if in_holdout else 0)] if in_holdout else [
                    ('blind',keep_random),
                    ('interaction',keep_interactive if f['past_only_interactive'] else 0),
                    ('braking_chain',keep_chain if f['scenario_strata']['braking_chain'] else 0),
                    ('crossing',keep_crossing if f['scenario_strata']['crossing'] else 0),
                    ('non_ego_linear_cpa',keep_cpa if f['scenario_strata']['non_ego_linear_cpa'] else 0)]
                compact_item=None
                for cohort_name,k in choices:
                    if k<=0:continue
                    h=_stable_hash(salt+'|'+cohort_name+'|'+s['scenario_id'])
                    arr=selected[cohort_name]
                    if len(arr)>=k and h>=-arr[0][0]:continue
                    if compact_item is None:
                        # Crucial for 150 shards: never retain the 128-actor
                        # state tensor in a reservoir of thousands of entries.
                        compact_item=(compact_selected_state(s,f),f,str(shard.name),j,
                                      int(s['ego_index']),observational_metrics(s,f['neighbor_indices']))
                    tup=(-h,s['scenario_id'],compact_item)
                    if len(arr)<k:heapq.heappush(arr,tup)
                    else:heapq.heapreplace(arr,tup)
            except Exception as e:
                counts['bad_records']+=1
                if len(errors)<10: errors.append({'shard':shard.name,'index':j,'error':str(e)})
        per_shard_counts.append({'name':shard.name,'records_processed':counts['examined']-before})
        print(f'[native_womd] {len(per_shard_counts)}/{len(shards)} {shard.name}: {counts["examined"]} total records',flush=True)
    if counts['examined'] and counts['bad_records'] > 0.05*counts['examined']:
        raise ValueError(f'{record_format} decoding failures {counts["bad_records"]}/{counts["examined"]}; inspect schema: {errors[:3]}')
    coverage['record_iteration_uncapped']=max_records_per_shard==0
    coverage['bad_decoded_records']=counts['bad_records']
    coverage['decode_integrity_pass']=counts['bad_records']==0
    if not counts['usable']:
        raise ValueError('No usable scenes from raw TFExample records. Check WOMD version and feature schema.')
    merged={}
    for mode,arr in selected.items():
        for _,sid,item in arr:
            if sid not in merged:merged[sid]={'item':item,'cohorts':[]}
            merged[sid]['cohorts'].append(mode)
    entries=[]
    for sid,rec in sorted(merged.items()):
        arrays,f,source,record_index,source_ego_idx,observed=rec['item']
        # CRITICAL: raw WOMD orders the SDC arbitrarily.  Preserve the
        # explicit selection: ego at local index zero, then past/current-only
        # selected vehicle neighbors.  Never set ego_index=0 on unpermuted
        # full-scene arrays; that silently simulates the wrong actor.
        np.savez_compressed(output/f'{sid}.npz',**{
            k:v for k,v in arrays.items() if k!='scenario_id'})
        entries.append({'scenario_id':sid,'file':f'{sid}.npz','source_shard':source,
            'cohorts':rec['cohorts'],
            'evaluation_partition':'sealed_holdout' if 'blind_holdout' in rec['cohorts'] else 'development',
            'source_record_index':int(record_index),
            'source_ego_index':int(source_ego_idx),
            'selected_source_indices':[int(source_ego_idx)]+[int(i) for i in f['neighbor_indices']],
            'features_from_current':f,
            'observed_future_diagnostic':observed})
    result={'status':'OBSERVATIONAL_SELECTION_BLIND_NOT_COUNTERFACTUAL',
        'split':split,'source_format':record_format,'raw_womd_root':str(root),'data_root':str(output),'salt':salt,
        'shards_scanned':[str(s) for s in shards],'max_records_per_shard':max_records_per_shard,
        'coverage':coverage,'per_shard_counts':per_shard_counts,
        'sampling_parameters':{'max_neighbors':max_neighbors,'neighbor_radius_m':neighbor_radius,
            'holdout_fraction':holdout_fraction,'keep_random_per_partition':keep_random,
            'keep_interactive':keep_interactive,'keep_chain':keep_chain,'keep_crossing':keep_crossing,
            'keep_cpa':keep_cpa},
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
    p.add_argument('--expected-shards',type=int,help='Override only for tiny synthetic tests; normal validation=150, training=1000')
    p.add_argument('--require-complete-split',action='store_true')
    p.add_argument('--max-records-per-shard',type=int,default=350)
    p.add_argument('--keep-random',type=int,default=120)
    p.add_argument('--keep-interactive',type=int,default=120)
    p.add_argument('--keep-chain',type=int,default=0)
    p.add_argument('--keep-crossing',type=int,default=0)
    p.add_argument('--keep-cpa',type=int,default=0,help='Current-only non-ego near-CPA stratum, not an outcome label')
    p.add_argument('--max-neighbors',type=int,default=10)
    p.add_argument('--neighbor-radius',type=float,default=50.)
    p.add_argument('--holdout-fraction',type=float,default=0.)
    p.add_argument('--exclude-ids',type=Path,help='Optional previous dev/Compact IDs, for anti-leakage selection')
    args=p.parse_args()
    r=build_cohort(args.raw_root,args.output,args.split,args.max_shards,args.max_records_per_shard,args.keep_random,args.keep_interactive,data_format=args.format,exclude_ids=load_exclusion_file(args.exclude_ids),require_complete_split=args.require_complete_split,keep_chain=args.keep_chain,keep_crossing=args.keep_crossing,keep_cpa=args.keep_cpa,max_neighbors=args.max_neighbors,neighbor_radius=args.neighbor_radius,holdout_fraction=args.holdout_fraction,expected_shards=args.expected_shards)
    print(json.dumps({'counts':r['counts'],'selected':len(r['entries']),'manifest':str(args.output/'cohort_manifest.json'),'coverage':r['coverage']},indent=2))

if __name__=='__main__':main()
