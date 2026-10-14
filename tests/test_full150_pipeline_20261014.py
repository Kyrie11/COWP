from __future__ import annotations
import json,sys,types
from pathlib import Path
import numpy as np
from research_probe.native_womd import build_cohort
from research_probe.split_inventory import inventory
from research_probe.verify_cohort_identity import check_raw_state


def _dummy_state(sid):
    n=3; ego=2
    state={'scenario_id':sid,'ego_index':ego,'agent_ids':np.array([10,11,12]),
           'types':np.ones(n,dtype=int)}
    for stage,count in [('past',10),('current',1),('future',80)]:
        for key in ('x','y','velocity_x','velocity_y','bbox_yaw','length','width','valid'):
            a=np.zeros((n,count))
            if key=='x':a=np.repeat(np.array([[9.],[19.],[0.]]),count,axis=1)
            if key=='velocity_x':a[:]=5.
            if key=='length':a[:]=4.0
            if key=='width':a[:]=1.8
            if key=='valid':a[:]=1.
            state[stage+'_'+key]=a
    return state


def test_inventory_reports_missing_and_complete(tmp_path):
    folder=tmp_path/'tf_example'/'validation'
    folder.mkdir(parents=True)
    (folder/'validation_tfexample.tfrecord-00000-of-00002').write_bytes(b'x')
    assert inventory(tmp_path,expected=2)['missing_indices']==[1]
    (folder/'validation_tfexample.tfrecord-00001-of-00002').write_bytes(b'x')
    assert inventory(tmp_path,expected=2)['complete_names']


def test_full_scanner_streaming_compacts_sdc_and_seals_holdout(monkeypatch,tmp_path):
    folder=tmp_path/'tf_example'/'validation'
    folder.mkdir(parents=True)
    for num in range(2):
        (folder/f'validation_tfexample.tfrecord-{num:05d}-of-00002').touch()
    from research_probe import native_womd as nw
    # No large TensorFlow installation required for a small synthetic stream test.
    class FakeDataset:
        def __init__(self,path,num_parallel_reads=1):self.path=path
        def as_numpy_iterator(self):
            ix=int(self.path[-10:-6]) if False else int(self.path.split('tfrecord-')[1][:5])
            return iter([f's{ix}_{j}'.encode() for j in range(3)])
    fake=types.SimpleNamespace(executing_eagerly=lambda:True,
            data=types.SimpleNamespace(TFRecordDataset=FakeDataset),
            train=types.SimpleNamespace(Example=types.SimpleNamespace(FromString=lambda b:b)))
    monkeypatch.setitem(sys.modules,'tensorflow',fake)
    monkeypatch.setattr(nw,'decode_example',lambda raw:_dummy_state(raw.decode()))
    out=tmp_path/'cohort'
    manifest=build_cohort(tmp_path,out,split='validation',max_shards=0,max_records_per_shard=0,
            require_complete_split=True,expected_shards=2,keep_random=3,keep_interactive=2,
            keep_chain=2,keep_crossing=0,holdout_fraction=.4)
    assert manifest['coverage']['complete']
    assert manifest['counts']['examined']==6
    assert len(manifest['per_shard_counts'])==2
    assert all(e['selected_source_indices'][0]==2 for e in manifest['entries'])
    assert all(e['source_ego_index']==2 for e in manifest['entries'])
    assert all(('blind_holdout' in e['cohorts'])==(e['evaluation_partition']=='sealed_holdout')
               for e in manifest['entries'])
    assert any(e['evaluation_partition']=='sealed_holdout' for e in manifest['entries'])
    for e in manifest['entries']:
        with np.load(out/e['file']) as npz:
            assert int(npz['ego_index'])==0
            assert int(npz['agent_ids'][0])==12
    assert sum(int(x['records_processed']) for x in manifest['per_shard_counts'])==6


def test_consistent_raw_sdc_comparison(tmp_path):
    st=_dummy_state('demo')
    e={'scenario_id':'demo','selected_source_indices':[2,0,1],'source_ego_index':2}
    assert check_raw_state(e,[12,10,11],[0,0],st)['source_sdc_id']==12
    try:check_raw_state(e,[10,12,11],[0,0],st)
    except ValueError as exc:assert 'IDs' in str(exc)
    else:assert False
