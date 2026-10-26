import json,numpy as np
from research_probe.verify_cohort_identity import verify


def test_provenance_pass_and_unpatched_fail(tmp_path):
    sid='synthetic_000';entry={'scenario_id':sid,'file':sid+'.npz','source_record_index':4,
        'source_shard':'v.tfrecord','selected_source_indices':[3,1,2],
        'features_from_current':{'neighbor_indices':[1,2]}}
    np.savez_compressed(tmp_path/entry['file'],agent_ids=np.array([13,11,12]),ego_index=np.array(0),
         current_x=np.array([[10.],[20.],[30.]]),current_y=np.array([[0.],[0.],[0.]]))
    (tmp_path/'cohort_manifest.json').write_text(json.dumps({'entries':[entry]}))
    assert verify(tmp_path)['pass']
    assert not verify(tmp_path,exclusion_ids={sid})['pass']
    del entry['source_record_index']
    (tmp_path/'cohort_manifest.json').write_text(json.dumps({'entries':[entry]}))
    result=verify(tmp_path)
    assert not result['pass'] and 'MISSING' in result['errors'][0]['error']
