from ride.v13_aggregate import aggregate_reports
from ride.v13_model_transfer import VERSION
import pytest


def report(x,seed,status=True):
    return {'protocol':VERSION,'source_family':'behavior','target_family':'tm','mode':'POSTHOC_RETROSPECTIVE_DIAGNOSTIC_NOT_PRECOMMITTED',
            'locked_params':{'threshold':0.25},
            'frozen_context':{'map':'Carla/Maps/Town04','anchors':[{'anchor':0,'key':[x,0.,0.]}]},
            'results':[{'anchor':'0','seed':str(seed),'scientific_status':'AVOIDABLE_C_BURDEN_LOSS' if status else 'UNKNOWN',
                        'paired_model_induced_selection_error':status}]}


def test_clusters_overlapping_contexts():
    res=aggregate_reports([report(0,23),report(3.5,41),report(200,73,False)])
    assert res['summary']['physical_corridor_clusters']==2
    assert res['summary']['corridors_with_diagnostic_model_error']==1


def test_reject_mixed_prospective_and_retrospective():
    r1=report(0,23);r2=report(2,41)
    r2['mode']='PROSPECTIVE_SOURCE_SIMULATOR_LOOKAHEAD'
    with pytest.raises(ValueError,match='pool'):
        aggregate_reports([r1,r2])


def test_endpoint_mismatch_rejected():
    r1=report(0,23);r2=report(5,41)
    r2['locked_params']={'threshold':0.1}
    with pytest.raises(ValueError,match='endpoint'):
        aggregate_reports([r1,r2])
