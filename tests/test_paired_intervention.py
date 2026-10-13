import numpy as np
from research_probe.reactive_stress import simulate
from research_probe.paired_evidence import summarize


def test_ghost_ego_keeps_ego_physics_but_changes_follower_response():
    state=np.array([[15.,0.,3.,0.,0.],[0.,0.,10.,0.,0.],[60.,0.,7.,0.,0.]])
    L=np.full(3,4.5);W=np.full(3,1.9);valid=np.ones(3,bool)
    a=simulate(state,L,W,valid,0,[0,1,2],-3.,'cautious',steps=7)
    b=simulate(state,L,W,valid,0,[0,1,2],-3.,'cautious',steps=7,ghost_ego_to_others=True)
    assert a['ego_progress_m']==b['ego_progress_m']
    assert a['accelerations'][0,1]<b['accelerations'][0,1]
    assert 'non_ego_collision_first_step' in a


def test_pair_sensitive_requires_action_specific_pair_and_safe_matched_control():
    def a(acc,non=False,coll=False,progress=10):
        return {'collision':coll,'ego_collision':False,'non_ego_collision':non,
           'non_ego_collision_pairs':[[1,2]] if non else [], 'ego_progress_m':progress,
           'near_term_ego_clear':True,'ego_acceleration_mps2':acc}
    actions=[a(-3,True,True,9),a(0,False,False,10)]
    sc={'rows':[{'scenario_id':'x','eligible':True,'stable_intervention_proxy_actions':[],
               'regimes':{'cautious':{'actions':actions}}}], 'errors':[]}
    out=summarize(sc)
    assert out['regimes']['cautious']['scenes_with_specific_pair_variation']==1
    assert out['regimes']['cautious']['scenes_with_symmetric_progress_matched_pair_sensitive_preventability']==1
    sc['rows'][0]['regimes']['cautious']['actions'][1]=a(0,True,True,10)
    out=summarize(sc)
    assert out['regimes']['cautious']['scenes_with_specific_pair_variation']==0
    assert out['regimes']['cautious']['scenes_with_symmetric_progress_matched_pair_sensitive_preventability']==0


def test_noninferior_but_not_similar_progress_is_not_matched():
    def act(acc, collision, progress):
        return {'collision':collision,'ego_collision':False, 'non_ego_collision':collision,
                'non_ego_collision_pairs':[[1,2]] if collision else [],
                'ego_progress_m':progress,'near_term_ego_clear':True,
                'ego_acceleration_mps2':acc}
    sc={'rows':[{'scenario_id':'s','eligible':True, 'regimes':{'test':{'actions':[
        act(-3,True,30),act(-1.5,False,42)]}}}], 'errors':[]}
    res=summarize(sc,tolerance=3)
    assert res['regimes']['test']['scenes_with_progress_noninferior_pair_sensitive_preventability']==1
    assert res['regimes']['test']['scenes_with_symmetric_progress_matched_pair_sensitive_preventability']==0
