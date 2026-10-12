import json
import numpy as np
from research_probe.reactive_stress import analyze_scene, run

def test_reactive_pipeline_on_synthetic_cohort(tmp_path):
    co=tmp_path/'cohort';co.mkdir()
    n=3;p=np.array([0.,28.,60.])
    d={'agent_ids':np.array([1,2,3]),'types':np.ones(n,int),'ego_index':np.array(0)}
    for phase,times in [('past',10),('current',1),('future',80)]:
        t={'past':np.arange(-10,0)*.1,'current':np.array([0.]),'future':np.arange(1,81)*.1}[phase]
        d.update({phase+'_x':p[:,None]+np.array([8.,6.,5.])[:,None]*t[None,:],
                  phase+'_y':np.zeros((n,times)),phase+'_velocity_x':np.repeat(np.array([[8.],[6.],[5.]]),times,axis=1),
                  phase+'_velocity_y':np.zeros((n,times)),phase+'_bbox_yaw':np.zeros((n,times)),
                  phase+'_length':np.full((n,times),4.5),phase+'_width':np.full((n,times),1.9),
                  phase+'_valid':np.ones((n,times))})
    np.savez(co/'test.npz',**d)
    (co/'cohort_manifest.json').write_text(json.dumps({'entries':[{'scenario_id':'test','file':'test.npz'}]}))
    result=analyze_scene(co/'test.npz')
    assert result['eligible']
    assert 'cautious' in result['regimes']
    summary=run(co,tmp_path/'out.json',10,3.,4.)
    assert summary['n_eligible']==1
    assert summary['scientific_claim_allowed'] is False
