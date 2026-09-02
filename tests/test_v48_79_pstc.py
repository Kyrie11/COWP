from __future__ import annotations
import json
import numpy as np
import pytest
import torch

from ocrap.algorithms.ocmero import oc_mero
from ocrap.algorithms.lcv import weighted_lcvar
from ocrap.cli.train import _absolute_feasibility_supervision_loss, _absolute_feasibility_supervision_mask
from ocrap.models.data import _load_absolute_truth_index
from ocrap.v48_79_truth_contract import (
    STRUCT_HIDDEN_BRANCH, STRUCT_RECOVERY_FLOOR, nested_tail_truth_contract,
    structural_root_option_reason_bits, weighted_lcvar_influence_np,
)


def _sample(M, modes, metas, C=None):
    M=np.asarray(M,dtype=np.float32);K,L=M.shape;p=np.ones(K,dtype=np.float32)/K
    C=np.ones((K,K),dtype=np.float32) if C is None else np.asarray(C,dtype=np.float32)
    res=oc_mero(M,p,C,alpha=.2,beta=.2,option_valid=np.ones(L,bool),root_valid=np.ones(K,bool),top_m=8)
    return {'m_star':M,'root_probs':p,'root_valid':np.ones(K,bool),'c_star':C,'option_valid':np.ones(L,bool),
            'root_assignments':np.arange(K,dtype=np.int64),'future_metadata':np.asarray(json.dumps(metas)),
            'recovery_modes':np.asarray(modes),'r_dep_star':np.float32(res.r_dep)}


def test_v4879_numpy_lcvar_influence_is_exact():
    s=np.asarray([-2.,0.,3.]);w=np.asarray([.2,.5,.3]);a=.6
    inf=weighted_lcvar_influence_np(s,w,a)
    assert abs(float(inf.sum())-1.0) < 1e-12
    assert abs(float((inf*s).sum())-weighted_lcvar(s,w,a)) < 1e-12


def test_v4879_clean_nested_tail_is_physically_identifiable():
    d=_sample([[.4,.2],[.3,.1]],['stop','brake_lane'],[{},{}])
    r=nested_tail_truth_contract(d)
    assert r.valid and r.physical_identifiable
    assert r.structural_exposure_mass == 0.0
    assert r.r_dep_abs_error <= 1e-5


def test_v4879_structural_rule_only_censors_when_selected_tail_exposed():
    exposed=_sample([[.8,-1.0],[.8,-1.0]],['yield_rejoin','stop'],[{},{}])
    re=nested_tail_truth_contract(exposed)
    assert re.valid and not re.physical_identifiable and re.structural_exposure_mass > 0
    assert re.structural_reason_mass['recovery_mode_floor_0p6'] > 0
    off=_sample([[-5.0,.8],[-5.0,.7]],['yield_rejoin','stop'],[{},{}])
    ro=nested_tail_truth_contract(off)
    assert ro.valid and ro.physical_identifiable and ro.structural_exposure_mass == 0.0


def test_v4879_hidden_branch_is_fail_closed_structural():
    d=_sample([[1.2,-6.0],[1.2,-6.0]],['yield_rejoin','stop'],[{'artifact_branch':'yield'},{'artifact_branch':'yield'}])
    bits=structural_root_option_reason_bits(d)
    assert np.all((bits & STRUCT_HIDDEN_BRANCH) != 0)
    # Hidden branch suppresses the generic 0.6 floor flag, matching teacher code.
    assert np.all((bits[:,0] & STRUCT_RECOVERY_FLOOR) == 0)
    r=nested_tail_truth_contract(d)
    assert r.valid and not r.physical_identifiable


def test_v4879_supervision_censors_structural_tail_not_exact_numeric_floor():
    batch={'r_dep_star':torch.tensor([.5,.2,-.7]),'is_nominal':torch.zeros(3),'bucket_id':torch.tensor([1,1,2]),
           'time_index':torch.zeros(3,dtype=torch.long),'absolute_truth_physical_identifiable':torch.tensor([1.,0.,1.])}
    cfg={'direct_value_absolute_feasibility_truth_contract':'censor_structural_tail','direct_value_absolute_feasibility_supervision_objective':'signed_margin_huber'}
    mask,_target,censored=_absolute_feasibility_supervision_mask(batch,cfg)
    assert mask.tolist()==[True,False,True]
    assert censored.tolist()==[False,True,False]
    # Exact 0.5 survives if its active teacher tail is physical-identifiable.
    out={'direct_recovery_absolute_feasibility_logit':torch.tensor([.4,99.,-.2])}
    got=_absolute_feasibility_supervision_loss(out,batch,cfg)
    exp=torch.nn.functional.smooth_l1_loss(torch.tensor([.4,-.2]),torch.tensor([.5,-.7]),beta=1.0)
    assert torch.equal(got,exp)


def test_v4879_truth_index_loader_is_fail_closed(tmp_path):
    p=tmp_path/'truth.jsonl'; sample=tmp_path/'x.npz'; sample.write_bytes(b'x')
    row={'sample_path':str(sample),'valid':True,'physical_identifiable':True,'structural_exposure_mass':0.0}
    p.write_text(json.dumps(row)+'\n')
    d=_load_absolute_truth_index(str(p)); assert str(sample.resolve()) in d
    p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    _load_absolute_truth_index.cache_clear()
    with pytest.raises(ValueError,match='duplicate sample_path'):
        _load_absolute_truth_index(str(p))
