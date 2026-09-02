from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np
import pytest
import torch
from ocrap.cli.train import _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_DIM,DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_SCHEMA,OPTION_FEATURE_DIM,direct_semantic_recovery_witness_features_from_sample,option_features_from_sample
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

def _layout(): return FlatFeatureLayout(feature_max_agents=2)
def _sample(*,x=18.0,acc_hist=(-2.0,)):
 ego=np.zeros(9,np.float32);ego[6]=4.;ego[7]=4.8;ego[8]=2.
 states=np.zeros((10,9),np.float32);states[:,0]=np.arange(1,11)*.4;states[:,6]=4.;states[:,7]=4.8;states[:,8]=2.
 controls=np.zeros((9,4),np.float32);H=len(acc_hist);hist=np.zeros((H,2,16),np.float32);valid=np.ones((H,2),bool)
 hist[:,1,0]=x;hist[:,1,5]=np.asarray(acc_hist,np.float32);hist[:,1,10]=4.8;hist[:,1,11]=2.
 return {'ego_state':ego,'prefix_states':states,'prefix_controls':controls,'agent_history':hist,'agent_valid':valid,'recovery_modes':np.asarray(['stop','lateral_escape'],object),'recovery_params':np.asarray([[-5.,5.,0.],[3.5,5.,1.5]],np.float32),'option_valid':np.asarray([1,1],bool),'prefix_macro_id':0,'prefix_macro_name':'candidate','prefix_param':np.zeros(0,np.float32),'utility':0.,'feasible':1.,'hard_violation':0.,'harm_proxy':0.}
def _cfg(*,boundary=False,history=False):
 return {'sample_rate_hz':10.,'recovery_horizon_s':4.,'prefix_horizon_s':1.,'route_dev_max_m':2.5,'control_limits':{'a_max':3.,'a_min':-6.,'delta_max':.55,'j_max':6.,'steer_rate_max':.5},'model':{'feature_max_agents':2,'direct_recovery_semantic_witness_route_alignment':True,'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,'direct_recovery_semantic_witness_boundary_transport':False,'direct_recovery_semantic_witness_projection_fidelity_weighting':True,'direct_recovery_semantic_witness_demand_normalized_fidelity':False,'direct_recovery_semantic_witness_robust_occupancy':False,'direct_recovery_semantic_witness_soft_occupancy_disagreement':False,'direct_recovery_semantic_witness_boundary_localized_occupancy_trust':boundary,'direct_recovery_semantic_witness_history_occupancy_reachability':history},'default_available_distance_m':60.}
def _model(*,boundary=False,history=False):
 L=_layout();return OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_active_set_alignment=True,direct_recovery_semantic_witness_path_stop_alignment=False,direct_recovery_semantic_witness_classlocal_transport=False,direct_recovery_semantic_witness_route_alignment=True,direct_recovery_semantic_witness_reentry_alignment=True,direct_recovery_semantic_witness_control_projection=True,direct_recovery_semantic_witness_boundary_transport=False,direct_recovery_semantic_witness_projection_fidelity_weighting=True,direct_recovery_semantic_witness_demand_normalized_fidelity=False,direct_recovery_semantic_witness_robust_occupancy=False,direct_recovery_semantic_witness_soft_occupancy_disagreement=False,direct_recovery_semantic_witness_boundary_localized_occupancy_trust=boundary,direct_recovery_semantic_witness_history_occupancy_reachability=history,direct_recovery_evidence_native_certificate_preservation=True)
def _force(m):
 m.root_logit_head.forward=MethodType(lambda self,z:torch.zeros((*z.shape[:-1],1),device=z.device,dtype=z.dtype),m.root_logit_head);m.obs_embed_head.forward=MethodType(lambda self,z:torch.zeros((*z.shape[:-1],8),device=z.device,dtype=z.dtype),m.obs_embed_head)
 def mf(self,z):return torch.zeros((z.shape[0],3,2,1),device=z.device,dtype=z.dtype)
 m.margin_head.forward=MethodType(mf,m.margin_head)
def _support(m,feat):
 _force(m);x=torch.zeros((1,_layout().total_dim));opt=torch.from_numpy(option_features_from_sample(_sample())).float().unsqueeze(0);rv=torch.ones((1,3),dtype=torch.bool);ov=torch.ones((1,2),dtype=torch.bool);out=m._direct_semantic_witness_absolute_feasibility(m._scene_tokens(x),x,opt,feat,root_valid=rv,option_valid=ov);return out[4],out[5]
def _feat18(cur=(0.,1.),hist_raw=(0.,1.),hist_bd=(0.,2.)):
 f=torch.full((1,2,18),.6);f[...,4]=-float(np.tanh(1.));f[...,8]=.3;f[...,9]=.3;f[...,11]=0.;f[...,12]=.6;f[...,13]=.6
 for j in range(2):f[0,j,15]=float(np.tanh(cur[j]));f[0,j,16]=float(np.tanh(hist_raw[j]));f[0,j,17]=float(np.tanh(hist_bd[j]))
 return f

def test_v4871_schema7_contract_all_factorial_arms():
 base={'direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_route_alignment':True,'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,'direct_recovery_semantic_witness_projection_fidelity_weighting':True}
 for boundary,history in ((True,False),(False,True),(True,True)):
  assert _semantic_witness_checkpoint_feature_contract({**base,'direct_recovery_semantic_witness_boundary_localized_occupancy_trust':boundary,'direct_recovery_semantic_witness_history_occupancy_reachability':history})==(7,'boundary_localized_history_reachability_projected_recovery_witness')
 assert DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_SCHEMA==7 and DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_DIM==18

def test_v4871_first14_coordinates_are_execution_exact_historical_cv():
 d=_sample(acc_hist=(-2.,1.,-1.,.5));base=direct_semantic_recovery_witness_features_from_sample(d,_cfg(),num_options=2)
 for boundary,history in ((True,False),(False,True),(True,True)):
  f=direct_semantic_recovery_witness_features_from_sample(d,_cfg(boundary=boundary,history=history),num_options=2);assert f.shape==(2,18);assert np.array_equal(base,f[:,:14])

def test_v4871_boundary_localization_does_not_penalize_far_safe_disagreement():
 f=direct_semantic_recovery_witness_features_from_sample(_sample(x=80.,acc_hist=(-2.,)),_cfg(boundary=True),num_options=2)
 assert np.any(f[:,14]>0.9)  # raw CV-vs-CA disagreement is very large
 assert np.all(f[:,15]==0.0) # but the alternate future never threatens the clearance boundary

def test_v4871_history_tube_separates_raw_uncertainty_from_boundary_exposure():
 f=direct_semantic_recovery_witness_features_from_sample(_sample(x=80.,acc_hist=(-2.,1.,-1.,.5)),_cfg(boundary=True,history=True),num_options=2)
 assert np.any(f[:,16]>0.9);assert np.all(f[:,17]==0.0)
 near=direct_semantic_recovery_witness_features_from_sample(_sample(x=18.,acc_hist=(-2.,1.,-1.,.5)),_cfg(boundary=True,history=True),num_options=2)
 assert np.any(near[:,17]>0.0)

def test_v4871_factorial_model_selects_expected_trust_coordinate_and_keeps_sign():
 feat=_feat18();h=_model(boundary=True,history=False).eval();j=_model(boundary=False,history=True).eval();k=_model(boundary=True,history=True).eval();vh,sh=_support(h,feat);vj,sj=_support(j,feat);vk,sk=_support(k,feat)
 assert torch.equal(vh>0,vj>0) and torch.equal(vh>0,vk>0)
 # projection fidelity contributes 1/2; option1 H risk=1, J risk=1, K risk=2.
 assert torch.allclose(sh[...,1],torch.tensor([[.25]]),atol=2e-5);assert torch.allclose(sj[...,1],torch.tensor([[.25]]),atol=2e-5);assert torch.allclose(sk[...,1],torch.tensor([[1/6]]),atol=2e-5)

def test_v4871_rejects_stacking_rejected_or_previous_occupancy_mechanisms():
 L=_layout();kw=dict(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_control_projection=True,direct_recovery_semantic_witness_projection_fidelity_weighting=True,direct_recovery_semantic_witness_boundary_localized_occupancy_trust=True)
 with pytest.raises(ValueError,match='replaces'):
  OCRAPModel(**kw,direct_recovery_semantic_witness_soft_occupancy_disagreement=True)
 with pytest.raises(ValueError,match='rejected hard'):
  OCRAPModel(**kw,direct_recovery_semantic_witness_robust_occupancy=True)

def test_v4871_runner_is_preregistered_2x2_and_boundary_transport_off():
 text=(Path(__file__).resolve().parents[1]/'scripts/run_v48_71_dcp_drfc_bcde_rifa_borw_two_gpu.sh').read_text();assert 'H71_BOUNDARY_LOCAL' in text and 'J71_HISTORY_TUBE' in text and 'K71_Main_OCBORW' in text;assert 'SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false' in text;assert 'SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false' in text;assert 'SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false' in text;assert 'SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false' in text;assert 'PROPOSAL_TOP_K=5' in text and 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' in text
