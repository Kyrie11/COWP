from __future__ import annotations
import numpy as np
import torch

from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success
from ocrap.models.losses import _exact_teacher_shared_success, direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select


def test_exact_teacher_shared_success_matches_numpy_contract():
    q=torch.tensor([[[0.2,-0.4],[0.1,0.8]]],dtype=torch.float32)
    m=torch.tensor([[[-1.0,1.0],[1.0,-1.0]]],dtype=torch.float32)
    p=torch.tensor([[0.7,0.3]],dtype=torch.float32)
    rv=torch.ones((1,2),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    got=float(_exact_teacher_shared_success(q,m,p,rv,ov,gamma=0.0).item())
    opt=best_shared_option_index(q[0].numpy(),p[0].numpy(),gamma=0.0,root_valid=rv[0].numpy(),option_valid=ov[0].numpy())
    expected=deployable_recovery_success(m[0].numpy(),p[0].numpy(),opt,rv[0].numpy())
    assert abs(got-expected)<1e-7


def test_preference_head_is_exact_warm_start_identity():
    torch.manual_seed(4)
    model=OCRAPModel(input_dim=12,num_roots=2,num_options=3,d_model=8,d_obs=4,encoder_type='mlp',dropout=0.0,direct_recovery_value_head=True,direct_recovery_value_output='score',direct_recovery_preference_head=True).eval()
    with torch.no_grad():
        out=model(torch.randn(4,12),direct_only=True)
    assert torch.allclose(out['direct_recovery_rank_logit'],out['direct_recovery_value_logit'],atol=1e-8)
    assert torch.count_nonzero(out['direct_recovery_rank_residual'])==0


def _preference_loss(rank):
    pred=torch.zeros(3)
    return direct_uncertainty_recovery_value_loss(
        pred_logit=pred,pred_logvar=torch.zeros(3),pred_rank_logit=torch.tensor(rank,dtype=torch.float32),
        teacher_r_dep=torch.tensor([-2.0,2.0,0.5]),teacher_r_orc=torch.tensor([-2.0,2.0,0.5]),
        teacher_q=torch.ones((3,1,1)),teacher_m_star=torch.tensor([[[-1.0]],[[1.0]],[[1.0]]]),
        root_probs=torch.ones((3,1)),root_valid=torch.ones((3,1),dtype=torch.bool),option_valid=torch.ones((3,1),dtype=torch.bool),
        scene_hash=torch.tensor([1,1,1]),time_index=torch.tensor([2,2,2]),macro_type_id=torch.tensor([0,5,5]),is_nominal=torch.tensor([1.0,0.0,0.0]),bucket_id=torch.tensor([2,2,2]),
        macro_ids=(5,),bucket_ids=(2,),output_mode='score',exact_teacher_pcd=True,positive_gain=0.01,negative_gain=0.01,
        point_weight=0.0,centered_weight=0.0,listwise_weight=0.0,advantage_weight=0.0,pairwise_weight=0.0,top_rank_weight=0.0,opportunity_weight=0.0,harm_weight=0.0,setwise_admission_weight=0.0,policy_distill_weight=0.0,policy_regret_weight=0.0,
        preference_weight=2.0,preference_regret_weight=1.0,preference_min_gap=0.001,preference_margin=0.02,preference_temperature=0.05,
    )


def test_ecpr_prefers_exact_teacher_best_recovery():
    correct=_preference_loss([0.0,1.5,0.2])
    wrong=_preference_loss([0.0,0.2,1.5])
    assert correct.item()+0.05 < wrong.item()


def test_selector_uses_rank_for_top1_and_value_for_admission():
    sel=calibrated_constrained_select(
        utility=np.array([1.0,0.9,0.85]),r_dep=np.array([0.2,0.2,0.2]),hard=np.zeros(3),harm=np.zeros(3),feasible=np.ones(3,dtype=bool),gamma_rec=0.0,
        pred_gap=np.zeros(3),pred_drs=np.ones(3),nominal_deviation=np.array([0.0,0.02,0.02]),
        pred_direct_value=np.array([0.0,0.8,0.7]),pred_direct_rank=np.array([0.0,0.1,0.9]),pred_direct_std=np.zeros(3),pred_direct_opportunity=np.array([0.0,0.9,0.9]),pred_direct_harm=np.zeros(3),candidate_macro_names=['nominal','yield','brake'],
        regime_name='contact',direct_value_certificate=True,direct_value_macro_allowlist='yield,brake',direct_value_score_mode=True,direct_value_uncertainty_mode='risk_controlled',direct_value_min_advantage_lcb=0.5,direct_value_opportunity_threshold=0.5,direct_value_harm_threshold=0.2,direct_value_top1_only=True,direct_value_challenge_nominal=True,direct_value_bonus=1.0,stress_rescue_challenge_nominal=True,
    )
    assert sel.selected_index==2
