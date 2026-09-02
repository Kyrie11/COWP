from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.models.data import (
    DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(privileged: bool = False):
    ego = np.zeros(9, np.float32)
    ego[6] = 4.0
    ego[7] = 4.8
    ego[8] = 2.0
    states = np.zeros((10, 9), np.float32)
    states[:, 0] = np.arange(1, 11) * 0.4
    states[:, 6] = 4.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((9, 4), np.float32)
    hist = np.zeros((1, 2, 16), np.float32)
    hist[0, 1, 0] = 30.0
    hist[0, 1, 10] = 4.8
    hist[0, 1, 11] = 2.0
    d = {
        "ego_state": ego,
        "prefix_states": states,
        "prefix_controls": controls,
        "agent_history": hist,
        "agent_valid": np.asarray([[1, 1]], bool),
        "recovery_modes": np.asarray(["stop", "lateral_escape"], object),
        "recovery_params": np.asarray([[-5.0, 5.0, 0.0], [3.5, 5.0, 1.5]], np.float32),
        "option_valid": np.asarray([1, 1], bool),
        "prefix_macro_id": 0,
        "prefix_macro_name": "candidate",
        "prefix_param": np.zeros(0, np.float32),
        "utility": 0.0,
        "feasible": 1.0,
        "hard_violation": 0.0,
        "harm_proxy": 0.0,
    }
    if privileged:
        d.update({
            "m_star": np.ones((3, 2), np.float32) * 99.0,
            "root_future_signature": np.ones((3, 8), np.float32) * 77.0,
            "r_dep_star": np.float32(-999.0),
            "bucket_id": np.int64(2),
        })
    return d


def _cfg(*, route=False, reentry=False):
    return {
        "sample_rate_hz": 10.0,
        "recovery_horizon_s": 4.0,
        "route_dev_max_m": 2.5,
        "model": {
            "feature_max_agents": 2,
            "direct_recovery_semantic_witness_route_alignment": route,
            "direct_recovery_semantic_witness_reentry_alignment": reentry,
        },
        "default_available_distance_m": 60.0,
    }


def _model(*, route=False, reentry=False):
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout=asdict(L),
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_active_set_alignment=True,
        direct_recovery_semantic_witness_path_stop_alignment=False,
        direct_recovery_semantic_witness_classlocal_transport=False,
        direct_recovery_semantic_witness_route_alignment=route,
        direct_recovery_semantic_witness_reentry_alignment=reentry,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _opt(batch=1):
    z = torch.from_numpy(option_features_from_sample(_sample())).float()
    return z.unsqueeze(0).repeat(batch, 1, 1)


def _force_common_support(m):
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype),
        m.root_logit_head,
    )
    m.obs_embed_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 8), device=z.device, dtype=z.dtype),
        m.obs_embed_head,
    )

    def margins(self, z):
        vals = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
            device=z.device,
            dtype=z.dtype,
        )
        return vals.unsqueeze(-1).expand(z.shape[0], -1, -1, 1)

    m.margin_head.forward = MethodType(margins, m.margin_head)


def _manual_features(*, route0=0.6, route1=0.6, reentry0=0.6, reentry1=0.6, batch=1):
    # v48.64 first 12 coordinates + v48.66 route/re-entry tail.
    f = torch.full((batch, 2, 14), 0.5, dtype=torch.float32)
    f[..., 8] = 0.2  # clearance floor gain
    f[..., 9] = 0.2  # stability floor gain
    f[..., 11] = 0.0 # stability inactive
    f[:, 0, 12] = route0
    f[:, 1, 12] = route1
    f[:, 0, 13] = reentry0
    f[:, 1, 13] = reentry1
    return f


def test_acrw_schema2_is_privilege_free_and_preserves_v4865_prefix_exactly():
    legacy = direct_semantic_recovery_witness_features_from_sample(
        _sample(), _cfg(route=False, reentry=False), num_options=2
    )
    a = direct_semantic_recovery_witness_features_from_sample(
        _sample(), _cfg(route=True, reentry=True), num_options=2
    )
    b = direct_semantic_recovery_witness_features_from_sample(
        _sample(True), _cfg(route=True, reentry=True), num_options=2
    )
    assert DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA == 2
    assert legacy.shape == (2, 12) and a.shape == (2, 14)
    assert np.array_equal(a[:, :12], legacy)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all() and np.all(np.abs(a[:, 12:14]) <= 1.0)


def test_acrw_route_factor_is_noncompensatory_and_isolated():
    L = _layout(); x = torch.zeros((1, L.total_dim)); feat = _manual_features(route0=-0.4, route1=0.6)
    on = _model(route=True, reentry=False).eval(); _force_common_support(on)
    off = _model(route=False, reentry=False).eval(); _force_common_support(off)
    a = on._direct_semantic_witness_absolute_feasibility(on._scene_tokens(x), x, _opt(), feat,
        root_valid=torch.ones((1,3),dtype=torch.bool), option_valid=torch.ones((1,2),dtype=torch.bool))
    b = off._direct_semantic_witness_absolute_feasibility(off._scene_tokens(x), x, _opt(), feat[..., :12],
        root_valid=torch.ones((1,3),dtype=torch.bool), option_valid=torch.ones((1,2),dtype=torch.bool))
    assert a is not None and b is not None
    assert float(a[4][0,0]) < 0.0 and float(b[4][0,0]) > 0.0
    assert a[10].shape[-1] == 5 and b[10].shape[-1] == 4


def test_acrw_reentry_factor_rejects_post_reentry_secondary_deterioration():
    L = _layout(); x = torch.zeros((1, L.total_dim)); feat = _manual_features(reentry0=-0.3, reentry1=0.6)
    on = _model(route=False, reentry=True).eval(); _force_common_support(on)
    off = _model(route=False, reentry=False).eval(); _force_common_support(off)
    a = on._direct_semantic_witness_absolute_feasibility(on._scene_tokens(x), x, _opt(), feat,
        root_valid=torch.ones((1,3),dtype=torch.bool), option_valid=torch.ones((1,2),dtype=torch.bool))
    b = off._direct_semantic_witness_absolute_feasibility(off._scene_tokens(x), x, _opt(), feat[..., :12],
        root_valid=torch.ones((1,3),dtype=torch.bool), option_valid=torch.ones((1,2),dtype=torch.bool))
    assert a is not None and b is not None
    assert float(a[4][0,0]) < 0.0 and float(b[4][0,0]) > 0.0
    assert a[10].shape[-1] == 5 and b[10].shape[-1] == 4


def test_acrw_zero_gain_is_execution_exact_native_b():
    torch.manual_seed(4866)
    m = _model(route=True, reentry=True).eval()
    L = _layout(); x = torch.randn((3, L.total_dim)); mem = m._scene_tokens(x)
    rv = torch.ones((3,3),dtype=torch.bool); ov = torch.ones((3,2),dtype=torch.bool)
    _, native = m._direct_recovery_option_compatibility_evidence(mem, x, _opt(3), root_valid=rv, option_valid=ov)
    feat0 = torch.from_numpy(direct_semantic_recovery_witness_features_from_sample(
        _sample(), _cfg(route=True,reentry=True), num_options=2)).float()
    feat = feat0.unsqueeze(0).repeat(3,1,1)
    out = m._direct_semantic_witness_absolute_feasibility(mem, x, _opt(3), feat, root_valid=rv, option_valid=ov)
    assert out is not None and torch.equal(out[3], torch.zeros(2))
    assert torch.allclose(out[1], native[:,1], atol=0.0, rtol=0.0)


def test_acrw_bce_gradient_stays_on_two_shared_gains():
    torch.manual_seed(4866)
    m = _model(route=True, reentry=True).train(); _force_common_support(m)
    for n,p in m.named_parameters(): p.requires_grad_(n == "direct_absolute_semantic_witness_gain")
    L = _layout(); x = torch.randn((4,L.total_dim))
    out = m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x), x, _opt(4), _manual_features(batch=4),
        root_valid=torch.ones((4,3),dtype=torch.bool), option_valid=torch.ones((4,2),dtype=torch.bool))
    assert out is not None
    loss = _absolute_feasibility_bce(
        {"direct_recovery_absolute_feasibility_logit": out[0]},
        {"r_dep_star": torch.tensor([-.5,.5,-.5,.5]), "is_nominal": torch.zeros(4),
         "bucket_id": torch.tensor([1,1,2,2]), "time_index": torch.arange(4)},
    )
    loss.backward(); g = m.direct_absolute_semantic_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g != 0)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 2


def test_acrw_checkpoint_schema2_and_factor_roundtrip(tmp_path):
    from ocrap.models.inference import load_model_bundle
    m = _model(route=True,reentry=True).eval(); L = _layout()
    model_cfg = {
        'transformer_layers':1,'transformer_heads':4,'dropout':0.0,'encoder_type':'structured_transformer',
        'option_feature_dim':OPTION_FEATURE_DIM,'direct_recovery_value_head':True,
        'direct_recovery_absolute_semantic_witness_correction':True,
        'direct_recovery_semantic_witness_active_set_alignment':True,
        'direct_recovery_semantic_witness_path_stop_alignment':False,
        'direct_recovery_semantic_witness_classlocal_transport':False,
        'direct_recovery_semantic_witness_route_alignment':True,
        'direct_recovery_semantic_witness_reentry_alignment':True,
        'direct_recovery_evidence_native_certificate_preservation':True,
    }
    ckpt = {
        'model_state':m.state_dict(),'input_dim':L.total_dim,'num_roots':3,'num_options':2,
        'd_model':16,'d_obs':8,'tau_obs':1.0,'encoder_type':'structured_transformer',
        'feature_layout':asdict(L),'d_signature':0,'d_future_signature':0,'option_feature_dim':OPTION_FEATURE_DIM,
        **model_cfg,
        'direct_recovery_absolute_semantic_witness_feature_schema':2,
        'direct_recovery_absolute_semantic_witness_feature_source':'active_constraint_coverage_common_executable_recovery_witness',
        'cfg':{'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':model_cfg,'runtime':{'device':'cpu'}},
    }
    p = tmp_path/'acrw.pt'; torch.save(ckpt,p); b = load_model_bundle(p)
    assert b.model.direct_recovery_semantic_witness_route_alignment is True
    assert b.model.direct_recovery_semantic_witness_reentry_alignment is True
    assert b.model.direct_recovery_semantic_witness_classlocal_transport is False
