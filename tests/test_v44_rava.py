from pathlib import Path

import numpy as np
import torch

from ocrap.cli import train as train_mod
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select
from ocrap.simulation import closed_loop_runner as clr


def test_group_sampler_never_merges_same_scene_time_across_buckets(monkeypatch):
    class DS:
        paths = [
            Path('/tmp/train_near_contact/samples/a.npz'),
            Path('/tmp/train_contact/samples/b.npz'),
        ]

    def metadata(path, key, default=None):
        values = {
            'scene_id': 'shared_scene', 'time_index': 7,
            'i_art_star': 0.0, 'r_dep_star': 0.1,
        }
        return values.get(key, default)

    monkeypatch.setattr(train_mod, 'scalar_metadata_for_path', metadata)
    sampler = train_mod._make_group_batch_sampler(
        DS(), {'training': {'group_batching': True, 'group_batching_replacement': False}}, batch_size=8
    )
    assert sampler is not None
    assert sorted(sorted(g) for g in sampler.groups) == [[0], [1]]


def test_shared_value_head_does_not_receive_oracle_bucket_when_conditioning_disabled():
    layout = FlatFeatureLayout(feature_max_agents=2)
    model = OCRAPModel(
        input_dim=layout.total_dim,
        num_roots=2,
        num_options=3,
        d_model=16,
        d_obs=8,
        encoder_type='structured_transformer',
        feature_layout=layout.__dict__,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling='candidate_concat',
        direct_recovery_value_output='score',
        direct_recovery_value_regime_conditioning=False,
        direct_recovery_opportunity_head=True,
    ).eval()
    x = torch.randn(1, layout.total_dim).repeat(2, 1)
    with torch.no_grad():
        out = model(x, bucket_id=torch.tensor([1, 2]))
    assert torch.allclose(out['direct_recovery_value_logit'][0], out['direct_recovery_value_logit'][1])
    assert torch.allclose(out['direct_recovery_opportunity_logit'][0], out['direct_recovery_opportunity_logit'][1])


def _select(opportunity: float):
    return calibrated_constrained_select(
        utility=np.array([1.0, 0.05]),
        r_dep=np.array([0.5, -1.0]),
        hard=np.zeros(2), harm=np.zeros(2), feasible=np.ones(2, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.zeros(2), pred_drs=np.ones(2),
        nominal_deviation=np.array([0.0, 0.02]),
        pred_direct_value=np.array([0.0, -0.05]),
        pred_direct_std=np.zeros(2),
        pred_direct_opportunity=np.array([0.0, opportunity]),
        candidate_macro_names=['nominal', 'merge'],
        regime_name='near_contact',
        direct_value_certificate=True,
        direct_value_macro_allowlist='merge',
        direct_value_uncertainty_mode='risk_selective',
        direct_value_min_advantage_lcb=-0.10,
        direct_value_opportunity_threshold=0.80,
        direct_value_score_mode=True,
        direct_value_top1_only=True,
        direct_value_risk_controlled_admission=True,
        direct_value_challenge_nominal=True,
        direct_value_bonus=0.20,
        stress_rescue_challenge_nominal=True,
    )


def test_opportunity_gate_abstains_below_threshold_and_executes_verified_negative_score_rule():
    assert _select(0.79).selected_index == 0
    selected = _select(0.90)
    assert selected.selected_index == 1
    assert 'direct_value' in selected.reason


def test_observable_regime_router_uses_current_geometry_only(monkeypatch):
    cfg = {
        'selection': {'auto_regime_from_observation': True},
        'regime_thresholds': {'tau_contact': 0.8, 'tau_d': 2.0, 'tau_ttc': 3.0},
    }
    monkeypatch.setattr(clr, '_state_geometry_metrics', lambda *_: {'min_clearance_m': 0.4, 'ttc_s': 5.0})
    assert clr._observable_regime_name(object(), 0, cfg, fallback='safe') == 'contact'
    monkeypatch.setattr(clr, '_state_geometry_metrics', lambda *_: {'min_clearance_m': 1.5, 'ttc_s': 5.0})
    assert clr._observable_regime_name(object(), 0, cfg, fallback='safe') == 'near_contact'
    monkeypatch.setattr(clr, '_state_geometry_metrics', lambda *_: {'min_clearance_m': 5.0, 'ttc_s': 8.0})
    assert clr._observable_regime_name(object(), 0, cfg, fallback='contact') == 'safe'
