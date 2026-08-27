from __future__ import annotations

import numpy as np
import torch

from cowp.external_baselines.adapters import make_external_batch
from cowp.external_baselines.pluto_cowp import COWPPLUTO, pluto_loss
from cowp.external_baselines.plant2_cowp import COWPPlanT2, plant2_loss
from cowp.external_baselines.waymax_policy import ExternalWaymaxPolicy


def _synthetic_batch(B=1, N=8, H=11, T=24, K=6):
    hist = torch.zeros(B, N, H, 11)
    for n in range(N):
        hist[:, n, :, 0] = torch.linspace(0, 1, H) + n * 4
        hist[:, n, :, 1] = n * 1.5
        hist[:, n, :, 3:6] = torch.tensor([4.8, 1.9, 1.6])
        hist[:, n, :, 7] = 1.0
        hist[:, n, :, 9] = 1.0
        hist[:, n, :, 10] = 1.0
    is_sdc = torch.zeros(B, N)
    is_sdc[:, 0] = 1
    fx = torch.zeros(B, N, T)
    fy = torch.zeros(B, N, T)
    for n in range(N):
        fx[:, n] = hist[:, n, -1, 0, None] + torch.arange(1, T + 1) * 0.1
        fy[:, n] = hist[:, n, -1, 1, None]
    cand = torch.zeros(B, K, T, 7)
    for k in range(K):
        cand[:, k, :, 0] = hist[:, 0, -1, 0, None] + torch.arange(1, T + 1) * 0.1 * (0.8 + 0.05 * k)
        cand[:, k, :, 3] = 0.8 + 0.05 * k
    rg = torch.zeros(B, 200, 3)
    rg[..., 0] = torch.linspace(-20, 100, 200)
    return {
        "state/history": hist,
        "state/is_sdc": is_sdc,
        "state/type": torch.ones(B, N),
        "state/future/x": fx,
        "state/future/y": fy,
        "state/future/valid": torch.ones(B, N, T),
        "cowp/candidates/trajectory": cand,
        "cowp/candidates/valid": torch.ones(B, K, dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones(B, K, dtype=torch.bool),
        "roadgraph_samples/xyz": rg,
        "roadgraph_samples/valid": torch.ones(B, 200, dtype=torch.bool),
        # Deliberately inject COWP-only labels; planner_inputs must not expose them.
        "cowp/candidates/false_safe": torch.ones(B, K, dtype=torch.bool),
        "cowp/witness/opr": torch.zeros(B, K, 2),
    }


def test_womd_adapter_excludes_cowp_mechanism_labels_and_runs_planners():
    T = 24
    batch = _synthetic_batch(T=T)
    cfg = {"limits": {"max_agents": 64}, "model": {"history_steps": 11}, "time": {"future_steps": T}}
    ext = make_external_batch(batch, cfg, device=torch.device("cpu"), max_neighbors=4, max_candidates=6, horizon=T)
    assert set(ext.planner_inputs) == {"agents", "agent_valid", "map_lanes", "map_lanes_valid", "route", "neighbors_future_xy", "neighbors_future_valid"}
    assert ext.ego_future_xy.shape == (1, T, 2)
    assert ext.origin.shape == (1, 2)
    assert ext.yaw0.shape == (1,)

    pluto = COWPPLUTO(future_len=T, d_model=32, num_heads=4, encoder_layers=1, lateral_queries=2, longitudinal_queries=2)
    loss, _ = pluto_loss(pluto, ext.planner_inputs, ext.ego_future_xy, ext.ego_future_valid, contrast_weight=0.0)
    assert torch.isfinite(loss)

    plant2 = COWPPlanT2(future_len=T, d_model=32, num_heads=4, layers=1)
    loss2, _ = plant2_loss(plant2, ext.planner_inputs, ext.ego_future_xy, ext.ego_future_valid)
    assert torch.isfinite(loss2)


def test_direct_local_waypoints_convert_to_global_contract():
    local = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    out = ExternalWaymaxPolicy._local_xy_to_global_traj(local, np.array([10.0, 20.0], dtype=np.float32), np.pi / 2, dt=0.1)
    assert out.shape == (3, 7)
    np.testing.assert_allclose(out[:, 0], 10.0, atol=1e-5)
    np.testing.assert_allclose(out[:, 1], [21.0, 22.0, 23.0], atol=1e-5)
    np.testing.assert_allclose(out[:, 2], np.pi / 2, atol=1e-5)
    assert np.isfinite(out).all()


def test_direct_adapter_uses_sdc_path_route_without_candidates_or_future():
    T = 24
    batch = _synthetic_batch(T=T)
    # Remove all training targets/proposals: direct Waymax inference must remain valid.
    for k in list(batch):
        if k.startswith("state/future/") or k.startswith("cowp/"):
            del batch[k]
    P, Q = 2, 40
    path_xyz = torch.zeros(1, P, Q, 3)
    path_xyz[0, 0, :, 0] = torch.linspace(0.0, 39.0, Q)
    path_xyz[0, 1, :, 1] = torch.linspace(0.0, 39.0, Q)
    batch["path_samples/xyz"] = path_xyz
    batch["path_samples/valid"] = torch.ones(1, P, Q, dtype=torch.bool)
    batch["path_samples/on_route"] = torch.tensor([[[True], [False]]])
    cfg = {"limits": {"max_agents": 64}, "model": {"history_steps": 11}, "time": {"future_steps": T}}
    ext = make_external_batch(
        batch, cfg, device=torch.device("cpu"), max_neighbors=4, max_candidates=6,
        horizon=T, baseline="pluto", require_candidates=False, require_future=False,
    )
    assert ext.candidates.shape[1] == 0
    assert ext.ego_future_xy.shape[1] == 0
    route = ext.planner_inputs["route"][0]
    valid = route[:, 3] > 0.5
    assert bool(valid.any())
    # Ego is at (0,0), yaw=0 in the synthetic batch, so the on-route x-path is retained.
    assert float(route[valid, 0].max()) > 20.0
    assert float(route[valid, 1].abs().max()) < 1e-4


def test_dtpp_vectorized_score_decoder_matches_reference_loop():
    from cowp.external_baselines.dtpp_cowp import ScoreDecoder

    torch.manual_seed(123)
    B, K, A, T = 2, 5, 4, 12
    dec = ScoreDecoder(variable_cost=True).eval()
    ego = torch.randn(B, K, T, 7)
    # columns used as speed / accel / curvature should stay in benign ranges
    ego[..., 3] = torch.rand(B, K, T) * 8.0
    ego[..., 4] = torch.randn(B, K, T) * 0.5
    ego[..., 5] = torch.randn(B, K, T) * 0.02
    enc = torch.randn(B, 256)
    agents = torch.randn(B, K, A, T, 3)
    states = torch.randn(B, A, 11)

    with torch.no_grad():
        got, weights = dec(ego, enc, agents, states, T)
        hard_all = dec.get_hardcoded_features(ego, T)
        ref_rows = []
        for i in range(K):
            latent = dec.get_latent_interaction_features(ego[:, i], agents[:, i], states, T)
            feat = torch.cat((hard_all[:, i], latent), dim=-1)
            score = -torch.sum(feat * weights, dim=-1)
            score += -10.0 * dec.calculate_collision(ego[:, i], agents[:, i], states, T)
            ref_rows.append(score)
        ref = torch.stack(ref_rows, dim=1)
        mask = torch.ne(ego.sum(-1).sum(-1), 0)
        ref = torch.where(mask, ref, torch.full_like(ref, -1e9))
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def _assert_tensor_dict_close(a, b):
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0.0, atol=0.0)


def test_baseline_specific_feature_pruning_preserves_consumed_tensors():
    T = 24
    batch = _synthetic_batch(T=T)
    cfg = {"limits": {"max_agents": 64}, "model": {"history_steps": 11}, "time": {"future_steps": T}}
    full = make_external_batch(
        batch, cfg, device=torch.device("cpu"), max_neighbors=4, max_candidates=6,
        horizon=T, baseline=None, require_candidates=True, require_future=True,
    )
    gf = make_external_batch(
        batch, cfg, device=torch.device("cpu"), max_neighbors=4, max_candidates=6,
        horizon=T, baseline="gameformer", require_candidates=False, require_future=True,
    )
    _assert_tensor_dict_close(full.gameformer_inputs, gf.gameformer_inputs)
    torch.testing.assert_close(full.ego_future_xy, gf.ego_future_xy)
    torch.testing.assert_close(full.neighbors_future_xy, gf.neighbors_future_xy)

    dtpp = make_external_batch(
        batch, cfg, device=torch.device("cpu"), max_neighbors=4, max_candidates=6,
        horizon=T, baseline="dtpp", require_candidates=True, require_future=True,
    )
    _assert_tensor_dict_close(full.dtpp_inputs, dtpp.dtpp_inputs)
    torch.testing.assert_close(full.candidates, dtpp.candidates)
    torch.testing.assert_close(full.dtpp_candidate_tree, dtpp.dtpp_candidate_tree)

    for name in ("pluto", "plant2"):
        direct = make_external_batch(
            batch, cfg, device=torch.device("cpu"), max_neighbors=4, max_candidates=6,
            horizon=T, baseline=name, require_candidates=False, require_future=True,
        )
        _assert_tensor_dict_close(full.planner_inputs, direct.planner_inputs)
        torch.testing.assert_close(full.ego_future_xy, direct.ego_future_xy)
