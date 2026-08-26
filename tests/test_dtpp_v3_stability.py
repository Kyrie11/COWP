from __future__ import annotations

import torch

from cowp.external_baselines.dtpp_cowp import COWPDTPP, DTPPEncoder, ScoreDecoder, VectorMapEncoder


def test_dtpp_public_default_uses_global_cost_weights():
    model = COWPDTPP(neighbors=2, max_branch=4)
    assert model.decoder.scorer.variable_cost is False

    scorer = ScoreDecoder(variable_cost=False).eval()
    a = torch.randn(2, 256)
    b = torch.randn(2, 256) * 100.0
    # Fixed/global DTPP replaces scene encoding by ones before the weight MLP.
    with torch.no_grad():
        wa = scorer.weights_decoder(torch.ones_like(a))
        wb = scorer.weights_decoder(torch.ones_like(b))
    torch.testing.assert_close(wa, wb)


def test_dtpp_stopped_ego_in_ego_frame_is_not_padding():
    enc = DTPPEncoder(dim=32, layers=1, heads=4, dropout=0.0).eval()
    inputs = {
        "ego_agent_past": torch.zeros(1, 11, 7),
        "neighbor_agents_past": torch.zeros(1, 2, 11, 11),
        "map_lanes": torch.zeros(1, 50, 50, 7),
        "map_crosswalks": torch.zeros(1, 20, 30, 3),
    }
    # Stopped SDC at the ego-frame origin: all kinematics are zero, but valid=1.
    inputs["ego_agent_past"][0, -1, 6] = 1.0
    with torch.no_grad():
        out = enc(inputs)
    assert bool(out["mask"][0, 0]) is False
    assert torch.isfinite(out["encoding"]).all()


def test_dtpp_map_padding_uses_all_zero_point_not_local_x_zero():
    encoder = VectorMapEncoder(map_dim=7, map_len=10, dim=8)
    map_tensor = torch.zeros(1, 1, 10, 7)
    # Every valid point lies exactly on local x=0.  This is a legal lane crossing
    # the ego y-axis and must not be interpreted as padding.
    map_tensor[0, 0, :, 1] = torch.linspace(-5.0, 5.0, 10)
    map_encoding = torch.randn(1, 1, 10, 8)
    _, mask = encoder.segment_map(map_tensor, map_encoding)
    assert mask.shape == (1, 1)
    assert bool(mask[0, 0]) is False


def test_dtpp_fixed_cost_forward_backward_stays_finite_on_stopped_scene():
    torch.manual_seed(7)
    B, A, K, T = 1, 2, 4, 20
    model = COWPDTPP(neighbors=A, max_branch=K, variable_cost=False)
    inputs = {
        "ego_agent_past": torch.zeros(B, 11, 7),
        "neighbor_agents_past": torch.zeros(B, A, 11, 11),
        "map_lanes": torch.zeros(B, 50, 50, 7),
        "map_crosswalks": torch.zeros(B, 20, 30, 3),
    }
    inputs["ego_agent_past"][..., 6] = 1.0
    tree = torch.zeros(B, K, T, 6)
    tree[..., 0] = torch.arange(1, T + 1).float()[None, None] * 0.1
    tree[..., 3] = 1.0
    outputs = model(inputs, tree, timesteps=T)
    loss = sum(x.float().square().mean() for x in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    assert torch.isfinite(grad_norm)


def test_dtpp_stopped_valid_neighbor_participates_in_collision_cost():
    scorer = ScoreDecoder(variable_cost=False).eval()
    T = 6
    ego = torch.zeros(1, T, 6)
    agent = torch.zeros(1, 1, T, 3)
    states = torch.zeros(1, 1, 11)
    states[..., 10] = 1.0  # explicit WOMD validity, all kinematics can stay zero
    with torch.no_grad():
        cost = scorer.calculate_collision(ego, agent, states, T)
    assert float(cost.item()) > 0.0


def test_dtpp_explicit_candidate_valid_keeps_stationary_stop_branch():
    torch.manual_seed(11)
    B, A, K, T = 1, 1, 2, 20
    model = COWPDTPP(neighbors=A, max_branch=K, variable_cost=False).eval()
    inputs = {
        "ego_agent_past": torch.zeros(B, 11, 7),
        "neighbor_agents_past": torch.zeros(B, A, 11, 11),
        "map_lanes": torch.zeros(B, 50, 50, 7),
        "map_crosswalks": torch.zeros(B, 20, 30, 3),
    }
    inputs["ego_agent_past"][..., 6] = 1.0
    # Branch 0 is a legitimate full stop at the ego-frame origin: all six
    # trajectory features are exactly zero. Branch 1 is padding.
    tree = torch.zeros(B, K, T, 6)
    candidate_valid = torch.tensor([[True, False]])
    with torch.no_grad():
        scores = model.score_candidates(inputs, tree, candidate_valid, timesteps=T)
    assert torch.isfinite(scores[0, 0])
    assert float(scores[0, 0]) > -1e8
    assert float(scores[0, 1]) <= -1e8
