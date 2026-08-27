from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import torch

from cowp.external_baselines.adapters import build_dtpp_map, build_gameformer_map, external_map_topology_report
from cowp.external_baselines.gameformer_cowp import COWPGameFormer, CrossTransformer, InteractionDecoder, gameformer_loss
from cowp.external_baselines.waymax_policy import _minimal_external_online_batch
from cowp.waymax_eval.policy_wrapper import _extract_roadgraph_tokens, _roadgraph_womd_batch_fields


def _structured_roadgraph_batch() -> dict[str, torch.Tensor]:
    # Four different WOMD vector-map features in one flat roadgraph stream.
    # id=10: lane center near agent 0, id=20: lane center near agent 1,
    # id=99: road line (must not be encoded as lane center), id=30: crosswalk.
    specs = [
        (10, 1, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
        (99, 6, [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]),
        (20, 2, [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0)]),
        (30, 18, [(0.0, 2.0), (1.0, 2.0), (1.0, 3.0), (0.0, 3.0)]),
    ]
    xyz, direction, ids, types = [], [], [], []
    for fid, typ, pts in specs:
        for x, y in pts:
            xyz.append((x, y, 0.0))
            direction.append((1.0, 0.0, 0.0))
            ids.append(fid)
            types.append(typ)
    return {
        "roadgraph_samples/xyz": torch.tensor([xyz], dtype=torch.float32),
        "roadgraph_samples/dir": torch.tensor([direction], dtype=torch.float32),
        "roadgraph_samples/id": torch.tensor([ids], dtype=torch.int64),
        "roadgraph_samples/type": torch.tensor([types], dtype=torch.int64),
        "roadgraph_samples/valid": torch.ones(1, len(xyz), dtype=torch.bool),
    }


def test_gameformer_map_preserves_feature_ids_types_and_agent_locality():
    batch = _structured_roadgraph_batch()
    anchors = torch.tensor([[[0.0, 0.0], [0.0, 10.0]]])
    lanes, cross, lane_valid, cross_valid = build_gameformer_map(
        batch,
        num_agents_to_predict=2,
        device=torch.device("cpu"),
        n_lanes=2,
        lane_points=8,
        n_crosswalks=1,
        agent_xy=anchors,
        agent_valid=torch.ones(1, 2, dtype=torch.bool),
        return_valid=True,
    )
    assert lane_valid[0, 0, 0].sum().item() == 3
    assert lane_valid[0, 1, 0].sum().item() == 3
    # Each predicted actor gets its own nearest real lane-center feature.
    assert torch.allclose(lanes[0, 0, 0, :3, 1], torch.zeros(3))
    assert torch.allclose(lanes[0, 1, 0, :3, 1], torch.full((3,), 10.0))
    # The type-6 road line at y=1 is not spliced into either lane polyline.
    lane_y = lanes[..., 1][lane_valid]
    assert not torch.any(torch.isclose(lane_y, torch.tensor(1.0)))
    # Crosswalk type=18 remains a separate map element.
    assert cross_valid[0, 0, 0].sum().item() == 4
    assert torch.all(cross[0, 0, 0, :4, 1] >= 2.0)


def test_dtpp_map_preserves_lane_and_crosswalk_elements_instead_of_flat_chunks():
    batch = _structured_roadgraph_batch()
    lanes, cross, lane_valid, cross_valid = build_dtpp_map(
        batch,
        torch.device("cpu"),
        n_lanes=4,
        lane_points=8,
        n_crosswalks=2,
        cross_points=8,
        return_valid=True,
    )
    assert lane_valid.sum().item() == 6  # only the two 3-point lane centers
    lane_y = lanes[..., 1][lane_valid]
    assert set(torch.unique(lane_y).tolist()) == {0.0, 10.0}
    assert cross_valid.sum().item() == 4
    assert torch.all(cross[..., 1][cross_valid] >= 2.0)


def test_v6_map_topology_contract_report_requires_aligned_womd_metadata():
    good = external_map_topology_report(_structured_roadgraph_batch())
    assert good["has_xy"] and good["has_id"] and good["has_type"] and good["has_dir"] and good["has_valid"]
    assert good["aligned"]
    assert good["points"] == 13

    legacy = dict(_structured_roadgraph_batch())
    legacy.pop("roadgraph_samples/id")
    legacy.pop("roadgraph_samples/type")
    legacy.pop("roadgraph_samples/dir")
    bad = external_map_topology_report(legacy)
    assert bad["has_xy"] and bad["has_valid"]
    assert not bad["has_id"] and not bad["has_type"] and not bad["has_dir"]
    assert not bad["aligned"]


def test_gameformer_cross_transformer_matches_source_no_query_residual():
    block = CrossTransformer(dim=8, heads=2, dropout=0.0).eval()
    # If cross-attention and FFN emit zero, source GameFormer returns zero after
    # norm.  V5's extra ``+ query`` would return normalized non-zero query state.
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    query = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    key = torch.zeros(1, 3, 8)
    out = block(query, key, key)
    assert torch.equal(out, torch.zeros_like(out))


def test_gameformer_interaction_decoder_uses_source_modal_mean_not_sum():
    source = inspect.getsource(InteractionDecoder.forward)
    assert ".mean(dim=2)" in source
    assert ".sum(dim=2)" not in source


def test_waymax_roadgraph_preserves_ids_types_and_directions_for_external_adapter():
    rg = SimpleNamespace(
        x=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        y=np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float32),
        dir_x=np.ones(4, dtype=np.float32),
        dir_y=np.zeros(4, dtype=np.float32),
        ids=np.asarray([10, 10, 20, 20], dtype=np.int64),
        types=np.asarray([1, 1, 2, 2], dtype=np.int32),
        valid=np.ones(4, dtype=bool),
    )
    state = SimpleNamespace(roadgraph_points=rg)
    tokens = _extract_roadgraph_tokens(state, {"limits": {"max_roadgraph_points": 100}})
    assert tokens["ids"].tolist() == [10, 10, 20, 20]
    assert tokens["types"].tolist() == [1, 1, 2, 2]
    assert np.allclose(tokens["dir_xy"], np.asarray([[1.0, 0.0]] * 4))

    fields = _roadgraph_womd_batch_fields(tokens)
    assert fields["roadgraph_samples/id"].shape == (1, 4)
    assert fields["roadgraph_samples/type"].shape == (1, 4)
    assert fields["roadgraph_samples/dir"].shape == (1, 4, 3)

    history = np.zeros((4, 11, 11), dtype=np.float32)
    state11 = np.zeros((4, 11), dtype=np.float32)
    state11[:, 10] = 1.0
    online = _minimal_external_online_batch(history, state11, 0, tokens, None, {"limits": {"max_agents": 4}})
    assert np.array_equal(online["roadgraph_samples/id"], fields["roadgraph_samples/id"])
    assert np.array_equal(online["roadgraph_samples/type"], fields["roadgraph_samples/type"])
    assert np.array_equal(online["roadgraph_samples/dir"], fields["roadgraph_samples/dir"])


def test_waymax_missing_ids_types_does_not_invent_one_fake_feature():
    rg = SimpleNamespace(
        x=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        y=np.zeros(3, dtype=np.float32),
        valid=np.ones(3, dtype=bool),
    )
    tokens = _extract_roadgraph_tokens(SimpleNamespace(roadgraph_points=rg), {})
    assert tokens["ids"].size == 0
    assert tokens["types"].size == 0
    fields = _roadgraph_womd_batch_fields(tokens)
    assert "roadgraph_samples/id" not in fields
    assert "roadgraph_samples/type" not in fields



def test_full_gameformer_structured_map_backward_is_finite():
    torch.manual_seed(23)
    B, N, H, T = 1, 2, 11, 8
    batch = _structured_roadgraph_batch()
    anchors = torch.tensor([[[0.0, 0.0], [0.0, 10.0]]])
    lanes, cross, lane_valid, cross_valid = build_gameformer_map(
        batch,
        num_agents_to_predict=N,
        device=torch.device("cpu"),
        n_lanes=6,
        lane_points=100,
        n_crosswalks=4,
        agent_xy=anchors,
        agent_valid=torch.ones(B, N, dtype=torch.bool),
        return_valid=True,
    )

    ego = torch.zeros(B, H, 9)
    neighbors = torch.zeros(B, 1, H, 9)
    ego[0, :, 0] = torch.linspace(-5.0, 0.0, H)
    ego[0, :, 3:6] = torch.tensor([4.8, 1.9, 1.6])
    ego[0, :, 7] = 5.0
    ego[0, :, 8] = 1.0
    neighbors[0, 0, :, 0] = torch.linspace(-5.0, 0.0, H)
    neighbors[0, 0, :, 1] = 10.0
    neighbors[0, 0, :, 3:6] = torch.tensor([4.8, 1.9, 1.6])
    neighbors[0, 0, :, 7] = 5.0
    neighbors[0, 0, :, 8] = 1.0
    inputs = {
        "ego_state": ego,
        "neighbors_state": neighbors,
        "actors_valid": torch.ones(B, N, H, dtype=torch.bool),
        "map_lanes": lanes,
        "map_crosswalks": cross,
        "map_lanes_valid": lane_valid,
        "map_crosswalks_valid": cross_valid,
    }
    model = COWPGameFormer(
        modalities=3,
        neighbors_to_predict=1,
        future_len=T,
        encoder_layers=1,
        decoder_levels=2,
    )
    outputs = model(inputs)
    t = torch.arange(1, T + 1, dtype=torch.float32) * 0.5
    ego_gt = torch.stack((t, torch.zeros_like(t)), dim=-1).unsqueeze(0)
    neighbor_gt = torch.stack((t, torch.full_like(t, 10.0)), dim=-1).reshape(1, 1, T, 2)
    valid = torch.ones(B, T, dtype=torch.bool)
    neighbor_valid = torch.ones(B, 1, T, dtype=torch.bool)
    loss, metrics = gameformer_loss(outputs, ego_gt, valid, neighbor_gt, neighbor_valid)
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["plannerADE"])
    loss.backward()
    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert bad == []
