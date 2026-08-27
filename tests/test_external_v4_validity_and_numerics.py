from __future__ import annotations

import numpy as np
import torch

from cowp.external_baselines.adapters import build_gameformer_map, build_dtpp_map
from cowp.external_baselines.dtpp_cowp import VectorMapEncoder
from cowp.external_baselines.gameformer_cowp import GameFormerEncoder
from cowp.external_baselines.rule_based import rule_costs_for_batch


def _roadgraph_batch():
    xyz = torch.zeros(1, 12, 3)
    xyz[0, 0, :2] = torch.tensor([0.0, 0.0])
    xyz[0, 1, :2] = torch.tensor([1.0, 0.0])
    valid = torch.zeros(1, 12, dtype=torch.bool)
    valid[0, :2] = True
    return {"roadgraph_samples/xyz": xyz, "roadgraph_samples/valid": valid}


def test_map_builders_preserve_source_validity_at_local_origin():
    batch = _roadgraph_batch()
    origin = torch.zeros(1, 2)
    yaw0 = torch.zeros(1)
    _, _, gf_valid, _ = build_gameformer_map(batch, 2, torch.device("cpu"), origin=origin, yaw0=yaw0, return_valid=True)
    _, _, dt_valid, _ = build_dtpp_map(batch, torch.device("cpu"), origin=origin, yaw0=yaw0, return_valid=True)
    assert bool(gf_valid[0, 0, 0, 0])
    assert bool(gf_valid[0, 0, 0, 1])
    assert bool(dt_valid[0, 0, 0])
    assert bool(dt_valid[0, 0, 1])


def test_gameformer_segment_mask_does_not_drop_partial_valid_segment():
    enc = GameFormerEncoder(neighbors_to_predict=0, layers=1, dim=256)
    x = torch.zeros(1, 1, 10, 16)
    e = torch.randn(1, 1, 10, 256)
    valid = torch.zeros(1, 1, 10, dtype=torch.bool)
    valid[..., 0] = True
    pooled, mask = enc.segment_map(x, e, valid)
    assert pooled.shape == (1, 1, 256)
    assert mask.tolist() == [[False]]
    assert torch.isfinite(pooled).all()


def test_dtpp_segment_mask_does_not_drop_valid_zero_point():
    enc = VectorMapEncoder(7, 50, 256)
    x = torch.zeros(1, 1, 10, 7)
    e = torch.randn(1, 1, 10, 256)
    valid = torch.zeros(1, 1, 10, dtype=torch.bool)
    valid[..., 0] = True
    pooled, mask = enc.segment_map(x, e, valid)
    assert mask.tolist() == [[False]]
    assert torch.isfinite(pooled).all()


def test_rule_baseline_rejects_nonfinite_candidate_geometry():
    cand = np.zeros((1, 2, 5, 7), dtype=np.float32)
    cand[0, 0, 2, 0] = np.nan
    batch = {
        "cowp/candidates/trajectory": cand,
        "cowp/candidates/valid": np.ones((1, 2), dtype=bool),
        "cowp/candidates/conventional_safe": np.ones((1, 2), dtype=bool),
        "state/current": np.zeros((1, 1, 11), dtype=np.float32),
        "state/is_sdc": np.ones((1, 1), dtype=bool),
    }
    _, accept, valid = rule_costs_for_batch(batch, {}, "pdm_closed", require_conventional_safe=False)
    assert not bool(valid[0, 0])
    assert not bool(accept[0, 0])
    assert bool(valid[0, 1])
