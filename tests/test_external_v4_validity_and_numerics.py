from __future__ import annotations

import numpy as np
import torch

from cowp.external_baselines.adapters import build_gameformer_map, build_dtpp_map, candidate_geometry_finite
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


def test_candidate_geometry_finite_reduces_only_trajectory_tail():
    cand = torch.zeros(2, 3, 5, 7)
    cand[0, 1, 2, 4] = float("nan")
    cand[1, 2, 4, 6] = float("inf")
    finite = candidate_geometry_finite(cand)
    assert finite.shape == (2, 3)
    assert finite.tolist() == [[True, False, True], [True, True, False]]


def test_external_baseline_runtime_has_no_tuple_dim_boolean_reduction():
    # Regression for environments where Tensor.all/any accept only one int dim.
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "cowp" / "external_baselines",
        root / "cowp" / "scripts" / "20_train_external_baseline.py",
        root / "cowp" / "scripts" / "21_eval_external_baseline.py",
    ]
    for target in targets:
        paths = target.rglob("*.py") if target.is_dir() else [target]
        for path in paths:
            text = path.read_text()
            assert ".all(dim=(" not in text, str(path)
            assert ".any(dim=(" not in text, str(path)


def test_dtpp_adapter_invalidates_only_nonfinite_candidate_branch():
    from cowp.external_baselines.adapters import make_external_batch

    B, N, H, T, K = 1, 3, 11, 8, 2
    hist = torch.zeros(B, N, H, 11)
    hist[..., 10] = 1.0
    hist[:, 0, -1, 0] = 5.0
    is_sdc = torch.zeros(B, N)
    is_sdc[:, 0] = 1.0
    future_x = torch.zeros(B, N, T)
    future_y = torch.zeros(B, N, T)
    candidates = torch.zeros(B, K, T, 7)
    candidates[0, 1, 3, 0] = float("nan")
    batch = {
        "state/history": hist,
        "state/is_sdc": is_sdc,
        "state/type": torch.ones(B, N),
        "state/future/x": future_x,
        "state/future/y": future_y,
        "state/future/valid": torch.ones(B, N, T),
        "roadgraph_samples/xyz": torch.zeros(B, 20, 3),
        "roadgraph_samples/valid": torch.ones(B, 20, dtype=torch.bool),
        "cowp/candidates/trajectory": candidates,
        "cowp/candidates/valid": torch.ones(B, K, dtype=torch.bool),
    }
    cfg = {"limits": {"max_agents": 64}, "model": {"history_steps": H}, "time": {"future_steps": T}}
    ext = make_external_batch(
        batch, cfg, device=torch.device("cpu"), max_neighbors=2, max_candidates=K,
        horizon=T, baseline="dtpp", require_candidates=True, require_future=True,
    )
    assert ext.candidate_valid.tolist() == [[True, False]]
    assert bool(torch.isfinite(ext.candidates).all())
