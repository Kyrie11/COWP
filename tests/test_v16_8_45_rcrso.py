from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cowp.label.safe_responses import (
    build_root_control_knot_residual_trajectory,
    expand_root_control_knots,
)
from cowp.models.recourse_set_operator import (
    RCRSOConfig,
    RootConditionedRecourseSetTransformer,
    build_rcrso_features_np,
)
import cowp.waymax_eval.policy_wrapper as pw


def _root(h: int = 8) -> np.ndarray:
    tr = np.zeros((h, 7), dtype=np.float32)
    tr[:, 0] = np.arange(h, dtype=np.float32)
    tr[:, 2] = 0.0
    tr[:, 3] = 10.0
    tr[:, 5] = 4.5
    tr[:, 6] = 1.8
    return tr


def test_control_knots_are_bounded_and_zero_is_exact_identity():
    cfg = {"time": {"dt": 0.1}, "candidate": {"max_decel_mps2": 6.0, "max_accel_mps2": 4.0}}
    seq = expand_root_control_knots(np.array([-2.0, 0.0, 2.0], np.float32), 9, cfg)
    assert float(seq.min()) >= -6.0
    assert float(seq.max()) <= 4.0
    root = _root(9)
    out, zero_seq = build_root_control_knot_residual_trajectory(root, cfg, control_knots=np.zeros(8, np.float32))
    assert np.array_equal(out, root)
    assert np.array_equal(zero_seq, np.zeros(9, np.float32))


def test_rcrso_environment_encoder_is_permutation_invariant():
    torch.manual_seed(7)
    cfg = RCRSOConfig(d_model=32, nhead=4, encoder_layers=1, max_queries=4, control_knots=5)
    model = RootConditionedRecourseSetTransformer(cfg).eval()
    root = torch.randn(1, 8, cfg.root_feature_dim)
    ego = torch.randn(1, 16, cfg.ego_feature_dim)
    env = torch.randn(1, 6, cfg.environment_feature_dim)
    blocker = torch.randn(1, cfg.blocker_feature_dim)
    conflict = torch.randn(1, cfg.conflict_feature_dim)
    valid = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        a = model(root_tokens=root, ego_tokens=ego, environment_tokens=env, environment_valid=valid,
                  blocker_state=blocker, conflict_features=conflict, query_count=4)["control_knots"]
        perm = torch.tensor([4, 1, 5, 0, 3, 2])
        b = model(root_tokens=root, ego_tokens=ego, environment_tokens=env[:, perm], environment_valid=valid[:, perm],
                  blocker_state=blocker, conflict_features=conflict, query_count=4)["control_knots"]
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_conflict_features_use_causal_unsafe_event_support(monkeypatch):
    def fake_unsafe(left, right, cfg, agent_type=0, **kwargs):
        n = min(len(left), len(right))
        mask = np.zeros(n, dtype=bool)
        if n >= 5:
            mask[2:5] = True
        return SimpleNamespace(event_mask=mask)

    import cowp.geometry.collision as collision
    monkeypatch.setattr(collision, "unsafe_between", fake_unsafe)
    root = _root(8)
    bs = np.zeros(11, np.float32); bs[3] = 10.0; bs[7:10] = [4.5, 1.8, 1.6]; bs[10] = 1.0
    env = [{"trajectory": root.copy(), "shifted_trajectory": root.copy(), "object_type": 1, "agent_index": 2}]
    f = build_rcrso_features_np(
        root=root, root_mass=0.8, root_source=2, blocker_state=bs,
        current_ego_trajectory=root.copy(), shifted_ego_trajectory=root.copy(), environment=env,
        cfg=RCRSOConfig(), verifier_cfg={"time": {"dt": 0.1}}, blocker_object_type=1,
    )
    # current, shifted and environment all expose the same fake [2,4] event interval.
    assert f["conflict_features"].shape == (10,)
    assert f["conflict_features"][0] == 1.0
    assert np.isclose(f["conflict_features"][1], 2 / 7)
    assert np.isclose(f["conflict_features"][2], 4 / 7)
    assert f["conflict_features"][6] == 1.0
    assert f["root_tokens"].shape[1] == 9
    assert f["blocker_state"].shape[0] == 13


def test_hard_verifier_rejects_learned_proposal_before_interaction(monkeypatch):
    root = _root(6)
    state = np.zeros((2, 11), np.float32); state[:, 10] = 1.0
    monkeypatch.setattr(pw, "build_root_control_knot_residual_trajectory", lambda root, cfg, control_knots: (root.copy(), np.zeros(len(root), np.float32)))
    monkeypatch.setattr(pw, "prepare_root_recovery_burden_bank", lambda *args, **kwargs: [(1.5, np.zeros(4, np.float32))])
    monkeypatch.setattr(pw, "_roadgraph_drivable_mask", lambda *args, **kwargs: True)
    # If burden rejection is respected, no collision predicate is ever needed.
    monkeypatch.setattr(pw, "unsafe_between_bool", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not reach interaction check")))
    profiles, detail = pw._verified_root_conditioned_recourse_set_profiles_np(
        state, 1, 1, root, 0.5, root, root, [], {}, {"time": {"dt": 0.1}},
        np.zeros((3, 8), np.float32), root_ordinal=0, compatibility_cache={},
    )
    assert profiles == []
    assert detail["proposal_count"] == 3
    assert detail["burden_rejects"] == 3
    assert detail["failure_reason"] == "no_low_burden_static_control"


def test_rcrso_cache_identity_tracks_actual_control_sequence():
    base = {
        "profile_index": 20000,
        "rcrso_extension": True,
        "rcrso_control_knots": np.array([0.0, 0.25], np.float32),
        "rcrso_accel_sequence_mps2": np.array([0.0, 1.0, 1.0], np.float32),
    }
    same = dict(base)
    changed = dict(base)
    changed["rcrso_control_knots"] = np.array([0.0, 0.5], np.float32)
    assert pw._response_profile_compatibility_identity_np(base) == pw._response_profile_compatibility_identity_np(same)
    assert pw._response_profile_compatibility_identity_np(base) != pw._response_profile_compatibility_identity_np(changed)


def test_v45_method_is_registered_and_sidecar_never_truncates_inside_group():
    canonical, gate = pw._canonical_online_method("cowp_verified_root_conditioned_recourse_set_operator", "priority")
    assert canonical == "cowp_verified_root_conditioned_recourse_set_operator"
    source = Path("cowp/scripts/104_build_rcrso_sidecar.py").read_text(encoding="utf-8")
    assert "group_contexts" in source
    assert "Never truncate a hypothesis midway" in source
    assert "if made>=args.max_examples_per_scene" not in source
