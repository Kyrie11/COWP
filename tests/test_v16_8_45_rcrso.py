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


def test_sidecar_restores_flat_raw_womd_roadgraph_and_aligned_heading():
    import importlib
    sidecar = importlib.import_module("cowp.scripts.104_build_rcrso_sidecar")
    xyz = np.array([0.,0.,0., 1.,0.,0., 1.,1.,0.], np.float32)
    direction = np.array([1.,0.,0., 1.,0.,0., 0.,1.,0.], np.float32)
    data = {
        "womd/roadgraph_samples/xyz": xyz,
        "womd/roadgraph_samples/dir": direction,
        "womd/roadgraph_samples/valid": np.array([1,1,1], np.int64),
        "womd/roadgraph_samples/type": np.array([1,1,2], np.int64),
    }
    rg = sidecar._roadgraph(data)
    assert rg["xy"].shape == (3,2)
    assert rg["heading"].shape == (3,)
    np.testing.assert_allclose(rg["xy"], np.array([[0,0],[1,0],[1,1]], np.float32))
    np.testing.assert_allclose(rg["heading"], np.array([0,0,np.pi/2], np.float32), atol=1e-6)
    assert rg["valid"].tolist() == [True, True, True]


def test_sidecar_roadgraph_accepts_waymax_shaped_vectors_and_rejects_malformed_flat_dir():
    import importlib
    sidecar = importlib.import_module("cowp.scripts.104_build_rcrso_sidecar")
    data = {
        "roadgraph_samples/xyz": np.array([[[0.,0.,0.],[1.,0.,0.]]], np.float32),
        "roadgraph_samples/dir": np.array([[[1.,0.,0.],[0.,1.,0.]]], np.float32),
        "roadgraph_samples/valid": np.array([[1,1]], np.int64),
        "roadgraph_samples/type": np.array([[1,2]], np.int64),
    }
    rg = sidecar._roadgraph(data)
    assert rg["xy"].shape == (2,2)
    np.testing.assert_allclose(rg["heading"], np.array([0,np.pi/2], np.float32), atol=1e-6)
    bad = dict(data)
    bad["roadgraph_samples/dir"] = np.arange(5, dtype=np.float32)
    import pytest
    with pytest.raises(ValueError, match="roadgraph_samples/dir"):
        sidecar._roadgraph(bad)


def test_sidecar_uses_model_input_index_not_scenario_track_index():
    import importlib
    sidecar = importlib.import_module("cowp.scripts.104_build_rcrso_sidecar")
    data = {
        "cowp/critical/track_index": np.array([9, 7], np.int64),
        "cowp/critical/input_index": np.array([2, 4], np.int64),
    }
    np.testing.assert_array_equal(sidecar._critical_input_indices(data), np.array([2,4], np.int64))
    np.testing.assert_array_equal(
        sidecar._critical_input_indices({"cowp/critical/track_index": np.array([3], np.int64)}),
        np.array([3], np.int64),
    )


def test_sidecar_shifted_environment_matches_online_non_ego_cv_successor():
    import importlib
    sidecar = importlib.import_module("cowp.scripts.104_build_rcrso_sidecar")
    state = np.zeros((3,11), np.float32)
    state[:,10] = 1.0
    state[1,0:2] = [10.,2.]
    state[1,3:5] = [5.,-1.]
    cfg = {"time":{"dt":0.1}}
    nxt = sidecar._one_step_cv_successor_for_environment(state, 0, cfg)
    np.testing.assert_allclose(nxt[1,0:2], [10.5,1.9], atol=1e-6)
    # SDC is intentionally not advanced by this helper; only environment actors
    # are consumed by the sidecar's shifted environment construction.
    np.testing.assert_allclose(nxt[0,0:2], state[0,0:2])


def test_rcrso_blocker_heading_uses_online_state_yaw_slot_six():
    root = _root(8)
    root[:,2] = 0.2
    blocker = np.zeros(11, np.float32)
    blocker[2] = -1.1  # reserved/z slot: must not be interpreted as yaw
    blocker[6] = 0.7   # actual online yaw
    blocker[3:5] = [4.0, 0.0]
    blocker[7:10] = [4.5, 1.8, 1.6]
    blocker[10] = 1.0
    f = build_rcrso_features_np(
        root=root, root_mass=0.8, root_source=1, blocker_state=blocker,
        current_ego_trajectory=root.copy(), shifted_ego_trajectory=root.copy(), environment=[],
        cfg=RCRSOConfig(), verifier_cfg={"time":{"dt":0.1}}, blocker_object_type=1,
    )
    dyaw = 0.7 - 0.2
    assert np.isclose(f["blocker_state"][2], np.sin(dyaw), atol=1e-6)
    assert np.isclose(f["blocker_state"][3], np.cos(dyaw), atol=1e-6)


def test_sidecar_roadgraph_subset_keeps_nearest_lane_when_local_box_has_none():
    import importlib
    sidecar = importlib.import_module("cowp.scripts.104_build_rcrso_sidecar")
    road = {
        "xy": np.array([[100.,0.],[150.,0.]], np.float32),
        "heading": np.zeros(2, np.float32),
        "types": np.array([1,1], np.int32),
        "valid": np.array([True,True]),
    }
    root = _root(8); root[:,0] = np.linspace(0,5,8); root[:,1] = 0
    xy,h,t,v = sidecar._sidecar_roadgraph_subset(road, root, root, root)
    assert xy.shape[0] == 1
    np.testing.assert_allclose(xy[0], [100.,0.])
    assert v.tolist() == [True]


def test_rcrso_query_cross_ignores_invalid_environment_padding():
    torch.manual_seed(17)
    cfg = RCRSOConfig(d_model=32, nhead=4, encoder_layers=1, max_queries=4, control_knots=5)
    model = RootConditionedRecourseSetTransformer(cfg).eval()
    root = torch.randn(1, 8, cfg.root_feature_dim)
    ego = torch.randn(1, 16, cfg.ego_feature_dim)
    env_a = torch.randn(1, 6, cfg.environment_feature_dim)
    env_b = env_a.clone()
    env_b[:, 2:] = torch.randn_like(env_b[:, 2:]) * 1000.0
    valid = torch.tensor([[True, True, False, False, False, False]])
    blocker = torch.randn(1, cfg.blocker_feature_dim)
    conflict = torch.randn(1, cfg.conflict_feature_dim)
    with torch.no_grad():
        a = model(root_tokens=root, ego_tokens=ego, environment_tokens=env_a,
                  environment_valid=valid, blocker_state=blocker,
                  conflict_features=conflict, query_count=4)["control_knots"]
        b = model(root_tokens=root, ego_tokens=ego, environment_tokens=env_b,
                  environment_valid=valid, blocker_state=blocker,
                  conflict_features=conflict, query_count=4)["control_knots"]
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_sidecar_retained_roots_use_canonical_mass_not_raw_threshold_reimplementation():
    import importlib
    from cowp.label.audit_relevance import canonical_root_weights
    sidecar = importlib.import_module("cowp.scripts.104_build_rcrso_sidecar")
    raw = np.array([[0.50, 0.30, 0.19, 0.01]], np.float32)
    valid = np.ones_like(raw, dtype=bool)
    cfg = {"ncf": {"min_alt_weight": 0.03, "root_probability_floor": 0.02}}
    canonical = canonical_root_weights({"valid": valid, "weight": raw}, cfg)[0]
    selected = sidecar._retained_roots_from_canonical(canonical, valid[0], 0.75, 2, 24)
    # The 0.01 mode is removed by canonical support and floor smoothing is already
    # represented in `canonical`; sidecar must not construct a second measure.
    assert 3 not in selected
    assert len(selected) >= 2
    assert float(canonical[selected].sum()) >= 0.75 - 1e-9


def test_stage0_distinguishes_static_online_callback_readiness_from_candidate_fixed_domain(monkeypatch):
    import importlib
    stage0 = importlib.import_module("cowp.scripts.106_eval_rcrso_support")
    root = _root(6)
    item = {
        "root_trajectory": root,
        "blocker_state_global": np.array([0,0,0,10,0,10,0,4.5,1.8,1.6,1], np.float32),
        "blocker_object_type": np.array(1, np.int64),
        "beta": np.array(0.8, np.float32),
        "ego_current": root.copy(),
        "ego_shifted": root.copy(),
        "environment_current": np.zeros((0,6,7), np.float32),
        "environment_shifted": np.zeros((0,6,7), np.float32),
        "environment_object_type": np.zeros((0,), np.int64),
        "environment_agent_index": np.zeros((0,), np.int64),
        "roadgraph_xy": np.zeros((1,2), np.float32),
        "roadgraph_heading": np.zeros((1,), np.float32),
        "roadgraph_types": np.ones((1,), np.int32),
        "roadgraph_valid": np.ones((1,), bool),
    }
    monkeypatch.setattr(stage0, "build_root_recovery_trajectory_bank", lambda root,cfg: [root.copy()])
    monkeypatch.setattr(stage0, "prepare_root_recovery_burden_bank", lambda *args, **kwargs: [(0.1, np.zeros(4, np.float32))])
    monkeypatch.setattr(stage0, "_roadgraph_drivable_mask", lambda *args, **kwargs: True)
    monkeypatch.setattr(stage0, "_trajectory_waymax_kinematic_safe_np", lambda *args, **kwargs: (True, {}))
    monkeypatch.setattr(stage0, "unsafe_between_bool", lambda *args, **kwargs: True)
    assert len(stage0._fixed_static_bank_profiles(item, {"time":{"dt":0.1}})) == 1
    # Candidate-specific collision can empty the fixed domain while the online
    # actor/root support is still structurally ready; this is exactly the domain
    # where RCRSO is allowed to extend the certificate.
    assert stage0._fixed_bank_profiles(item, {"time":{"dt":0.1}}) == []


def test_prepare_support_can_preserve_empty_fixed_root_domains_for_verified_rcrso(monkeypatch):
    state = np.zeros((2, 11), np.float32)
    state[:, 10] = 1.0
    state[1, 7:10] = [4.5, 1.8, 1.6]
    root_a = _root(6)
    root_b = _root(6); root_b[:, 1] += 2.0
    roots = np.stack([root_a, root_b], axis=0)[None, ...]
    logits = np.array([[0.2, 0.1]], np.float32)
    monkeypatch.setattr(pw, "adaptive_beta", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(pw, "build_root_recovery_trajectory_bank", lambda *args, **kwargs: [])
    monkeypatch.setattr(pw, "prepare_root_recovery_burden_bank", lambda *args, **kwargs: [])
    cfg = {
        "time": {"dt": 0.1},
        "natural": {"certificate_min_low_burden_roots": 2, "root_dedup_mean_distance_m": 0.0},
        "ncf": {"min_alt_weight": 0.0, "root_probability_floor": 0.0, "cvar_tail_mass": 0.25},
        "planning": {"set_transport_cvar_tail_mass": 0.25},
        "response": {"root_conditioned_transport": {"max_roots_per_agent": 2}},
    }
    args = (
        state, 0, np.array([1]), np.array([True]), roots, logits,
        np.array([1, 1]), {"xy": np.zeros((0,2), np.float32), "heading": np.zeros(0), "types": np.zeros(0), "valid": np.zeros(0,bool)}, cfg,
    )
    frozen, _ = pw._prepare_interaction_response_support_np(*args)
    assert not frozen[1]["ready"]
    learned, detail = pw._prepare_interaction_response_support_np(*args, allow_empty_profile_roots=True)
    assert learned[1]["ready"]
    assert learned[1]["reason"] == "ready_with_verified_proposal_holes"
    assert all(root["profiles"] == [] for root in learned[1]["roots"])
    assert detail["agents_ready_with_proposal_holes"] == 1


def test_rcrso_wrapper_exactly_nests_v43_before_learned_pass(monkeypatch):
    sentinel = {"parent_index": 3, "trajectory": _root(4), "target": np.zeros(5, np.float32), "accel": 0.0}
    monkeypatch.setattr(
        pw, "_construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np",
        lambda *args, **kwargs: (sentinel, {"selected_certificate_kind": "blocker_conditioned_interaction_aware_reachable_response_envelope"}),
    )
    def forbidden_decoder(_):
        raise AssertionError("learned full-support pass must not run when V43 already certifies")
    out, detail = pw._construct_verified_root_conditioned_recourse_set_operator_np(
        np.zeros((1,11),np.float32), 0, np.zeros((1,4,7),np.float32), np.array([True]), np.array([True]),
        np.array([0]), np.array([0.]), np.array([0.]), np.zeros((1,5),np.float32), np.zeros(1), {}, {}, 0.0,
        base_candidate_index=0, critical_track_index=np.zeros(0,np.int64), critical_valid=np.zeros(0,bool),
        natural_trajectories=np.zeros((0,0,0,7),np.float32), natural_logits=np.zeros((0,0),np.float32),
        blocker_query_track_index=np.array([1],np.int64), blocker_query_trajectories=None, blocker_query_logits=None,
        object_types=np.zeros(1,np.int32), blocker_query_decoder=forbidden_decoder,
        verified_recourse_set_proposal_fn=lambda **kwargs: ([], {}),
    )
    assert out is sentinel
    assert detail["rcrso_nested_v43_selected"] is True
    assert detail["rcrso_full_support_pass_attempted"] is False



def test_rcrso_wrapper_full_pass_allows_static_holes_and_augments_verified_sets(monkeypatch):
    monkeypatch.setattr(
        pw, "_construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np",
        lambda *args, **kwargs: (None, {"interaction_failure_reason": "frozen_v43_empty"}),
    )
    captured = {}
    sentinel = {"parent_index": 1, "trajectory": _root(4), "target": np.zeros(5, np.float32), "accel": 0.0}
    def fake_construct(*args, **kwargs):
        captured.update(kwargs)
        return sentinel, {
            "interaction_rcrso_attempts": 3,
            "interaction_rcrso_verified_profiles": 2,
            "interaction_selected_uses_rcrso": True,
            "interaction_selected_rcrso_assignment_profiles": 1,
            "interaction_failure_reason": "none",
        }
    monkeypatch.setattr(pw, "_construct_interaction_aware_reachable_response_envelope_np", fake_construct)
    out, detail = pw._construct_verified_root_conditioned_recourse_set_operator_np(
        np.zeros((2,11),np.float32), 0, np.zeros((1,4,7),np.float32), np.array([True]), np.array([True]),
        np.array([0]), np.array([0.]), np.array([0.]), np.zeros((1,5),np.float32), np.zeros(1), {}, {}, 0.0,
        base_candidate_index=0, critical_track_index=np.array([1],np.int64), critical_valid=np.array([True]),
        natural_trajectories=np.zeros((1,2,4,7),np.float32), natural_logits=np.zeros((1,2),np.float32),
        blocker_query_track_index=np.zeros(0,np.int64), blocker_query_trajectories=None, blocker_query_logits=None,
        object_types=np.zeros(2,np.int32), blocker_query_decoder=lambda idx: (np.zeros((0,0,0,7),np.float32),np.zeros((0,0),np.float32)),
        verified_recourse_set_proposal_fn=lambda **kwargs: ([], {}),
    )
    assert out is sentinel
    assert captured["allow_empty_fixed_root_domains"] is True
    assert captured["augment_with_verified_recourse_set_proposals"] is True
    assert captured["known_nested_v39_empty"] is True
    assert detail["rcrso_full_support_pass_selected"] is True
    assert detail["blocker_conditioned_query_rcrso_verified_profiles"] == 2

def test_stage0_uses_verified_fixed_plus_learned_set_and_keeps_static_holes():
    src = Path("cowp/scripts/106_eval_rcrso_support.py").read_text(encoding="utf-8")
    assert 'online=list(fixed_profiles) + list(learned)' in src
    assert 'empty_frozen_static_root_domains_are_valid_proposal_completeness_targets' in src
    assert 'fixed_if_nonempty_else_rcrso' not in src
    assert 'all(bool(x["online_static_support_nonempty"]) for x in rr)' not in src


def test_v45r2_vectorized_waymax_kinematics_matches_literal_reference_randomized():
    import numpy as np
    from cowp.core.config import load_config
    from cowp.waymax_eval.policy_wrapper import (
        _trajectory_waymax_kinematic_safe_np,
        _trajectory_waymax_kinematic_safe_literal_np,
    )
    cfg = load_config('configs/label_cowp_v16_8.yaml','configs/data.yaml','configs/eval_cowp_v16_8.yaml')
    rng = np.random.default_rng(20260845)
    for _ in range(120):
        h = int(rng.integers(1, 81))
        cur = np.zeros(11, np.float32)
        cur[0:2] = rng.normal(0, 5, 2)
        speed0 = float(rng.uniform(0, 16))
        yaw0 = float(rng.uniform(-np.pi, np.pi))
        cur[3:5] = speed0 * np.array([np.cos(yaw0), np.sin(yaw0)], np.float32)
        cur[5] = speed0; cur[6] = yaw0; cur[7:10] = [4.5,1.8,1.6]; cur[10] = 1
        tr = np.zeros((h,7), np.float32)
        pos = cur[:2].astype(np.float64).copy(); yaw = yaw0; speed = speed0
        for t in range(h):
            speed = max(0.0, speed + float(rng.normal(0, 0.35)))
            yaw = float((yaw + rng.normal(0, 0.025) + np.pi) % (2*np.pi) - np.pi)
            vel = speed * np.array([np.cos(yaw), np.sin(yaw)])
            pos = pos + vel * 0.1
            tr[t,:2] = pos; tr[t,2] = yaw; tr[t,3:5] = vel; tr[t,5:7] = [4.5,1.8]
        a, da = _trajectory_waymax_kinematic_safe_literal_np(cur, tr, cfg)
        b, db = _trajectory_waymax_kinematic_safe_np(cur, tr, cfg)
        assert a == b
        assert da['failure_step'] == db['failure_step']
        assert da['contract_source'] == db['contract_source']
        assert np.isclose(da['max_abs_accel_mps2'], db['max_abs_accel_mps2'], rtol=0, atol=1e-6)
        assert np.isclose(da['max_abs_steering_curvature'], db['max_abs_steering_curvature'], rtol=0, atol=1e-6)


def test_v45r2_precomputed_environment_event_steps_preserve_rcrso_features():
    import numpy as np
    from cowp.geometry.collision import unsafe_between
    root = _root(12)
    blocker = np.zeros(11, np.float32); blocker[3]=4.0; blocker[5]=4.0; blocker[7:10]=[4.5,1.8,1.6]; blocker[10]=1.0
    ego = root.copy(); ego[:,1] += 1.0
    shift = np.concatenate([ego[1:], ego[-1:]], axis=0)
    actor = root.copy(); actor[:,1] += 2.0
    actor_shift = np.concatenate([actor[1:], actor[-1:]], axis=0)
    env=[{"agent_index":2,"object_type":1,"trajectory":actor,"shifted_trajectory":actor_shift}]
    vcfg={"time":{"dt":0.1},"unsafe":{}}
    steps=[]
    shifted_root=np.concatenate([root[1:],root[-1:]],axis=0)
    for left,right,right_type in ((root,actor,1),(actor,root,1),(shifted_root,actor_shift,1),(actor_shift,shifted_root,1)):
        steps.extend(np.flatnonzero(np.asarray(unsafe_between(left,right,vcfg,agent_type=right_type).event_mask,bool)).tolist())
    a=build_rcrso_features_np(root=root,root_mass=.8,root_source=1,blocker_state=blocker,current_ego_trajectory=ego,shifted_ego_trajectory=shift,environment=env,cfg=RCRSOConfig(),verifier_cfg=vcfg,blocker_object_type=1)
    b=build_rcrso_features_np(root=root,root_mass=.8,root_source=1,blocker_state=blocker,current_ego_trajectory=ego,shifted_ego_trajectory=shift,environment=env,cfg=RCRSOConfig(),verifier_cfg=vcfg,blocker_object_type=1,precomputed_environment_event_steps=steps)
    for k in a:
        np.testing.assert_allclose(a[k],b[k],rtol=0,atol=0)


def test_v45r2_scene_shared_verifier_cache_preserves_candidate_results():
    import numpy as np
    from cowp.core.config import load_config
    from cowp.waymax_eval.policy_wrapper import _verified_root_conditioned_recourse_set_profiles_np
    cfg=load_config('configs/label_cowp_v16_8.yaml','configs/data.yaml','configs/eval_cowp_v16_8.yaml')
    H=16; dt=0.1
    state=np.zeros((7,11),np.float32); state[:,10]=1.; state[:,7:10]=[4.5,1.8,1.6]
    state[1,3]=4.; state[1,5]=4.
    root=np.zeros((H,7),np.float32); root[:,0]=np.arange(1,H+1)*dt*4.; root[:,3]=4.; root[:,5:7]=[4.5,1.8]
    env=[]
    for e in range(2,6):
        tr=root.copy(); tr[:,1]=30.+e*4
        env.append({'agent_index':e,'object_type':1,'trajectory':tr,'shifted_trajectory':np.concatenate([tr[1:],tr[-1:]],axis=0)})
    road={'xy':np.zeros((0,2),np.float32),'heading':np.zeros(0,np.float32),'valid':np.zeros(0,bool),'types':np.zeros(0,np.int32)}
    props=np.linspace(-.2,.2,8*8,dtype=np.float32).reshape(8,8)
    egos=[]
    for dy in (10.,11.,12.):
        ego=root.copy(); ego[:,1]=dy; egos.append((ego,np.concatenate([ego[1:],ego[-1:]],axis=0)))
    fresh=[]
    for ego,shift in egos:
        v,d=_verified_root_conditioned_recourse_set_profiles_np(state,1,1,root,100.,ego,shift,env,road,cfg,props,root_ordinal=0,compatibility_cache={})
        fresh.append(([int(x['profile_index']) for x in v], list(d['proposal_outcomes'])))
    cache={}; shared=[]
    for ego,shift in egos:
        v,d=_verified_root_conditioned_recourse_set_profiles_np(state,1,1,root,100.,ego,shift,env,road,cfg,props,root_ordinal=0,compatibility_cache=cache)
        shared.append(([int(x['profile_index']) for x in v], list(d['proposal_outcomes'])))
    assert fresh==shared
