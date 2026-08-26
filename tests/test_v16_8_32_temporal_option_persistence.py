from __future__ import annotations


def test_trihorizon_product_order_requires_nonregression_everywhere():
    from cowp.waymax_eval.policy_wrapper import _trihorizon_option_persistence_dominates

    b1 = (0, 0, 0, 10)
    b2 = (0, 0, 0, 8)
    # Current-prefix improvement with equal future option support is admissible.
    assert _trihorizon_option_persistence_dominates(b1, b1, b2, b2, 3, 7)
    # First-successor improvement with all else equal is admissible.
    assert _trihorizon_option_persistence_dominates(b1, (0, 0, 0, 11), b2, b2, 3, 3)
    # Second-successor improvement is independently admissible.
    assert _trihorizon_option_persistence_dominates(b1, b1, b2, (0, 0, 0, 9), 3, 3)
    # A delayed V2 regression blocks an otherwise-valid BHOV switch.
    assert not _trihorizon_option_persistence_dominates(b1, (1, 1, 1, 80), (1, 1, 2, 20), (0, 9, 9, 80), 3, 80)
    # Exact ties preserve COWP.
    assert not _trihorizon_option_persistence_dominates(b1, b1, b2, b2, 3, 3)


def test_v32_methods_keep_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    for method in ("cowp_trihorizon_option_persistence", "cowp_sov_recovery_commitment"):
        assert _canonical_online_method(method, "hard") == (method, "priority")
        assert _method_gate_defaults(method, "hard") == (method, "priority")


def test_second_successor_uses_two_causal_emitted_steps(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    seen = []

    def fake_target(current, desired, cfg, previous_longitudinal_accel=0.0):
        # Emit a deterministic second action one meter ahead.
        return np.asarray([current[0] + 1.0, current[1], current[6], 1.0, 0.0], dtype=np.float32), 0.0

    def fake_candidates(agent_state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        seen.append(np.array(agent_state, copy=True))
        traj = np.zeros((1, 3, 5), dtype=np.float32)
        valid = np.asarray([True])
        conv = np.asarray([True])
        macro = np.asarray([1], dtype=np.int64)
        util = np.asarray([0.0], dtype=np.float32)
        road = np.asarray([True])
        collision = np.asarray([True])
        prefix = np.asarray([3], dtype=np.int32)
        margin = np.asarray([1.0], dtype=np.float32)
        return traj, valid, conv, macro, util, road, collision, prefix, margin

    monkeypatch.setattr(pw, "_consistent_one_step_target", fake_target)
    monkeypatch.setattr(pw, "_route_lane_aware_candidates", fake_candidates)
    state = np.zeros((2, 11), dtype=np.float32)
    state[:, 10] = 1.0
    state[0, 3] = 1.0  # other agent CV
    traj = np.zeros((3, 5), dtype=np.float32)
    first = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    sig, detail = pw._second_successor_option_signature(state, 1, first, 0.0, traj, {}, {"time": {"dt": 0.1}})
    assert sig[0] == 1 and detail["conventional_candidates"] == 1
    assert len(seen) == 1
    # Other agent advances by CV on both causal steps; ego follows emitted targets.
    assert abs(float(seen[0][0, 0]) - 0.2) < 1e-6
    assert abs(float(seen[0][1, 0]) - 2.0) < 1e-6


def test_v32_fresh37_is_disjoint_from_all_prior_development_panels():
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "reference_manifests"
    def ids(name: str) -> list[str]:
        return [x.strip() for x in (root / name).read_text(encoding="utf-8").splitlines() if x.strip()]

    fresh = ids("waymax_v16_8_32_fresh37_ids.txt")
    prior = set(ids("waymax_v16_8_30_equivalence16_ids.txt"))
    prior |= set(ids("waymax_v16_8_30_counterfactual48_ids.txt"))
    prior |= set(ids("waymax_v16_8_30_balanced_dev96_ids.txt"))
    prior |= set(ids("waymax_v16_8_31_holdout64_ids.txt"))
    assert len(fresh) == 37 and len(set(fresh)) == 37
    assert set(fresh).isdisjoint(prior)
    assert hashlib.sha256("\n".join(fresh).encode()).hexdigest() == "ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481"


def test_v32_fresh37_reference_hash_and_row_order_match_manifest():
    import hashlib
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    ids = [x.strip() for x in (repo / "reference_manifests/waymax_v16_8_32_fresh37_ids.txt").read_text().splitlines() if x.strip()]
    expected = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    for name in ("v16_8_32_fresh37_cowp_reference.json", "v16_8_32_fresh37_rvr_reference.json"):
        payload = json.loads((repo / "reference_results" / name).read_text())
        assert payload["scenario_ids_sha256"] == expected
        assert [str(r["scenario_id"]) for r in payload["scenario_results"]] == ids
        prov = payload["reference_subset_provenance"]
        assert prov["subset_logical_sha256"] == expected
        assert prov["subset_num_rollouts"] == 37
