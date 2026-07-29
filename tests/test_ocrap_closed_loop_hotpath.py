from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ocrap.data.build.history import construct_history, construct_history_from_waymax_state
from ocrap.data.schema import RawScenario
from ocrap.data.waymax_loader import raw_scenario_from_waymax_state
from ocrap.models.data import OPTION_FEATURE_DIM, sample_to_feature
from ocrap.models.inference import ModelBundle, predict_samples
from ocrap.models.ocrap import OCRAPModel
from ocrap.simulation.closed_loop_runner import _state_geometry_snapshot, _step_metrics_geometry_snapshot
from ocrap.simulation.observation.visibility import ego_centered_grid_geometry, grid_coords


def _fake_waymax_state(*, agents: int = 5, steps: int = 32, timestep: int = 13):
    rng = np.random.default_rng(12)
    base_x = np.linspace(-5.0, 45.0, steps, dtype=np.float32)[None, :] + np.arange(agents, dtype=np.float32)[:, None] * 4.0
    base_y = np.arange(agents, dtype=np.float32)[:, None] * 2.5 + np.zeros((agents, steps), dtype=np.float32)
    log = SimpleNamespace(
        x=base_x,
        y=base_y,
        z=np.zeros((agents, steps), dtype=np.float32),
        vel_x=np.gradient(base_x, 0.1, axis=1).astype(np.float32),
        vel_y=np.zeros((agents, steps), dtype=np.float32),
        yaw=np.zeros((agents, steps), dtype=np.float32),
        valid=np.ones((agents, steps), dtype=bool),
        length=np.full((agents, steps), 4.8, dtype=np.float32),
        width=np.full((agents, steps), 2.0, dtype=np.float32),
        height=np.full((agents, steps), 1.6, dtype=np.float32),
        timestamp_micros=np.arange(steps, dtype=np.int64)[None, :] * 100_000,
    )
    sim = SimpleNamespace(**{name: np.array(getattr(log, name), copy=True) for name in vars(log)})
    sim.x[:, : timestep + 1] += rng.normal(0.0, 0.15, size=(agents, timestep + 1)).astype(np.float32)
    sim.y[:, : timestep + 1] += rng.normal(0.0, 0.08, size=(agents, timestep + 1)).astype(np.float32)
    sim.vel_x[:, : timestep + 1] += rng.normal(0.0, 0.05, size=(agents, timestep + 1)).astype(np.float32)
    metadata = SimpleNamespace(
        ids=np.arange(agents, dtype=np.int64),
        object_types=np.arange(agents, dtype=np.int32) % 3 + 1,
        is_sdc=np.asarray([True] + [False] * (agents - 1)),
    )
    return SimpleNamespace(
        log_trajectory=log,
        sim_trajectory=sim,
        object_metadata=metadata,
        timestep=np.int32(timestep),
        is_done=np.bool_(False),
        num_objects=agents,
    )


def _static_raw(state, cfg: dict) -> RawScenario:
    agents = int(state.num_objects)
    steps = int(state.log_trajectory.x.shape[1])
    route = np.zeros((80, 6), dtype=np.float32)
    route[:, 0] = np.linspace(-10.0, 100.0, 80, dtype=np.float32)
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    maps = np.zeros((4, 12, 8), dtype=np.float32)
    maps[..., 0] = np.linspace(-10.0, 60.0, 12, dtype=np.float32)[None, :]
    maps[:, :, 1] = np.arange(4, dtype=np.float32)[:, None] * 3.5
    return RawScenario(
        scenario_id="scene",
        timestamps=np.arange(steps, dtype=np.float32) * 0.1,
        sdc_track_index=0,
        agent_states=np.zeros((steps, agents, 16), dtype=np.float32),
        agent_valid=np.ones((steps, agents), dtype=bool),
        map_polylines=maps,
        map_valid=np.ones((4, 12), dtype=bool),
        route=route,
        dynamic_map=np.zeros((steps, 3, 8), dtype=np.float32),
        object_ids=[str(i) for i in range(agents)],
        metadata={"source": "womd_waymax"},
    )


def test_fast_waymax_history_matches_legacy_conversion() -> None:
    cfg = {
        "sample_rate_hz": 10.0,
        "history_horizon_s": 1.0,
        "prefix_horizon_s": 1.0,
        "recovery_horizon_s": 1.2,
        "max_agents": 4,
        "route_points": 80,
        "bev_channels": 7,
        "local_radius_m": 20.0,
        "bev_resolution_m": 2.0,
        "route_width": 3.5,
    }
    state = _fake_waymax_state()
    static = _static_raw(state, cfg)
    t = int(state.timestep)
    sid = "scene__cl0000"
    legacy_raw = raw_scenario_from_waymax_state(
        state, sid, 0, cfg,
        trajectory_mode="closed_loop_splice",
        splice_until=t,
        static_template=static,
    )
    legacy = construct_history(legacy_raw, t, cfg)
    fast = construct_history_from_waymax_state(state, static, t, cfg, scenario_id=sid, scenario_index=0)
    for name in (
        "agent_history", "agent_valid", "map_polylines", "map_valid", "dynamic_map",
        "route", "occ_mask", "ego_state", "future_agent_states", "future_agent_valid",
    ):
        np.testing.assert_allclose(getattr(fast, name), getattr(legacy, name), rtol=0.0, atol=1e-6, equal_nan=True)
    assert fast.scene_id == legacy.scene_id
    assert fast.original_scenario_id == legacy.original_scenario_id
    assert fast.metadata["agent_order"] == legacy.metadata["agent_order"]
    assert fast.metadata["ego_global_xy"] == legacy.metadata["ego_global_xy"]


def test_bev_grid_and_polar_geometry_are_cached_read_only() -> None:
    X1, Y1 = grid_coords(20.0, 1.0)
    X2, Y2 = grid_coords(20.0, 1.0)
    assert X1 is X2 and Y1 is Y2
    X3, Y3, R1, T1, M1 = ego_centered_grid_geometry(20.0, 1.0)
    X4, Y4, R2, T2, M2 = ego_centered_grid_geometry(20.0, 1.0)
    assert X3 is X4 and Y3 is Y4 and R1 is R2 and T1 is T2 and M1 is M2
    assert not X1.flags.writeable and not R1.flags.writeable and not M1.flags.writeable


def _feature_sample(candidate_index: int) -> dict:
    rng = np.random.default_rng(123)
    K, L = 2, 3
    sample = {
        "time_index": np.int64(17),
        "candidate_index": np.int64(candidate_index),
        "is_nominal": np.int64(candidate_index == 0),
        "ego_state": rng.normal(size=(9,)).astype(np.float32),
        "prefix_param": (rng.normal(size=(5,)) + candidate_index).astype(np.float32),
        "prefix_macro_id": np.int64(candidate_index),
        "prefix_macro_type_id": np.int64(candidate_index % 4),
        "utility": np.float32(2.5 - 0.1 * candidate_index),
        "hard_violation": np.float32(0.2 * candidate_index),
        "harm_proxy": np.float32(0.05 * candidate_index),
        "feasible": np.int64(1),
        "prefix_states": (rng.normal(size=(10, 9)) + candidate_index).astype(np.float32),
        "prefix_controls": (rng.normal(size=(9, 4)) + candidate_index).astype(np.float32),
        "agent_history": rng.normal(size=(10, 6, 16)).astype(np.float32),
        "agent_valid": np.ones((10, 6), dtype=np.float32),
        "bev_occ": rng.normal(size=(7, 21, 21)).astype(np.float32),
        "route": rng.normal(size=(20, 6)).astype(np.float32),
        "map_polylines": rng.normal(size=(8, 12, 8)).astype(np.float32),
        "map_valid": np.ones((8, 12), dtype=np.float32),
        "dynamic_map": rng.normal(size=(10, 6, 8)).astype(np.float32),
        "root_probs": np.asarray([0.4, 0.6], dtype=np.float32),
        "root_valid": np.ones((K,), dtype=np.float32),
        "root_signature": np.zeros((K, 0), dtype=np.float32),
        "root_future_signature": np.zeros((K, 0), dtype=np.float32),
        "c_star": np.eye(K, dtype=np.float32),
        "y_obs": np.eye(K, dtype=np.float32),
        "m_star": np.zeros((K, L), dtype=np.float32),
        "option_valid": np.ones((L,), dtype=np.float32),
        "recovery_modes": np.asarray(["stop", "lateral_escape", "avoid_secondary"]),
        "recovery_params": np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
    }
    return sample


def test_shared_geometry_and_single_transfer_prediction_is_exact() -> None:
    samples = [_feature_sample(i) for i in range(5)]
    shared_keys = (
        "agent_history", "agent_valid", "bev_occ", "route", "map_polylines", "map_valid",
        "dynamic_map", "ego_state", "root_probs", "root_valid", "root_signature",
        "root_future_signature", "c_star", "y_obs", "m_star", "option_valid",
        "recovery_modes", "recovery_params",
    )
    for key in shared_keys:
        for sample in samples[1:]:
            sample[key] = samples[0][key]
    input_dim = int(sample_to_feature(samples[0], {}).size)
    torch.manual_seed(4)
    model = OCRAPModel(
        input_dim=input_dim,
        num_roots=2,
        num_options=3,
        d_model=16,
        d_obs=4,
        num_heads=4,
        dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=True,
    )
    model.eval()
    bundle = ModelBundle(model=model, cfg={}, device=torch.device("cpu"))
    reference = predict_samples(samples, bundle, {}, shared_scene_features=True, shared_geometry=False)
    optimized = predict_samples(samples, bundle, {}, shared_scene_features=True, shared_geometry=True)
    for a, b in zip(reference, optimized):
        assert a.r_dep == b.r_dep
        assert a.r_orc == b.r_orc
        assert a.gap == b.gap
        assert a.direct_recovery_value == b.direct_recovery_value
        assert a.direct_recovery_std == b.direct_recovery_std
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.root_probs, b.root_probs)
        np.testing.assert_array_equal(a.c_star, b.c_star)
        np.testing.assert_array_equal(a.margins, b.margins)


def test_state_geometry_snapshot_uses_current_slice_and_returns_trace_xy() -> None:
    state = _fake_waymax_state(agents=4, steps=20, timestep=7)
    metrics, xy = _state_geometry_snapshot(state, 0, timestep=7)
    assert xy == [float(state.sim_trajectory.x[0, 7]), float(state.sim_trajectory.y[0, 7])]
    assert set(metrics) == {"min_clearance_m", "ttc_s", "ego_speed_mps", "ego_yaw_rad"}
    assert np.isfinite(metrics["ego_speed_mps"])


def test_combined_step_snapshot_preserves_waymax_metric_selection() -> None:
    state = _fake_waymax_state(agents=4, steps=20, timestep=7)
    class FakeEnv:
        def metrics(self, _state):
            return {
                "overlap": SimpleNamespace(value=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
                "offroad": SimpleNamespace(value=np.asarray([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32)),
            }
    metrics, xy, timestep, done = _step_metrics_geometry_snapshot(FakeEnv(), state, 0)
    assert timestep == 7 and done is False
    assert metrics["overlap"] == 0.0
    assert metrics["offroad"] == np.float32(0.2)
    assert xy == [float(state.sim_trajectory.x[0, 7]), float(state.sim_trajectory.y[0, 7])]
