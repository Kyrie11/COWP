from types import SimpleNamespace

import numpy as np

from ocrap.data.waymax_loader import raw_scenario_from_waymax_state


def test_raw_scenario_from_waymax_state_normalizes_agent_time_metadata_fields():
    A, T = 2, 5
    t = np.arange(T, dtype=np.float32)
    x = np.stack([t, t + 10.0], axis=0)
    y = np.stack([t * 0.0, t * 0.0 + 1.0], axis=0)
    valid = np.ones((A, T), dtype=bool)
    tr = SimpleNamespace(
        x=x,
        y=y,
        z=np.zeros((A, T), dtype=np.float32),
        vel_x=np.ones((A, T), dtype=np.float32),
        vel_y=np.zeros((A, T), dtype=np.float32),
        yaw=np.zeros((A, T), dtype=np.float32),
        valid=valid,
        # These often arrive from Waymax as per-agent metadata rather than
        # full (A, T) fields.  This previously crashed via missing
        # _agent_time_array.
        length=np.array([4.5, 4.0], dtype=np.float32),
        width=np.array([2.0, 1.8], dtype=np.float32),
        height=np.array([1.6, 1.5], dtype=np.float32),
        timestamp_micros=(np.arange(T, dtype=np.int64) * 100000),
    )
    meta = SimpleNamespace(
        ids=np.array([101, 202], dtype=np.int64),
        object_types=np.array([1, 2], dtype=np.int32),
        is_sdc=np.array([True, False]),
    )
    state = SimpleNamespace(
        log_trajectory=tr,
        object_metadata=meta,
        roadgraph_points=None,
        sdc_paths=None,
        num_objects=A,
    )

    raw = raw_scenario_from_waymax_state(
        state,
        "dummy_waymax",
        0,
        {"max_map_polylines": 4, "max_polyline_points": 8, "route_points": 6},
    )

    assert raw.agent_states.shape == (T, A, 16)
    assert np.allclose(raw.agent_states[:, 0, 10], 4.5)
    assert np.allclose(raw.agent_states[:, 1, 11], 1.8)
    assert np.allclose(raw.agent_states[:, :, 13], np.array([[1.0, 2.0]] * T))
    assert raw.metadata["_waymax_state"] is state
