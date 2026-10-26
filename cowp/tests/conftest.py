from __future__ import annotations

import numpy as np
import pytest

from cowp.core.types import Lane, MapData, ScenarioData
from cowp.core.constants import ObjectType
from cowp.core.config import load_config


@pytest.fixture
def cfg():
    
    c = load_config('configs/label.yaml')
    c['limits']['max_candidates'] = 8
    c['limits']['max_critical_agents'] = 2
    c['limits']['max_natural_alternatives'] = 6
    c['limits']['max_safe_responses'] = 8
    c['time']['future_steps'] = 30
    c['candidate']['min_valid_horizon_steps'] = 20
    c['response']['response_acc_values_mps2'] = [-4.5, -2.5, 0.0, 1.0]
    c['response']['response_start_delay_s'] = [0.0, 0.3]
    c['response']['response_duration_s'] = [1.0]
    c['natural']['max_obs_samples'] = 2
    c['natural']['max_neutral_samples'] = 2
    c['natural']['prio_max_samples'] = 2
    c['natural']['min_natural_alternatives'] = 2
    return c


@pytest.fixture
def toy_scene():
    T = 91
    dt = 0.1
    timestamps = np.arange(T, dtype=np.float32) * dt
    states = np.zeros((2, T, 11), dtype=np.float32)
    # ego moves along x, starts at 0 at current t=10
    for t in range(T):
        tau = (t - 10) * dt
        x = 5.0 * tau
        states[0, t] = [x, 0, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, 1]
        # agent target lane rear vehicle starts behind ego in adjacent lane, moves along x.
        xa = -8.0 + 8.0 * tau
        states[1, t] = [xa, 3.5, 0, 8, 0, 8, 0, 4.8, 1.9, 1.6, 1]
    lane0 = Lane(1, np.array([[-20, 0, 0], [100, 0, 0]], dtype=np.float32), exit_lanes=(3,))
    lane1 = Lane(2, np.array([[-20, 3.5, 0], [100, 3.5, 0]], dtype=np.float32), exit_lanes=(3,))
    lane2 = Lane(3, np.array([[100, 1.75, 0], [200, 1.75, 0]], dtype=np.float32), entry_lanes=(1,2))
    return ScenarioData(
        scenario_id='toy',
        timestamps=timestamps,
        current_time_index=10,
        states=states,
        object_type=np.array([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.array([0, 1], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.array([1], dtype=np.int64),
        tracks_to_predict=np.array([1], dtype=np.int32),
        map_data=MapData(lanes={1: lane0, 2: lane1, 3: lane2}),
    )
