from __future__ import annotations

import numpy as np

from cowp.core.constants import PriorityRelation
from cowp.label.priority import determine_priority
from cowp.core.types import Lane, MapData, ScenarioData
from cowp.core.constants import ObjectType


def test_target_lane_rear_vehicle_priority(toy_scene, cfg):
    ego = toy_scene.states[0, 11:, :]
    agent = toy_scene.states[1, 11:, :]
    rho = determine_priority(toy_scene, 1, ego[:, [0,1,6,3,4,7,8]], agent[:, [0,1,6,3,4,7,8]], cfg)
    assert rho in (PriorityRelation.AGENT_PRIORITY, PriorityRelation.EQUAL_OR_NEGOTIATED)


def test_stop_control_priority(cfg):
    T = 91
    st = np.zeros((2, T, 11), dtype=np.float32)
    st[:, :, 10] = 1
    st[0, :, 7:9] = [4.8, 1.9]
    st[1, :, 7:9] = [4.8, 1.9]
    lane_ego = Lane(1, np.array([[0,0,0],[50,0,0]], dtype=np.float32), controlled_by_stop=True)
    lane_agent = Lane(2, np.array([[0,4,0],[50,4,0]], dtype=np.float32), controlled_by_stop=False)
    scene = ScenarioData('s', np.arange(T)*0.1, 10, st, np.array([1,1]), np.array([0,1]), 0, np.array([], dtype=np.int64), np.array([], dtype=np.int32), MapData(lanes={1: lane_ego, 2: lane_agent}))
    scene.states[0,10] = [0,0,0,5,0,5,0,4.8,1.9,1.6,1]
    scene.states[1,10] = [0,4,0,5,0,5,0,4.8,1.9,1.6,1]
    assert determine_priority(scene, 1, None, None, cfg) == PriorityRelation.AGENT_PRIORITY
