from types import SimpleNamespace

from ocrap.data.womd.scenario_parser import parse_scenario_proto


def state(x, valid=True):
    return SimpleNamespace(center_x=x, center_y=0.0, center_z=0.0, velocity_x=0.0, velocity_y=0.0, heading=0.0, length=4.0, width=2.0, height=1.5, valid=valid)


def test_sdc_original_index_beyond_max_agents_is_index_zero():
    tracks = [SimpleNamespace(id=i, object_type=1, states=[state(float(i))]) for i in range(5)]
    scenario = SimpleNamespace(scenario_id="s", timestamps_seconds=[0.0], sdc_track_index=4, tracks=tracks, map_features=[], dynamic_map_states=[])
    raw = parse_scenario_proto(scenario, max_agents=2, max_polylines=4, max_points=8)
    assert raw.sdc_track_index == 0
    assert raw.object_ids[0] == "4"
    assert raw.metadata["original_sdc_track_index"] == 4
    assert raw.metadata["agent_index_map"][0] == 4


def test_invalid_states_remain_invalid_and_shapes_exist():
    tracks = [SimpleNamespace(id=0, object_type=1, states=[state(0.0), state(1.0, valid=False)]), SimpleNamespace(id=1, object_type=1, states=[state(2.0), state(3.0)])]
    scenario = SimpleNamespace(scenario_id="s2", timestamps_seconds=[0.0, 0.1], sdc_track_index=0, tracks=tracks, map_features=[], dynamic_map_states=[])
    raw = parse_scenario_proto(scenario, max_agents=4, max_polylines=5, max_points=7, max_dynamic_signals=3)
    assert raw.agent_valid[1, 0] == False
    assert raw.map_polylines.shape == (5, 7, 10)
    assert raw.dynamic_map.shape == (2, 3, 6)
    assert raw.route.shape[0] == 7
