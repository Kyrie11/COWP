import numpy as np

from ocrap.data.build.builder import select_planning_times_with_reasons
from ocrap.data.schema import RawScenario


def test_biased_time_is_not_discarded_by_earliest_uniform_time_when_budget_is_one():
    T, A = 80, 2
    states = np.zeros((T, A, 16), dtype=np.float32)
    valid = np.ones((T, A), dtype=bool)
    states[:, 0, 0] = np.arange(T, dtype=np.float32)
    states[:, 1, 0] = 100.0
    # Make a later frame strongly interaction-biased. The previous implementation
    # sorted the union of uniform and biased times, then kept the earliest frame,
    # so this frame was lost whenever max_times_per_scenario == 1.
    states[25, 1, 0] = states[25, 0, 0] + 2.0
    raw = RawScenario(
        scenario_id="s",
        timestamps=np.arange(T, dtype=np.float32) * 0.1,
        sdc_track_index=0,
        agent_states=states,
        agent_valid=valid,
        map_polylines=np.zeros((0, 0, 10), dtype=np.float32),
        map_valid=np.zeros((0, 0), dtype=bool),
        route=np.zeros((0, 6), dtype=np.float32),
        dynamic_map=np.zeros((T, 0, 8), dtype=np.float32),
    )
    times, reasons = select_planning_times_with_reasons(
        raw,
        {
            "sample_rate_hz": 10,
            "history_horizon_s": 1.0,
            "prefix_horizon_s": 1.0,
            "recovery_horizon_s": 4.0,
            "planning_time_stride_s": 0.5,
            "max_times_per_scenario": 1,
            "max_biased_times_per_scenario": 1,
        },
    )

    assert times == [25]
    assert "near_contact" in reasons[25]
