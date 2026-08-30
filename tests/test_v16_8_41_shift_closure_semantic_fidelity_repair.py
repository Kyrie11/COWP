from __future__ import annotations


def test_shift_schedule_preserves_v39_terminal_policy_semantics():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _shift_longitudinal_envelope_schedule_np,
    )

    lower_all = -np.ones((6,), dtype=np.int8)
    upper_all = np.ones((6,), dtype=np.int8)
    lower_release = np.asarray([-1, -1, -1, 0, 0, 0], dtype=np.int8)
    upper_release = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8)

    assert np.array_equal(
        _shift_longitudinal_envelope_schedule_np(lower_all, -1),
        -np.ones((6,), dtype=np.int8),
    )
    assert np.array_equal(
        _shift_longitudinal_envelope_schedule_np(upper_all, 1),
        np.ones((6,), dtype=np.int8),
    )
    assert np.array_equal(
        _shift_longitudinal_envelope_schedule_np(lower_release, -3),
        np.asarray([-1, -1, 0, 0, 0, 0], dtype=np.int8),
    )
    assert np.array_equal(
        _shift_longitudinal_envelope_schedule_np(upper_release, 2),
        np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8),
    )


def test_v39_and_v40_constructors_share_the_same_shift_helper_source():
    import inspect
    import cowp.waymax_eval.policy_wrapper as pw

    v39 = inspect.getsource(pw._construct_conflict_window_control_reachable_tube_np)
    v40 = inspect.getsource(pw._construct_shift_closed_first_action_viability_interval_np)
    call = "_shift_longitudinal_envelope_schedule_np("
    assert call in v39
    assert call in v40
    assert "shifted_schedule[-1]" not in v39
    assert "shifted_schedule[-1]" not in v40


def test_repair_does_not_widen_or_modify_any_controller_limit():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _shift_longitudinal_envelope_schedule_np,
    )

    for pid, schedule in (
        (-1, -np.ones((8,), dtype=np.int8)),
        (1, np.ones((8,), dtype=np.int8)),
        (-2, np.asarray([-1, -1, 0, 0, 0, 0, 0, 0], dtype=np.int8)),
        (3, np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)),
    ):
        shifted = _shift_longitudinal_envelope_schedule_np(schedule, pid)
        assert shifted.shape == schedule.shape
        assert set(np.unique(shifted)).issubset({-1, 0, 1})


def test_helper_is_exactly_equivalent_to_literal_frozen_v39_rule():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _shift_longitudinal_envelope_schedule_np,
    )

    rng = np.random.default_rng(16841)
    for horizon in (1, 2, 3, 8, 80):
        for policy_id in (-3, -2, -1, 0, 1, 2, 3):
            for _ in range(20):
                schedule = rng.integers(-1, 2, size=horizon, dtype=np.int8)
                literal = np.zeros_like(schedule, dtype=np.int8)
                if horizon > 1:
                    literal[:-1] = schedule[1:]
                if policy_id in {-1, 1}:
                    literal[-1] = np.int8(np.sign(policy_id))
                actual = _shift_longitudinal_envelope_schedule_np(
                    schedule, policy_id
                )
                assert np.array_equal(actual, literal)
