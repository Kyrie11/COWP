from __future__ import annotations


def test_macro_recovery_representatives_are_existing_bank_per_macro_and_roadgraph_first():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _macro_recovery_representatives_np
    from cowp.core.constants import MacroType

    valid = np.asarray([1, 1, 1, 1, 1, 1, 1], dtype=bool)
    road = np.asarray([1, 1, 1, 1, 0, 0, 1], dtype=bool)
    prefix = np.asarray([4, 6, 6, 3, 9, 10, 8], dtype=np.float32)
    macro = np.asarray([0, 0, 1, 1, 2, 2, int(MacroType.PAD)], dtype=np.int64)
    fallback = np.asarray([1.0, 3.0, 4.0, 2.0, 0.1, 0.0, -10.0], dtype=np.float32)

    reps = _macro_recovery_representatives_np(valid, road, prefix, macro, fallback)
    # roadgraph-safe pool is nonempty, so macros that only occur off-road are not
    # manufactured into recovery support; PAD is never an option.
    assert reps == [1, 2]

    # Tie on prefix inside a macro is broken by the already-frozen fallback score.
    prefix2 = np.asarray([6, 6, 6, 3], dtype=np.float32)
    macro2 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    reps2 = _macro_recovery_representatives_np(
        np.ones(4, dtype=bool), np.ones(4, dtype=bool), prefix2, macro2,
        np.asarray([5.0, 1.0, 2.0, 0.0], dtype=np.float32),
    )
    assert reps2 == [1, 2]


def test_recovery_frontier_entry_selects_least_fallback_cost_not_largest_profile():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _recovery_frontier_mode_choice_np

    macro = np.asarray([0, 1, 2, 3], dtype=np.int64)
    scores = np.asarray([0.0, 5.0, 1.5, 3.0], dtype=np.float64)
    strict = {1: True, 2: True, 3: True}
    weak = {1: True, 2: True, 3: True}
    chosen, active, entered, continued, exited = _recovery_frontier_mode_choice_np(
        0, [1, 2, 3], macro, scores, strict, weak, -1
    )
    assert chosen == 2
    assert active == 2
    assert entered and not continued and not exited


def test_recovery_frontier_continuation_preserves_semantic_mode_and_exits_on_dominance_loss():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _recovery_frontier_mode_choice_np

    macro = np.asarray([0, 1, 2, 2, 3], dtype=np.int64)
    scores = np.asarray([0.0, 0.1, 4.0, 1.0, -5.0], dtype=np.float64)
    strict = {1: True, 2: False, 3: False, 4: True}
    weak = {1: True, 2: True, 3: True, 4: True}

    # Active macro 2 cannot jump to the cheaper/dominating macro 3; it continues
    # with the least-cost weakly-dominating representative of macro 2.
    chosen, active, entered, continued, exited = _recovery_frontier_mode_choice_np(
        0, [1, 2, 3, 4], macro, scores, strict, weak, 2
    )
    assert chosen == 3 and active == 2
    assert not entered and continued and not exited

    # Once the active semantic branch loses weak physical dominance, exit to COWP.
    weak[2] = False
    weak[3] = False
    chosen, active, entered, continued, exited = _recovery_frontier_mode_choice_np(
        0, [1, 2, 3, 4], macro, scores, strict, weak, 2
    )
    assert chosen == 0 and active == -1
    assert not entered and not continued and exited


def test_v36_method_keeps_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    method = "cowp_control_projected_recovery_frontier"
    assert _canonical_online_method(method, "hard") == (method, "priority")
    assert _method_gate_defaults(method, "hard") == (method, "priority")
