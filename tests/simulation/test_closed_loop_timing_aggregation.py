from __future__ import annotations

from ocrap.simulation.closed_loop_runner import _aggregate_scene_results


def test_closed_loop_timing_is_summed_and_normalized() -> None:
    scenes = [
        {
            "num_decisions": 2,
            "num_metric_steps": 2,
            "label_mode": "all",
            "timing": {"wall_s": 5.0, "totals_s": {"teacher_labels": 4.0, "policy_selection": 0.2}},
            "metric_summary": {},
            "macro_counts": {},
            "selection_reason_counts": {},
        },
        {
            "num_decisions": 3,
            "num_metric_steps": 3,
            "label_mode": "all",
            "timing": {"wall_s": 7.0, "totals_s": {"teacher_labels": 6.0, "policy_selection": 0.3}},
            "metric_summary": {},
            "macro_counts": {},
            "selection_reason_counts": {},
        },
    ]
    out = _aggregate_scene_results(scenes, "marc_lite", "test")
    assert out["timing"]["scene_wall_sum_s"] == 12.0
    assert out["timing"]["totals_s"]["teacher_labels"] == 10.0
    assert out["timing"]["per_decision_s"]["teacher_labels"] == 2.0
