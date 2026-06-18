from __future__ import annotations

from cowp.waymax_eval.metrics_cowp import policy_diagnostic_summary


def test_policy_diagnostic_summary_aggregates_rows():
    rollouts = [
        {"policy_diagnostics": [{"max_witness_prob": 0.7, "min_opr": 0.2, "mean_opr": 0.5, "max_predicted_burden": 0.9, "fallback_used": True}]},
        {"policy_diagnostics": [{"max_witness_prob": 0.1, "min_opr": 0.8, "mean_opr": 0.9, "max_predicted_burden": 0.2, "fallback_used": False}]},
    ]
    out = policy_diagnostic_summary(rollouts)
    assert out["ClosedLoopPredFSR"] == 0.5
    assert out["ClosedLoopFallbackStepRate"] == 0.5
    assert out["ClosedLoopPolicySteps"] == 2.0
