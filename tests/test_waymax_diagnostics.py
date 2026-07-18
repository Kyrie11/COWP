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


def test_standard_metric_accumulator_resolves_sdc_once_per_step(monkeypatch):
    import cowp.waymax_eval.metrics_standard as metrics_standard

    calls = {"sdc": 0}

    def fake_sdc_index(state):
        calls["sdc"] += 1
        return 0

    class Result:
        def __init__(self, value):
            self.value = value
            self.valid = [True]

    class Metric:
        def __init__(self, value):
            self.value = value

        def compute(self, state):
            return Result([self.value])

    monkeypatch.setattr(metrics_standard, "_sdc_index_from_state", fake_sdc_index)
    acc = metrics_standard.WaymaxStandardMetricAccumulator(
        metric_objects=[("OverlapMetric", Metric(0.0)), ("OffroadMetric", Metric(1.0))],
        jit_metrics=False,
    )
    acc.update(object())
    out = acc.finalize()

    assert calls["sdc"] == 1
    assert out["CR"] == 1.0
