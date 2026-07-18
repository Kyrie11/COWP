from importlib import import_module


def test_compare_eval_outputs_ignores_performance_metadata_and_accepts_tolerance():
    mod = import_module("cowp.scripts.23_compare_eval_outputs")
    reference = {"mode": "waymax", "metric": 0.5, "steps": [80, 79]}
    candidate = {
        "mode": "waymax",
        "metric": 0.5000001,
        "steps": [80, 79],
        "jit_waymax_env": True,
    }
    assert mod.compare_eval_outputs(reference, candidate, atol=1e-5, rtol=0.0) == []


def test_compare_eval_outputs_reports_real_metric_change():
    mod = import_module("cowp.scripts.23_compare_eval_outputs")
    mismatches = mod.compare_eval_outputs({"CR": 0.1}, {"CR": 0.2}, atol=1e-6, rtol=0.0)
    assert mismatches and "CR" in mismatches[0]
