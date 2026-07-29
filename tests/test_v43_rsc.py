import importlib.util
from pathlib import Path

import numpy as np

from ocrap.planning.selector import calibrated_constrained_select


def _select(risk_admission: bool):
    return calibrated_constrained_select(
        utility=np.array([1.0, 0.05]),
        r_dep=np.array([0.5, -1.0]),  # recovery lacks the old scalar admission
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.ones(2, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.zeros(2),
        pred_drs=np.ones(2),
        nominal_deviation=np.array([0.0, 0.02]),
        pred_direct_value=np.array([0.0, 0.8]),
        pred_direct_std=np.array([100.0, 100.0]),  # ignored in selective mode
        candidate_macro_names=["nominal", "yield"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="yield",
        direct_value_uncertainty_mode="risk_selective",
        direct_value_min_advantage_lcb=0.5,
        direct_value_score_mode=True,
        direct_value_top1_only=True,
        direct_value_risk_controlled_admission=risk_admission,
        direct_value_challenge_nominal=True,
        direct_value_bonus=1.0,
        stress_rescue_challenge_nominal=True,
    )


def test_risk_controlled_certificate_can_augment_stress_admission():
    sel = _select(True)
    assert sel.selected_index == 1
    assert bool(sel.admitted[1])
    assert "direct_value" in sel.reason


def test_preference_only_mode_reproduces_v42_abstention():
    sel = _select(False)
    assert sel.selected_index == 0
    assert not bool(sel.admitted[1])


def _load_calibrator_module():
    path = Path(__file__).parents[1] / "tools" / "calibrate_direct_value_risk_v43.py"
    spec = importlib.util.spec_from_file_location("v43_cal", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_selective_threshold_fit_uses_disjoint_risk_constraints():
    mod = _load_calibrator_module()
    rows = [
        {"pred_adv": 0.9, "teacher_adv": 0.20, "oracle_best_teacher_adv": 0.20},
        {"pred_adv": 0.8, "teacher_adv": 0.15, "oracle_best_teacher_adv": 0.15},
        {"pred_adv": 0.7, "teacher_adv": 0.10, "oracle_best_teacher_adv": 0.10},
        {"pred_adv": 0.6, "teacher_adv": 0.08, "oracle_best_teacher_adv": 0.08},
        {"pred_adv": 0.5, "teacher_adv": -0.10, "oracle_best_teacher_adv": 0.10},
    ]
    class Args:
        min_score_advantage = 0.0
        positive_gain = 0.025
        min_fit_selected = 4
        min_fit_precision = 0.75
        max_fit_harmful_selected_rate = 0.10
    threshold, metrics, _ = mod._fit_threshold(rows, Args())
    assert threshold == 0.6
    assert metrics["num_selected"] == 4
    assert metrics["challenge_precision"] == 1.0
