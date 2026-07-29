import numpy as np

from ocrap.evaluation.baselines import select_baseline
from ocrap.planning.selector import calibrated_constrained_select


def test_budget_rate_by_bucket_is_applied_without_global_scalar():
    utility = np.array([1.0, 1.5])
    pred_r_dep = np.array([-0.01, 0.05])
    teacher = np.array([0.1, 0.1])
    hard = np.array([0.0, 0.0])
    harm = np.array([0.0, 0.0])
    feasible = np.array([True, True])
    cfg = {
        "selection": {
            "ocrap_selector": "calibrated_constrained",
            "active_bucket_name": "test_safe_v2",
            "intervention_budget_rate_by_bucket": {"safe": 0.05},
            "intervention_budget_used": 10,
            "intervention_budget_steps": 20,
            "intervention_budget_penalty": 100.0,
            "budget_nominal_slack": 0.20,
            "safe_nominal_slack": 0.20,
        }
    }
    sel = select_baseline(
        "ocrap",
        utility,
        pred_r_dep,
        teacher,
        teacher,
        hard,
        harm,
        feasible,
        gamma_rec=0.0,
        gamma_H=0.0,
        gamma_D=5.0,
        cfg=cfg,
        pred_gap=np.zeros(2),
        nominal_deviation=np.zeros(2),
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_budget_preserved"


def test_safe_switch_guard_requires_material_recovery_gain():
    sel = calibrated_constrained_select(
        utility=np.array([1.00, 1.40]),
        r_dep=np.array([-0.01, 0.00]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.0,
        pred_gap=np.array([0.05, 0.05]),
        nominal_deviation=np.array([0.0, 0.02]),
        regime_name="test_safe",
        safe_nominal_slack=0.15,
        safe_switch_score_margin=0.10,
        safe_min_rec_lcb_gain=0.10,
        safe_min_gap_reduction=0.20,
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_safe_switch_guard"


def test_prefer_admitted_ranks_within_admitted_candidates():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 4.0, 2.0]),
        r_dep=np.array([-0.4, -0.3, 0.2]),
        hard=np.zeros(3),
        harm=np.zeros(3),
        feasible=np.array([True, True, True]),
        gamma_rec=0.0,
        pred_gap=np.zeros(3),
        nominal_deviation=np.zeros(3),
        regime_name="test_contact",
        prefer_admitted=True,
    )
    assert sel.selected_index == 2
    assert sel.reason == "best_calibrated_prefer_admitted"


def test_option_drs_certificate_can_admit_conservative_scalar_candidate():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 1.2]),
        r_dep=np.array([0.0, -0.15]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([0.9, 0.4]),
        pred_drs=np.array([0.55, 0.88]),
        nominal_deviation=np.zeros(2),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        option_drs_certificate=True,
        option_drs_certificate_threshold=0.8,
        option_drs_certificate_max_gap=0.6,
        option_drs_certificate_rec_slack=0.5,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert sel.reason == "best_option_drs_certified_prefer_admitted"


def test_relative_recovery_certificate_admits_contact_opportunity_without_soft_fallback():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 0.8, 0.7]),
        r_dep=np.array([-0.95, -0.55, -0.90]),
        hard=np.zeros(3),
        harm=np.zeros(3),
        feasible=np.array([True, True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([1.30, 1.25, 1.50]),
        pred_drs=np.array([0.76, 0.78, 0.91]),
        nominal_deviation=np.zeros(3),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_nominal_rec_lcb_max=-0.60,
        relative_recovery_nominal_gap_min=1.10,
        relative_recovery_min_rec_gain=0.25,
        relative_recovery_min_drs=0.70,
        relative_recovery_max_gap=1.40,
        relative_recovery_max_gap_increase=0.20,
        relative_recovery_bonus=1.0,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert sel.reason == "best_relative_recovery_certified_prefer_admitted"


def test_relative_recovery_certificate_uses_recovery_pool_in_contact():
    # Candidate 1 is a plausible post-contact recovery: it has a nominal hard-rule
    # violation, so v16's `safe & ...` gate would reject it.  The v17 recovery
    # pool allows it only under explicit contact-regime hard/harm bounds.
    sel = calibrated_constrained_select(
        utility=np.array([2.0, 0.6]),
        r_dep=np.array([-1.0, -0.82]),
        hard=np.array([0.0, 2.0]),
        harm=np.array([0.0, 0.2]),
        feasible=np.array([True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([1.30, 1.10]),
        pred_drs=np.array([0.70, 0.78]),
        nominal_deviation=np.array([0.0, 0.4]),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_use_recovery_pool=True,
        recovery_cert_max_hard=2.0,
        recovery_cert_max_harm=1.0,
        relative_recovery_nominal_rec_lcb_max=-0.5,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_min_rec_gain=0.02,
        relative_recovery_min_gap_reduction=0.05,
        relative_recovery_gate="any_gain",
        relative_recovery_min_drs=0.70,
        relative_recovery_max_gap=1.35,
        relative_recovery_max_gap_increase=0.20,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert sel.reason == "best_relative_recovery_certified_prefer_admitted"


def test_relative_recovery_gap_dominance_can_rescue_overconfident_nominal():
    # The model sometimes saturates DRS for a nominal action that has high gap and
    # low teacher recovery.  A candidate that materially reduces the predicted
    # oracle-deployability gap should be certifiable even when scalar R_dep gain is
    # small, but only when the explicit dominance gate is requested.
    sel = calibrated_constrained_select(
        utility=np.array([5.0, 0.5]),
        r_dep=np.array([-0.80, -0.79]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([1.50, 1.05]),
        pred_drs=np.array([0.82, 0.83]),
        nominal_deviation=np.array([0.0, 0.2]),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_nominal_rec_lcb_max=-0.5,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_min_rec_gain=0.10,
        relative_recovery_min_gap_reduction=0.20,
        relative_recovery_gate="any_gain",
        relative_recovery_min_drs=0.70,
        relative_recovery_max_gap=1.35,
        relative_recovery_max_gap_increase=0.20,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]

def test_v18_relative_certificate_requires_drs_gain_when_configured():
    utility = np.array([0.0, -0.1, -0.2], dtype=float)
    r_dep = np.array([-0.70, -0.65, -0.68], dtype=float)
    hard = np.zeros(3, dtype=float)
    harm = np.zeros(3, dtype=float)
    feasible = np.ones(3, dtype=bool)
    gap = np.array([1.20, 1.00, 0.90], dtype=float)
    drs = np.array([0.80, 0.79, 0.83], dtype=float)
    sel = calibrated_constrained_select(
        utility, r_dep, hard, harm, feasible,
        gamma_rec=0.0, gamma_H=0.0, gamma_D=0.0,
        pred_gap=gap, pred_drs=drs,
        lcb_beta=0.1,
        prefer_admitted=True,
        require_admitted_intervention=True,
        relative_recovery_certificate=True,
        relative_recovery_use_recovery_pool=True,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_gate="drs_or_gap",
        relative_recovery_min_drs=0.60,
        relative_recovery_min_drs_gain=0.02,
        relative_recovery_max_drs_drop=0.0,
        relative_recovery_max_gap=1.50,
        relative_recovery_max_gap_increase=0.50,
        relative_recovery_min_gap_reduction=0.05,
        relative_recovery_min_rec_gain=0.0,
        recovery_cert_max_hard=2.0,
        recovery_cert_max_harm=2.0,
        intervention_min_pred_drs=0.60,
        intervention_max_pred_gap=1.50,
        require_intervention_evidence=True,
        relative_recovery_counts_as_evidence=True,
        unadmitted_fallback_to_nominal=True,
    )
    # Candidate 1 closes the gap but lowers DRS, so it is blocked; candidate 2
    # improves DRS and is certified.
    assert sel.selected_index == 2
    assert "relative_recovery" in sel.reason


def test_v18_relative_certificate_abstains_when_only_gap_improves_but_drs_drops():
    utility = np.array([0.0, 1.0], dtype=float)
    r_dep = np.array([-0.70, -0.65], dtype=float)
    hard = np.zeros(2, dtype=float)
    harm = np.zeros(2, dtype=float)
    feasible = np.ones(2, dtype=bool)
    gap = np.array([1.20, 0.80], dtype=float)
    drs = np.array([0.80, 0.75], dtype=float)
    sel = calibrated_constrained_select(
        utility, r_dep, hard, harm, feasible,
        gamma_rec=0.0, gamma_H=0.0, gamma_D=0.0,
        pred_gap=gap, pred_drs=drs,
        lcb_beta=0.1,
        prefer_admitted=True,
        require_admitted_intervention=True,
        relative_recovery_certificate=True,
        relative_recovery_use_recovery_pool=True,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_gate="drs_or_gap",
        relative_recovery_min_drs=0.60,
        relative_recovery_min_drs_gain=0.01,
        relative_recovery_max_drs_drop=0.0,
        relative_recovery_max_gap=1.50,
        relative_recovery_max_gap_increase=0.50,
        relative_recovery_min_gap_reduction=0.05,
        recovery_cert_max_hard=2.0,
        recovery_cert_max_harm=2.0,
        intervention_min_pred_drs=0.60,
        intervention_max_pred_gap=1.50,
        require_intervention_evidence=True,
        relative_recovery_counts_as_evidence=True,
        unadmitted_fallback_to_nominal=True,
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_no_certified_intervention_preserved"


def test_v19_relative_recovery_blocks_nominal_like_macro_even_when_certificate_passes():
    sel = calibrated_constrained_select(
        utility=np.array([0.0, 2.0, 0.1], dtype=float),
        r_dep=np.array([-0.70, -0.62, -0.66], dtype=float),
        hard=np.zeros(3, dtype=float),
        harm=np.zeros(3, dtype=float),
        feasible=np.ones(3, dtype=bool),
        gamma_rec=0.0, gamma_H=0.0, gamma_D=0.0,
        pred_gap=np.array([1.20, 0.90, 1.00], dtype=float),
        pred_drs=np.array([0.80, 0.84, 0.83], dtype=float),
        nominal_deviation=np.array([0.0, 0.1, 0.1], dtype=float),
        candidate_macro_names=["nominal", "perturb_nominal", "brake"],
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_use_recovery_pool=True,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_gate="drs_or_gap",
        relative_recovery_min_drs=0.60,
        relative_recovery_min_drs_gain=0.02,
        relative_recovery_min_gap_reduction=0.05,
        recovery_cert_max_hard=2.0,
        recovery_cert_max_harm=2.0,
        intervention_min_pred_drs=0.60,
        intervention_max_pred_gap=1.50,
        relative_recovery_macro_blocklist="perturb_nominal,keep,pull_over",
        relative_recovery_require_macro=True,
        relative_recovery_counts_as_evidence=True,
        relative_recovery_bonus=1.0,
        unadmitted_fallback_to_nominal=True,
    )
    assert sel.selected_index == 2
    assert sel.admitted[2]
    assert not sel.admitted[1]
    assert "relative_recovery" in sel.reason


def test_v19_relative_recovery_macro_allowlist_abstains_without_recovery_macro():
    sel = calibrated_constrained_select(
        utility=np.array([0.0, 2.0], dtype=float),
        r_dep=np.array([-0.70, -0.62], dtype=float),
        hard=np.zeros(2, dtype=float),
        harm=np.zeros(2, dtype=float),
        feasible=np.ones(2, dtype=bool),
        gamma_rec=0.0, gamma_H=0.0, gamma_D=0.0,
        pred_gap=np.array([1.20, 0.90], dtype=float),
        pred_drs=np.array([0.80, 0.84], dtype=float),
        candidate_macro_names=["nominal", "perturb_nominal"],
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_use_recovery_pool=True,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_gate="drs_or_gap",
        relative_recovery_min_drs=0.60,
        relative_recovery_min_drs_gain=0.02,
        relative_recovery_min_gap_reduction=0.05,
        recovery_cert_max_hard=2.0,
        recovery_cert_max_harm=2.0,
        intervention_min_pred_drs=0.60,
        intervention_max_pred_gap=1.50,
        relative_recovery_macro_allowlist="brake,stabilize,yield,merge",
        relative_recovery_require_macro=True,
        relative_recovery_counts_as_evidence=True,
        unadmitted_fallback_to_nominal=True,
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_no_certified_intervention_preserved"


def test_v20_protective_macro_certificate_rescues_brake_when_relative_is_strict():
    sel = calibrated_constrained_select(
        utility=np.array([0.0, 0.2, 2.0], dtype=float),
        r_dep=np.array([-0.80, -0.82, -0.70], dtype=float),
        hard=np.zeros(3, dtype=float),
        harm=np.zeros(3, dtype=float),
        feasible=np.ones(3, dtype=bool),
        gamma_rec=0.0, gamma_H=0.0, gamma_D=0.0,
        pred_gap=np.array([1.30, 1.15, 0.80], dtype=float),
        pred_drs=np.array([0.76, 0.74, 0.84], dtype=float),
        candidate_macro_names=["nominal", "brake", "perturb_nominal"],
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        # v19-style relative gate is strict and would not admit candidate 1.
        relative_recovery_certificate=True,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_gate="pareto",
        relative_recovery_min_improvement_axes=2,
        relative_recovery_min_rec_gain=0.04,
        relative_recovery_min_drs_gain=0.04,
        relative_recovery_min_gap_reduction=0.20,
        relative_recovery_min_drs=0.70,
        relative_recovery_macro_allowlist="brake,stabilize,merge",
        relative_recovery_macro_blocklist="perturb_nominal,keep,pull_over",
        relative_recovery_require_macro=True,
        # v20 protective channel admits the semantic brake with gap improvement and
        # bounded DRS/LCB non-inferiority, but still blocks perturb_nominal.
        protective_macro_certificate=True,
        protective_macro_allowlist="brake,stabilize",
        protective_macro_blocklist="nominal,keep,perturb_nominal,pull_over",
        protective_macro_nominal_gap_min=1.0,
        protective_macro_min_gap_reduction=0.10,
        protective_macro_min_drs=0.70,
        protective_macro_max_drs_drop=0.05,
        protective_macro_max_rec_lcb_drop=0.15,
        protective_macro_max_gap=1.40,
        protective_macro_max_hard=0.0,
        protective_macro_max_harm=0.0,
        protective_macro_counts_as_evidence=True,
        intervention_min_pred_drs=0.70,
        intervention_max_pred_gap=1.40,
        unadmitted_fallback_to_nominal=True,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert not sel.admitted[2]
    assert sel.reason == "best_protective_macro_recovery_certified_prefer_admitted"


def test_v20_protective_macro_certificate_does_not_admit_pull_over_or_perturbation():
    sel = calibrated_constrained_select(
        utility=np.array([0.0, 2.0, 1.5], dtype=float),
        r_dep=np.array([-0.80, -0.70, -0.68], dtype=float),
        hard=np.zeros(3, dtype=float),
        harm=np.zeros(3, dtype=float),
        feasible=np.ones(3, dtype=bool),
        gamma_rec=0.0, gamma_H=0.0, gamma_D=0.0,
        pred_gap=np.array([1.30, 0.80, 0.75], dtype=float),
        pred_drs=np.array([0.76, 0.80, 0.82], dtype=float),
        candidate_macro_names=["nominal", "pull_over", "perturb_nominal"],
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        protective_macro_certificate=True,
        protective_macro_allowlist="brake,stabilize",
        protective_macro_blocklist="nominal,keep,perturb_nominal,pull_over",
        protective_macro_nominal_gap_min=1.0,
        protective_macro_min_gap_reduction=0.10,
        protective_macro_min_drs=0.70,
        protective_macro_max_gap=1.40,
        protective_macro_max_hard=0.0,
        protective_macro_max_harm=0.0,
        protective_macro_counts_as_evidence=True,
        intervention_min_pred_drs=0.70,
        intervention_max_pred_gap=1.40,
        unadmitted_fallback_to_nominal=True,
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_no_certified_intervention_preserved"
