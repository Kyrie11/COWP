from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select
from ocrap.simulation.closed_loop_runner import _aggregate_scene_results


def _calibrator():
    path = Path(__file__).parents[1] / "tools" / "calibrate_direct_value_risk_v47.py"
    spec = importlib.util.spec_from_file_location("calibrate_v47", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v47_model_exposes_gain_harm_and_robust_disagreement() -> None:
    layout = FlatFeatureLayout(feature_max_agents=2)
    model = OCRAPModel(
        input_dim=layout.total_dim,
        num_roots=2,
        num_options=2,
        d_model=32,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout={"feature_max_agents": 2},
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_value_output="score",
        direct_recovery_opportunity_head=True,
        direct_recovery_harm_head=True,
        direct_recovery_value_experts=True,
        direct_recovery_value_num_experts=2,
        direct_recovery_value_expert_routing="uniform_robust",
        direct_recovery_value_router_pooling="ego_shared_raw",
        direct_recovery_expert_disagreement_penalty=0.5,
    ).eval()
    with torch.no_grad():
        out = model(torch.randn(3, layout.total_dim))
    assert out["direct_recovery_value_logit"].shape == (3,)
    assert out["direct_recovery_opportunity_logit"].shape == (3,)
    assert out["direct_recovery_harm_logit"].shape == (3,)
    assert out["direct_expert_outputs"].shape == (3, 2, 4)
    assert out["direct_expert_output_std"].shape == (3, 4)
    assert torch.allclose(out["direct_expert_weights"], torch.full((3, 2), 0.5))


def test_harm_veto_blocks_high_gain_candidate() -> None:
    common = dict(
        utility=np.array([1.0, 0.8]),
        r_dep=np.array([0.2, 0.2]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.0,
        pred_gap=np.zeros(2),
        pred_drs=np.ones(2),
        nominal_deviation=np.array([0.0, 0.1]),
        pred_direct_value=np.array([0.0, 1.0]),
        pred_direct_std=np.zeros(2),
        pred_direct_opportunity=np.array([0.5, 0.99]),
        candidate_macro_names=["nominal", "brake"],
        regime_name="near_contact",
        direct_value_certificate=True,
        direct_value_score_mode=True,
        direct_value_top1_only=True,
        direct_value_opportunity_threshold=0.8,
        direct_value_harm_threshold=0.2,
        direct_value_macro_allowlist="brake",
        direct_value_min_nominal_deviation=0.002,
        direct_value_min_advantage_lcb=0.1,
        direct_value_challenge_nominal=True,
        stress_rescue_challenge_nominal=True,
        direct_value_bonus=0.5,
    )
    blocked = calibrated_constrained_select(pred_direct_harm=np.array([0.5, 0.9]), **common)
    admitted = calibrated_constrained_select(pred_direct_harm=np.array([0.5, 0.05]), **common)
    assert blocked.selected_index == 0
    assert admitted.selected_index == 1


def test_tri_state_calibration_harm_gate_changes_top1() -> None:
    mod = _calibrator()
    groups = [{
        "scene": "s", "time": 1, "fold": 0, "oracle_best_teacher_adv": 0.2,
        "pairs": [
            {"candidate": 1, "pred_adv": 0.9, "teacher_adv": -0.2, "opportunity": 0.9, "harm_probability": 0.9},
            {"candidate": 2, "pred_adv": 0.7, "teacher_adv": 0.2, "opportunity": 0.9, "harm_probability": 0.1},
        ],
    }]
    no_veto = mod._select_top1(groups, 0.5, 1.0)
    veto = mod._select_top1(groups, 0.5, 0.2)
    assert no_veto[0]["candidate"] == 1
    assert veto[0]["candidate"] == 2


def test_fit_rule_uses_policy_level_precision_and_positive_gain() -> None:
    mod = _calibrator()
    groups = []
    for i in range(8):
        groups.append({
            "scene": f"s{i}", "time": 0, "fold": 0, "oracle_best_teacher_adv": 0.1,
            "pairs": [{
                "candidate": 1, "pred_adv": 0.3 + i * 0.01, "teacher_adv": 0.1,
                "opportunity": 0.9, "harm_probability": 0.05,
            }],
        })
    args = SimpleNamespace(
        min_opportunity=0.0, max_predicted_harm=0.95, min_score_advantage=-0.1,
        positive_gain=0.015, negative_gain=0.01,
        min_fit_selected=4, min_fit_precision=0.5, min_fit_precision_lcb=0.2,
        min_fit_teacher_advantage_mean=0.0,
        max_fit_harmful_selected_rate=0.2, max_fit_harmful_selected_ucb=0.75,
    )
    opp, harm, score, metrics, _ = mod._fit_rule(groups, args)
    assert np.isfinite(opp) and np.isfinite(harm) and np.isfinite(score)
    assert metrics["challenge_precision"] == 1.0
    assert metrics["selected_teacher_advantage_mean"] > 0.0


def test_closed_loop_route_free_metrics_are_aggregated_for_all_regimes() -> None:
    scenes = [
        {"num_decisions": 2, "num_metric_steps": 2, "label_mode": "fast",
         "metric_summary": {"route_free_path_length_m": 5.0, "hard_brake_rate": 0.5,
                            "longitudinal_jerk_abs_max_mps3": 3.0},
         "macro_counts": {}, "selection_reason_counts": {}, "timing": {}},
        {"num_decisions": 2, "num_metric_steps": 2, "label_mode": "fast",
         "metric_summary": {"route_free_path_length_m": 7.0, "hard_brake_rate": 0.0,
                            "longitudinal_jerk_abs_max_mps3": 4.0},
         "macro_counts": {}, "selection_reason_counts": {}, "timing": {}},
    ]
    out = _aggregate_scene_results(scenes, "nominal", "reference")
    assert out["waymax_metrics"]["route_free_path_length_m"] == 6.0
    assert out["waymax_metrics"]["hard_brake_rate"] == 0.25
    assert out["waymax_metrics"]["longitudinal_jerk_abs_max_mps3"] == 4.0


def test_closed_loop_explicit_collision_and_offroad_aliases() -> None:
    scenes = [
        {"num_decisions": 2, "num_metric_steps": 2, "label_mode": "fast",
         "metric_summary": {"overlap_any": 1.0, "overlap_mean": 0.5,
                            "offroad_any": 0.0, "offroad_mean": 0.0,
                            "min_clearance_m_min": 0.1, "ttc_s_min": 0.8},
         "macro_counts": {}, "selection_reason_counts": {}, "timing": {}},
        {"num_decisions": 4, "num_metric_steps": 4, "label_mode": "fast",
         "metric_summary": {"overlap_any": 0.0, "overlap_mean": 0.0,
                            "offroad_any": 1.0, "offroad_mean": 0.25,
                            "min_clearance_m_min": 1.2, "ttc_s_min": 2.0},
         "macro_counts": {}, "selection_reason_counts": {}, "timing": {}},
    ]
    out = _aggregate_scene_results(scenes, "nominal", "reference")
    assert out["collision_scene_rate"] == 0.5
    assert np.isclose(out["collision_step_rate"], 1.0 / 6.0)
    assert out["offroad_scene_rate"] == 0.5
    assert np.isclose(out["offroad_step_rate"], 1.0 / 6.0)
    assert out["minimum_clearance_m"] == 0.1
    assert out["minimum_ttc_s"] == 0.8


def test_sampler_teacher_pcd_uses_same_composite_target(tmp_path: Path) -> None:
    from ocrap.cli.train import _sampler_teacher_pcd

    path = tmp_path / "sample.npz"
    np.savez_compressed(
        path,
        m_star=np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32),
        root_probs=np.asarray([0.5, 0.5], dtype=np.float32),
        c_star=np.eye(2, dtype=np.float32),
        root_valid=np.asarray([1, 1], dtype=np.int64),
        option_valid=np.asarray([1, 1], dtype=np.int64),
        r_dep_star=np.asarray(1.0, dtype=np.float32),
        r_orc_star=np.asarray(1.0, dtype=np.float32),
    )
    value = _sampler_teacher_pcd(path, {"ocmero": {"alpha": 0.2, "beta": 0.2, "top_m": 2}})
    assert 0.0 <= value <= 1.0
    assert value > 0.5


def test_closed_loop_target_loader_respects_explicit_val_split(tmp_path: Path) -> None:
    from ocrap.simulation.closed_loop_runner import _load_closed_loop_targets

    root = tmp_path / "val_safe"
    root.mkdir()
    for split, scene in (("val", "scene-val"), ("test", "scene-test")):
        np.savez_compressed(
            root / f"{scene}.npz",
            split_id=np.asarray(split),
            scene_id=np.asarray(scene),
            time_index=np.asarray(7, dtype=np.int64),
        )
    cfg = {"closed_loop": {"bucket_split": "val", "max_targets_per_scene": 1}}
    targets = _load_closed_loop_targets(str(root), cfg)
    assert len(targets) == 1
    assert targets[0]["scene_id"] == "scene-val"


def test_stress_macro_schedule_frontloads_distinct_recovery_variants() -> None:
    from ocrap.planning.prefix_generation import _macro_params, _macro_sequence_from_cfg

    cfg = {
        "prefix_macro_whitelist": "brake,yield,merge,stabilize",
        "prefix_macro_schedule": "merge,brake,stabilize,yield,merge,brake,stabilize,yield",
    }
    seq = _macro_sequence_from_cfg(cfg, 9)
    assert seq[0] == ("nominal", 0)
    assert seq[1:] == [
        ("merge", 0), ("brake", 0), ("stabilize", 0), ("yield", 0),
        ("merge", 1), ("brake", 1), ("stabilize", 1), ("yield", 1),
    ]
    for macro in ("merge", "brake", "stabilize", "yield"):
        p0 = _macro_params(macro, 0, 12.0, {})
        p1 = _macro_params(macro, 1, 12.0, {})
        assert not np.allclose(p0, p1), macro


def test_v47_orchestrator_separates_train_and_eval_dataset_roots():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "run_v47_two_gpu_fast_commands.txt").read_text(encoding="utf-8")
    assert "TRAIN_OCRAP_ROOT" in text
    assert "EVAL_OCRAP_ROOT" in text
    assert '${TRAIN_OCRAP_ROOT}/train_near_contact' in text
    assert '$EVAL_OCRAP_ROOT/val_near_contact' in text or '${EVAL_OCRAP_ROOT}/val_near_contact' in text
    assert '$EVAL_OCRAP_ROOT/test_near_contact' in text or '${EVAL_OCRAP_ROOT}/test_near_contact' in text
