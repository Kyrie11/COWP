from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_V16_8_20_LABEL_FP = "c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51"


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _probe_inputs(tmp_path: Path, *, eligible_rate: float = 639 / 800, ncf_rate: float = 398 / 800, pbtr: float = 241 / 639):
    paired = {
        "num_representative_scenes": 800,
        "pairing_completeness": {"complete": True, "build_error_count": 0},
        "new": {
            "any_valid_scene_rate": 1.0,
            "any_ncf_scene_rate": 0.315,
            "best_case_selected_false_safe_lower_bound": 0.63875,
            "any_priority_eligible_scene_rate": eligible_rate,
            "any_priority_ncf_scene_rate": ncf_rate,
            "best_case_pbtr_lower_bound": pbtr,
        },
        "paired": {"old_hard_scene_count": 400, "hard_scene_ncf_recovery_rate": 0.2525},
    }
    ablation = {
        "ablations": {
            "all": {
                "any_ncf_scene_rate": 0.32,
                "best_case_selected_false_safe_lower_bound": 0.64,
                "best_case_pbtr_lower_bound": 0.40,
            },
            "without_priority_smooth_yield": {
                "any_ncf_scene_rate": 0.31,
                "best_case_selected_false_safe_lower_bound": 0.65,
                "best_case_pbtr_lower_bound": 0.41,
            },
            "without_joint_route_ncf": {},
        },
        "proposal_source_candidate_counts": {"JOINT_ROUTE_NCF": 1000},
        "joint_route_ncf_increment": {"delta_priority_ncf_scene_rate": 0.001},
    }
    profile = {
        "unique_scenarios": 1200,
        "critical_selection_reference_modes": {"causal_anchor_v2": 1200},
        "conflict_region_selection": {
            "profiled_scenes": 1200,
            "ego_reference_used_rate": 1.0,
            "candidate_pool_saturation_rate": 0.0,
        },
    }
    audit = {
        "pair_rates": {"relevant": 0.44, "burden_only_root_fraction": 0.0, "burden_only_scene_rate": 0.0},
        "root_counts": {"burden_only": 0},
        "integrity": {
            "affected_definition_consistent": True,
            "burden_only_definition_consistent": True,
            "no_read_errors": True,
            "no_silent_blockers": True,
            "no_irrelevant_blockers": True,
            "transport_affected_matches_audit": True,
            "transport_conflict_matches_audit": True,
            "transport_retain_matches_audit": True,
            "canonical_root_weight_matches_transport": True,
            "no_responses_for_irrelevant_pairs": True,
        },
    }
    return (
        _write_json(tmp_path / "paired.json", paired),
        _write_json(tmp_path / "ablation.json", ablation),
        _write_json(tmp_path / "profile.json", profile),
        _write_json(tmp_path / "audit.json", audit),
    )


def _run_screen(tmp_path: Path, **kwargs):
    paired, ablation, profile, audit = _probe_inputs(tmp_path, **kwargs)
    out = tmp_path / "screen.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cowp.scripts.58_screen_v16_8_9_causal_audit_probe",
            "--paired-probe",
            str(paired),
            "--source-ablation",
            str(ablation),
            "--profile-summary",
            str(profile),
            "--audit-diagnostic",
            str(audit),
            "--output",
            str(out),
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc, json.loads(out.read_text(encoding="utf-8"))


def test_v16_8_21_policy_does_not_change_label_semantics() -> None:
    gate = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    assert gate.current_label_semantic_fingerprint(Path.cwd()) == EXPECTED_V16_8_20_LABEL_FP


def test_strict_gates_conditional_protected_support_not_womd_prevalence(tmp_path: Path) -> None:
    proc, result = _run_screen(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert result["screen_pass"] is True
    assert result["observed"]["new_priority_eligible_scene_rate"] < 0.90
    assert result["observed"]["new_priority_ncf_scene_rate"] < 0.50
    assert result["observed"]["priority_eligible_scene_count"] == 639
    assert result["observed"]["priority_ncf_scene_count"] == 398
    assert abs(result["observed"]["protected_priority_ncf_given_eligible"] - (398 / 639)) < 1e-12
    assert result["observed"]["protected_partition_error"] < 1e-12
    assert result["checks"]["protected_eligible_support_count"] is True
    assert result["checks"]["protected_ncf_support_count"] is True
    assert "priority_eligible" not in result["checks"]
    assert "priority_ncf" not in result["checks"]


def test_evidence_count_floor_prevents_tiny_eligible_denominator_from_passing(tmp_path: Path) -> None:
    eligible = 160 / 800
    ncf = 152 / 800
    pbtr = 8 / 160
    proc, result = _run_screen(tmp_path, eligible_rate=eligible, ncf_rate=ncf, pbtr=pbtr)
    assert proc.returncode == 2
    assert result["screen_pass"] is False
    assert result["checks"]["protected_eligible_support_count"] is False
    assert result["checks"]["pbtr_floor"] is True
