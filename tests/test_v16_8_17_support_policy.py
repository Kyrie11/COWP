from __future__ import annotations

import importlib
from pathlib import Path


def test_v16_8_17_smoke_coverage_policy_treats_current_smoke_as_uncertain_not_failure():
    mod = importlib.import_module("cowp.scripts.65_audit_model_support")
    unaud_ci = mod._wilson(21, 538)
    cert_ci = mod._wilson(80, 96)
    assert mod._coverage_check_max(21 / 538, unaud_ci, 0.05, "wilson_gross_failure")
    assert mod._coverage_check_min(80 / 96, cert_ci, 0.75, "wilson_gross_failure")
    # The retired policy really was stricter than the observed support.
    assert not mod._coverage_check_max(21 / 538, unaud_ci, 0.01, "point")
    assert not mod._coverage_check_min(80 / 96, cert_ci, 0.95, "point")


def test_v16_8_17_strict_point_policy_requires_high_pair_coverage_but_not_all_scene_completion():
    mod = importlib.import_module("cowp.scripts.65_audit_model_support")
    unaud_ci = mod._wilson(21, 538)
    cert_ci = mod._wilson(80, 96)
    assert mod._coverage_check_max(21 / 538, unaud_ci, 0.05, "point")
    assert mod._coverage_check_min(80 / 96, cert_ci, 0.75, "point")


def test_v16_8_17_hard_random_missingness_gaps_are_small_on_uploaded_smoke_counts():
    hard_unaud = 12 / 277
    random_unaud = 9 / 261
    hard_cert = 39 / 48
    random_cert = 41 / 48
    assert abs(hard_unaud - random_unaud) < 0.03
    assert abs(hard_cert - random_cert) < 0.10


def test_v16_8_19_label_semantic_fingerprint_matches_constructive_support_patch():
    gate = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    got = gate.current_label_semantic_fingerprint(Path(__file__).resolve().parents[1])
    assert got == "51844462540c083592280a7a8c24da962aba9d743a92b41d1a7f27095f0c2452"


def test_v16_8_17_wrappers_use_policy_checks_not_hardcoded_one_percent():
    root = Path(__file__).resolve().parents[1]
    smoke = (root / "NEXT_RUN_COMMANDS_V16_8_17_CAUSAL_AUDIT_SMOKE_CN.sh").read_text(encoding="utf-8")
    strict = (root / "NEXT_RUN_COMMANDS_V16_8_17_STRICT_PROPOSAL_PROBE_CN.sh").read_text(encoding="utf-8")
    pilot = (root / "NEXT_TRAIN_PILOT_V16_8_17_CN.sh").read_text(encoding="utf-8")
    assert "--max-unauditable-critical-rate 0.05" in smoke
    assert "--min-certificate-complete-scene-rate 0.75" in smoke
    assert "--coverage-gate-mode wilson_gross_failure" in smoke
    for text in (strict, pilot):
        assert "--max-unauditable-critical-rate 0.05" in text
        assert "--min-certificate-complete-scene-rate 0.75" in text
        assert "--coverage-gate-mode point" in text
    assert "float(nat.get('mechanism_unauditable_rate',1.0)) <= max_unauditable" not in smoke
    assert "float(nat.get('mechanism_unauditable_rate',1.0))<=max_unauditable" not in strict
