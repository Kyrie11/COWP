from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = "v4836-1000-deadbeefcafe"
LEGACY = "legacy-untracked"


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_failure(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    source = tmp_path / "source"
    protocol = tmp_path / "protocol"
    source.mkdir()
    protocol.mkdir()
    variants = {}
    repair_variants = {}
    for variant in ("balanced", "precision"):
        ckpt = run / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes((variant + "-checkpoint").encode())
        digest = _sha(ckpt)
        variants[variant] = {"checkpoint": str(ckpt), "sha256": digest}
        repair_variants[variant] = {"final_sha256": digest}
        cal = run / "candidates" / variant / "calibration"
        _write(
            cal / "CERTIFICATE_CALIBRATION_COMPLETE.json",
            {
                "event": "v48_36_certificate_pool_calibration_complete",
                "attempt_id": LEGACY,
                "near_exit_code": 3,
                "contact_exit_code": 3,
                "certificate_executed": True,
                "gate_evaluated": True,
                "certificate_data_valid": True,
                "test_roots_read": False,
            },
        )
        _write(cal / "SAFE_REGIME_STATUS.json", {"event": "v48_36_safe_regime_status", "attempt_id": LEGACY, "test_roots_read": False})
        for bucket in ("near", "contact"):
            _write(
                cal / f"direct_value_risk_{bucket}_v48.json",
                {"certificate_data_valid": True, "gate_evaluated": True, "valid_for_deployment": False},
            )

    failure = {
        "event": "v48_36_pipeline_failed",
        "attempt_id": ATTEMPT,
        "stage": "terminal_state_contract",
        "raw_exit_code": 4,
        "normalized_exit_code": 30,
        "pipeline_exit_code": 30,
        "adaptation_exit_codes": {"balanced": 0, "precision": 0},
        "certificate_executed": True,
        "gate_evaluated": True,
        "pipeline_valid": False,
        "test_roots_read": False,
    }
    complete = {
        "event": "v48_36_ocaf_controller_complete",
        "attempt_id": ATTEMPT,
        "source_run": str(source),
        "protocol_root": str(protocol),
        "variants": variants,
        "raw_certificate_exit_code": 4,
        "certificate_exit_code": 30,
        "pipeline_exit_code": 30,
        "certificate_executed": True,
        "gate_evaluated": True,
        "gate_passed": False,
        "pipeline_valid": False,
        "failure_stage": "terminal_state_contract",
        "test_roots_read": False,
    }
    selection = {
        "event": "v48_36_certificate_candidate_selection",
        "attempt_id": LEGACY,
        "requested_variants": ["balanced", "precision"],
        "controller_exit_codes": {"balanced": 20, "precision": 20},
        "certificate_executed": True,
        "gate_evaluated": True,
        "valid_candidates": [],
        "test_roots_read": False,
    }
    _write(run / "PIPELINE_FAILED.json", failure)
    _write(run / "V48_36_COMPLETE.json", complete)
    _write(run / "AUTHORITATIVE_RUN_STATUS.json", {"valid": True, "authoritative_exit_code": 30, "test_roots_read": False})
    _write(run / "dedicated_recalibration_status.json", selection)
    _write(run / "GATE_SPEC.json", {"event": "v48_36_gate_protocol_preregistered", "attempt_id": LEGACY, "protocol_sha256": "abc", "protocol": {}, "test_roots_read": False})
    _write(
        run / "V48_36_2_STAGE_TRANSFER_REPAIR.json",
        {
            "valid": True,
            "retraining_performed": False,
            "source_run": str(source),
            "protocol_root": str(protocol),
            "variants": repair_variants,
            "test_roots_read": False,
        },
    )
    _write(run / "V48_36_RESUME_CONTRACT.json", {"valid": True, "failure_mode": "repaired_stage_transfer", "test_roots_read": False})
    _write(
        run / "ATTEMPT_STARTED.json",
        {"event": "v48_36_attempt_started", "attempt_id": "v4836-2000-abandoned0000", "test_roots_read": False},
    )
    _write(run / "NEXT_COMMANDS_STATUS.json", {"attempt_id": ATTEMPT, "reason": "pipeline_failure", "test_roots_read": False})
    _write(run / "NEXT_COMMANDS_BLOCKED.json", {"attempt_id": ATTEMPT, "reason": "pipeline_failure", "test_roots_read": False})
    archived = run / "status_history" / f"overridden-by-{ATTEMPT}-1" / "GATE_FAILED.json"
    _write(archived, selection)
    failed_audit = {
        "valid": False,
        "authoritative_exit_code": 20,
        "pipeline_valid": True,
        "attempt_id": ATTEMPT,
        "checks": {
            "json_readable": True,
            "terminal_completion_present": True,
            "no_active_status_contradictions": False,
            "pipeline_valid": True,
            "certificate_executed": True,
            "gate_evaluated": True,
            "gate_passed_false": True,
            "next_commands_not_generated": True,
            "next_commands_absent": True,
            "next_commands_blocked": True,
            "next_status_is_natural_gate_failure": True,
            "gate_failure_marker_present": False,
            "no_calibration_failure_marker": True,
            "no_pipeline_failure_marker": True,
            "expected_exit_code": True,
            "expected_attempt_id": True,
        },
        "active_contradictions": ["gate_failed belongs to an older attempt but is required for RC=20"],
        "test_roots_read": False,
    }
    (run / "logs").mkdir(exist_ok=True)
    (run / "logs" / "authoritative_run_state.log").write_text(json.dumps(failed_audit) + "\n")
    return run


def _run_status_contract(run: Path, attempt: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_certificate_status_contract.py"),
            "--run",
            str(run),
            "--expected-attempt-id",
            attempt,
            "--output",
            str(run / "status_contract.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_certificate_status_contract_rejects_legacy_attempt(tmp_path: Path) -> None:
    run = _build_failure(tmp_path)
    completed = _run_status_contract(run, ATTEMPT)
    doc = json.loads((run / "status_contract.json").read_text())
    assert completed.returncode == 4
    assert doc["valid"] is False
    assert doc["checks"]["selection_attempt_matches"] is False


def test_exact_terminal_state_failure_repairs_to_authoritative_rc20(tmp_path: Path) -> None:
    run = _build_failure(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "repair_v48_36_2_terminal_state_failure.py"),
            "--run",
            str(run),
            "--repo",
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    repair = json.loads((run / "V48_36_3_TERMINAL_STATE_REPAIR.json").read_text())
    state = json.loads((run / "AUTHORITATIVE_RUN_STATUS.json").read_text())
    complete = json.loads((run / "V48_36_COMPLETE.json").read_text())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert repair["valid"] is True
    assert repair["retraining_performed"] is False
    assert repair["recalibration_performed"] is False
    assert state["valid"] is True and state["authoritative_exit_code"] == 20
    assert complete["pipeline_valid"] is True and complete["pipeline_exit_code"] == 20
    assert not (run / "PIPELINE_FAILED.json").exists()
    assert json.loads((run / "GATE_FAILED.json").read_text())["attempt_id"] == ATTEMPT
    assert json.loads((run / "ATTEMPT_STARTED.json").read_text())["attempt_id"] == ATTEMPT
    assert list((run / "repair_history").glob("v48.36.3-terminal-state-*"))


def test_terminal_repair_rejects_unknown_algorithm_result(tmp_path: Path) -> None:
    run = _build_failure(tmp_path)
    risk = run / "candidates" / "precision" / "calibration" / "direct_value_risk_contact_v48.json"
    doc = json.loads(risk.read_text())
    doc["valid_for_deployment"] = True
    _write(risk, doc)
    before = (run / "V48_36_COMPLETE.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "repair_v48_36_2_terminal_state_failure.py"), "--run", str(run), "--repo", str(ROOT)],
        cwd=ROOT,
        check=False,
    )
    repair = json.loads((run / "V48_36_3_TERMINAL_STATE_REPAIR.json").read_text())
    assert completed.returncode == 4
    assert repair["valid"] is False
    assert "precision_contact_certificate_is_natural_rejection" in repair["errors"]
    assert (run / "V48_36_COMPLETE.json").read_bytes() == before


def test_v48_36_controller_and_calibration_use_same_attempt_namespace() -> None:
    calibration = (ROOT / "scripts" / "calibrate_v48_36_shared_certificate_pool.sh").read_text()
    controller = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    assert "V4835_ATTEMPT_ID" not in calibration
    assert calibration.count("V4836_ATTEMPT_ID") >= 5
    assert 'V4836_ATTEMPT_ID="$ATTEMPT_ID"' in controller
    assert "check_v48_36_certificate_status_contract.py" in controller
    assert "certificate_status_contract" in controller
    assert "v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX" in controller


def test_calibration_launcher_fails_closed_without_attempt_id(tmp_path: Path) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "OUTPUTDIR": str(tmp_path / "out"),
        "CAL_SAFE": str(tmp_path / "safe"),
        "CERT_NEAR": str(tmp_path / "near"),
        "CERT_CONTACT": str(tmp_path / "contact"),
        "DEV_NEAR": str(tmp_path / "dev-near"),
        "DEV_CONTACT": str(tmp_path / "dev-contact"),
        "OCRAP_REPO": str(ROOT),
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "calibrate_v48_36_shared_certificate_pool.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 30
    assert "V4836_ATTEMPT_ID is required" in completed.stderr


def test_terminal_repair_rolls_back_when_post_repair_resolver_fails(tmp_path: Path) -> None:
    run = _build_failure(tmp_path)
    fake_repo = tmp_path / "fake-repo"
    tools = fake_repo / "tools"
    tools.mkdir(parents=True)
    (tools / "check_v48_36_certificate_status_contract.py").write_bytes(
        (ROOT / "tools" / "check_v48_36_certificate_status_contract.py").read_bytes()
    )
    (tools / "resolve_v48_36_authoritative_result.py").write_text(
        "import sys\nraise SystemExit(4)\n", encoding="utf-8"
    )
    tracked = [
        run / "ATTEMPT_STARTED.json",
        run / "GATE_SPEC.json",
        run / "dedicated_recalibration_status.json",
        run / "NEXT_COMMANDS_STATUS.json",
        run / "NEXT_COMMANDS_BLOCKED.json",
        run / "PIPELINE_FAILED.json",
        run / "V48_36_COMPLETE.json",
        run / "AUTHORITATIVE_RUN_STATUS.json",
    ]
    before = {str(path.relative_to(run)): path.read_bytes() for path in tracked}
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "repair_v48_36_2_terminal_state_failure.py"),
            "--run",
            str(run),
            "--repo",
            str(fake_repo),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    repair = json.loads((run / "V48_36_3_TERMINAL_STATE_REPAIR.json").read_text())
    assert completed.returncode == 4
    assert repair["valid"] is False and repair["rolled_back"] is True
    for rel, payload in before.items():
        assert (run / rel).read_bytes() == payload
    assert not (run / "GATE_FAILED.json").exists()
