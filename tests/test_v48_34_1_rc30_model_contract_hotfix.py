from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, check=False)


def _scene(key: str, *, intervention: float, nup: float, near_gain: float, unsafe: float = 0.0) -> dict:
    return {
        "target_key": key,
        "scene_id": key.split(":")[0],
        "target_time_index": 10,
        "intervention_rate": intervention,
        "closed_loop_bounded_NUP": nup,
        "metric_summary": {
            "ttc_s_p05": 1.0 + near_gain,
            "terminal_ttc_s": 1.5 + near_gain,
            "min_clearance_m_p05": 0.3 + near_gain,
            "terminal_clearance_m": 0.5 + near_gain,
            "critical_ttc_exposure_duration_s": max(0.0, 1.0 - near_gain),
            "near_zero_clearance_exposure_rate": max(0.0, 0.2 - near_gain),
            "clearance_deficit_auc_m_s": max(0.0, 2.0 - near_gain),
            "ttc_deficit_auc_s2": max(0.0, 2.0 - near_gain),
            "overlap_any": unsafe,
            "offroad_any": 0.0,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps({"scene": x}) + "\n" for x in rows), encoding="utf-8")


def test_v48_34_model_checker_accepts_barrier_mode_at_parse_time(tmp_path: Path) -> None:
    proc = _run(
        sys.executable,
        "tools/check_v48_34_model_contract.py",
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--support-contract", str(tmp_path / "missing.json"),
        "--output", str(tmp_path / "out.json"),
        "--expect-admission-prior-mode", "barrier_gated_slack",
    )
    assert proc.returncode != 2
    assert "invalid choice" not in proc.stderr
    assert "support contract not found" in (proc.stderr + proc.stdout)


def test_v48_34_controllers_use_version_specific_model_checker() -> None:
    for name in ("run_v48_34_barrier_crossfit_dedicated.sh", "run_v48_34_barrier_crossfit_ablations.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "check_v48_34_model_contract.py" in text
        assert "check_v48_32_model_contract.py" not in text


def test_exploratory_adaptation_scope_uses_correct_split_and_standard_validation() -> None:
    text = (ROOT / "scripts/run_v48_34_1_exploratory_closed_loop_baselines_and_videos.sh").read_text()
    adaptation = text[text.index("if [[ \"$EXPLORATORY_DATA_SCOPE\" == adaptation_dev") : text.index("elif [[ \"$EXPLORATORY_DATA_SCOPE\" == heldout_test")]
    assert "CONTACT_BUCKET_SPLIT:=evidence_adapt_dev" in adaptation
    assert "CONTACT_WOMD_SOURCE:=$WOMD_VAL" in adaptation
    assert "WOMD_VAL_INTERACTIVE" not in adaptation
    assert "closed_loop.label_mode=fast" in text
    assert "AUDIT_LABEL_MODE=fast" in text
    assert "build_v48_34_1_progress_tables.py" in text


def test_paired_report_emits_absolute_and_delta_metrics(tmp_path: Path) -> None:
    control = tmp_path / "control.json.scenes.jsonl"
    method = tmp_path / "method.json.scenes.jsonl"
    _write_jsonl(control, [_scene("s1:t10", intervention=0.0, nup=0.8, near_gain=0.0), _scene("s2:t10", intervention=0.0, nup=0.7, near_gain=0.0)])
    _write_jsonl(method, [_scene("s1:t10", intervention=0.2, nup=0.9, near_gain=0.2), _scene("s2:t10", intervention=0.1, nup=0.8, near_gain=0.1)])
    out_json = tmp_path / "report.json"; out_csv = tmp_path / "report.csv"
    proc = _run(
        sys.executable, "tools/build_v48_34_paired_baseline_report.py",
        "--regime", "near", "--reference", f"scalar={control}", "--method", f"ocrap_precision={method}",
        "--output-json", str(out_json), "--output-csv", str(out_csv), "--bootstrap", "20", "--require-core-metrics",
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out_json.read_text())
    rep = doc["methods"][0]["metrics"]["ttc_s_p05"]
    assert rep["reference_mean"] == 1.0
    assert rep["method_mean"] > rep["reference_mean"]
    assert rep["raw_delta_mean"] > 0
    with out_csv.open() as f:
        fields = next(csv.reader(f))
    assert "method_mean" in fields and "reference_mean" in fields and "raw_delta_mean" in fields
    assert out_csv.with_name("report.wide.csv").is_file()


def test_critical_scene_selection_has_complete_metrics_and_no_duplicate_categories(tmp_path: Path) -> None:
    control = tmp_path / "control.json.scenes.jsonl"
    method = tmp_path / "method.json.scenes.jsonl"
    _write_jsonl(control, [_scene("s1:t10", intervention=0.0, nup=0.8, near_gain=0.0), _scene("s2:t10", intervention=0.0, nup=0.8, near_gain=0.0)])
    _write_jsonl(method, [_scene("s1:t10", intervention=0.2, nup=0.9, near_gain=0.3), _scene("s2:t10", intervention=0.2, nup=0.6, near_gain=-0.2, unsafe=1.0)])
    out = tmp_path / "selection.json"
    proc = _run(
        sys.executable, "tools/select_critical_scenes_v48_34.py",
        "--method-scenes", str(method), "--control-scenes", str(control), "--regime", "near",
        "--num-positive", "1", "--num-failure", "1", "--output", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    selected = json.loads(out.read_text())["selected"]
    assert len(selected) == 2
    assert len({x["target_key"] for x in selected}) == 2
    positive = next(x for x in selected if x["category"] == "positive_toy_example")
    assert positive["missing_required_metrics"] == []
    assert positive["score"] > 0


def test_audit_runner_exposes_max_targets_per_scene() -> None:
    text = (ROOT / "scripts/run_ocrap_v48_trac_sr.sh").read_text()
    assert "AUDIT_MAX_TARGETS_PER_SCENE" in text


def test_repair_script_refuses_retraining_and_requires_known_signature() -> None:
    text = (ROOT / "scripts/repair_v48_34_rc30_model_contract_with_v48_34_1.sh").read_text()
    assert "adaptation_reused_without_retraining" in text
    assert "unexpected_failure_stage" in text
    assert "check_v48_34_model_contract.py" in text
    assert "adapt_ocrap_v48_34" not in text
