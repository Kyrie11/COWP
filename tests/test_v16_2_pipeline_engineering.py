from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


def test_natural_mode_usage_avoids_old_torch_tuple_any_api() -> None:
    source = Path("cowp/models/losses.py").read_text(encoding="utf-8")
    assert ".any(dim=(" not in source
    assert "mode_mask.any(dim=0).any(dim=0)" in source


def test_full_wrapper_really_enables_full_waymax() -> None:
    source = Path("NEXT_RUN_COMMANDS_V16_2_FULL_CN.sh").read_text(encoding="utf-8")
    assert 'RUN_FULL="${RUN_FULL:-1}"' in source
    assert "REQUIRE_WAYMAX_PREFLIGHT=1" in source


def test_parallel_pipeline_waits_all_children_and_reports_offroad() -> None:
    runner = Path("run_cowp_v16_2_dual_gpu.sh").read_text(encoding="utf-8")
    summary = Path("cowp/scripts/24_summarize_planner_delta.py").read_text(encoding="utf-8")
    assert "wait_all()" in runner
    assert "validate_pipeline_outputs" in runner
    assert '"Offroad"' in summary


def test_pipeline_output_validator_accepts_required_closed_loop_metrics(tmp_path: Path) -> None:
    root = tmp_path / "run"
    required = [
        "checkpoints/natural/cowp_natural_best.pt",
        "checkpoints/natural/history_natural.json",
        "eval/learned_offline/natural_basis_gate.json",
        "eval/learned_offline/natural_effectiveness_gate.json",
        "checkpoints/transport/cowp_witness_best.pt",
        "checkpoints/planner/cowp_planner_best.pt",
        "eval/learned_offline/mechanism_verification.json",
    ]
    for rel in required:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]" if "history" in rel else "{}", encoding="utf-8")
    delta = {
        "CR": {"reference": 0.1, "candidate": 0.08, "delta": -0.02},
        "OffroadRate": {"reference": 0.02, "candidate": 0.02, "delta": 0.0},
        "EP": {"reference": 0.8, "candidate": 0.81, "delta": 0.01},
    }
    p = root / "eval/probe/delta_conventional_vs_root_transport.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(delta), encoding="utf-8")
    out = root / "eval/completion.json"
    subprocess.run(
        [sys.executable, "-m", "cowp.scripts.44_validate_pipeline_outputs", "--out-root", str(root), "--level", "probe", "--output", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(out.read_text(encoding="utf-8"))["pass"] is True
