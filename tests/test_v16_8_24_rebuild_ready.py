from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_active_v24_shell_chain_is_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "active.json"
    subprocess.run(
        [sys.executable, "-m", "cowp.scripts.77_audit_active_execution_chain", "--repo-root", str(ROOT), "--output", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["missing_modules"] == []
    assert payload["missing_shells"] == []
    assert payload["missing_configs"] == []


def test_v23_missing_profile_module_regression_is_fixed() -> None:
    active = [
        ROOT / "BENCHMARK_V16_8_24_FASTPATHS_CN.sh",
        ROOT / "PREPARE_COWP_V16_8_24_FAST_DATA_CN.sh",
    ]
    text = "\n".join(p.read_text(encoding="utf-8") for p in active)
    assert "cowp.scripts.44_summarize_label_build_profile" not in text
    assert "cowp.scripts.49_summarize_label_build_profile" in text
    assert (ROOT / "cowp" / "scripts" / "49_summarize_label_build_profile.py").is_file()


def test_v24_persists_scenario_indices_and_caps_workers() -> None:
    text = (ROOT / "PREPARE_COWP_V16_8_24_FAST_DATA_CN.sh").read_text(encoding="utf-8")
    assert ".cowp_v131_indices" in text
    assert "SCENARIO_INDEX_TRAIN" in text
    assert "SCENARIO_INDEX_VAL" in text
    assert "DEFAULT_LABEL_WORKERS > 40" in text
    assert "OMP_NUM_THREADS" in text and "OPENBLAS_NUM_THREADS" in text
    assert "womd_scenario_index" in text  # self-contained fallback if old caches are too small


def test_v24_waymax_outcomes_cover_heldout_test() -> None:
    text = (ROOT / "ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh").read_text(encoding="utf-8")
    assert "tensor_cache_heldout_test" in text
    assert 'attach_split heldout_test' in text
    assert "WAYMAX_GPUS" in text


def test_v24_benchmark_supports_report_only_recovery() -> None:
    text = (ROOT / "BENCHMARK_V16_8_24_FASTPATHS_CN.sh").read_text(encoding="utf-8")
    assert 'REPORT_ONLY="${REPORT_ONLY:-0}"' in text
    assert 'if [[ "$REPORT_ONLY" == "1" ]]' in text
    assert "semantic_equivalence_pass" in text


def test_v24_full_verdict_hard_gates_causal_integrity() -> None:
    text = (ROOT / "PREPARE_COWP_V16_8_24_FAST_DATA_CN.sh").read_text(encoding="utf-8")
    assert "causal_audit_train" in text
    assert "causal_audit_val" in text
    assert "causal_audit_heldout_test" in text
    assert "all(bool(v) for v in integ.values())" in text
