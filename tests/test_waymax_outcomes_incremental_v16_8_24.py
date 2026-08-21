from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cowp.waymax_eval.candidate_replay import _JitCallableWithFallback
from cowp.waymax_eval.outcome_attach import attach_rows_to_cache_file, restore_key


def _make_core_cache(path: Path, *, sid: str = "scene_a", k: int = 4) -> None:
    arrays = {
        "scenario__id": np.asarray(sid),
        "cowp__candidates__valid": np.asarray([True] * k, dtype=bool),
        "cowp__candidates__trajectory": np.zeros((k, 80, 7), dtype=np.float32),
        "some__unrelated__tensor": np.arange(6, dtype=np.int32),
    }
    np.savez(path, **arrays)


def _canonical_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {restore_key(k): z[k] for k in z.files}


def test_incremental_attachment_matches_final_attach_schema(tmp_path: Path) -> None:
    core = tmp_path / "core"
    inc = tmp_path / "inc"
    ref = tmp_path / "ref"
    core.mkdir()
    src = core / "scene_a.npz"
    _make_core_cache(src)

    rows = [
        {
            "scenario_id": "scene_a",
            "candidate_index": 0,
            "rollout_valid": True,
            "collision": False,
            "offroad": True,
            "log_divergence": float("nan"),
            "rollout_seconds": 1.25,
        },
        {
            "scenario_id": "scene_a",
            "candidate_index": 2,
            "rollout_valid": True,
            "collision": True,
            "offroad": False,
            "log_divergence": 0.125,
            "rollout_seconds": 2.5,
        },
        {
            "scenario_id": "scene_a",
            "candidate_index": 3,
            "rollout_valid": False,
            "error": "synthetic failure",
            "rollout_seconds": 0.2,
        },
    ]

    inc.mkdir()
    attach_rows_to_cache_file(src, inc / src.name, rows)

    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in rows), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cowp.scripts.12_attach_waymax_candidate_outcomes",
            "--cache-dir",
            str(core),
            "--output-dir",
            str(ref),
            "--outcomes-jsonl",
            str(outcomes),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    a = _canonical_npz(inc / src.name)
    b = _canonical_npz(ref / src.name)
    waymax_keys = sorted(k for k in b if k.startswith("waymax/"))
    assert waymax_keys
    assert set(waymax_keys) == {k for k in a if k.startswith("waymax/")}
    for key in waymax_keys:
        av = np.asarray(a[key])
        bv = np.asarray(b[key])
        if av.dtype.kind == "f" or bv.dtype.kind == "f":
            assert np.array_equal(av, bv, equal_nan=True), key
        else:
            assert np.array_equal(av, bv), key
    assert np.array_equal(a["some/unrelated/tensor"], b["some/unrelated/tensor"])


def test_incremental_resume_preserves_previous_candidate_rows(tmp_path: Path) -> None:
    core = tmp_path / "core"
    out = tmp_path / "out"
    core.mkdir()
    out.mkdir()
    src = core / "scene_a.npz"
    dst = out / "scene_a.npz"
    _make_core_cache(src)

    row0 = {"scenario_id": "scene_a", "candidate_index": 0, "rollout_valid": True, "collision": False, "offroad": False}
    row1 = {"scenario_id": "scene_a", "candidate_index": 1, "rollout_valid": True, "collision": True, "offroad": False}
    attach_rows_to_cache_file(src, dst, [row0])
    attach_rows_to_cache_file(src, dst, [row0, row1])

    a = _canonical_npz(dst)
    sel = np.asarray(a["waymax/candidate_selected_for_rollout"], dtype=bool)
    col = np.asarray(a["waymax/candidate_collision"], dtype=bool)
    assert sel.tolist()[:2] == [True, True]
    assert col.tolist()[:2] == [False, True]
    assert int(np.asarray(a["waymax/outcomes_attached_count"]).item()) == 2


def test_jit_wrapper_falls_back_permanently() -> None:
    calls = {"jit": 0, "eager": 0}

    def bad(x):
        calls["jit"] += 1
        raise RuntimeError("trace unsupported")

    def eager(x):
        calls["eager"] += 1
        return x + 1

    f = _JitCallableWithFallback(bad, eager)
    assert f(1) == 2
    assert f(2) == 3
    assert calls == {"jit": 1, "eager": 2}
    assert not f.using_jit


def test_v24_shell_keeps_exact_metrics_and_live_progress_path() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh").read_text(encoding="utf-8")
    assert "mapfile -t files < <(replay_split" not in text
    assert "--attach-output-dir" in text
    assert "--profile-detail" in text
    assert "--metric-eval-mode step" in text
    assert "--metric-eval-interval 1" in text
    assert "--done-check-interval 1" in text
    assert "COWP_TQDM_POSITION" in text
    rollout_text = (root / "cowp" / "waymax_eval" / "rollout.py").read_text(encoding="utf-8")
    assert "requires waymax.dynamics.StateDynamics" in rollout_text
    assert 'dynamics_names = ("StateDynamics", "DeltaGlobal")' not in rollout_text
