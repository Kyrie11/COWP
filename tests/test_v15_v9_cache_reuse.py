from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _write_split(root: Path, name: str) -> tuple[Path, Path]:
    raw = root / f"raw_{name}"
    overlay = root / f"overlay_{name}"
    side = overlay / ".transport_v9"
    raw.mkdir()
    overlay.mkdir()
    side.mkdir()

    base = {
        "womd__state__is_sdc": np.asarray([True, False]),
        "cowp__critical__valid": np.asarray([True]),
        "cowp__critical__input_index": np.asarray([1], dtype=np.int64),
        "cowp__natural__traj": np.zeros((1, 2, 4, 7), dtype=np.float32),
        "cowp__natural__valid": np.ones((1, 2), dtype=bool),
        "cowp__natural__weight": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "cowp__natural__source": np.asarray([[0, 1]], dtype=np.int32),
        "cowp__response__valid": np.ones((2, 1, 2), dtype=bool),
        "cowp__witness__exists": np.zeros((2, 1), dtype=bool),
        "cowp__candidates__noncoercive_feasible": np.asarray([True, False]),
        "waymax__candidate_rollout_valid": np.asarray([True, True]),
        "waymax__candidate_log_divergence": np.asarray([np.nan, np.nan], dtype=np.float32),
    }
    np.savez(raw / "scene.npz", **base)
    os.symlink((raw / "scene.npz").resolve(), overlay / "scene.npz")
    np.savez(
        side / "scene.npz",
        cowp__transport__mode_valid=np.ones((2, 1, 2), dtype=bool),
        cowp__transport__mode_conflict=np.zeros((2, 1, 2), dtype=bool),
        cowp__transport__response_root_index=np.zeros((2, 1, 2), dtype=np.int32),
    )
    (overlay / "transport_augmentation_summary.json").write_text(
        json.dumps({
            "storage_mode": "overlay",
            "sidecar_subdir": ".transport_v9",
            "files_total": 1,
            "files_completed": 1,
            "error_count": 0,
            "complete": True,
        }),
        encoding="utf-8",
    )
    return raw, overlay


def test_v9_reuse_gate_and_protocol_audit(tmp_path: Path) -> None:
    train_raw, train_overlay = _write_split(tmp_path, "train")
    val_raw, val_overlay = _write_split(tmp_path, "val")
    # Avoid filename overlap in the synthetic split.
    (val_raw / "scene.npz").rename(val_raw / "val_scene.npz")
    (val_overlay / "scene.npz").unlink()
    os.symlink((val_raw / "val_scene.npz").resolve(), val_overlay / "val_scene.npz")
    (val_overlay / ".transport_v9" / "scene.npz").rename(val_overlay / ".transport_v9" / "val_scene.npz")

    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    gate = tmp_path / "reuse.json"
    subprocess.run(
        [
            sys.executable, "-m", "cowp.scripts.38_gate_cache_reuse",
            "--raw-train", str(train_raw), "--raw-val", str(val_raw),
            "--transport-train", str(train_overlay), "--transport-val", str(val_overlay),
            "--sample-scenes", "1", "--min-train-scenes", "1", "--min-val-scenes", "1",
            "--output", str(gate),
        ],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
    )
    report = json.loads(gate.read_text())
    assert report["decisions"]["reuse_for_v14_or_v15_model_with_v9_labels"]["pass"] is True
    assert report["decisions"]["reuse_as_true_v15_causal_label_dataset"]["pass"] is False

    model = tmp_path / "model.yaml"
    label = tmp_path / "label.yaml"
    train = tmp_path / "train.yaml"
    eval_cfg = tmp_path / "eval.yaml"
    model.write_text(yaml.safe_dump({"model": {"allow_label_only_state_fallback": False, "require_explicit_sdc_index": True}}))
    label.write_text(yaml.safe_dump({"natural": {"map_filter_enabled": True, "obs_decontamination_enabled": True}}))
    train.write_text(yaml.safe_dump({"loss_weights": {"branch_source_ce": 0.0, "outcome_logdiv": 0.0}}))
    eval_cfg.write_text(yaml.safe_dump({
        "planning": {"online_other_future_source": "constant_velocity", "causal_log_trajectory_fallback": False},
        "eval": {
            "reported_collision_source": "waymax_standard_metrics_only",
            "label_space_candidate_safety_metric": "OfflineConventionalUnsafeRate",
            "actual_non_ego_policy": "logged_replay",
            "reactive_mixture_implemented": False,
        },
    }))
    audit = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable, "-m", "cowp.scripts.36_audit_causal_protocol",
            "--model-config", str(model), "--label-config", str(label),
            "--train-config", str(train), "--eval-config", str(eval_cfg),
            "--data-protocol", "v9_reuse", "--cache-reuse-report", str(gate),
            "--output", str(audit),
        ],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
    )
    audited = json.loads(audit.read_text())
    assert audited["pass"] is True
    assert audited["engineering_pass"] is True
    assert audited["full_v15_label_protocol_pass"] is False
