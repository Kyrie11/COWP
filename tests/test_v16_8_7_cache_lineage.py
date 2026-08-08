from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cowp.data.dataset import COWPNpzDataset
from cowp.label.label_engine import build_labels_for_scene


def test_dataset_fails_early_on_broken_overlay_symlink(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    missing = tmp_path / "deleted_backing" / "scene.npz"
    os.symlink(str(missing), str(overlay / "scene.npz"))
    with pytest.raises(FileNotFoundError, match="broken NPZ symlink"):
        COWPNpzDataset(overlay)


def test_fresh_label_contains_complete_inline_transport(toy_scene, cfg) -> None:
    label = build_labels_for_scene(toy_scene, cfg)
    required = {
        "cowp/transport/mode_valid",
        "cowp/transport/mode_conflict",
        "cowp/transport/mode_retained_low_safe",
        "cowp/transport/response_root_index",
        "cowp/transport/response_is_min_burden",
        "cowp/transport/root_recovery_mass",
        "cowp/transport/root_low_safe_score",
        "cowp/transport/root_target_confidence",
        "cowp/transport/root_min_safe_burden",
        "cowp/transport/transported_opr",
        "cowp/transport/canonical_root_weight",
    }
    assert required.issubset(label)
    w = np.asarray(label["cowp/transport/canonical_root_weight"], dtype=np.float32)
    valid = np.asarray(label["cowp/natural/valid"], dtype=bool)
    assert w.shape == valid.shape
    assert np.all(w[~valid] == 0.0)
    for a in range(len(w)):
        if valid[a].any():
            assert np.isclose(float(w[a].sum()), 1.0, atol=1e-5)


def test_rebase_transport_overlay_uses_surviving_base(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    base = tmp_path / "base"
    old = tmp_path / "old_overlay"
    out = tmp_path / "rebased"
    side = old / ".transport_v16_8"
    base.mkdir(); old.mkdir(); side.mkdir()
    np.savez(base / "scene.npz", scenario__id=np.asarray("scene"), cowp__candidates__valid=np.ones(1, dtype=bool))
    os.symlink(str(tmp_path / "deleted_waymax" / "scene.npz"), str(old / "scene.npz"))
    transport = {k.replace("/", "__"): np.zeros((1,), dtype=np.float32) for k in (
        "cowp/transport/mode_valid", "cowp/transport/mode_conflict", "cowp/transport/mode_retained_low_safe",
        "cowp/transport/response_root_index", "cowp/transport/response_is_min_burden", "cowp/transport/root_recovery_mass",
        "cowp/transport/root_low_safe_score", "cowp/transport/root_target_confidence", "cowp/transport/root_min_safe_burden",
        "cowp/transport/transported_opr",
    )}
    np.savez(side / "scene.npz", **transport)
    (old / "transport_augmentation_summary.json").write_text(json.dumps({"storage_mode":"overlay","sidecar_subdir":".transport_v16_8","input_dir":"/deleted"}))
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "cowp.scripts.54_rebase_transport_overlay",
                    "--base-cache", str(base), "--old-overlay", str(old), "--output-dir", str(out),
                    "--verify-all-sidecars"], cwd=repo, check=True, capture_output=True, text=True)
    assert (out / "scene.npz").is_symlink()
    assert (out / "scene.npz").resolve() == (base / "scene.npz").resolve()
    assert (out / ".transport_v16_8" / "scene.npz").is_file()
