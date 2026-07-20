from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset


def _traj(x: float, *, t: int = 4) -> np.ndarray:
    out = np.zeros((t, 7), dtype=np.float32)
    out[:, 0] = x
    out[:, 5] = 4.5
    out[:, 6] = 2.0
    return out


def test_overlay_augmentation_and_transparent_loading(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    overlay = tmp_path / "overlay"
    raw.mkdir()

    cand = np.stack([_traj(0.0), _traj(100.0)], axis=0)
    natural = np.stack([_traj(0.0), _traj(20.0)], axis=0)[None, ...]
    response = np.zeros((2, 1, 2, 4, 7), dtype=np.float32)
    response[:, 0, 0] = _traj(0.0)
    response[:, 0, 1] = _traj(20.0)

    arrays = {
        "cowp__candidates__trajectory": cand,
        "cowp__candidates__valid": np.ones(2, dtype=bool),
        "cowp__critical__valid": np.ones(1, dtype=bool),
        "cowp__critical__agent_type": np.ones(1, dtype=np.int32),
        "cowp__natural__traj": natural,
        "cowp__natural__valid": np.ones((1, 2), dtype=bool),
        "cowp__natural__weight": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "cowp__natural__burden_neutral": np.zeros((1, 2), dtype=np.float32),
        "cowp__natural__beta": np.asarray([0.65], dtype=np.float32),
        "cowp__response__traj": response,
        "cowp__response__valid": np.ones((2, 1, 2), dtype=bool),
        "cowp__response__is_safe": np.ones((2, 1, 2), dtype=bool),
        "cowp__response__is_low_burden": np.ones((2, 1, 2), dtype=bool),
        "cowp__response__burden_total": np.asarray([[[0.1, 0.2]], [[0.1, 0.2]]], dtype=np.float32),
        "cowp__witness__rho": np.zeros((2, 1), dtype=np.int32),
    }
    src = raw / "scene.npz"
    np.savez(src, **arrays)

    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cowp.scripts.26_augment_transport_labels",
            "--data-config",
            str(repo / "configs/data.yaml"),
            "--label-config",
            str(repo / "configs/label_cowp_v9.yaml"),
            "--input-dir",
            str(raw),
            "--output-dir",
            str(overlay),
            "--num-workers",
            "1",
            "--storage-mode",
            "overlay",
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (overlay / "scene.npz").is_symlink()
    sidecar = overlay / ".transport_v9" / "scene.npz"
    assert sidecar.is_file()
    summary = json.loads((overlay / "transport_augmentation_summary.json").read_text())
    assert summary["complete"] is True
    assert summary["files_completed"] == 1

    ds = COWPNpzDataset(overlay)
    loaded = ds.load(0, {"cowp/candidates/valid", "cowp/transport/"})
    assert "cowp/candidates/valid" in loaded
    assert "cowp/transport/mode_conflict" in loaded
    conflict = loaded["cowp/transport/mode_conflict"]
    retained = loaded["cowp/transport/mode_retained_low_safe"]
    assert bool(conflict[0, 0, 0]) is True
    assert bool(conflict[1, 0, 0]) is False
    assert np.all(~(conflict & retained))
    roots = loaded["cowp/transport/response_root_index"]
    assert roots[0, 0].tolist() == [0, 1]
