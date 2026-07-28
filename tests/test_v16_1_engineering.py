from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from cowp.core.constants import NaturalSource
from cowp.models.losses import natural_loss
from cowp.models.natural_decoder import NaturalDecoder


def test_realistic_natural_batch_shape_contract_and_backward() -> None:
    """The exact A/M pattern from the failed server run must survive forward+loss."""
    torch.manual_seed(2026)
    B, A, M, R, T = 2, 6, 24, 24, 10
    decoder = NaturalDecoder(d_model=32, modes=M, future_steps=T, decoder_type="typed_causal_dynamics")
    z = torch.randn(B, 10, 32)
    idx = torch.arange(A).view(1, A).expand(B, -1)
    anchor = torch.zeros(B, A, 7)
    anchor[..., 3] = 4.0
    pred = decoder(z, idx, anchor7=anchor, dt=0.1)
    source = torch.tensor([0] * 8 + [1] * 8 + [2] * 8).view(1, 1, R).expand(B, A, -1)
    batch = {
        "cowp/natural/traj": pred["base_traj"].detach().clone(),
        "cowp/natural/valid": torch.ones(B, A, R, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(B, A, R),
        "cowp/natural/source": source,
        "cowp/natural/priority_preserved": (source == int(NaturalSource.PRIO)).float(),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
    }
    out = natural_loss(pred, batch, {"natural_dt": 0.1, "natural_mode_usage": 0.03})
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["mode_usage"])
    out["loss"].backward()
    assert decoder.logit.weight.grad is not None


def test_threaded_transport_diagnostic_keeps_exact_core_aggregates(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    mv = np.array([[[1, 1]]], dtype=bool)
    mc = np.array([[[1, 0]]], dtype=bool)
    mr = np.array([[[0, 1]]], dtype=bool)
    w = np.array([[0.4, 0.6]], dtype=np.float32)
    payload = {
        "cowp__transport__mode_valid": mv,
        "cowp__transport__mode_conflict": mc,
        "cowp__transport__mode_retained_low_safe": mr,
        "cowp__transport__response_root_index": np.array([[[0, 1]]], dtype=np.int64),
        "cowp__transport__response_is_min_burden": np.array([[[1, 0]]], dtype=bool),
        "cowp__transport__root_recovery_mass": np.array([[0.25]], dtype=np.float32),
        "cowp__response__valid": np.array([[[1, 1]]], dtype=bool),
        "cowp__natural__weight": w,
        "cowp__witness__natural_conflict_mass": np.array([[0.4]], dtype=np.float32),
        "cowp__witness__opr": np.array([[0.6]], dtype=np.float32),
    }
    for i in range(3):
        np.savez(cache / f"scene_{i}.npz", **payload)
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "cowp.scripts.27_diagnose_transport_labels",
            "--cache-dir", str(cache), "--output", str(report), "--workers", "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(report.read_text())
    assert data["pass"] is True
    assert data["files_ok"] == 3
    assert data["mode_conflict_rate"] == 0.5
    assert data["mode_retained_low_safe_rate"] == 0.5
    assert data["response_root_assignment_coverage"] == 1.0
    assert data["aggregate_conflict_mae"] < 1e-7
    assert data["aggregate_opr_mae"] < 1e-7
    assert abs(data["root_recovery_mean"] - 0.25) < 1e-7
    assert data["root_recovery_quantile_bin_width"] <= 1e-4


def test_run_provenance_rejects_mixed_code_under_same_output_root(tmp_path) -> None:
    a = tmp_path / "a.py"
    a.write_text("x=1\n")
    out = tmp_path / "run_provenance.json"
    base = [
        sys.executable, "-m", "cowp.scripts.42_write_run_provenance",
        "--output", str(out), "--strict-existing", "--file", str(a),
    ]
    first = subprocess.run(base, check=False, capture_output=True, text=True)
    assert first.returncode == 0
    a.write_text("x=2\n")
    second = subprocess.run(base, check=False, capture_output=True, text=True)
    assert second.returncode == 2
    assert "different code/config signature" in second.stdout


def test_run_provenance_uses_logical_names_not_temporary_paths(tmp_path) -> None:
    a = tmp_path / "candidate_a.yaml"
    b = tmp_path / "candidate_b.yaml"
    a.write_text("seed: 2026\n")
    b.write_text("seed: 2026\n")
    out = tmp_path / "run_provenance.json"
    first = subprocess.run([
        sys.executable, "-m", "cowp.scripts.42_write_run_provenance",
        "--output", str(out), "--strict-existing", "--file", f"configs/train.yaml={a}",
    ], check=False, capture_output=True, text=True)
    assert first.returncode == 0
    second = subprocess.run([
        sys.executable, "-m", "cowp.scripts.42_write_run_provenance",
        "--output", str(out), "--strict-existing", "--file", f"configs/train.yaml={b}",
    ], check=False, capture_output=True, text=True)
    assert second.returncode == 0
