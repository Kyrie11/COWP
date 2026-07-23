from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cowp.label.trajectory_primitives import resample_logged
from cowp.models.cowp_model import COWPModel
from cowp.waymax_eval.metrics_cowp import metrics_from_labels
from cowp.waymax_eval.policy_wrapper import _traj_arrays


def _tiny_cfg(**model_overrides):
    model = {
        "d_state": 11,
        "history_steps": 2,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "max_agents": 2,
        "max_natural_alternatives": 3,
        "future_steps": 4,
        "max_safe_responses": 2,
        "token_count": 7,
    }
    model.update(model_overrides)
    return {"model": model, "ablation": {}}


def test_model_refuses_future_label_as_encoder_input():
    batch = {
        "cowp/critical/track_index": torch.tensor([[0]]),
        "cowp/critical/valid": torch.tensor([[True]]),
        "cowp/natural/traj": torch.zeros(1, 1, 3, 4, 7),
    }
    with pytest.raises(RuntimeError, match="future cowp/natural/traj"):
        COWPModel(_tiny_cfg())(batch, stage="representation")


def test_strict_coordinate_mode_requires_explicit_sdc_marker():
    hist = torch.zeros(1, 2, 2, 11)
    hist[..., 10] = 1.0
    batch = {
        "state/history": hist,
        "state/agent_valid": torch.ones(1, 2, dtype=torch.bool),
        "cowp/critical/track_index": torch.tensor([[1]]),
        "cowp/critical/valid": torch.tensor([[True]]),
    }
    with pytest.raises(RuntimeError, match="state/is_sdc"):
        COWPModel(_tiny_cfg(require_explicit_sdc_index=True))(batch, stage="representation")


def test_main_waymax_path_does_not_fall_back_to_logged_future():
    fake = SimpleNamespace(log_trajectory=object(), timestep=np.array(0))
    with pytest.raises(ValueError, match="fallback is disabled"):
        _traj_arrays(fake)
    traj, t = _traj_arrays(fake, allow_logged_fallback=True, prefer_logged=True)
    assert traj is fake.log_trajectory and t == 0


def test_resampled_observation_is_continuous_and_kinematically_consistent():
    dt = 0.1
    current = np.zeros(11, dtype=np.float32)
    current[3] = current[5] = 10.0
    current[6] = 0.0
    current[7:9] = [4.8, 1.9]
    logged = np.zeros((20, 7), dtype=np.float32)
    logged[:, 0] = np.arange(1, 21) * dt * 10.0
    logged[:, 2] = 0.0
    logged[:, 3] = 10.0
    logged[:, 5:7] = [4.8, 1.9]
    out = resample_logged(
        logged, 10, time_shift_steps=5, speed_scale=0.8,
        lateral_offset=0.5, current=current, dt=dt,
    )
    first_jump = np.linalg.norm(out[0, :2] - current[:2])
    assert first_jump < 1.1
    prev = np.vstack([current[:2], out[:-1, :2]])
    finite_diff_v = (out[:, :2] - prev) / dt
    np.testing.assert_allclose(out[:, 3:5], finite_diff_v, atol=1e-5)
    assert np.isfinite(out).all()


def test_label_space_metric_cannot_be_mistaken_for_closed_loop_cr():
    K, A, T = 1, 1, 2
    label = {
        "cowp/candidates/trajectory": np.zeros((K, T, 7), dtype=np.float32),
        "cowp/candidates/valid": np.ones(K, dtype=bool),
        "cowp/candidates/conventional_safe": np.zeros(K, dtype=bool),
        "cowp/critical/valid": np.zeros(A, dtype=bool),
        "cowp/witness/exists": np.zeros((K, A), dtype=bool),
    }
    out = metrics_from_labels([0], [label])
    assert out["OfflineConventionalUnsafeRate"] == 1.0
    assert out["MetricProtocol/ClosedLoopCollisionAvailable"] == 0.0
    assert out["CR_proxy_deprecated"] == out["CR"]
