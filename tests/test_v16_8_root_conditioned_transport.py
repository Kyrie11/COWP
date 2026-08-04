from __future__ import annotations

import numpy as np
import pytest
import torch

from cowp.label.safe_responses import _root_residual_trajectory
from cowp.models.losses import _root_low_safe_target, paper_aligned_supervision_batch


def _paper_batch(*, recovery: float) -> dict[str, torch.Tensor]:
    # One candidate, one protected agent, one natural root.  The root conflicts
    # with ego and is not directly retained, so OPR must equal q exactly.
    return {
        "cowp/candidates/valid": torch.ones(1, 1, dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones(1, 1, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(1, 1, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(1, 1, 1),
        "cowp/natural/valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "cowp/natural/beta": torch.full((1, 1), 0.65),
        "cowp/transport/mode_valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/mode_conflict": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/mode_retained_low_safe": torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/response_root_index": torch.zeros(1, 1, 1, 1, dtype=torch.long),
        "cowp/transport/root_low_safe_score": torch.full((1, 1, 1, 1), recovery),
        "cowp/response/valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/is_safe": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/is_low_burden": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/burden_total": torch.full((1, 1, 1, 1), 0.2),
        "cowp/witness/exists": torch.zeros(1, 1, 1, dtype=torch.bool),
    }


def test_paper_aligned_opr_transports_conflict_root_recovery():
    recovered = paper_aligned_supervision_batch(_paper_batch(recovery=1.0), {})
    lost = paper_aligned_supervision_batch(_paper_batch(recovery=0.0), {})
    assert recovered["cowp/witness/opr"].item() == pytest.approx(1.0)
    assert lost["cowp/witness/opr"].item() == pytest.approx(0.0)
    assert not recovered["cowp/witness/exists"].item()
    assert lost["cowp/witness/exists"].item()


def test_soft_root_target_is_used_without_collapsing_to_binary():
    batch = {"cowp/transport/root_low_safe_score": torch.tensor([[[[0.15, 0.80]]]])}
    target = _root_low_safe_target(batch, 2)
    assert target is not None
    assert torch.allclose(target, torch.tensor([[[[0.15, 0.80]]]]))


def test_root_residual_preserves_root_geometry_and_identity():
    t = np.arange(8, dtype=np.float32)
    root = np.zeros((8, 7), dtype=np.float32)
    root[:, 0] = t
    root[:, 2] = 0.0
    root[:, 3] = 10.0
    identity = _root_residual_trajectory(root, dt=0.1, accel=0.0, start_delay_s=0.0, duration_s=1.0)
    slowed = _root_residual_trajectory(root, dt=0.1, accel=-2.0, start_delay_s=0.0, duration_s=0.4)
    assert np.array_equal(identity, root)
    assert np.allclose(slowed[:, 1], root[:, 1])
    assert np.allclose(slowed[:, 2], root[:, 2])
    assert np.all(slowed[:, 3] <= root[:, 3] + 1.0e-6)


def test_planner_freezes_validated_transport_and_response_modules():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("cowp_train_script", Path("cowp/scripts/03_train.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = torch.nn.Linear(2, 2)
            self.candidate_encoder = torch.nn.Linear(2, 2)
            self.natural_decoder = torch.nn.Linear(2, 2)
            self.witness_decoder = torch.nn.Linear(2, 2)
            self.set_transport = torch.nn.Linear(2, 2)
            self.response_decoder = torch.nn.Linear(2, 2)

    model = Dummy()
    module._set_stage_freeze(
        model, "planner", False,
        freeze_transport_during_planner=True,
        freeze_response_during_planner=True,
    )
    assert not any(p.requires_grad for p in model.set_transport.parameters())
    assert not any(p.requires_grad for p in model.response_decoder.parameters())
    assert any(p.requires_grad for p in model.candidate_encoder.parameters())


def _controlled_head_output(*, conflict: float, retain_conditional: float, recovery: float, root_min_burden: float):
    from cowp.models.set_transport_head import SetTransportCertificateHead

    b = k = a = m = r = 1
    d, t = 8, 6
    head = SetTransportCertificateHead(d_model=d, hidden=8, geometry_steps=4, response_topk=1)
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        bias = head.mode_out[-1].bias
        bias[0] = torch.logit(torch.tensor(conflict).clamp(1.0e-5, 1.0 - 1.0e-5))
        bias[1] = torch.logit(torch.tensor(retain_conditional).clamp(1.0e-5, 1.0 - 1.0e-5))
        bias[2] = 0.0
        bias[3] = torch.logit(torch.tensor(recovery).clamp(1.0e-5, 1.0 - 1.0e-5))
        bias[4] = torch.logit(torch.tensor(root_min_burden / 2.0).clamp(1.0e-5, 1.0 - 1.0e-5))
    natural = {
        "mode_latent": torch.zeros(b, a, m, d),
        "logits": torch.zeros(b, a, m),
        "source_logits": torch.zeros(b, a, m, 4),
        "priority_logits": torch.zeros(b, a, m),
        "traj": torch.zeros(b, a, m, t, 7),
    }
    response = {
        "safe_logits": torch.full((b, k, a, r), 8.0),
        "low_logits": torch.full((b, k, a, r), 8.0),
        "valid_logits": torch.full((b, k, a, r), 8.0),
        "mode_logits": torch.zeros(b, k, a, r),
        "root_logits": torch.zeros(b, k, a, r, m),
        "burden_total": torch.full((b, k, a, r), root_min_burden),
    }
    return head(
        z_agent=torch.zeros(b, 1, d),
        z_candidate=torch.zeros(b, k, d),
        z_graph=torch.zeros(b, d),
        critical_indices=torch.zeros(b, a, dtype=torch.long),
        natural=natural,
        response=response,
        beta=torch.full((b, a), 0.65),
        candidate_traj=torch.zeros(b, k, t, 7),
        natural_traj=natural["traj"],
        calibration_scale=0.0,
        root_probability_floor=0.0,
    )


def test_model_opr_implements_full_root_transport_equation():
    out = _controlled_head_output(
        conflict=0.8,
        retain_conditional=0.25,
        recovery=0.5,
        root_min_burden=0.2,
    )
    expected = (1.0 - 0.8) * 0.25 + 0.8 * 0.5
    assert out["opr"].item() == pytest.approx(expected, abs=2.0e-5)
    assert out["transported_root_prob"].item() == pytest.approx(expected, abs=2.0e-5)


def test_root_recovery_probability_and_safe_burden_are_distinct_outputs():
    # q=0 means no *low-burden* recovery; it does not imply that no finite safe
    # same-root response exists.  The model must therefore preserve b* instead
    # of replacing it with the no-response sentinel.
    absent_low_burden = _controlled_head_output(
        conflict=0.9,
        retain_conditional=0.1,
        recovery=1.0e-5,
        root_min_burden=1.2,
    )
    recovered_low_burden = _controlled_head_output(
        conflict=0.9,
        retain_conditional=0.1,
        recovery=1.0 - 1.0e-5,
        root_min_burden=0.1,
    )
    assert absent_low_burden["root_min_safe_burden"].item() == pytest.approx(1.2, abs=2.0e-5)
    assert recovered_low_burden["root_min_safe_burden"].item() == pytest.approx(0.1, abs=2.0e-5)
    assert recovered_low_burden["opr"].item() > absent_low_burden["opr"].item()


def test_causal_audit_accepts_v16_8_overlay_as_mechanism_protocol(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    gate = tmp_path / "reuse.json"
    gate.write_text(json.dumps({
        "train": {
            "overlay_summary": {"complete": True, "error_count": 0},
            "missing_required_key_counts": {"cowp/transport/mode_valid": 0},
        },
        "val": {
            "overlay_summary": {"complete": True, "error_count": 0},
            "missing_required_key_counts": {"cowp/transport/mode_valid": 0},
        },
        "cross_split_filename_overlap": 0,
        "decisions": {
            "reuse_for_v14_or_v15_model_with_v9_labels": {"pass": True},
            "reuse_as_true_v15_causal_label_dataset": {"pass": False},
        },
    }), encoding="utf-8")
    output = tmp_path / "audit.json"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, "-m", "cowp.scripts.36_audit_causal_protocol",
        "--model-config", "configs/model_cowp_v16_8.yaml",
        "--label-config", "configs/label_cowp_v16_8.yaml",
        "--train-config", "configs/train_cowp_v16_8.yaml",
        "--eval-config", "configs/eval_cowp_v16_8.yaml",
        "--data-protocol", "v16_8_root_conditioned_overlay",
        "--cache-reuse-report", str(gate),
        "--output", str(output),
    ], cwd=repo, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["mechanism_overlay_protocol_pass"] is True
    assert report["full_v15_label_protocol_pass"] is False


def test_v16_8_launcher_background_parent_exits_and_online_restart_is_self_contained():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    main = (repo / "NEXT_RUN_COMMANDS_V16_8_CN.sh").read_text(encoding="utf-8")
    mechanism = (repo / "NEXT_RUN_COMMANDS_V16_8_MECHANISM_CN.sh").read_text(encoding="utf-8")
    probe = (repo / "NEXT_RUN_COMMANDS_V16_8_PROBE_CN.sh").read_text(encoding="utf-8")
    full = (repo / "NEXT_RUN_COMMANDS_V16_8_FULL_CN.sh").read_text(encoding="utf-8")

    background_block = main.split("if [[ \"$BACKGROUND\" == \"1\"", 1)[1].split("fi", 1)[0]
    assert "exit 0" in background_block
    assert "'natural_history':str(history_path)" in mechanism
    assert "natural_attribution_transfer_manifest.json" in probe
    assert "export RUN_PROBE=1" in probe and "export RUN_FULL=0" in probe
    assert "delta_conventional_vs_root_transport.json" in full
    assert "export RUN_PROBE=0" in full and "export RUN_FULL=1" in full
