from __future__ import annotations

import numpy as np
import pytest

from cowp.label.label_engine import NoValidEgoCandidatesError, _make_ego_neutral


def test_no_valid_candidate_is_scene_local_diagnostic_error() -> None:
    candidates = {
        "valid": np.zeros(4, dtype=bool),
        "trajectory": np.zeros((4, 8, 7), dtype=np.float32),
        "_proposal_debug": {"attempted": 12, "accepted": 0, "rejection_counts": {"map_filter": 12}},
    }
    with pytest.raises(NoValidEgoCandidatesError) as ei:
        _make_ego_neutral(candidates, "scene-zero-valid")
    assert ei.value.scenario_id == "scene-zero-valid"
    assert ei.value.diagnostics["accepted"] == 0
    assert ei.value.diagnostics["rejection_counts"]["map_filter"] == 12


def test_ego_neutral_prefers_valid_neutral_candidate() -> None:
    traj = np.zeros((3, 5, 7), dtype=np.float32)
    traj[1, :, 0] = 7.0
    candidates = {
        "valid": np.asarray([True, True, False]),
        "is_neutral": np.asarray([False, True, True]),
        "trajectory": traj,
    }
    out = _make_ego_neutral(candidates, "scene-ok")
    assert np.all(out[:, 0] == 7.0)


def test_learned_shared_forward_ablation_names_fail_loudly():
    import pytest
    from cowp.waymax_eval.rollout import _method_gate_defaults
    # Canonicalization itself remains cheap/pure; the selector guard is exercised
    # through a tiny fake batch below so historical ablation names cannot become
    # silent COWP aliases in learned_offline.
    assert _method_gate_defaults("cowp_wo_neutral_branch", "priority")[0] == "cowp_wo_neutral_branch"
    torch = pytest.importorskip("torch")
    from cowp.waymax_eval.rollout import _select_from_learned
    batch = {
        "cowp/candidates/valid": torch.ones((1, 1), dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones((1, 1), dtype=torch.bool),
        "cowp/candidates/ego_utility_prior": torch.zeros((1, 1)),
        "cowp/critical/valid": torch.ones((1, 1), dtype=torch.bool),
    }
    pred = {
        "planner_score": torch.zeros((1, 1)),
        "witness": {
            "exist_logits": torch.zeros((1, 1, 1)),
            "opr": torch.ones((1, 1, 1)),
        },
    }
    with pytest.raises(ValueError, match="not a valid shared-forward"):
        _select_from_learned(batch, pred, method="cowp_wo_neutral_branch")


def test_online_shared_forward_ablation_names_fail_loudly():
    import pytest
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    with pytest.raises(ValueError, match="not a valid shared-forward online ablation"):
        _canonical_online_method("cowp_wo_dual_edge", "priority")
