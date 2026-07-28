from __future__ import annotations

import json
from pathlib import Path

import torch

from cowp.models.losses import candidate_certificate_loss


def test_candidate_certificate_ignores_physical_safe_as_ncf_label() -> None:
    # Candidate 0 is NCF; candidate 1 is false-safe but also replay-safe.
    pred = {
        "candidate_ncf_logit": torch.tensor([[0.2, -0.2]], requires_grad=True),
        "candidate_false_safe_logit": torch.tensor([[-0.2, 0.2]], requires_grad=True),
        "candidate_quality_logit": torch.tensor([[0.2, -0.2]], requires_grad=True),
    }
    batch = {
        "cowp/candidates/valid": torch.tensor([[True, True]]),
        "cowp/candidates/noncoercive_feasible": torch.tensor([[True, False]]),
        "cowp/candidates/false_safe": torch.tensor([[False, True]]),
        # Both candidates are physically safe.  This must not turn candidate 1 into
        # a certificate positive or create a positive/negative self-comparison.
        "waymax/candidate_rollout_valid": torch.tensor([[True, True]]),
        "waymax/candidate_collision": torch.tensor([[False, False]]),
        "waymax/candidate_offroad": torch.tensor([[False, False]]),
    }
    out = candidate_certificate_loss(pred, batch, {
        "candidate_certificate_rank": 1.0,
        "candidate_certificate_risk_rank": 1.0,
        "candidate_certificate_risk_bce": 1.0,
    })
    assert torch.isfinite(out["loss"])
    assert float(out["overlap_rate"]) == 0.0
    out["loss"].backward()
    assert pred["candidate_ncf_logit"].grad is not None
    # The false-safe candidate should be pushed toward lower NCF probability.
    assert float(pred["candidate_ncf_logit"].grad[0, 1]) > 0.0


def test_merge_diagnostic_episode_count_is_summed(tmp_path: Path) -> None:
    # Import by file path because scripts use numeric filenames.
    import importlib.util
    module_path = Path(__file__).parents[1] / "cowp" / "scripts" / "17_merge_waymax_shards.py"
    spec = importlib.util.spec_from_file_location("merge_waymax_shards_v6", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    paths = []
    for i in range(2):
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps({
            "num_rollouts": 150,
            "steps": [],
            "standard_metrics": [],
            "policy_diagnostic_summary": {"ClosedLoopPolicySteps": 100.0},
            "closed_loop_cowp_metric_summary": {
                "EpisodesWithDiagnostics": 150.0,
                "FallbackEpisodeRate": 0.2 + 0.2 * i,
            },
        }))
        paths.append(p)
    merged = mod.merge_payloads(paths)
    assert merged["num_rollouts"] == 300
    assert merged["closed_loop_cowp_metric_summary"]["EpisodesWithDiagnostics"] == 300.0
