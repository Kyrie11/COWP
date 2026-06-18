import torch

from cowp.models.cowp_model import COWPModel


def _batch_with_out_of_range_critical():
    B, N, H, D = 1, 128, 11, 11
    K, A, M, R, T = 4, 2, 3, 2, 6
    hist = torch.zeros(B, N, H, D)
    hist[..., 10] = 1.0
    return {
        "state/history": hist,
        "state/agent_valid": torch.ones(B, N, dtype=torch.bool),
        "cowp/candidates/trajectory": torch.zeros(B, K, T, 7),
        "cowp/candidates/macro_type": torch.zeros(B, K, dtype=torch.long),
        "cowp/candidates/valid": torch.ones(B, K, dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones(B, K, dtype=torch.bool),
        "cowp/candidates/ego_utility_prior": torch.zeros(B, K),
        "cowp/critical/track_index": torch.tensor([[149, 2]]),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
        "cowp/natural/traj": torch.zeros(B, A, M, T, 7),
        "cowp/natural/weight": torch.ones(B, A, M) / M,
        "cowp/natural/source": torch.zeros(B, A, M, dtype=torch.long),
        "cowp/natural/valid": torch.ones(B, A, M, dtype=torch.bool),
        "cowp/natural/priority_preserved": torch.ones(B, A, M, dtype=torch.bool),
    }


def test_model_masks_out_of_range_critical_without_crashing():
    cfg = {
        "model": {
            "d_state": 11,
            "history_steps": 11,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "max_agents": 128,
            "max_natural_alternatives": 3,
            "max_safe_responses": 2,
            "future_steps": 6,
            "token_count": 7,
        },
        "ablation": {},
    }
    model = COWPModel(cfg)
    out = model(_batch_with_out_of_range_critical(), stage="representation")
    assert out["natural"]["traj"].shape[:3] == (1, 2, 3)
    assert out["critical_mask"].tolist() == [[False, True]]
