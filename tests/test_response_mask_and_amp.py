import torch

from cowp.models.losses import response_loss
from cowp.scripts import __dict__ as _scripts_pkg


def test_response_loss_uses_critical_valid_mask():
    pred = {
        "safe_logits": torch.zeros(1, 1, 2, 1),
        "low_logits": torch.zeros(1, 1, 2, 1),
        "burden_total": torch.zeros(1, 1, 2, 1),
        "traj": torch.zeros(1, 1, 2, 1, 2, 7),
        "burden_components": torch.zeros(1, 1, 2, 1, 6),
    }
    batch = {
        "cowp/critical/valid": torch.tensor([[True, False]]),
        "cowp/response/valid": torch.ones(1, 1, 2, 1, dtype=torch.bool),
        "cowp/response/is_safe": torch.zeros(1, 1, 2, 1),
        "cowp/response/is_low_burden": torch.zeros(1, 1, 2, 1),
        "cowp/response/burden_total": torch.tensor([[[[1.0], [1000.0]]]]),
        "cowp/response/traj": torch.zeros(1, 1, 2, 1, 2, 7),
        "cowp/response/burden_components": torch.zeros(1, 1, 2, 1, 6),
    }
    losses = response_loss(pred, batch, {"response_burden_l1": 1.0, "response_safe_bce": 0.0, "response_low_bce": 0.0, "response_components_l1": 0.0, "response_traj_l1": 0.0})
    assert torch.allclose(losses["burden"], torch.tensor(1.0))


def test_amp_helpers_do_not_require_torch_amp_grad_scaler_when_disabled():
    import importlib
    train_mod = importlib.import_module("cowp.scripts.03_train")
    assert train_mod._make_grad_scaler(False) is None
    with train_mod._autocast_context(torch.device("cpu"), False):
        x = torch.tensor(1.0) + 1.0
    assert x.item() == 2.0
