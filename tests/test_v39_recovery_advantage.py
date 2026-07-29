import torch

from ocrap.models.losses import observation_consistent_recovery_advantage_loss


def _loss(pred_r_dep, pred_q):
    # One near-contact scene-time group: nominal (macro 0) and brake (macro 2).
    teacher_r_dep = torch.tensor([-2.0, 0.5])
    teacher_r_orc = torch.tensor([-1.8, 0.6])
    teacher_q = torch.tensor([[[-2.0]], [[2.0]]])
    return observation_consistent_recovery_advantage_loss(
        torch.tensor(pred_r_dep, requires_grad=True),
        torch.tensor([0.02, 0.15], requires_grad=True),
        torch.tensor(pred_q, requires_grad=True),
        teacher_r_dep,
        teacher_r_orc,
        teacher_q,
        torch.ones((2, 1)),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
        torch.tensor([7, 7]),
        torch.tensor([4, 4]),
        torch.tensor([0, 2]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1, 1]),
        success_temperature=0.1,
    )


def test_recovery_advantage_loss_penalizes_near_contact_pairwise_inversion():
    inverted = _loss([1.5, -0.5], [[[2.0]], [[-1.0]]])
    aligned = _loss([-1.5, 0.5], [[[-2.0]], [[2.0]]])
    assert inverted.item() > aligned.item()
    inverted.backward()


def test_recovery_advantage_loss_ignores_unrequested_bucket():
    pred = torch.tensor([1.0, -1.0], requires_grad=True)
    loss = observation_consistent_recovery_advantage_loss(
        pred,
        torch.tensor([0.0, 0.1]),
        torch.tensor([[[1.0]], [[-1.0]]]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-0.8, 1.1]),
        torch.tensor([[[-1.0]], [[1.0]]]),
        torch.ones((2, 1)),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
        torch.tensor([0, 2]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0, 0]),
        bucket_ids=(1, 2),
    )
    assert loss.item() == 0.0
