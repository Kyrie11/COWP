from __future__ import annotations

import torch
import pytest

from ocrap.models.ocrap import OCRAPModel


def _model(**kw):
    base = dict(
        input_dim=16,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        encoder_type='structured_transformer',
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_root_tail_source=True,
        direct_recovery_semantic_witness_tail_localization=True,
        direct_recovery_semantic_witness_structured_tail_field=True,
        direct_recovery_semantic_witness_signed_tail_channels=True,
        direct_recovery_semantic_witness_counterfactual_tail_response=True,
    )
    base.update(kw)
    return OCRAPModel(**base)


def test_v4883_counterfactual_latent_response_exact_and_nominal_zero():
    x = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [2.0, 3.0]]],
        [[[2.0, 4.0], [4.0, 7.0]], [[1.0, 3.0], [5.0, 9.0]]],
        [[[0.0, 0.0], [5.0, 1.0]], [[-1.0, 2.0], [0.0, 1.0]]],
    ])
    g = torch.tensor([[7, 8], [7, 8], [7, 8]])
    n = torch.tensor([1.0, 0.0, 0.0])
    got = OCRAPModel._counterfactual_tail_response(x, g, n)
    assert torch.equal(got, x - x[:1])
    assert torch.count_nonzero(got[0]).item() == 0


def test_v4883_counterfactual_latent_response_malformed_group_fails_closed():
    x = torch.randn(3, 2, 2, 4)
    g = torch.tensor([[1, 2], [1, 2], [1, 2]])
    two_nominals = torch.tensor([1.0, 1.0, 0.0])
    got = OCRAPModel._counterfactual_tail_response(x, g, two_nominals)
    assert torch.count_nonzero(got).item() == 0


def test_v4883_counterfactual_field_requires_signed_structured_field():
    with pytest.raises(ValueError):
        _model(direct_recovery_semantic_witness_signed_tail_channels=False)
    with pytest.raises(ValueError):
        _model(direct_recovery_semantic_witness_structured_tail_field=False)


def test_v4883_parameter_contract_is_only_two_channel_field():
    m = _model()
    w = m.direct_absolute_structured_tail_field_weight
    assert w is not None
    assert tuple(w.shape) == (2, 16)
    assert torch.count_nonzero(w).item() == 0
    assert m.direct_absolute_root_tail_source_scale is None
