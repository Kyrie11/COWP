import numpy as np
import torch

from ocrap.algorithms.ocmero import oc_mero, torch_oc_mero


def test_ocmero_detects_oracle_artifact_under_shared_observation():
    M = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    p = np.array([0.5, 0.5])
    C = np.ones((2, 2), dtype=float)
    res = oc_mero(M, p, C, alpha=0.5, beta=0.5, option_valid=np.array([True, True]))
    assert res.r_orc > 0.9
    assert res.r_dep < -0.9
    assert res.gap > 1.9


def test_invalid_option_masked_and_no_oracle_assert():
    M = np.array([[1.0, 10.0], [-1.0, -10.0]], dtype=float)
    res = oc_mero(M, np.array([0.5, 0.5]), np.eye(2), option_valid=np.array([True, False]))
    assert np.all(res.q[:, 1] < -1e8)
    assert isinstance(res.gap, float)


def test_torch_ocmero_matches_numpy_top_m_path():
    M = np.array([[1.0, -1.0], [-0.5, 0.3], [0.2, 0.4]], dtype=np.float32)
    p = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    C = np.array([[1.0, 0.8, 0.1], [0.7, 1.0, 0.4], [0.2, 0.9, 1.0]], dtype=np.float32)
    valid = np.array([True, True])
    ref = oc_mero(M, p, C, alpha=0.5, beta=0.5, option_valid=valid, top_m=2)
    r_dep, r_orc, gap, q = torch_oc_mero(
        torch.from_numpy(M).unsqueeze(0),
        torch.from_numpy(p).unsqueeze(0),
        torch.from_numpy(C).unsqueeze(0),
        alpha=0.5,
        beta=0.5,
        option_valid=torch.from_numpy(valid).unsqueeze(0),
        top_m=2,
    )
    assert np.isclose(float(r_dep.item()), ref.r_dep, atol=1e-5)
    assert np.isclose(float(r_orc.item()), ref.r_orc, atol=1e-5)
    assert np.isclose(float(gap.item()), ref.gap, atol=1e-5)
    assert np.allclose(q.squeeze(0).numpy(), ref.q, atol=1e-5)
