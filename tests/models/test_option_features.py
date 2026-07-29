import numpy as np
import torch

from ocrap.models.data import OPTION_FEATURE_DIM, option_features_from_sample
from ocrap.models.ocrap import OCRAPModel


def test_option_features_condition_margin_decoder():
    sample = {
        "m_star": np.zeros((2, 3), dtype=np.float32),
        "option_valid": np.array([1, 0, 1], dtype=np.float32),
        "recovery_modes": np.array(["stop", "lateral_escape", "avoid_secondary"]),
        "recovery_params": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    }
    feat = option_features_from_sample(sample)
    assert feat.shape == (3, OPTION_FEATURE_DIM)
    assert feat[0, 0] == 1.0
    assert feat[1, -2] == 0.0

    model = OCRAPModel(input_dim=5, num_roots=2, num_options=3, d_model=16, d_obs=4, num_heads=4, option_feature_dim=OPTION_FEATURE_DIM)
    x = torch.randn(2, 5)
    opt = torch.from_numpy(np.stack([feat, feat], axis=0))
    out = model(x, opt)
    assert out["margins"].shape == (2, 2, 3)
