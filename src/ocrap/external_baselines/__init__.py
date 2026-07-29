"""External baseline adapters for OC-RAP.

The modules in this package intentionally keep the baseline implementations
separate from the OC-RAP model.  They consume the same OC-RAP dataset shards
(.npz samples grouped by scene_id/time_index) and report the same regime-level
metrics, but they do not use OC-RAP's observation-consistent selector.
"""

from .train import train_external_baseline
from .evaluate import evaluate_external_baselines

__all__ = ["train_external_baseline", "evaluate_external_baselines"]
