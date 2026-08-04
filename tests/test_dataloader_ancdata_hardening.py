from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as torch_mp

from cowp.utils.dataloader_runtime import configure_dataloader_runtime


def test_auto_sharing_strategy_avoids_file_descriptor_ancdata_path() -> None:
    before = torch_mp.get_sharing_strategy()
    try:
        available = torch_mp.get_all_sharing_strategies()
        if "file_descriptor" in available:
            torch_mp.set_sharing_strategy("file_descriptor")
        runtime = configure_dataloader_runtime("auto")
        if "file_system" in available:
            assert runtime["selected"] == "file_system"
        assert runtime["nofile_soft"] is None or runtime["nofile_soft"] > 0
    finally:
        torch_mp.set_sharing_strategy(before)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
def test_validation_loader_can_release_workers_after_each_pass() -> None:
    train = importlib.import_module("cowp.scripts.03_train")

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int):
            return {"x": torch.tensor([index], dtype=torch.float32)}

    dl = train._make_loader(
        TinyDataset(),
        {"train": {"num_workers": 1, "prefetch_factor": 1}},
        2,
        shuffle=False,
        oversample=False,
        use_cuda=False,
        progress=False,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
    )
    assert dl.num_workers == 1
    assert dl.persistent_workers is False
    assert sum(int(batch["x"].numel()) for batch in dl) == 8


def test_v16_2_launcher_uses_separate_safe_validation_pipeline() -> None:
    path = Path("run_cowp_v16_2_dual_gpu.sh")
    if not path.exists():
        pytest.skip("legacy v16.2 launcher is not shipped in the active v16.8 package")
    source = path.read_text(encoding="utf-8")
    assert 'TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"' in source
    assert '--val-num-workers "$NATURAL_VAL_NUM_WORKERS"' in source
    assert '--val-prefetch-factor "$NATURAL_VAL_PREFETCH_FACTOR"' in source
    assert '--sharing-strategy "$TORCH_SHARING_STRATEGY"' in source
