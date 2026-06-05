from __future__ import annotations

import json
import random
from pathlib import Path


def make_splits(index_jsonl: str | Path, output_dir: str | Path, train: float = 0.70, val: float = 0.15, test: float = 0.15, seed: int = 2026) -> dict[str, list[str]]:
    ids = []
    with Path(index_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["scenario_id"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    splits = {"train": ids[:n_train], "val": ids[n_train : n_train + n_val], "test": ids[n_train + n_val :]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (output_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for sid in values:
                f.write(sid + "\n")
    return splits
