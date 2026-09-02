from __future__ import annotations

from ocrap.utils.seed import stable_seed

_VALID_SPLITS = {"train", "val", "calibration", "test"}


def scenario_split(scene_id: str, ratios: dict | None = None) -> str:
    ratios = ratios or {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}
    keys = ["train", "val", "calibration", "test"]
    total = sum(float(ratios.get(k, 0.0)) for k in keys)
    if total <= 0:
        return "train"
    u = stable_seed("split", scene_id) / float(2**32 - 1)
    acc = 0.0
    for k in keys:
        acc += float(ratios.get(k, 0.0)) / total
        if u <= acc:
            return k
    return "test"


def resolve_split(scene_id: str, cfg: dict | None = None) -> str:
    """Resolve the split for a scenario, honoring explicit dataset-shard overrides.

    Hash-based scene splits are fine for large one-shot builds, but with small
    smoke or regime-specific shards they can easily put every scenario in
    ``train``.  ``split.force_id`` / ``force_split_id`` lets builders create
    independent train/val/calibration/test shards while still writing the split
    label expected by train/evaluate/calibrate.
    """
    cfg = cfg or {}
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split", {}), dict) else {}
    forced = (
        cfg.get("force_split_id", None)
        or cfg.get("split_id_override", None)
        or split_cfg.get("force_id", None)
        or split_cfg.get("id", None)
    )
    if forced is not None and str(forced).strip():
        out = str(forced).strip()
        if out not in _VALID_SPLITS:
            raise ValueError(f"Unsupported forced split id {out!r}; expected one of {sorted(_VALID_SPLITS)}")
        return out
    return scenario_split(scene_id, cfg.get("split_ratios"))
