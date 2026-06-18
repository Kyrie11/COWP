from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cowp.data.dataset import _infer_agent_count, _infer_agent_visible_mask, _restore_key
from cowp.utils.progress import tqdm_iter


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {_restore_key(k): data[k] for k in data.files}


def diagnose_cache(cache_dir: str | Path, pattern: str = "*.npz", *, progress: bool = True, output: str | Path | None = None) -> dict[str, object]:
    cache_dir = Path(cache_dir)
    paths = sorted(cache_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No tensor-cache npz files found in {cache_dir} matching {pattern}")

    stats = {
        "cache_dir": str(cache_dir),
        "num_files": len(paths),
        "scenes_with_any_valid_critical": 0,
        "scenes_with_all_valid_critical_invisible": 0,
        "scenes_with_any_out_of_range_critical": 0,
        "scenes_with_any_current_invisible_critical": 0,
        "valid_critical_slots": 0,
        "visible_critical_slots": 0,
        "out_of_range_critical_slots": 0,
        "current_invisible_critical_slots": 0,
        "max_agent_count": 0,
        "min_agent_count": None,
        "max_critical_index": -1,
        "examples": [],
    }

    for path in tqdm_iter(paths, enabled=progress, total=len(paths), desc="Diagnose critical visibility", unit="file"):
        d = _load_npz(path)
        idx = np.asarray(d.get("cowp/critical/track_index", []), dtype=np.int64).reshape(-1)
        if idx.size == 0:
            continue
        valid = np.asarray(d.get("cowp/critical/valid", np.ones_like(idx, dtype=bool))).reshape(-1).astype(bool)
        n = _infer_agent_count(d)
        if n is None:
            continue
        stats["max_agent_count"] = max(int(stats["max_agent_count"]), int(n))
        stats["min_agent_count"] = int(n) if stats["min_agent_count"] is None else min(int(stats["min_agent_count"]), int(n))
        if idx.size:
            stats["max_critical_index"] = max(int(stats["max_critical_index"]), int(idx.max()))
        agent_visible = _infer_agent_visible_mask(d, n)
        in_range = (idx >= 0) & (idx < int(n))
        visible = in_range.copy()
        if agent_visible is not None and len(agent_visible) > 0:
            safe_idx = np.clip(idx, 0, len(agent_visible) - 1)
            current_vis = agent_visible[safe_idx]
            visible = visible & current_vis
        else:
            current_vis = np.ones_like(visible, dtype=bool)

        valid_slots = valid
        visible_valid = valid_slots & visible
        oor_valid = valid_slots & ~in_range
        current_invis_valid = valid_slots & in_range & ~current_vis
        if valid_slots.any():
            stats["scenes_with_any_valid_critical"] += 1
        if valid_slots.any() and not visible_valid.any():
            stats["scenes_with_all_valid_critical_invisible"] += 1
        if oor_valid.any():
            stats["scenes_with_any_out_of_range_critical"] += 1
        if current_invis_valid.any():
            stats["scenes_with_any_current_invisible_critical"] += 1
        stats["valid_critical_slots"] += int(valid_slots.sum())
        stats["visible_critical_slots"] += int(visible_valid.sum())
        stats["out_of_range_critical_slots"] += int(oor_valid.sum())
        stats["current_invisible_critical_slots"] += int(current_invis_valid.sum())
        if (oor_valid.any() or current_invis_valid.any()) and len(stats["examples"]) < 20:
            stats["examples"].append({
                "file": path.name,
                "num_agents": int(n),
                "critical_index": idx.tolist(),
                "critical_valid": valid.tolist(),
                "input_visible": visible.tolist(),
            })

    denom = max(int(stats["valid_critical_slots"]), 1)
    scene_denom = max(int(stats["scenes_with_any_valid_critical"]), 1)
    stats["visible_critical_slot_ratio"] = float(stats["visible_critical_slots"] / denom)
    stats["out_of_range_critical_slot_ratio"] = float(stats["out_of_range_critical_slots"] / denom)
    stats["current_invisible_critical_slot_ratio"] = float(stats["current_invisible_critical_slots"] / denom)
    stats["all_valid_critical_invisible_scene_ratio"] = float(stats["scenes_with_all_valid_critical_invisible"] / scene_denom)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose whether COWP critical-agent labels are visible in tensor-cache WOMD input tensors.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    stats = diagnose_cache(args.cache_dir, args.pattern, progress=not args.no_progress, output=args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
