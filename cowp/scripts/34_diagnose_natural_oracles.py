from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset
from cowp.utils.progress import tqdm_iter


def _indices(n: int, limit: int) -> list[int]:
    return list(range(n)) if limit <= 0 or limit >= n else sorted(set(np.linspace(0, n - 1, limit, dtype=np.int64).tolist()))


def _current(d: dict[str, np.ndarray]) -> np.ndarray | None:
    h = d.get("state/history")
    if h is not None:
        h = np.asarray(h)
        if h.ndim == 3 and h.shape[-1] >= 11:
            return h[:, -1, :11].astype(np.float32, copy=False)
        if h.ndim == 2 and h.shape[-1] >= 11:
            return h[:, :11].astype(np.float32, copy=False)

    def f(name: str) -> np.ndarray | None:
        x = d.get(f"state/current/{name}")
        return None if x is None else np.asarray(x).reshape(-1).astype(np.float32, copy=False)

    x, y = f("x"), f("y")
    if x is None or y is None:
        return None
    n = len(x)
    s = np.zeros((n, 11), np.float32)
    s[:, 0], s[:, 1] = x, y
    for c, names in [
        (3, ("length",)), (4, ("width",)), (5, ("height",)),
        (6, ("bbox_yaw", "heading", "yaw")),
        (7, ("velocity_x", "vx")), (8, ("velocity_y", "vy")),
        (10, ("valid",)),
    ]:
        for name in names:
            v = f(name)
            if v is not None and len(v) >= n:
                s[:, c] = v[:n]
                break
    s[:, 9] = np.linalg.norm(s[:, 7:9], axis=-1)
    if not np.any(s[:, 10] > 0.5):
        s[:, 10] = 1.0
    return s


def _bank(cur: np.ndarray, horizon: int, dt: float) -> np.ndarray:
    """Vectorized 15-trajectory diagnostic bank used by the original v13 oracle."""
    t = (np.arange(horizon, dtype=np.float32) + 1.0) * float(dt)
    accels = np.asarray((-3.0, -1.5, 0.0, 1.5, 3.0), np.float32)
    yaw_rates = np.asarray((-0.14, 0.0, 0.14), np.float32)
    aa, ww = np.meshgrid(accels, yaw_rates, indexing="ij")
    a = aa.reshape(-1, 1)
    w = ww.reshape(-1, 1)
    yaw0 = float(cur[6])
    speed0 = float(max(cur[9], np.linalg.norm(cur[7:9]), 0.0))
    speed = np.maximum(0.0, speed0 + a * t[None])
    yaw = yaw0 + w * t[None]
    vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
    out = np.zeros((len(a), horizon, 7), np.float32)
    out[..., 0] = float(cur[0]) + np.cumsum(vx, axis=1) * float(dt)
    out[..., 1] = float(cur[1]) + np.cumsum(vy, axis=1) * float(dt)
    out[..., 2], out[..., 3], out[..., 4] = yaw, vx, vy
    out[..., 5], out[..., 6] = max(float(cur[3]), 0.1), max(float(cur[4]), 0.1)
    return out


def _stats(x: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(len(a)), "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def _process_scene(ds: COWPNpzDataset, idx: int, dt: float) -> tuple[dict[str, list[float]], Counter]:
    wanted = {"state/", "womd/state/", "cowp/natural/", "cowp/critical/"}
    d = ds.load(idx, wanted)
    s = _current(d)
    nat = np.asarray(d.get("cowp/natural/traj", []), np.float32)
    valid = np.asarray(d.get("cowp/natural/valid", []), bool)
    source = np.asarray(d.get("cowp/natural/source", np.zeros(valid.shape, np.int64)), np.int64)
    inp = np.asarray(d.get("cowp/critical/input_index", d.get("cowp/critical/track_index", [])), np.int64).reshape(-1)
    values: dict[str, list[float]] = defaultdict(list)
    counts: Counter = Counter()
    if s is None or nat.ndim != 4 or valid.shape != nat.shape[:2]:
        counts["invalid_scene"] += 1
        return values, counts

    horizon = int(nat.shape[2])
    hs = {sec: min(horizon, max(1, round(sec / float(dt)))) for sec in (1, 3, 5, 8)}
    for a in range(min(len(inp), nat.shape[0])):
        j = int(inp[a])
        roots = np.flatnonzero(valid[a])
        if j < 0 or j >= len(s) or roots.size == 0:
            continue
        gt = nat[a, roots]
        finite = np.isfinite(gt[..., :2]).all(axis=(1, 2))
        gt, roots = gt[finite], roots[finite]
        if not len(gt):
            continue
        bank = _bank(s[j], horizon, dt)
        # [R,B,T], then one cumulative reduction serves every requested horizon.
        dist = np.linalg.norm(gt[:, None, :, :2] - bank[None, :, :, :2], axis=-1)
        cum = np.cumsum(dist, axis=-1)
        for sec, h in hs.items():
            best = (cum[:, :, h - 1] / float(h)).min(axis=1)
            values[f"all/{sec}s"].extend(best.astype(float).tolist())
            src = source[a, roots]
            for src_id in np.unique(src):
                sel = src == src_id
                values[f"source_{int(src_id)}/{sec}s"].extend(best[sel].astype(float).tolist())
        counts["natural_roots"] += int(len(gt))
        counts["critical_agents"] += 1
    counts["valid_scene"] += 1
    return values, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast vectorized kinematic-oracle coverage diagnostic for COWP natural labels.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-scenes", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8, help="Threaded NPZ readers; 4-16 is usually optimal on local SSD.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    ds = COWPNpzDataset(args.cache_dir)
    idxs = _indices(len(ds), int(args.max_scenes))
    values: dict[str, list[float]] = defaultdict(list)
    counts: Counter = Counter()

    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = tqdm_iter(idxs, enabled=not args.no_progress, total=len(idxs), desc="natural oracle", unit="scene")
        results = (_process_scene(ds, idx, float(args.dt)) for idx in iterator)
        for local_values, local_counts in results:
            counts.update(local_counts)
            for key, vals in local_values.items():
                values[key].extend(vals)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_scene, ds, idx, float(args.dt)) for idx in idxs]
            iterator = tqdm_iter(as_completed(futures), enabled=not args.no_progress, total=len(futures), desc="natural oracle", unit="scene")
            for future in iterator:
                local_values, local_counts = future.result()
                counts.update(local_counts)
                for key, vals in local_values.items():
                    values[key].extend(vals)

    distributions = {k: _stats(v) for k, v in sorted(values.items())}
    full = distributions.get("all/8s", {})
    report = {
        "cache_dir": str(Path(args.cache_dir)),
        "sampled_scenes": len(idxs),
        "workers": workers,
        "counts": dict(counts),
        "kinematic_bank_minade_m": distributions,
        "interpretation": {
            "oracle_8s_mean_m": full.get("mean"),
            "recommended_model_gate_m": None if full.get("mean") is None else float(full["mean"]) + 6.0,
            "warning": "This label-space oracle is necessary but not sufficient. Run 35_diagnose_model_anchor.py to test the exact model-facing critical index, anchor and typed basis before training.",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
