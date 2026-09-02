from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset
from cowp.utils.progress import tqdm_iter


def _sample_indices(n: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=limit, dtype=np.int64).tolist()))


def _current_state(d: dict[str, np.ndarray]) -> np.ndarray | None:
    h = d.get("state/history")
    if h is not None:
        h = np.asarray(h)
        if h.ndim == 3 and h.shape[-1] >= 11:
            return h[:, -1, :11].astype(np.float32, copy=False)
        if h.ndim == 2 and h.shape[-1] >= 11:
            return h[:, :11].astype(np.float32, copy=False)

    def cur(name: str) -> np.ndarray | None:
        x = d.get(f"state/current/{name}")
        return None if x is None else np.asarray(x).reshape(-1).astype(np.float32, copy=False)

    x, y = cur("x"), cur("y")
    if x is None or y is None:
        return None
    n = len(x)
    out = np.zeros((n, 11), dtype=np.float32)
    out[:, 0], out[:, 1] = x, y
    aliases = {
        3: ("length",), 4: ("width",), 5: ("height",),
        6: ("bbox_yaw", "heading", "yaw"),
        7: ("velocity_x", "vx"), 8: ("velocity_y", "vy"), 10: ("valid",),
    }
    for col, names in aliases.items():
        for name in names:
            v = cur(name)
            if v is not None and len(v) >= n:
                out[:, col] = v[:n]
                break
    out[:, 9] = np.linalg.norm(out[:, 7:9], axis=-1)
    if not np.any(out[:, 10] > 0.5):
        out[:, 10] = 1.0
    return out


def _finite_stats(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def _digest(arr: np.ndarray, mode: str) -> str:
    a = np.ascontiguousarray(arr)
    b = a.view(np.uint8).reshape(-1)
    h = hashlib.sha1()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    if mode == "full" or b.size <= 196608:
        h.update(b)
    else:
        n = min(65536, b.size)
        h.update(b[:n])
        mid = max(0, b.size // 2 - n // 2)
        h.update(b[mid: mid + n])
        h.update(b[-n:])
    return h.hexdigest()


def _same_storage(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        try:
            return a.resolve() == b.resolve()
        except OSError:
            return False


def _process_scene(
    raw: COWPNpzDataset,
    trans: COWPNpzDataset,
    ri: int,
    ti: int,
    dt: float,
    hash_mode: str,
) -> dict:
    wanted = {
        "state/", "womd/state/", "cowp/critical/", "cowp/natural/",
        "cowp/candidates/trajectory", "cowp/candidates/valid",
        "cowp/response/valid", "cowp/transport/",
        "waymax/candidate_selected_for_rollout", "waymax/candidate_rollout_valid",
        "waymax/candidate_log_divergence",
    }
    d0 = raw.load(ri, wanted)
    d1 = trans.load(ti, wanted)
    c = Counter(scenes=1)
    vals: dict[str, list[float]] = defaultdict(list)
    src = Counter()
    missing = Counter()
    examples: dict[str, list[str]] = {"base": [], "mapping": []}
    name = raw.paths[ri].name

    base_keys = (
        "cowp/natural/traj", "cowp/natural/valid", "cowp/natural/source",
        "cowp/candidates/trajectory", "cowp/candidates/valid",
        "cowp/critical/track_id", "cowp/critical/track_index",
    )
    if _same_storage(raw.paths[ri], trans.paths[ti]):
        c["base_payload_same_storage"] += 1
    else:
        same = True
        for key in base_keys:
            if key not in d0 or key not in d1:
                continue
            a, b = np.asarray(d0[key]), np.asarray(d1[key])
            if a.shape != b.shape or a.dtype != b.dtype or _digest(a, hash_mode) != _digest(b, hash_mode):
                same = False
                break
        if not same:
            c["base_payload_mismatch"] += 1
            examples["base"].append(name)

    for key in ("cowp/transport/response_root_index", "cowp/transport/mode_valid", "cowp/transport/mode_conflict"):
        if key not in d1:
            missing[key] += 1

    crit = np.asarray(d1.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
    inp = np.asarray(d1.get("cowp/critical/input_index", d1.get("cowp/critical/track_index", [])), np.int64).reshape(-1)
    visible = np.asarray(d1.get("cowp/critical/input_visible", np.ones_like(inp, bool)), bool).reshape(-1)
    n = min(len(crit), len(inp))
    crit, inp = crit[:n], inp[:n]
    visible = visible[:n] if len(visible) >= n else np.zeros(n, bool)
    c["critical_valid"] += int(crit.sum())
    c["critical_unmapped"] += int((crit & (inp < 0)).sum())
    c["critical_invisible"] += int((crit & ~visible).sum())
    if np.any(crit & (inp < 0)):
        examples["mapping"].append(name)

    state = _current_state(d1)
    nat = np.asarray(d1.get("cowp/natural/traj", []), dtype=np.float32)
    nat_valid = np.asarray(d1.get("cowp/natural/valid", []), dtype=bool)
    nat_source = np.asarray(d1.get("cowp/natural/source", np.zeros(nat_valid.shape, dtype=np.int64)), np.int64)
    if state is not None and nat.ndim == 4 and nat_valid.shape == nat.shape[:2]:
        for a in range(min(len(inp), nat.shape[0])):
            j = int(inp[a])
            roots = np.flatnonzero(nat_valid[a])
            if j < 0 or j >= len(state) or roots.size == 0:
                continue
            tr = nat[a, roots]
            finite = np.isfinite(tr[..., :2]).all(axis=(1, 2))
            c["natural_nonfinite"] += int((~finite).sum())
            tr, roots = tr[finite], roots[finite]
            if not len(tr):
                continue
            cur = state[j]
            cv1 = cur[:2] + cur[7:9] * float(dt)
            vals["natural_first_step_jump_m"].extend(np.linalg.norm(tr[:, 0, :2] - cur[None, :2], axis=-1).astype(float).tolist())
            vals["natural_first_step_cv_error_m"].extend(np.linalg.norm(tr[:, 0, :2] - cv1[None], axis=-1).astype(float).tolist())
            vals["natural_8s_displacement_m"].extend(np.linalg.norm(tr[:, -1, :2] - cur[None, :2], axis=-1).astype(float).tolist())
            if tr.shape[-1] >= 5:
                vals["natural_max_speed_mps"].extend(np.linalg.norm(tr[..., 3:5], axis=-1).max(axis=1).astype(float).tolist())
            src.update(map(str, nat_source[a, roots].astype(int).tolist()))

    root_idx = d1.get("cowp/transport/response_root_index")
    if root_idx is not None:
        r = np.asarray(root_idx, np.int64)
        m = int(nat.shape[1]) if nat.ndim == 4 else 0
        response_valid = np.asarray(d1.get("cowp/response/valid", np.ones_like(r, bool)), bool)
        c["response_root_valid_slots"] += int(response_valid.sum())
        c["response_root_out_of_range"] += int((response_valid & ((r < 0) | (r >= max(m, 1)))).sum())

    selected = np.asarray(d1.get("waymax/candidate_selected_for_rollout", []), bool)
    rollout = np.asarray(d1.get("waymax/candidate_rollout_valid", []), bool)
    if selected.size:
        c["waymax_selected"] += int(selected.sum())
        if rollout.shape == selected.shape:
            c["waymax_rollout_valid"] += int((selected & rollout).sum())
    logdiv = np.asarray(d1.get("waymax/candidate_log_divergence", []), np.float32)
    if logdiv.size and rollout.shape == logdiv.shape:
        finite = rollout & np.isfinite(logdiv)
        c["waymax_logdiv_finite"] += int(finite.sum())
        vals["waymax_logdiv"].extend(logdiv[finite].astype(float).tolist())
    return {"counts": c, "values": vals, "sources": src, "missing": missing, "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast raw/transport cache alignment and label-health diagnostic.")
    ap.add_argument("--raw-cache", required=True)
    ap.add_argument("--transport-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-scenes", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hash-mode", choices=["sampled", "full"], default="sampled", help="Full SHA1 is much slower; same-file/symlink overlays skip hashing entirely.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    raw = COWPNpzDataset(args.raw_cache)
    trans = COWPNpzDataset(args.transport_cache)
    trans_by_name = {p.name: i for i, p in enumerate(trans.paths)}
    pairs, count = [], Counter()
    for ri in _sample_indices(len(raw), int(args.max_scenes)):
        ti = trans_by_name.get(raw.paths[ri].name)
        if ti is None:
            count["transport_file_missing"] += 1
        else:
            pairs.append((ri, ti))

    values: dict[str, list[float]] = defaultdict(list)
    source_counts, missing_transport = Counter(), Counter()
    examples = {"base_payload_mismatch": [], "unmapped_or_negative_input_index": []}
    workers = max(1, int(args.workers))

    def merge(res: dict) -> None:
        count.update(res["counts"])
        source_counts.update(res["sources"])
        missing_transport.update(res["missing"])
        for k, v in res["values"].items():
            values[k].extend(v)
        examples["base_payload_mismatch"].extend(res["examples"]["base"][: max(0, 10 - len(examples["base_payload_mismatch"]))])
        examples["unmapped_or_negative_input_index"].extend(res["examples"]["mapping"][: max(0, 10 - len(examples["unmapped_or_negative_input_index"]))])

    if workers == 1:
        it = tqdm_iter(pairs, enabled=not args.no_progress, total=len(pairs), desc="cache alignment", unit="scene")
        for ri, ti in it:
            merge(_process_scene(raw, trans, ri, ti, float(args.dt), args.hash_mode))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs = [pool.submit(_process_scene, raw, trans, ri, ti, float(args.dt), args.hash_mode) for ri, ti in pairs]
            it = tqdm_iter(as_completed(fs), enabled=not args.no_progress, total=len(fs), desc="cache alignment", unit="scene")
            for f in it:
                merge(f.result())

    crit_valid = max(count["critical_valid"], 1)
    root_valid = max(count["response_root_valid_slots"], 1)
    rates = {
        "base_payload_mismatch": count["base_payload_mismatch"] / max(count["scenes"], 1),
        "critical_unmapped": count["critical_unmapped"] / crit_valid,
        "critical_invisible": count["critical_invisible"] / crit_valid,
        "response_root_out_of_range": count["response_root_out_of_range"] / root_valid,
        "waymax_selected_rollout_success": count["waymax_rollout_valid"] / max(count["waymax_selected"], 1),
        "waymax_logdiv_finite": count["waymax_logdiv_finite"] / max(count["waymax_rollout_valid"], 1),
    }
    distributions = {k: _finite_stats(v) for k, v in values.items()}
    report = {
        "raw_cache": str(Path(args.raw_cache)), "transport_cache": str(Path(args.transport_cache)),
        "raw_files": len(raw), "transport_files": len(trans), "sampled_scenes": count["scenes"],
        "workers": workers, "hash_mode": args.hash_mode,
        "counts": dict(count), "source_counts": dict(source_counts),
        "missing_transport_keys": dict(missing_transport), "rates": rates,
        "distributions": distributions, "examples": examples,
    }
    hard_fail = []
    if rates["base_payload_mismatch"] > 0.0:
        hard_fail.append("transport overlay changes base tensors")
    if rates["critical_unmapped"] > 0.02 or rates["critical_invisible"] > 0.05:
        hard_fail.append("critical-agent-to-WOMD row alignment is insufficient")
    if rates["response_root_out_of_range"] > 1e-4:
        hard_fail.append("response root indices are out of range")
    jump = distributions.get("natural_first_step_cv_error_m", {}).get("p90")
    if jump is not None and float(jump) > 5.0:
        hard_fail.append("natural first future step is inconsistent with current state/CV anchor")
    report["pass"] = not hard_fail
    report["hard_fail_reasons"] = hard_fail
    report["optimization_note"] = "Only required NPZ members are read; overlay symlinks bypass content hashing; per-scene I/O is threaded; natural statistics are vectorized."
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
