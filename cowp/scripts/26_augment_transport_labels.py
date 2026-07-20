from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cowp.core.config import load_config
from cowp.core.constants import PriorityRelation
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden
from cowp.utils.progress import tqdm_iter


def _decode_key(key: str) -> str:
    return key.replace("__", "/")


def _encode_key(key: str) -> str:
    return key.replace("/", "__")


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    raw_name: dict[str, str] = {}
    with np.load(path, allow_pickle=True) as data:
        for raw in data.files:
            key = _decode_key(raw)
            arrays[key] = data[raw]
            raw_name[key] = raw
    return arrays, raw_name


def _state_type(data: dict[str, np.ndarray], idx: int) -> int:
    for key in ("cowp/critical/agent_type",):
        arr = data.get(key)
        if arr is not None and 0 <= idx < len(np.asarray(arr).reshape(-1)):
            return int(np.asarray(arr).reshape(-1)[idx])
    cidx = data.get("cowp/critical/input_index", data.get("cowp/critical/track_index"))
    row = int(np.asarray(cidx).reshape(-1)[idx]) if cidx is not None else idx
    for key in ("state/type", "womd/state/type", "state/current/type", "womd/state/current/type"):
        arr = data.get(key)
        if arr is not None:
            flat = np.asarray(arr).reshape(-1)
            if 0 <= row < len(flat):
                return int(flat[row])
    return 1


def _derive(data: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    required = (
        "cowp/candidates/trajectory", "cowp/candidates/valid",
        "cowp/critical/valid", "cowp/natural/traj", "cowp/natural/valid",
        "cowp/natural/weight", "cowp/natural/burden_neutral", "cowp/natural/beta",
        "cowp/response/traj", "cowp/response/valid", "cowp/response/is_safe",
        "cowp/response/is_low_burden", "cowp/response/burden_total",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"missing required fields: {missing}")

    cand_traj = np.asarray(data["cowp/candidates/trajectory"])
    cand_valid = np.asarray(data["cowp/candidates/valid"], dtype=bool)
    crit_valid = np.asarray(data["cowp/critical/valid"], dtype=bool)
    nat_traj = np.asarray(data["cowp/natural/traj"])
    nat_valid = np.asarray(data["cowp/natural/valid"], dtype=bool)
    nat_weight = np.asarray(data["cowp/natural/weight"], dtype=np.float32)
    nat_burden = np.asarray(data["cowp/natural/burden_neutral"], dtype=np.float32)
    beta = np.asarray(data["cowp/natural/beta"], dtype=np.float32)
    resp_traj = np.asarray(data["cowp/response/traj"])
    resp_valid = np.asarray(data["cowp/response/valid"], dtype=bool)
    resp_safe = np.asarray(data["cowp/response/is_safe"], dtype=bool)
    resp_low = np.asarray(data["cowp/response/is_low_burden"], dtype=bool)
    resp_burden = np.asarray(data["cowp/response/burden_total"], dtype=np.float32)
    rho_arr = np.asarray(data.get("cowp/witness/rho", np.zeros((cand_traj.shape[0], crit_valid.shape[0]), dtype=np.int32)))

    K, A, M = cand_traj.shape[0], crit_valid.shape[0], nat_valid.shape[-1]
    R = resp_valid.shape[-1]
    mode_valid = np.zeros((K, A, M), dtype=bool)
    mode_conflict = np.zeros((K, A, M), dtype=bool)
    mode_retained = np.zeros((K, A, M), dtype=bool)
    root_idx = np.zeros((K, A, R), dtype=np.int32)
    is_min = np.zeros((K, A, R), dtype=bool)
    recovery = np.zeros((K, A), dtype=np.float32)
    min_weight = float(cfg.get("ncf", {}).get("min_alt_weight", 0.03))

    for a in range(A):
        if not crit_valid[a]:
            continue
        object_type = _state_type(data, a)
        roots = np.where(nat_valid[a])[0]
        if not len(roots):
            continue
        nat_xy = nat_traj[a, roots, :, :2]
        for k in np.where(cand_valid)[0]:
            ego = cand_traj[k]
            rho = PriorityRelation(int(rho_arr[k, a])) if rho_arr.ndim == 2 else PriorityRelation.UNKNOWN
            for m in roots:
                low_neutral = bool(nat_burden[a, m] <= beta[a] and nat_weight[a, m] >= min_weight)
                if not low_neutral:
                    continue
                mode_valid[k, a, m] = True
                unsafe = unsafe_between(ego, nat_traj[a, m], cfg, agent_type=object_type)
                b_under, _ = compute_burden(
                    nat_traj[a, m], ego, cfg, object_type,
                    natural_ref=nat_traj[a, m], rho=rho,
                )
                mode_conflict[k, a, m] = bool(unsafe.unsafe)
                mode_retained[k, a, m] = bool((not unsafe.unsafe) and b_under <= beta[a])

            valid_r = np.where(resp_valid[k, a])[0]
            root_low = np.zeros(M, dtype=bool)
            for r in valid_r:
                rxy = resp_traj[k, a, r, :, :2]
                dist = np.mean(np.linalg.norm(nat_xy - rxy[None, :, :], axis=-1), axis=-1)
                root = int(roots[int(np.argmin(dist))])
                root_idx[k, a, r] = root
                root_low[root] |= bool(resp_safe[k, a, r] and resp_low[k, a, r])
            for root in roots:
                members = [int(r) for r in valid_r if int(root_idx[k, a, r]) == int(root) and resp_safe[k, a, r]]
                if members:
                    best = min(members, key=lambda r: float(resp_burden[k, a, r]))
                    is_min[k, a, best] = True
            conflict_w = nat_weight[a] * mode_valid[k, a] * mode_conflict[k, a]
            denom = float(conflict_w.sum())
            if denom > 1.0e-8:
                recovery[k, a] = float((conflict_w * root_low).sum() / denom)

    return {
        "cowp/transport/mode_valid": mode_valid,
        "cowp/transport/mode_conflict": mode_conflict,
        "cowp/transport/mode_retained_low_safe": mode_retained,
        "cowp/transport/response_root_index": root_idx,
        "cowp/transport/response_is_min_burden": is_min,
        "cowp/transport/root_recovery_mass": recovery,
    }


def _write_atomic(path: Path, arrays: dict[str, np.ndarray], *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as f:
            if compress:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _process(src_s: str, dst_s: str, cfg: dict, compress: bool, force: bool) -> dict[str, object]:
    src, dst = Path(src_s), Path(dst_s)
    if dst.exists() and not force:
        try:
            with np.load(dst, allow_pickle=True) as z:
                if _encode_key("cowp/transport/mode_conflict") in z.files:
                    return {"status": "skipped", "file": src.name}
        except Exception:
            pass
    data, raw_names = _load(src)
    labels = _derive(data, cfg)
    raw: dict[str, np.ndarray] = {}
    for key, value in data.items():
        raw[raw_names.get(key, _encode_key(key))] = value
    for key, value in labels.items():
        raw[_encode_key(key)] = value
    _write_atomic(dst, raw, compress=compress)
    valid = labels["cowp/transport/mode_valid"]
    return {
        "status": "written", "file": src.name,
        "mode_pairs": int(valid.sum()),
        "mode_conflict_rate": float(labels["cowp/transport/mode_conflict"][valid].mean()) if valid.any() else 0.0,
        "mode_retain_rate": float(labels["cowp/transport/mode_retained_low_safe"][valid].mean()) if valid.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Add explicit same-root Set-Transport labels to existing COWP tensor caches.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label_cowp_v9.yaml")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config)
    src_dir, dst_dir = Path(args.input_dir), Path(args.output_dir)
    paths = sorted(src_dir.glob("*.npz"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    if args.num_workers <= 1:
        for src in tqdm_iter(paths, desc="transport labels"):
            results.append(_process(str(src), str(dst_dir / src.name), cfg, args.compress, args.force))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_process, str(src), str(dst_dir / src.name), cfg, args.compress, args.force) for src in paths]
            for fut in tqdm_iter(as_completed(futures), total=len(futures), desc="transport labels"):
                results.append(fut.result())
    written = [r for r in results if r["status"] == "written"]
    summary = {
        "input_dir": str(src_dir), "output_dir": str(dst_dir),
        "files_total": len(paths), "files_written": len(written),
        "files_skipped": sum(r["status"] == "skipped" for r in results),
        "mean_mode_conflict_rate": float(np.mean([r["mode_conflict_rate"] for r in written])) if written else None,
        "mean_mode_retain_rate": float(np.mean([r["mode_retain_rate"] for r in written])) if written else None,
    }
    out = Path(args.summary) if args.summary else dst_dir / "transport_augmentation_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
