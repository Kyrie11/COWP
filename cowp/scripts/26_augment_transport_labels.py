from __future__ import annotations

import argparse
import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from cowp.core.config import load_config
from cowp.core.constants import PriorityRelation
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden
from cowp.label.witness import _root_affinity
from cowp.label.safe_responses import root_conditioned_recovery_search
from cowp.utils.progress import tqdm_iter


TRANSPORT_KEYS = (
    "cowp/transport/mode_valid",
    "cowp/transport/mode_conflict",
    "cowp/transport/mode_retained_low_safe",
    "cowp/transport/response_root_index",
    "cowp/transport/response_is_min_burden",
    "cowp/transport/root_recovery_mass",
    "cowp/transport/root_low_safe_score",
    "cowp/transport/root_target_confidence",
    "cowp/transport/root_min_safe_burden",
    "cowp/transport/transported_opr",
)

REQUIRED_INPUT_KEYS = (
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/critical/valid",
    "cowp/natural/traj",
    "cowp/natural/valid",
    "cowp/natural/weight",
    "cowp/natural/burden_neutral",
    "cowp/natural/beta",
    "cowp/response/traj",
    "cowp/response/valid",
    "cowp/response/is_safe",
    "cowp/response/is_low_burden",
    "cowp/response/burden_total",
)
_WORKER_CFG: dict | None = None


def _init_worker(cfg: dict) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


OPTIONAL_INPUT_KEYS = (
    "cowp/witness/rho",
    "cowp/critical/agent_type",
    "cowp/response/root_index",
    "cowp/response/root_affinity",
    "cowp/critical/input_index",
    "cowp/critical/track_index",
    "state/type",
    "womd/state/type",
    "state/current/type",
    "womd/state/current/type",
)


def _decode_key(key: str) -> str:
    return key.replace("__", "/")


def _encode_key(key: str) -> str:
    return key.replace("/", "__")


def _load_selected(path: Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Read only tensors needed for transport-label construction.

    Old code materialized every array in the cache, including large state and
    roadgraph tensors.  Selective loading avoids a large amount of disk traffic
    and worker memory pressure.
    """
    wanted = set(keys)
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        for raw in data.files:
            key = _decode_key(raw)
            if key in wanted:
                arrays[key] = data[raw]
    return arrays


def _load_all(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    raw_name: dict[str, str] = {}
    with np.load(path, allow_pickle=True) as data:
        for raw in data.files:
            key = _decode_key(raw)
            arrays[key] = data[raw]
            raw_name[key] = raw
    return arrays, raw_name


def _state_type(data: dict[str, np.ndarray], idx: int) -> int:
    arr = data.get("cowp/critical/agent_type")
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


def _pair_broadphase_far(
    ego: np.ndarray,
    natural: np.ndarray,
    cfg: dict,
    agent_type: int,
) -> np.ndarray:
    """Exact negative broad phase for all candidate/root pairs.

    Returns [K,M] where True means the pair cannot trigger collision, near-miss,
    TTC, or RSS tests.  Near pairs are still passed to ``unsafe_between`` so the
    generated labels remain bit-for-bit compatible with the original geometry
    checks, apart from floating-point tie cases at the gate boundary.
    """
    if ego.size == 0 or natural.size == 0:
        return np.ones((len(ego), len(natural)), dtype=bool)
    t = min(ego.shape[-2], natural.shape[-2])
    e = np.asarray(ego[:, :t], dtype=np.float32)
    n = np.asarray(natural[:, :t], dtype=np.float32)
    delta = e[:, None, :, :2] - n[None, :, :, :2]
    dmin = np.linalg.norm(delta, axis=-1).min(axis=-1)

    e_half = 0.5 * np.sqrt(np.maximum(e[..., 5], 0.1) ** 2 + np.maximum(e[..., 6], 0.1) ** 2)
    n_half = 0.5 * np.sqrt(np.maximum(n[..., 5], 0.1) ** 2 + np.maximum(n[..., 6], 0.1) ** 2)
    max_half = (e_half[:, None, :] + n_half[None, :, :]).max(axis=-1)

    unsafe_cfg = cfg.get("unsafe", cfg)
    near_thresh = float(unsafe_cfg.get("near_miss_distance_vehicle_m", 1.0 if agent_type == 1 else 1.5))
    dist_gate = float(unsafe_cfg.get("ttc_distance_gate_vehicle_m", 15.0 if agent_type == 1 else 20.0))
    inflation = float(unsafe_cfg.get("collision_inflation_m", 0.1))
    gate = np.maximum(dist_gate, near_thresh + max_half + inflation + 1.0)
    return dmin > gate


def _derive(data: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    missing = [k for k in REQUIRED_INPUT_KEYS if k not in data]
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
    rho_arr = np.asarray(
        data.get(
            "cowp/witness/rho",
            np.zeros((cand_traj.shape[0], crit_valid.shape[0]), dtype=np.int32),
        )
    )

    K, A, M = cand_traj.shape[0], crit_valid.shape[0], nat_valid.shape[-1]
    R = resp_valid.shape[-1]
    min_weight = float(cfg.get("ncf", {}).get("min_alt_weight", 0.03))

    low_support = nat_valid & (nat_weight >= min_weight) & (nat_burden <= beta[:, None])
    mode_valid = (
        cand_valid[:, None, None]
        & crit_valid[None, :, None]
        & low_support[None, :, :]
    )
    mode_conflict = np.zeros((K, A, M), dtype=bool)
    intrinsic_low = np.zeros((A, M), dtype=bool)

    valid_k = np.where(cand_valid)[0]
    for a in np.where(crit_valid)[0]:
        a = int(a)
        object_type = _state_type(data, a)
        roots = np.where(low_support[a])[0]
        if not len(roots) or not len(valid_k):
            continue

        # The candidate-conditioned burden of an unchanged natural trajectory is
        # candidate-independent whenever the pair is not unsafe: TTC/RSS terms are
        # then exactly zero, while acceleration/jerk/progress/norm terms depend only
        # on the natural trajectory.  Compute it once per root rather than K times.
        rho0 = PriorityRelation(int(rho_arr[valid_k[0], a])) if rho_arr.ndim == 2 else PriorityRelation.UNKNOWN
        for m in roots:
            b0, _ = compute_burden(
                nat_traj[a, m],
                None,
                cfg,
                object_type,
                natural_ref=nat_traj[a, m],
                rho=rho0,
            )
            intrinsic_low[a, m] = bool(b0 <= beta[a])

        far = _pair_broadphase_far(cand_traj[valid_k], nat_traj[a, roots], cfg, object_type)
        conflict_local = np.zeros_like(far, dtype=bool)
        near_pairs = np.argwhere(~far)
        for kk, mm in near_pairs:
            k = int(valid_k[int(kk)])
            m = int(roots[int(mm)])
            conflict_local[int(kk), int(mm)] = bool(
                unsafe_between(cand_traj[k], nat_traj[a, m], cfg, agent_type=object_type).unsafe
            )
        mode_conflict[np.ix_(valid_k, np.asarray([a]), roots)] = conflict_local[:, None, :]

    mode_retained = mode_valid & (~mode_conflict) & intrinsic_low[None, :, :]

    root_idx = np.full((K, A, R), -1, dtype=np.int32)
    is_min = np.zeros((K, A, R), dtype=bool)
    recovery = np.zeros((K, A), dtype=np.float32)
    root_low_score = np.zeros((K, A, M), dtype=np.float32)
    root_confidence = np.zeros((K, A, M), dtype=np.float32)
    root_min_safe_burden = np.full((K, A, M), 2.0, dtype=np.float32)
    transported_opr = np.zeros((K, A), dtype=np.float32)
    explicit_root = data.get("cowp/response/root_index")
    explicit_affinity = data.get("cowp/response/root_affinity")
    rcfg = cfg.get("response", {}).get("root_conditioned_transport", {})
    root_search_enabled = bool(rcfg.get("enabled", True))

    for a in np.where(crit_valid)[0]:
        a = int(a)
        roots = np.where(nat_valid[a])[0]
        if not len(roots):
            continue
        nat_roots = np.asarray(nat_traj[a, roots], dtype=np.float32)
        min_affinity = float(cfg.get("response", {}).get("root_assignment_min_affinity", 0.35))
        for k in valid_k:
            k = int(k)
            valid_r = np.where(resp_valid[k, a])[0]
            if len(valid_r):
                valid_root_set = set(int(x) for x in roots.tolist())
                for ridx in valid_r:
                    affinity = np.zeros(M, dtype=np.float32)
                    explicit = int(explicit_root[k, a, ridx]) if explicit_root is not None else -1
                    if explicit in valid_root_set:
                        conf = 1.0
                        if explicit_affinity is not None:
                            conf = float(np.clip(explicit_affinity[k, a, ridx], 0.0, 1.0))
                        affinity[explicit] = max(conf, min_affinity)
                    else:
                        local = _root_affinity(resp_traj[k, a, ridx], nat_roots, cfg)
                        affinity[roots] = local
                    root = int(np.argmax(affinity))
                    root_idx[k, a, ridx] = root
                    root_confidence[k, a] = np.maximum(root_confidence[k, a], affinity)
                    if resp_safe[k, a, ridx] and resp_low[k, a, ridx]:
                        root_low_score[k, a] = np.maximum(root_low_score[k, a], affinity)

                safe_r = valid_r[resp_safe[k, a, valid_r]]
                for root in roots:
                    members = [
                        int(r) for r in safe_r
                        if int(root_idx[k, a, r]) == int(root)
                        and float(root_confidence[k, a, root]) >= min_affinity
                    ]
                    if members:
                        best = int(members[int(np.argmin(resp_burden[k, a, members]))])
                        is_min[k, a, best] = True
                        root_min_safe_burden[k, a, root] = float(resp_burden[k, a, best])

            # Re-evaluate q_ikm from root-conditioned controls rather than treating
            # a finite global response bank as the definition of recoverability.
            # This can be applied to reused caches because natural/candidate
            # trajectories and object type are already present.  It gives every
            # conflicting root an equal search budget and eliminates response-slot
            # truncation as a source of false negatives.
            if root_search_enabled:
                rho = PriorityRelation(int(rho_arr[k, a])) if rho_arr.ndim == 2 else PriorityRelation.UNKNOWN
                object_type = _state_type(data, a)
                conflict_roots = np.where(mode_valid[k, a] & mode_conflict[k, a])[0]
                for root in conflict_roots:
                    root = int(root)
                    root_confidence[k, a, root] = 1.0
                    best_b, low_ok, _ = root_conditioned_recovery_search(
                        nat_traj[a, root], cand_traj[k], cfg,
                        object_type=object_type, beta=float(beta[a]), rho=rho,
                    )
                    root_min_safe_burden[k, a, root] = min(
                        float(root_min_safe_burden[k, a, root]), float(best_b)
                    )
                    if low_ok:
                        root_low_score[k, a, root] = 1.0

            conflict_w = nat_weight[a] * mode_valid[k, a] * mode_conflict[k, a]
            denom = float(conflict_w.sum())
            if denom > 1.0e-8:
                recovery[k, a] = float((conflict_w * root_low_score[k, a]).sum() / denom)
            transported = mode_retained[k, a].astype(np.float32)
            transported = np.where(mode_conflict[k, a], root_low_score[k, a], transported)
            transported_opr[k, a] = float(np.clip((nat_weight[a] * transported).sum(), 0.0, 1.0))

    return {
        "cowp/transport/mode_valid": mode_valid,
        "cowp/transport/mode_conflict": mode_conflict,
        "cowp/transport/mode_retained_low_safe": mode_retained,
        "cowp/transport/response_root_index": root_idx,
        "cowp/transport/response_is_min_burden": is_min,
        "cowp/transport/root_recovery_mass": recovery,
        "cowp/transport/root_low_safe_score": root_low_score,
        "cowp/transport/root_target_confidence": root_confidence,
        "cowp/transport/root_min_safe_burden": root_min_safe_burden,
        "cowp/transport/transported_opr": transported_opr,
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


def _has_transport(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as z:
            files = set(z.files)
        return all(_encode_key(k) in files or k in files for k in TRANSPORT_KEYS)
    except Exception:
        return False


def _ensure_base_link(src: Path, dst: Path, *, force: bool) -> str:
    """Expose the raw cache in the overlay directory without copying it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return "link_existing"
        except Exception:
            pass
        if force:
            dst.unlink()
        else:
            return "link_conflict"
    elif dst.exists():
        # A partially completed old materialized augmentation is valid and can be
        # mixed with overlay samples.  Keep it to make interrupted jobs resumable.
        return "materialized_existing"
    if not dst.exists() and not dst.is_symlink():
        os.symlink(str(src.resolve()), str(dst))
        return "link_created"
    return "materialized_existing"


def _process_task(task: tuple[str, str, str, bool, bool, str]) -> dict[str, object]:
    src_s, dst_s, sidecar_s, compress, force, storage_mode = task
    if _WORKER_CFG is None:
        raise RuntimeError("transport worker config was not initialized")
    cfg = _WORKER_CFG
    src, dst, sidecar = Path(src_s), Path(dst_s), Path(sidecar_s)
    try:
        if storage_mode == "overlay":
            link_status = _ensure_base_link(src, dst, force=force)
            if link_status == "link_conflict":
                raise RuntimeError(f"destination symlink points elsewhere: {dst}")
            if not force and dst.exists() and not dst.is_symlink() and _has_transport(dst):
                return {"status": "skipped_materialized", "file": src.name, "mode_pairs": 0, "conflict_count": 0, "retain_count": 0}
            if not force and sidecar.exists() and _has_transport(sidecar):
                return {"status": "skipped_sidecar", "file": src.name, "mode_pairs": 0, "conflict_count": 0, "retain_count": 0}
            data = _load_selected(src, REQUIRED_INPUT_KEYS + OPTIONAL_INPUT_KEYS)
            labels = _derive(data, cfg)
            raw = {_encode_key(k): v for k, v in labels.items()}
            _write_atomic(sidecar, raw, compress=compress)
        else:
            if not force and dst.exists() and _has_transport(dst):
                return {"status": "skipped_materialized", "file": src.name, "mode_pairs": 0, "conflict_count": 0, "retain_count": 0}
            data, raw_names = _load_all(src)
            labels = _derive(data, cfg)
            raw: dict[str, np.ndarray] = {}
            for key, value in data.items():
                raw[raw_names.get(key, _encode_key(key))] = value
            for key, value in labels.items():
                raw[_encode_key(key)] = value
            _write_atomic(dst, raw, compress=compress)

        valid = labels["cowp/transport/mode_valid"]
        conflict = labels["cowp/transport/mode_conflict"] & valid
        retained = labels["cowp/transport/mode_retained_low_safe"] & valid
        return {
            "status": "written",
            "file": src.name,
            "mode_pairs": int(valid.sum()),
            "conflict_count": int(conflict.sum()),
            "retain_count": int(retained.sum()),
        }
    except Exception as exc:
        return {"status": "error", "file": src.name, "error": repr(exc), "mode_pairs": 0, "conflict_count": 0, "retain_count": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Add explicit same-root Set-Transport labels to existing COWP tensor caches.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label_cowp_v9.yaml")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--chunksize", type=int, default=2, help="ProcessPool map chunksize; small values improve resume granularity.")
    ap.add_argument("--storage-mode", choices=["overlay", "materialized"], default="overlay", help="overlay writes tiny sidecars and symlinks raw NPZ files; materialized rewrites the full cache.")
    ap.add_argument("--sidecar-subdir", default=".transport_v9")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    cfg = load_config(args.label_config, args.data_config)
    src_dir, dst_dir = Path(args.input_dir), Path(args.output_dir)
    # Hidden NPZ files in a cache directory are metadata/artifact caches (for
    # example .cowp_sampler_weights_*.npz), not WOMD scenario samples.
    # Treating them as samples makes augmentation fail on missing COWP fields.
    paths = sorted(p for p in src_dir.glob("*.npz") if not p.name.startswith("."))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = dst_dir / args.sidecar_subdir
    if args.storage_mode == "overlay":
        sidecar_dir.mkdir(parents=True, exist_ok=True)

    def tasks() -> Iterable[tuple[str, str, str, bool, bool, str]]:
        for src in paths:
            yield (
                str(src),
                str(dst_dir / src.name),
                str(sidecar_dir / src.name),
                bool(args.compress),
                bool(args.force),
                str(args.storage_mode),
            )

    counts: dict[str, int] = {}
    mode_pairs = conflict_count = retain_count = 0
    errors: list[dict[str, object]] = []
    workers = max(1, int(args.num_workers))
    if workers == 1:
        _init_worker(cfg)
        iterator = map(_process_task, tasks())
        for result in tqdm_iter(iterator, total=len(paths), desc="transport labels"):
            status = str(result["status"])
            counts[status] = counts.get(status, 0) + 1
            mode_pairs += int(result.get("mode_pairs", 0))
            conflict_count += int(result.get("conflict_count", 0))
            retain_count += int(result.get("retain_count", 0))
            if status == "error" and len(errors) < 50:
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg,)) as ex:
            iterator = ex.map(_process_task, tasks(), chunksize=max(1, int(args.chunksize)))
            for result in tqdm_iter(iterator, total=len(paths), desc="transport labels"):
                status = str(result["status"])
                counts[status] = counts.get(status, 0) + 1
                mode_pairs += int(result.get("mode_pairs", 0))
                conflict_count += int(result.get("conflict_count", 0))
                retain_count += int(result.get("retain_count", 0))
                if status == "error" and len(errors) < 50:
                    errors.append(result)

    completed = sum(v for k, v in counts.items() if k != "error")
    summary = {
        "input_dir": str(src_dir),
        "output_dir": str(dst_dir),
        "storage_mode": str(args.storage_mode),
        "sidecar_subdir": str(args.sidecar_subdir) if args.storage_mode == "overlay" else None,
        "files_total": len(paths),
        "files_completed": completed,
        "files_written": counts.get("written", 0),
        "files_skipped_sidecar": counts.get("skipped_sidecar", 0),
        "files_skipped_materialized": counts.get("skipped_materialized", 0),
        "error_count": counts.get("error", 0),
        "mode_pairs_written": mode_pairs,
        "mode_conflict_rate_written": conflict_count / max(mode_pairs, 1),
        "mode_retain_rate_written": retain_count / max(mode_pairs, 1),
        "errors": errors,
        "complete": bool(completed == len(paths) and counts.get("error", 0) == 0),
    }
    out = Path(args.summary) if args.summary else dst_dir / "transport_augmentation_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
