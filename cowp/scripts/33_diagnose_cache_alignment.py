from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset


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
        if x is None:
            return None
        return np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    x, y = cur("x"), cur("y")
    if x is None or y is None:
        return None
    n = len(x)
    out = np.zeros((n, 11), dtype=np.float32)
    out[:, 0], out[:, 1] = x, y
    aliases = {
        3: ("length",), 4: ("width",), 5: ("height",),
        6: ("bbox_yaw", "heading", "yaw"),
        7: ("velocity_x", "vx"), 8: ("velocity_y", "vy"),
        10: ("valid",),
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


def _digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha1(a.view(np.uint8)).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose raw/transport COWP cache alignment and label health.")
    ap.add_argument("--raw-cache", required=True)
    ap.add_argument("--transport-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-scenes", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    raw = COWPNpzDataset(args.raw_cache)
    trans = COWPNpzDataset(args.transport_cache)
    trans_by_name = {p.name: i for i, p in enumerate(trans.paths)}
    idxs = _sample_indices(len(raw), int(args.max_scenes))

    count = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    source_counts = Counter()
    missing_transport = Counter()
    base_mismatch_examples: list[str] = []
    mapping_examples: list[str] = []
    required_transport = (
        "cowp/transport/response_root_index", "cowp/transport/mode_valid",
        "cowp/transport/mode_conflict",
    )
    base_keys = (
        "cowp/natural/traj", "cowp/natural/valid", "cowp/natural/source",
        "cowp/candidates/trajectory", "cowp/candidates/valid",
        "cowp/critical/track_id", "cowp/critical/track_index",
    )

    for ri in idxs:
        name = raw.paths[ri].name
        ti = trans_by_name.get(name)
        if ti is None:
            count["transport_file_missing"] += 1
            continue
        d0 = raw.load(ri)
        d1 = trans.load(ti)
        count["scenes"] += 1

        same = True
        for key in base_keys:
            if key not in d0 or key not in d1:
                continue
            a, b = np.asarray(d0[key]), np.asarray(d1[key])
            if a.shape != b.shape or _digest(a) != _digest(b):
                same = False
                break
        if not same:
            count["base_payload_mismatch"] += 1
            if len(base_mismatch_examples) < 10:
                base_mismatch_examples.append(name)

        for key in required_transport:
            if key not in d1:
                missing_transport[key] += 1

        crit_orig = np.asarray(d1.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
        inp = np.asarray(d1.get("cowp/critical/input_index", d1.get("cowp/critical/track_index", [])), dtype=np.int64).reshape(-1)
        mapped = np.asarray(d1.get("cowp/critical/mapped_by_id", np.ones_like(inp, dtype=bool)), dtype=bool).reshape(-1)
        visible = np.asarray(d1.get("cowp/critical/input_visible", np.ones_like(inp, dtype=bool)), dtype=bool).reshape(-1)
        count["critical_slots"] += len(inp)
        count["critical_valid"] += int(crit_orig.sum())
        count["critical_unmapped"] += int((crit_orig & ~mapped[: len(crit_orig)]).sum()) if mapped.size >= crit_orig.size else 0
        count["critical_invisible"] += int((crit_orig & ~visible[: len(crit_orig)]).sum()) if visible.size >= crit_orig.size else 0
        if crit_orig.size and ((inp < 0) & crit_orig).any() and len(mapping_examples) < 10:
            mapping_examples.append(name)

        state = _current_state(d1)
        nat = np.asarray(d1.get("cowp/natural/traj", []), dtype=np.float32)
        nat_valid = np.asarray(d1.get("cowp/natural/valid", []), dtype=bool)
        nat_source = np.asarray(d1.get("cowp/natural/source", np.zeros(nat_valid.shape, dtype=np.int64))).astype(np.int64, copy=False)
        if state is not None and nat.ndim == 4 and nat_valid.shape == nat.shape[:2]:
            for a in range(min(len(inp), nat.shape[0])):
                j = int(inp[a])
                if j < 0 or j >= len(state):
                    continue
                for m in np.flatnonzero(nat_valid[a]):
                    tr = nat[a, m]
                    if tr.shape[0] == 0 or not np.isfinite(tr[:, :2]).all():
                        count["natural_nonfinite"] += 1
                        continue
                    cur = state[j]
                    cv1 = cur[:2] + cur[7:9] * float(args.dt)
                    values["natural_first_step_jump_m"].append(float(np.linalg.norm(tr[0, :2] - cur[:2])))
                    values["natural_first_step_cv_error_m"].append(float(np.linalg.norm(tr[0, :2] - cv1)))
                    values["natural_8s_displacement_m"].append(float(np.linalg.norm(tr[-1, :2] - cur[:2])))
                    if tr.shape[-1] >= 5:
                        values["natural_max_speed_mps"].append(float(np.linalg.norm(tr[:, 3:5], axis=-1).max()))
                    source_counts[str(int(nat_source[a, m]))] += 1

        rv = d1.get("cowp/transport/response_root_index")
        if rv is not None:
            r = np.asarray(rv, dtype=np.int64)
            M = int(nat.shape[1]) if nat.ndim == 4 else 0
            response_valid = np.asarray(d1.get("cowp/response/valid", np.ones_like(r, dtype=bool)), dtype=bool)
            count["response_root_valid_slots"] += int(response_valid.sum())
            count["response_root_out_of_range"] += int((response_valid & ((r < 0) | (r >= max(M, 1)))).sum())

        selected = np.asarray(d1.get("waymax/candidate_selected_for_rollout", []), dtype=bool)
        rollout_valid = np.asarray(d1.get("waymax/candidate_rollout_valid", []), dtype=bool)
        if selected.size:
            count["waymax_selected"] += int(selected.sum())
            count["waymax_rollout_valid"] += int((selected & rollout_valid).sum()) if rollout_valid.shape == selected.shape else 0
        logdiv = np.asarray(d1.get("waymax/candidate_log_divergence", []), dtype=np.float32)
        if logdiv.size and rollout_valid.shape == logdiv.shape:
            finite = rollout_valid & np.isfinite(logdiv)
            count["waymax_logdiv_finite"] += int(finite.sum())
            values["waymax_logdiv"].extend(logdiv[finite].astype(float).tolist())

    crit_valid = max(count["critical_valid"], 1)
    root_valid = max(count["response_root_valid_slots"], 1)
    report = {
        "raw_cache": str(Path(args.raw_cache)), "transport_cache": str(Path(args.transport_cache)),
        "raw_files": len(raw), "transport_files": len(trans), "sampled_scenes": count["scenes"],
        "counts": dict(count), "source_counts": dict(source_counts),
        "missing_transport_keys": dict(missing_transport),
        "rates": {
            "base_payload_mismatch": count["base_payload_mismatch"] / max(count["scenes"], 1),
            "critical_unmapped": count["critical_unmapped"] / crit_valid,
            "critical_invisible": count["critical_invisible"] / crit_valid,
            "response_root_out_of_range": count["response_root_out_of_range"] / root_valid,
            "waymax_selected_rollout_success": count["waymax_rollout_valid"] / max(count["waymax_selected"], 1),
            "waymax_logdiv_finite": count["waymax_logdiv_finite"] / max(count["waymax_rollout_valid"], 1),
        },
        "distributions": {k: _finite_stats(v) for k, v in values.items()},
        "examples": {"base_payload_mismatch": base_mismatch_examples, "unmapped_or_negative_input_index": mapping_examples},
    }
    hard_fail = []
    if report["rates"]["base_payload_mismatch"] > 0.0:
        hard_fail.append("transport overlay changes base tensors")
    if report["rates"]["critical_unmapped"] > 0.02 or report["rates"]["critical_invisible"] > 0.05:
        hard_fail.append("critical-agent-to-WOMD row alignment is insufficient")
    if report["rates"]["response_root_out_of_range"] > 1e-4:
        hard_fail.append("response root indices are out of range")
    jump = report["distributions"].get("natural_first_step_cv_error_m", {}).get("p90")
    if jump is not None and float(jump) > 5.0:
        hard_fail.append("natural first future step is inconsistent with current state/CV anchor")
    report["pass"] = not hard_fail
    report["hard_fail_reasons"] = hard_fail
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
