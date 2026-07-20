from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from cowp.core.config import load_config
from cowp.core.constants import PriorityRelation
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden


def _transport_arrays(data: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    cand = np.asarray(data["cowp/candidates/trajectory"], dtype=np.float32)
    cand_valid = np.asarray(data["cowp/candidates/valid"], dtype=bool)
    crit_valid = np.asarray(data["cowp/critical/valid"], dtype=bool)
    agent_type = np.asarray(data.get("cowp/critical/agent_type", np.ones_like(crit_valid)), dtype=np.int32)
    natural = np.asarray(data["cowp/natural/traj"], dtype=np.float32)
    natural_valid = np.asarray(data["cowp/natural/valid"], dtype=bool)
    natural_weight = np.asarray(data["cowp/natural/weight"], dtype=np.float32)
    natural_burden = np.asarray(data["cowp/natural/burden_neutral"], dtype=np.float32)
    beta = np.asarray(data["cowp/natural/beta"], dtype=np.float32)
    rho = data.get("cowp/witness/rho")
    base_priority = np.asarray(data.get("cowp/critical/base_priority", np.zeros_like(crit_valid)), dtype=np.int32)

    K, A, M = cand.shape[0], natural.shape[0], natural.shape[1]
    support = np.zeros((A, M), dtype=bool)
    conflict = np.zeros((K, A, M), dtype=bool)
    retained = np.zeros((K, A, M), dtype=bool)
    burden_under = np.full((K, A, M), 2.0, dtype=np.float32)
    min_w = float(cfg.get("ncf", {}).get("min_alt_weight", 0.03))

    for a in range(A):
        if not crit_valid[a]:
            continue
        valid_m = np.where(natural_valid[a] & (natural_weight[a] >= min_w))[0]
        support[a, valid_m] = natural_burden[a, valid_m] <= float(beta[a])
        for k in np.where(cand_valid)[0]:
            relation = int(rho[k, a]) if rho is not None else int(base_priority[a])
            try:
                priority = PriorityRelation(relation)
            except Exception:
                priority = PriorityRelation.UNKNOWN
            for m in valid_m:
                nat = natural[a, m]
                unsafe = unsafe_between(cand[k], nat, cfg, agent_type=int(agent_type[a]))
                b, _ = compute_burden(
                    nat,
                    cand[k],
                    cfg,
                    int(agent_type[a]),
                    natural_ref=nat,
                    rho=priority,
                )
                b = float(np.clip(b, 0.0, 2.0))
                burden_under[k, a, m] = b
                if support[a, m]:
                    conflict[k, a, m] = bool(unsafe.unsafe)
                    retained[k, a, m] = bool((not unsafe.unsafe) and b <= float(beta[a]))
    return {
        "cowp/transport/mode_support": support,
        "cowp/transport/mode_conflict": conflict,
        "cowp/transport/mode_retained": retained,
        "cowp/transport/mode_burden_under": burden_under,
    }


def _one(args: tuple[str, str, dict[str, Any], bool, bool]) -> dict[str, Any]:
    src_s, dst_s, cfg, compress, skip_existing = args
    src, dst = Path(src_s), Path(dst_s)
    if skip_existing and dst.is_file():
        return {"status": "existing", "file": src.name}
    try:
        with np.load(src, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        data.update(_transport_arrays(data, cfg))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            if compress:
                np.savez_compressed(f, **data)
            else:
                np.savez(f, **data)
        tmp.replace(dst)
        return {"status": "written", "file": src.name}
    except Exception as exc:
        return {"status": "error", "file": src.name, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Add per-natural-mode set-transport labels to existing COWP label NPZ files.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label_cowp_v9.yaml")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--start-method", choices=["fork", "forkserver", "spawn"], default="forkserver")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--profile-jsonl", default=None)
    args = ap.parse_args()

    cfg = load_config(args.label_config, args.data_config)
    paths = sorted(Path(args.input_dir).glob("*.npz"))
    if args.limit is not None:
        paths = paths[: max(int(args.limit), 0)]
    if not paths:
        raise FileNotFoundError(f"No NPZ labels found in {args.input_dir}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(out / p.name), cfg, bool(args.compress), bool(args.skip_existing)) for p in paths]
    counts = {"written": 0, "existing": 0, "error": 0}
    profile = Path(args.profile_jsonl) if args.profile_jsonl else None
    if profile:
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text("", encoding="utf-8")

    ctx = mp.get_context(args.start_method)
    with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers)), mp_context=ctx) as ex:
        for i, row in enumerate(ex.map(_one, tasks, chunksize=1), start=1):
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1
            if profile:
                with profile.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if status == "error":
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if i % 100 == 0 or i == len(tasks):
                print(f"transport labels {i}/{len(tasks)}: {counts}", flush=True)
    if counts.get("error", 0):
        raise SystemExit(f"Failed files: {counts['error']}; inspect {profile or 'stdout'}")


if __name__ == "__main__":
    main()
