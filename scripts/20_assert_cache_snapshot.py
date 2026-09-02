from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

REQUIRED = (
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/witness/exists",
    "cowp/witness/opr",
    "waymax/candidate_rollout_valid",
    "waymax/candidate_collision",
    "waymax/candidate_offroad",
)


def _scan(cache: Path, expected: int | None, sample: int, seed: int) -> dict:
    paths = sorted(cache.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files in {cache}")
    if expected is not None and len(paths) != expected:
        raise RuntimeError(
            f"Cache snapshot mismatch for {cache}: found {len(paths)} files, expected {expected}. "
            "Do not train until the analyzed cache and the training cache are the same snapshot."
        )
    rng = random.Random(seed)
    chosen = paths if sample <= 0 or sample >= len(paths) else rng.sample(paths, sample)
    missing: dict[str, int] = {}
    overlap = 0
    valid_candidates = 0
    ncf = 0
    fs = 0
    replayed = 0
    finite_logdiv = 0
    nonzero_logdiv = 0
    scenario_ids: set[str] = set()
    for path in chosen:
        with np.load(path, allow_pickle=True) as d:
            keys = set(d.files)
            for key in REQUIRED:
                if key not in keys:
                    missing[key] = missing.get(key, 0) + 1
            sid = str(d["scenario/id"].item()) if "scenario/id" in keys else path.stem
            scenario_ids.add(sid)
            valid = np.asarray(d.get("cowp/candidates/valid", []), dtype=bool)
            y_ncf = np.asarray(d.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool) & valid
            y_fs = np.asarray(d.get("cowp/candidates/false_safe", np.zeros_like(valid)), dtype=bool) & valid
            valid_candidates += int(valid.sum())
            ncf += int(y_ncf.sum())
            fs += int(y_fs.sum())
            overlap += int((y_ncf & y_fs).sum())
            rv = np.asarray(d.get("waymax/candidate_rollout_valid", np.zeros_like(valid)), dtype=bool) & valid
            replayed += int(rv.sum())
            if "waymax/candidate_log_divergence" in keys:
                ld = np.asarray(d["waymax/candidate_log_divergence"], dtype=np.float32)
                finite = rv & np.isfinite(ld)
                finite_logdiv += int(finite.sum())
                nonzero_logdiv += int((finite & (np.abs(ld) > 1e-6)).sum())
    if missing:
        raise RuntimeError(f"Required-key failures in sampled cache {cache}: {missing}")
    if overlap:
        raise RuntimeError(f"Found {overlap} NCF/false-safe overlapping candidates in sampled cache {cache}")
    return {
        "cache": str(cache),
        "files": len(paths),
        "sampled_files": len(chosen),
        "sample_unique_scenario_ids": len(scenario_ids),
        "valid_candidates": valid_candidates,
        "ncf_rate": ncf / max(valid_candidates, 1),
        "false_safe_rate": fs / max(valid_candidates, 1),
        "replay_coverage": replayed / max(valid_candidates, 1),
        "finite_logdiv_rate": finite_logdiv / max(replayed, 1),
        "nonzero_logdiv_rate": nonzero_logdiv / max(finite_logdiv, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-fast COWP cache snapshot and semantic audit.")
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--expected-train-files", type=int, default=None)
    ap.add_argument("--expected-val-files", type=int, default=None)
    ap.add_argument("--sample", type=int, default=512, help="Files sampled per split; <=0 scans all files.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    report = {
        "train": _scan(Path(args.train_cache), args.expected_train_files, args.sample, args.seed),
        "val": _scan(Path(args.val_cache), args.expected_val_files, args.sample, args.seed + 1),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
