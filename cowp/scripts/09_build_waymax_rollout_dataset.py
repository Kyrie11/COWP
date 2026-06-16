from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from cowp.core.config import load_config
from cowp.planning.cowp_planner import COWPPlanner
from cowp.utils.progress import tqdm_iter


def _selected_candidates(label: dict[str, np.ndarray], cfg: dict, selection: str) -> np.ndarray:
    valid = label["cowp/candidates/valid"].astype(bool)
    if selection == "all":
        return valid
    if selection == "noncoercive":
        return valid & label["cowp/candidates/noncoercive_feasible"].astype(bool)
    if selection == "false_safe":
        return valid & label["cowp/candidates/false_safe"].astype(bool)
    planner = COWPPlanner(cfg)
    dec = planner.select_from_labels(label)
    out = np.zeros_like(valid)
    if dec.candidate_index >= 0:
        out[dec.candidate_index] = True
    return out


def _copy_with_rollout_fields(path: Path, out_path: Path, cfg: dict, selection: str, status: str, background_policy: str, horizon: int) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        label = {k: data[k] for k in data.files}
    selected = _selected_candidates(label, cfg, selection)
    K = label["cowp/candidates/valid"].shape[0]
    arrays = dict(label)
    arrays.update(
        {
            "waymax/candidate_selected_for_rollout": selected.astype(bool),
            "waymax/candidate_rollout_valid": np.zeros(K, dtype=bool),
            "waymax/candidate_collision": np.zeros(K, dtype=bool),
            "waymax/candidate_offroad": np.zeros(K, dtype=bool),
            "waymax/candidate_log_divergence": np.full(K, np.nan, dtype=np.float32),
            "waymax/rollout_horizon_steps": np.asarray(int(horizon), dtype=np.int32),
            "waymax/background_policy": np.asarray(str(background_policy)),
            "waymax/rollout_status": np.asarray(str(status)),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {"scenario": path.stem, "status": status, "selected_candidates": int(selected.sum()), "output": str(out_path)}


def build_rollout_dataset(
    labels_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    *,
    tfexample_glob: str | None = None,
    candidate_selection: str = "selected",
    background_policy: str = "expert",
    horizon_steps: int | None = None,
    limit: int | None = None,
    require_waymax: bool = False,
    progress: bool = True,
    profile_jsonl: str | Path | None = None,
) -> int:
    """Create a rollout-augmented label directory.

    The script writes a dataset with explicit ``waymax/*`` fields and a manifest.
    When Waymax is installed, this is the insertion point for real candidate
    replay rollouts; when it is not installed and ``require_waymax`` is false, it
    still creates a deterministic manifest/label copy with ``rollout_status`` set
    to ``waymax_unavailable`` so downstream table code can fail gracefully instead
    of silently treating offline labels as closed-loop results.
    """
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(profile_jsonl) if profile_jsonl else output_dir / "waymax_rollout_profile.jsonl"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("", encoding="utf-8")
    horizon = int(horizon_steps if horizon_steps is not None else cfg.get("eval", {}).get("rollout_horizon_steps", cfg.get("time", {}).get("future_steps", 80)))

    status = "pending_real_waymax"
    try:
        from cowp.waymax_eval.dataloader import require_waymax

        require_waymax()
        status = "waymax_ready_candidate_replay_not_run"
    except Exception as exc:
        if require_waymax:
            raise
        status = "waymax_unavailable"
        unavailable_reason = str(exc)
    else:
        unavailable_reason = ""

    # This repository cannot execute Waymax in the packaging environment, but the
    # generated fields make the dataset contract concrete and prevent the README
    # from referencing a missing script.  Users with Waymax can plug candidate
    # replay actors here while keeping the same output schema.
    paths = sorted(labels_dir.glob("*.npz"))
    if limit is not None:
        paths = paths[: int(limit)]
    iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc="Build Waymax rollout dataset", unit="scene")
    count = 0
    for path in iterator:
        t0 = time.perf_counter()
        out_path = output_dir / path.name
        row = _copy_with_rollout_fields(path, out_path, cfg, candidate_selection, status, background_policy, horizon)
        row["seconds"] = time.perf_counter() - t0
        row["tfexample_glob"] = tfexample_glob or cfg.get("womd", {}).get("tfexample_glob", "")
        row["unavailable_reason"] = unavailable_reason
        with profile_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(status=status, written=count, selected=row["selected_candidates"], refresh=True)
    manifest = {
        "num_scenes": count,
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "candidate_selection": candidate_selection,
        "background_policy": background_policy,
        "rollout_horizon_steps": horizon,
        "status": status,
        "profile_jsonl": str(profile_path),
    }
    if unavailable_reason:
        manifest["unavailable_reason"] = unavailable_reason
    with (output_dir / "waymax_rollout_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a COWP label directory augmented with Waymax rollout fields and manifest.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/cowp/labels_waymax_rollout")
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--candidate-selection", choices=["selected", "all", "noncoercive", "false_safe"], default="selected")
    ap.add_argument("--background-policy", choices=["expert", "idm", "constant_speed", "learned_reactive"], default="expert")
    ap.add_argument("--rollout-horizon-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--require-waymax", action="store_true", help="Fail if Waymax/JAX dependencies are not importable.")
    ap.add_argument("--profile-jsonl", default=None)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    labels_dir = args.labels_dir or cfg["outputs"]["labels_dir"]
    n = build_rollout_dataset(
        labels_dir,
        args.output_dir,
        cfg,
        tfexample_glob=args.tfexample_glob,
        candidate_selection=args.candidate_selection,
        background_policy=args.background_policy,
        horizon_steps=args.rollout_horizon_steps,
        limit=args.limit,
        require_waymax=args.require_waymax,
        progress=not args.no_progress,
        profile_jsonl=args.profile_jsonl,
    )
    print(f"Wrote {n} rollout-augmented label files to {args.output_dir}")


if __name__ == "__main__":
    main()
