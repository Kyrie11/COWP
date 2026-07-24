from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cowp.core.config import load_config
from cowp.core.constants import NaturalSource
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.utils.progress import tqdm_iter


def _indices(n: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=limit, dtype=np.int64).tolist()))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(len(a)), "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def _typed_minade(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    gt_source: torch.Tensor,
    mode_source: torch.Tensor,
    steps: int,
) -> tuple[list[float], dict[int, list[float]]]:
    h = max(1, min(int(steps), pred.shape[-2], gt.shape[-2]))
    # [B,A,M_pred,M_gt]
    pair = torch.linalg.norm(
        pred[..., :h, :2].unsqueeze(3) - gt[..., :h, :2].unsqueeze(2), dim=-1
    ).mean(dim=-1)
    psrc = mode_source.to(pair.device)[None, None, :, None]
    gsrc = gt_source[..., None, :]
    allowed = (psrc == gsrc) | (gsrc == int(NaturalSource.PAD))
    pair = torch.where(allowed, pair, torch.full_like(pair, 1.0e4))
    best = pair.min(dim=2).values
    all_vals = best[valid].detach().cpu().float().tolist()
    by_source: dict[int, list[float]] = {}
    for src in (int(NaturalSource.OBS), int(NaturalSource.NEU), int(NaturalSource.PRIO)):
        m = valid & (gt_source == src)
        by_source[src] = best[m].detach().cpu().float().tolist()
    return all_vals, by_source


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard preflight for the exact model-facing critical index, current-state anchor and typed natural basis.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--train-config", default="configs/train_cowp_v14.yaml")
    ap.add_argument("--max-scenes", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-1s-minade-m", type=float, default=3.0)
    ap.add_argument("--max-8s-minade-m", type=float, default=8.5)
    ap.add_argument("--max-first-step-cv-p90-m", type=float, default=5.0)
    ap.add_argument("--max-critical-unmapped-rate", type=float, default=0.02)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.model_config, args.label_config, args.train_config, args.data_config)
    device = torch.device(args.device)
    model = COWPModel(cfg).to(device).eval()
    if not model.natural_decoder.uses_typed_basis:
        raise ValueError(
            f"Preflight requires a decoder with a typed analytic basis, got "
            f"{model.natural_decoder.decoder_type!r}. Configure a typed/CNOB decoder."
        )

    base = TorchCOWPDataset(args.cache_dir, stage="natural")
    idxs = _indices(len(base), int(args.max_scenes))
    ds = Subset(base, idxs)
    dl = DataLoader(
        ds, batch_size=max(1, int(args.batch_size)), shuffle=False,
        num_workers=max(0, int(args.num_workers)), collate_fn=collate_torch,
        pin_memory=False,
    )

    values: dict[str, list[float]] = defaultdict(list)
    counts: Counter = Counter()
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    horizons = {sec: max(1, round(sec / dt)) for sec in (1, 3, 5, 8)}

    iterator = tqdm_iter(dl, enabled=not args.no_progress, total=len(dl), desc="model anchor preflight", unit="batch")
    with torch.no_grad():
        for batch in iterator:
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            agent_history, agent_mask = model._agent_history_from_batch(batch)
            raw_idx = batch.get("cowp/critical/input_index", batch["cowp/critical/track_index"]).long()
            crit_valid = batch["cowp/critical/valid"].bool()
            safe_idx, safe_mask = model._safe_critical_indices(raw_idx, crit_valid, agent_mask)
            anchor = model._critical_anchor7(agent_history, safe_idx)
            basis_offset = model.natural_decoder.typed_kinematic_basis(anchor, dt)
            pred = basis_offset + anchor[:, :, None, None, :]

            gt = batch["cowp/natural/traj"].float()
            gt_valid = batch["cowp/natural/valid"].bool() & safe_mask[:, :, None]
            gt_source = torch.nan_to_num(
                batch["cowp/natural/source"].float(), nan=float(NaturalSource.PAD),
                posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD),
            ).long().clamp(0, int(NaturalSource.PAD))

            counts["scenes"] += int(gt.shape[0])
            counts["critical_label_valid"] += int(crit_valid.sum())
            counts["critical_model_visible"] += int(safe_mask.sum())
            counts["critical_unmapped_or_invisible"] += int((crit_valid & ~safe_mask).sum())
            counts["natural_roots"] += int(gt_valid.sum())

            first_cv = anchor[:, :, None, 0:2] + anchor[:, :, None, 3:5] * dt
            first_err = torch.linalg.norm(gt[..., 0, :2] - first_cv, dim=-1)
            values["gt_first_step_vs_model_cv_anchor_m"].extend(first_err[gt_valid].detach().cpu().float().tolist())

            # This comparison identifies exactly the common Scenario-index versus
            # tf.Example-row failure.  Large values are expected when rows are
            # reordered; using track_index in that case is incorrect.
            track_idx = batch["cowp/critical/track_index"].long()
            n_agent = agent_history.shape[1]
            track_in_range = (track_idx >= 0) & (track_idx < n_agent) & crit_valid
            track_safe = track_idx.clamp(0, max(n_agent - 1, 0))
            track_anchor = model._critical_anchor7(agent_history, track_safe)
            anchor_delta = torch.linalg.norm(anchor[..., :2] - track_anchor[..., :2], dim=-1)
            values["input_index_vs_original_track_index_anchor_delta_m"].extend(
                anchor_delta[track_in_range & safe_mask].detach().cpu().float().tolist()
            )

            for sec, steps in horizons.items():
                all_vals, by_source = _typed_minade(
                    pred, gt, gt_valid, gt_source, model.natural_decoder.mode_source, steps
                )
                values[f"typed_basis/{sec}s"].extend(all_vals)
                for src, vals in by_source.items():
                    values[f"source_{src}/{sec}s"].extend(vals)

    distributions = {k: _stats(v) for k, v in sorted(values.items())}
    unmapped_rate = counts["critical_unmapped_or_invisible"] / max(counts["critical_label_valid"], 1)
    minade_1s = distributions.get("typed_basis/1s", {}).get("mean")
    minade_8s = distributions.get("typed_basis/8s", {}).get("mean")
    first_p90 = distributions.get("gt_first_step_vs_model_cv_anchor_m", {}).get("p90")

    checks = {
        "critical_mapping_pass": unmapped_rate <= float(args.max_critical_unmapped_rate),
        "first_step_anchor_pass": first_p90 is not None and float(first_p90) <= float(args.max_first_step_cv_p90_m),
        "typed_basis_1s_pass": minade_1s is not None and float(minade_1s) <= float(args.max_1s_minade_m),
        "typed_basis_8s_pass": minade_8s is not None and float(minade_8s) <= float(args.max_8s_minade_m),
    }
    report = {
        "pass": bool(all(checks.values())),
        "cache_dir": str(Path(args.cache_dir)),
        "sampled_scenes": len(idxs),
        "decoder_type": model.natural_decoder.decoder_type,
        "decoder_family": "typed_causal_dynamics" if model.natural_decoder.uses_dynamic_residual else "typed_residual",
        "mode_source": model.natural_decoder.mode_source.detach().cpu().tolist(),
        "counts": dict(counts),
        "rates": {"critical_unmapped_or_invisible": unmapped_rate},
        "distributions": distributions,
        "checks": checks,
        "thresholds": {
            "max_1s_minade_m": args.max_1s_minade_m,
            "max_8s_minade_m": args.max_8s_minade_m,
            "max_first_step_cv_p90_m": args.max_first_step_cv_p90_m,
            "max_critical_unmapped_rate": args.max_critical_unmapped_rate,
        },
        "interpretation": (
            "This script follows TorchCOWPDataset -> COWPModel._agent_history_from_batch -> "
            "_safe_critical_indices -> _critical_anchor7 -> NaturalDecoder.typed_kinematic_basis. "
            "A failure therefore blocks training and localizes the problem to the exact model-facing data path."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
