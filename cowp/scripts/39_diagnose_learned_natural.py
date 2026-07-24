from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cowp.core.config import load_config
from cowp.core.constants import NaturalSource
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime


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
        "count": int(a.size), "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def _typed_pairwise(pred: torch.Tensor, gt: torch.Tensor, gt_source: torch.Tensor, mode_source: torch.Tensor, steps: int) -> torch.Tensor:
    h = max(1, min(int(steps), int(pred.shape[-2]), int(gt.shape[-2])))
    pair = torch.linalg.norm(
        pred[..., :h, :2].unsqueeze(3) - gt[..., :h, :2].unsqueeze(2), dim=-1
    ).mean(dim=-1)
    allowed = mode_source.to(pair.device)[None, None, :, None] == gt_source[..., None, :]
    return torch.where(allowed, pair, torch.full_like(pair, 1.0e4))


def _load_checkpoint(model: COWPModel, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/config mismatch for learned-natural diagnosis: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return ckpt if isinstance(ckpt, dict) else {}


def _effective_modes(mass: np.ndarray) -> float:
    mass = np.asarray(mass, np.float64)
    mass = mass[np.isfinite(mass) & (mass > 0)]
    if not mass.size:
        return 0.0
    p = mass / mass.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure whether the learned natural decoder improves its analytic typed basis.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model_cowp_v16.yaml")
    ap.add_argument("--label-config", default="configs/label_cowp_v16.yaml")
    ap.add_argument("--train-config", default="configs/train_cowp_v16.yaml")
    ap.add_argument("--max-scenes", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sharing-strategy", choices=["auto", "current", "file_descriptor", "file_system"], default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    loader_runtime = configure_dataloader_runtime(args.sharing_strategy)
    print("DataLoader IPC runtime: " + json.dumps(loader_runtime, sort_keys=True), flush=True)

    cfg = load_config(args.model_config, args.label_config, args.train_config, args.data_config)
    device = torch.device(args.device)
    model = COWPModel(cfg).to(device).eval()
    if not model.natural_decoder.uses_typed_basis:
        raise ValueError(f"Learned-natural diagnosis requires a typed decoder, got {model.natural_decoder.decoder_type!r}")
    ckpt = _load_checkpoint(model, args.checkpoint, device)

    base_ds = TorchCOWPDataset(args.cache_dir, stage="natural")
    idxs = _indices(len(base_ds), int(args.max_scenes))
    dl = DataLoader(
        Subset(base_ds, idxs), batch_size=max(1, int(args.batch_size)), shuffle=False,
        num_workers=max(0, int(args.num_workers)), collate_fn=collate_torch, pin_memory=False,
    )
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    horizons = {s: max(1, round(s / dt)) for s in (1, 3, 5, 8)}
    vals: dict[str, list[float]] = defaultdict(list)
    weighted_sum: dict[str, float] = defaultdict(float)
    weighted_den: dict[str, float] = defaultdict(float)
    assignment_mass = np.zeros(model.natural_decoder.modes, dtype=np.float64)
    counts = defaultdict(int)

    iterator = tqdm_iter(dl, enabled=not args.no_progress, total=len(dl), desc="learned natural", unit="batch")
    with torch.no_grad():
        for batch in iterator:
            if not batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            full = model(batch, stage="natural")
            natural = full["natural"]
            learned = natural["traj"].float()
            analytic = natural["base_traj"].float()
            gt = batch["cowp/natural/traj"].float()
            source = torch.nan_to_num(
                batch["cowp/natural/source"].float(), nan=float(NaturalSource.PAD),
                posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD),
            ).long().clamp(0, int(NaturalSource.PAD))
            valid = batch["cowp/natural/valid"].bool() & full["critical_mask"][:, :, None]
            weight = torch.nan_to_num(batch["cowp/natural/weight"].float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            mode_source = natural["mode_source"]
            counts["scenes"] += int(learned.shape[0])
            counts["roots"] += int(valid.sum())

            for sec, steps in horizons.items():
                lp = _typed_pairwise(learned, gt, source, mode_source, steps)
                bp = _typed_pairwise(analytic, gt, source, mode_source, steps)
                lb = lp.min(dim=2).values
                bb = bp.min(dim=2).values
                for name, tensor in (("learned", lb), ("base", bb), ("gain", bb - lb)):
                    key = f"all/{sec}s/{name}"
                    vals[key].extend(tensor[valid].cpu().tolist())
                    w = weight[valid]
                    weighted_sum[key] += float((tensor[valid] * w).sum().cpu())
                    weighted_den[key] += float(w.sum().cpu())
                    for src in (int(NaturalSource.OBS), int(NaturalSource.NEU), int(NaturalSource.PRIO)):
                        m = valid & (source == src)
                        skey = f"source_{src}/{sec}s/{name}"
                        vals[skey].extend(tensor[m].cpu().tolist())
                        sw = weight[m]
                        weighted_sum[skey] += float((tensor[m] * sw).sum().cpu())
                        weighted_den[skey] += float(sw.sum().cpu())
                if sec == 8:
                    best_mode = lp.argmin(dim=2)
                    for m in range(model.natural_decoder.modes):
                        assignment_mass[m] += float((weight * valid.float() * (best_mode == m).float()).sum().cpu())

            residual = natural["residual"].float()
            endpoint = torch.linalg.norm(residual[..., -1, :2], dim=-1)
            rms = torch.sqrt(residual[..., :5].square().mean(dim=(-1, -2)))
            mode_mask = full["critical_mask"][:, :, None]
            vals["residual/endpoint_m"].extend(endpoint[mode_mask.expand_as(endpoint)].cpu().tolist())
            vals["residual/rms_state"].extend(rms[mode_mask.expand_as(rms)].cpu().tolist())
            if "controls" in natural:
                controls = natural["controls"].float()
                ctrl = torch.linalg.norm(controls, dim=-1)
                vals["controls/norm"].extend(ctrl[mode_mask[:, :, :, None].expand_as(ctrl)].cpu().tolist())

            if learned.shape[-2] >= 2:
                fd = (learned[..., 1:, 0:2] - learned[..., :-1, 0:2]) / dt
                vm = 0.5 * (learned[..., 1:, 3:5] + learned[..., :-1, 3:5])
                verr = torch.linalg.norm(fd - vm, dim=-1)
                tm = mode_mask[:, :, :, None].expand_as(verr)
                vals["kinematic/velocity_error_mps"].extend(verr[tm].cpu().tolist())
                speed = torch.linalg.norm(learned[..., 1:, 3:5], dim=-1)
                vyaw = torch.atan2(learned[..., 1:, 4], learned[..., 1:, 3])
                yerr = torch.abs(torch.atan2(torch.sin(learned[..., 1:, 2] - vyaw), torch.cos(learned[..., 1:, 2] - vyaw)))
                ym = tm & (speed > 0.5)
                vals["kinematic/yaw_error_rad"].extend(yerr[ym].cpu().tolist())

    distributions = {k: _stats(v) for k, v in sorted(vals.items())}
    for key, row in distributions.items():
        den = float(weighted_den.get(key, 0.0))
        if den > 0.0:
            row["weighted_mean"] = float(weighted_sum[key] / den)
    mode_source_np = model.natural_decoder.mode_source.detach().cpu().numpy()
    usage = {}
    for src in (int(NaturalSource.OBS), int(NaturalSource.NEU), int(NaturalSource.PRIO)):
        mask = mode_source_np == src
        usage[f"source_{src}"] = {
            "assignment_mass": assignment_mass[mask].tolist(),
            "effective_modes": _effective_modes(assignment_mass[mask]),
            "active_modes": int((assignment_mass[mask] > 0).sum()),
        }
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": ckpt.get("epoch"),
        "cache_dir": str(Path(args.cache_dir)),
        "sampled_scenes": len(idxs),
        "decoder_type": model.natural_decoder.decoder_type,
        "decoder_family": "typed_causal_dynamics" if model.natural_decoder.uses_dynamic_residual else "typed_residual",
        "counts": dict(counts),
        "distributions": distributions,
        "mode_usage": usage,
        "interpretation": (
            "Positive gain means the learned decoder improves the exact analytic basis under source-restricted matching. "
            "This diagnostic is required before attributing performance to residual capacity or the new losses."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
