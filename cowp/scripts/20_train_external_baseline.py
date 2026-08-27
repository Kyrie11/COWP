from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cowp.core.config import load_config
from cowp.data.dataset import collate_torch
from cowp.external_baselines.adapters import ExternalCOWPDataset, best_candidate_to_logged_ego, make_external_batch
from cowp.external_baselines.dtpp_cowp import COWPDTPP, dtpp_loss
from cowp.external_baselines.gameformer_cowp import COWPGameFormer, gameformer_loss
from cowp.external_baselines.pluto_cowp import COWPPLUTO, pluto_loss
from cowp.external_baselines.plant2_cowp import COWPPlanT2, plant2_loss
from cowp.external_baselines.reference_metadata import baseline_reference_metadata
from cowp.external_baselines.training_contract import EXTERNAL_TRAINING_CONTRACT_VERSION
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _safe_len(obj: Any) -> int | None:
    try:
        return int(len(obj))
    except Exception:
        return None


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass


def _build_model(args: argparse.Namespace):
    if args.baseline == "gameformer":
        return COWPGameFormer(
            modalities=args.gameformer_modalities,
            neighbors_to_predict=args.max_neighbors,
            future_len=args.future_len,
            encoder_layers=args.gameformer_encoder_layers,
            decoder_levels=args.gameformer_decoder_levels,
        )
    if args.baseline == "dtpp":
        return COWPDTPP(
            neighbors=args.max_neighbors,
            max_branch=args.max_candidates,
            # Official DTPP defaults to non-variable/global cost weights.
            # --dtpp-variable-cost is an explicit opt-in for the ablation.
            variable_cost=bool(getattr(args, "dtpp_variable_cost", False)),
        )
    if args.baseline == "pluto":
        return COWPPLUTO(
            future_len=args.future_len, d_model=args.vector_d_model, num_heads=args.vector_heads,
            encoder_layers=args.vector_layers, lateral_queries=args.pluto_lateral_queries,
            longitudinal_queries=args.pluto_longitudinal_queries, dropout=args.vector_dropout,
        )
    if args.baseline == "plant2":
        return COWPPlanT2(
            future_len=args.future_len, d_model=args.vector_d_model, num_heads=args.vector_heads,
            layers=args.vector_layers, dropout=args.vector_dropout,
        )
    raise ValueError(args.baseline)


def _state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _num_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable




def _build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    """Use public-source LR schedules where the reference implementation specifies one."""
    if args.baseline == "gameformer":
        # MCZhi/GameFormer open_loop_planning/train.py
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)
    if args.baseline == "dtpp":
        # MCZhi/DTPP train.py
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # PLUTO/PlanT2 are cross-domain adapters here.  Do not invent a scheduler
    # that is not unambiguously specified by the public source contract.
    return None

def _make_grad_scaler(enabled: bool):
    # GradScaler is useful for FP16.  BF16 has FP32-like exponent range and
    # does not need loss scaling; keeping a scaler there only adds another
    # state machine around numerical failures.
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(device: torch.device, enabled: bool, dtype_name: str = "auto"):
    if dtype_name == "auto":
        use_bf16 = bool(device.type == "cuda" and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    except Exception:  # pragma: no cover - older PyTorch fallback
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)



class _FatalModelStateError(RuntimeError):
    pass


def _resolved_amp_dtype(device: torch.device, enabled: bool, dtype_name: str) -> torch.dtype | None:
    if not enabled or device.type != "cuda":
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return torch.bfloat16 if use_bf16 else torch.float16


def _first_nonfinite_parameter(model: torch.nn.Module) -> str | None:
    for name, p in model.named_parameters():
        if p.numel() and not bool(torch.isfinite(p.detach()).all()):
            return name
    return None




def _first_nonfinite_gradient(model: torch.nn.Module) -> str | None:
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.numel() and not bool(torch.isfinite(p.grad.detach()).all()):
            return name
    return None


def _nonfinite_gradient_paths(model: torch.nn.Module, limit: int = 16) -> list[str]:
    """Return parameter names whose *gradient entries* contain NaN/Inf.

    This deliberately distinguishes a true non-finite gradient from the much
    more common failure in ``clip_grad_norm_`` where every fp32 gradient entry
    is finite but the fp32 sum-of-squares used for the global L2 norm overflows.
    """
    bad: list[str] = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        try:
            if not bool(torch.isfinite(grad.detach()).all().item()):
                bad.append(name)
        except Exception:
            bad.append(name)
        if len(bad) >= int(limit):
            break
    return bad


def _clip_grad_norm_stable(
    model: torch.nn.Module, max_norm: float
) -> tuple[torch.Tensor, list[str], bool]:
    """Global L2 clip with an FP64 norm-reduction overflow fallback.

    PyTorch normally reduces gradient norms in the gradient dtype.  Thus a set
    of individually finite fp32 gradients can still make the *total* L2 norm
    become Inf when squares are accumulated in fp32.  ``error_if_nonfinite``
    must stay enabled: true NaN/Inf gradient entries are fatal.  Only when every
    entry is finite do we recompute the mathematically identical global L2 norm
    in float64 and scale the original gradients once.

    Returns ``(pre_clip_norm, bad_gradient_paths, used_fp64_fallback)``.
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        ref = next(model.parameters(), None)
        dev = ref.device if ref is not None else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=dev), [], False
    max_norm = max(float(max_norm), 1.0e-6)
    try:
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
        return norm, [], False
    except TypeError:
        # Compatibility with old local PyTorch builds lacking error_if_nonfinite.
        bad = _nonfinite_gradient_paths(model)
        if bad:
            return torch.full((), float("nan"), device=params[0].grad.device), bad, False
    except RuntimeError:
        # IMPORTANT: clip_grad_norm_ raises before scaling when
        # error_if_nonfinite=True, so the gradients are still available here.
        bad = _nonfinite_gradient_paths(model)
        if bad:
            return torch.full((), float("nan"), device=params[0].grad.device), bad, False

    device = params[0].grad.device
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for param in params:
        total_sq = total_sq + param.grad.detach().to(torch.float64).square().sum()
    norm64 = total_sq.sqrt()
    if not bool(torch.isfinite(norm64).item()):
        return norm64, ["<finite entries but non-finite float64 norm>"], True
    clip_coef = min(1.0, max_norm / (float(norm64.item()) + 1.0e-6))
    if clip_coef < 1.0:
        for param in params:
            param.grad.mul_(clip_coef)
    return norm64, [], True


def _metrics_preview(metrics: dict[str, float]) -> str:
    parts = []
    for key in ("plannerADE", "score_ce", "neighbor_cmp", "ego_reg", "weight_reg", "score_abs_max", "weight_max", "valid_samples"):
        if key in metrics:
            try:
                parts.append(f"{key}={float(metrics[key]):.6g}")
            except Exception:
                pass
    return " ".join(parts)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    scaler=None,
) -> dict[str, float]:
    train = optimizer is not None
    phase = "train" if train else "val"
    model.train(train)
    sums: dict[str, float] = {}
    metric_counts: dict[str, float] = {}
    n = 0
    skipped = 0
    numerical_skipped = 0
    malformed_skipped = 0
    empty_supervision_skipped = 0
    fp64_grad_norm_fallbacks = 0
    max_preclip_grad_norm = 0.0
    consecutive_numerical = 0
    total_batches = _safe_len(loader)
    log_every = max(int(getattr(args, "log_every", 0) or 0), 0)
    amp_enabled = bool(getattr(args, "amp", False) and device.type == "cuda")
    amp_dtype = str(getattr(args, "amp_dtype", "auto"))
    resolved_amp_dtype = _resolved_amp_dtype(device, amp_enabled, amp_dtype)
    _log(
        f"{phase} {args.baseline} epoch={epoch} start batches={total_batches if total_batches is not None else 'unknown'} "
        f"batch_size={args.batch_size} workers={args.num_workers} amp={amp_enabled} "
        f"amp_dtype={str(resolved_amp_dtype).replace('torch.', '') if resolved_amp_dtype is not None else 'fp32'}"
    )
    iterator = tqdm_iter(
        loader,
        enabled=not args.no_progress,
        total=total_batches,
        desc=f"{phase} {args.baseline} e{epoch}",
        unit="batch",
    )
    t0 = time.time()
    last_log = t0
    for batch_idx, batch in enumerate(iterator, start=1):
        if train:
            optimizer.zero_grad(set_to_none=True)
        try:
            ext = make_external_batch(
                batch, cfg, device=device, max_neighbors=args.max_neighbors, max_candidates=args.max_candidates,
                horizon=args.future_len, baseline=args.baseline,
                require_candidates=(args.baseline == "dtpp"), require_future=True,
            )
            with _autocast(device, amp_enabled, amp_dtype):
                if args.baseline == "gameformer":
                    outputs = model(ext.gameformer_inputs)
                    loss, metrics = gameformer_loss(outputs, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid)
                elif args.baseline == "dtpp":
                    best_idx = best_candidate_to_logged_ego(ext.candidates, ext.candidate_valid, ext.ego_future_xy, ext.ego_future_valid)
                    loss, metrics = dtpp_loss(model, ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, best_idx, ext.ego_future_xy, ext.ego_future_valid, ext.neighbors_future_xy, ext.neighbors_future_valid, timesteps=args.future_len)
                elif args.baseline == "pluto":
                    loss, metrics = pluto_loss(
                        model, ext.planner_inputs, ext.ego_future_xy, ext.ego_future_valid,
                        contrast_weight=args.pluto_contrast_weight, aux_weight=args.pluto_aux_weight,
                    )
                elif args.baseline == "plant2":
                    loss, metrics = plant2_loss(model, ext.planner_inputs, ext.ego_future_xy, ext.ego_future_valid)
                else:
                    raise ValueError(args.baseline)
            bs = int(ext.ego_future_xy.shape[0])
            # A completely unsupervised batch should never update the optimizer.
            # In particular DTPP can return a differentiable zero when no scene
            # has a valid candidate/future pair; counting its NaN ADE used to
            # contaminate epoch metrics.
            valid_samples = float(metrics.get("valid_samples", bs))
            if valid_samples <= 0:
                skipped += 1
                empty_supervision_skipped += 1
                consecutive_numerical = 0
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss="--", samples=n, skipped=skipped, refresh=False)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue
            if not bool(torch.isfinite(loss)):
                skipped += 1
                numerical_skipped += 1
                consecutive_numerical += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                preview = _metrics_preview(metrics)
                _log(
                    f"{phase} {args.baseline} epoch={epoch} batch={batch_idx} skipped non-finite loss "
                    f"consecutive={consecutive_numerical} amp={amp_enabled} amp_dtype={amp_dtype} {preview}"
                )
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss="nonfinite", samples=n, skipped=skipped, refresh=False)
                max_num_frac = float(getattr(args, "max_numerical_skip_fraction", 0.0))
                if max_num_frac <= 0.0 or consecutive_numerical >= int(args.max_consecutive_numerical_skips):
                    bad_param = _first_nonfinite_parameter(model)
                    raise _FatalModelStateError(
                        f"{args.baseline} produced a non-finite loss at {phase} epoch={epoch} batch={batch_idx}; "
                        f"numerical_skipped={numerical_skipped}, first_nonfinite_parameter={bad_param}, "
                        f"amp={amp_enabled}, amp_dtype={amp_dtype}. Numerical failures are fatal by default "
                        "so the first causal traceback is preserved."
                    )
                continue
            if train:
                try:
                    if scaler is not None and amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if args.baseline == "dtpp" and hasattr(model, "encoder") and hasattr(model, "decoder"):
                        # Public DTPP clips encoder and decoder independently at
                        # 5.0.  Preserve that contract, but perform each global-L2
                        # reduction robustly when fp32 sum-of-squares overflows.
                        enc_norm, enc_bad, enc_fp64 = _clip_grad_norm_stable(model.encoder, args.grad_clip)
                        dec_norm, dec_bad, dec_fp64 = _clip_grad_norm_stable(model.decoder, args.grad_clip)
                        bad_paths = [f"encoder.{x}" for x in enc_bad] + [f"decoder.{x}" for x in dec_bad]
                        if bad_paths:
                            raise FloatingPointError(f"non-finite gradient entries: {bad_paths[:16]}")
                        grad_norm = torch.maximum(enc_norm.to(torch.float64), dec_norm.to(torch.float64))
                        used_fp64_fallback = bool(enc_fp64 or dec_fp64)
                    else:
                        grad_norm, bad_paths, used_fp64_fallback = _clip_grad_norm_stable(model, args.grad_clip)
                        if bad_paths:
                            raise FloatingPointError(f"non-finite gradient entries: {bad_paths[:16]}")
                    if not bool(torch.isfinite(grad_norm)):
                        raise FloatingPointError(f"non-finite gradient norm={grad_norm}")
                    grad_norm_value = float(grad_norm.detach().cpu())
                    if math.isfinite(grad_norm_value):
                        max_preclip_grad_norm = max(max_preclip_grad_norm, grad_norm_value)
                    if used_fp64_fallback:
                        fp64_grad_norm_fallbacks += 1
                        if fp64_grad_norm_fallbacks <= 5 or (log_every and fp64_grad_norm_fallbacks % max(log_every, 1) == 0):
                            _log(
                                f"{phase} {args.baseline} epoch={epoch} batch={batch_idx} recovered finite-gradient "
                                f"fp32 norm overflow with float64 L2 clipping; preclip_norm={grad_norm_value:.6g}"
                            )
                    if scaler is not None and amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                except (FloatingPointError, RuntimeError) as grad_exc:
                    msg = str(grad_exc).lower()
                    if "non-finite" not in msg and "nonfinite" not in msg and "nan" not in msg and "inf" not in msg:
                        raise
                    skipped += 1
                    numerical_skipped += 1
                    consecutive_numerical += 1
                    # Diagnose BEFORE zero_grad(); the previous order guaranteed
                    # first_nonfinite_gradient=None even when a real bad tensor existed.
                    bad_grad = _first_nonfinite_gradient(model)
                    bad_grad_paths = _nonfinite_gradient_paths(model)
                    optimizer.zero_grad(set_to_none=True)
                    _log(
                        f"{phase} {args.baseline} epoch={epoch} batch={batch_idx} skipped non-finite gradients: "
                        f"{type(grad_exc).__name__}: {grad_exc}; first_nonfinite_gradient={bad_grad}; "
                        f"nonfinite_gradient_paths={bad_grad_paths}; {_metrics_preview(metrics)}"
                    )
                    if hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(loss="bad_grad", samples=n, skipped=skipped, refresh=False)
                    max_num_frac = float(getattr(args, "max_numerical_skip_fraction", 0.0))
                    if max_num_frac <= 0.0 or consecutive_numerical >= int(args.max_consecutive_numerical_skips):
                        raise _FatalModelStateError(
                            f"{args.baseline} hit a numerical-gradient failure at {phase} epoch={epoch} batch={batch_idx}; "
                            f"numerical_skipped={numerical_skipped}, first_nonfinite_gradient={bad_grad}, "
                            f"nonfinite_gradient_paths={bad_grad_paths}."
                        ) from grad_exc
                    continue
                bad_param = _first_nonfinite_parameter(model)
                if bad_param is not None:
                    raise _FatalModelStateError(
                        f"Optimizer step created a non-finite parameter {bad_param} at {phase} epoch={epoch} batch={batch_idx}."
                    )
                consecutive_numerical = 0
            else:
                consecutive_numerical = 0
            n += bs
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * bs
            metric_counts["loss"] = metric_counts.get("loss", 0.0) + bs
            metric_weight = valid_samples
            for k, v in metrics.items():
                try:
                    vf = float(v)
                except Exception:
                    continue
                if math.isfinite(vf):
                    sums[k] = sums.get(k, 0.0) + vf * metric_weight
                    metric_counts[k] = metric_counts.get(k, 0.0) + metric_weight
            mean_loss = sums["loss"] / max(n, 1)
            if hasattr(iterator, "set_postfix"):
                postfix = {"loss": f"{mean_loss:.4f}", "samples": n, "skipped": skipped}
                for mk in ("plannerADE", "plannerFDE", "score_ce", "neighbor_cmp", "ego_reg"):
                    if mk in metrics:
                        try:
                            postfix[mk] = f"{float(metrics[mk]):.4f}"
                        except Exception:
                            pass
                iterator.set_postfix(postfix, refresh=False)
            now = time.time()
            should_log = bool(log_every and (batch_idx == 1 or batch_idx % log_every == 0 or (total_batches is not None and batch_idx == total_batches)))
            if should_log:
                elapsed = max(now - t0, 1e-6)
                batch_rate = batch_idx / elapsed
                sample_rate = n / elapsed
                total_txt = str(total_batches) if total_batches is not None else "?"
                _log(
                    f"{phase} {args.baseline} epoch={epoch} batch={batch_idx}/{total_txt} "
                    f"samples={n} skipped={skipped} loss={mean_loss:.6f} "
                    f"batch_rate={batch_rate:.3f}/s sample_rate={sample_rate:.1f}/s"
                )
                last_log = now
            elif log_every == 0 and batch_idx == 1 and now - last_log > 30:
                _log(f"{phase} {args.baseline} epoch={epoch} first batch finished samples={n} loss={mean_loss:.6f}")
                last_log = now
        except KeyboardInterrupt:
            raise
        except _FatalModelStateError:
            if train:
                optimizer.zero_grad(set_to_none=True)
            raise
        except Exception as exc:
            if train:
                optimizer.zero_grad(set_to_none=True)
            # Do not turn arbitrary model/CUDA/programming failures into a
            # "malformed batch".  The V2 loop caught every Exception, which can
            # hide shape bugs or device errors until max_skip_fraction fires much
            # later and obscures the first real traceback.  Only adapter-level
            # data-contract exceptions are eligible for bounded skipping.
            skippable_data_error = isinstance(exc, (KeyError, ValueError, IndexError))
            if args.strict or not skippable_data_error:
                raise
            skipped += 1
            malformed_skipped += 1
            consecutive_numerical = 0
            if skipped <= 5 or (log_every and skipped % max(log_every, 1) == 0):
                _log(f"Warning: skipped malformed batch phase={phase} baseline={args.baseline} epoch={epoch} batch={batch_idx}: {type(exc).__name__}: {exc}")
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(samples=n, skipped=skipped, refresh=False)
            continue
    elapsed = max(time.time() - t0, 1e-6)
    if n == 0:
        raise RuntimeError(f"No usable samples in {phase} epoch {epoch}. skipped_batches={skipped}. Set --strict to expose the first malformed batch.")
    skip_fraction = float(skipped / max(int(total_batches or (skipped + 1)), 1))
    max_skip_fraction = float(getattr(args, "max_skip_fraction", 0.02))
    numerical_skip_fraction = float(numerical_skipped / max(int(total_batches or (skipped + 1)), 1))
    max_numerical_skip_fraction = float(getattr(args, "max_numerical_skip_fraction", 0.0))
    if numerical_skip_fraction > max_numerical_skip_fraction:
        raise RuntimeError(
            f"External baseline {args.baseline} {phase} epoch {epoch} numerical skips "
            f"{numerical_skipped}/{total_batches} ({numerical_skip_fraction:.3%}) exceed "
            f"max_numerical_skip_fraction={max_numerical_skip_fraction:.3%}."
        )
    if skip_fraction > max_skip_fraction:
        raise RuntimeError(
            f"External baseline {args.baseline} {phase} epoch {epoch} skipped {skipped}/{total_batches} batches "
            f"({skip_fraction:.3%}) > max_skip_fraction={max_skip_fraction:.3%}; "
            f"numerical={numerical_skipped}, malformed={malformed_skipped}, empty_supervision={empty_supervision_skipped}. "
            "Do not publish a checkpoint learned from a materially reduced subset. Use --strict to expose the first "
            "adapter/data-contract failure; numerical failures are fatal on their first occurrence by default."
        )
    out = {k: v / max(metric_counts.get(k, float(n)), 1.0) for k, v in sums.items()} | {
        "num_samples": float(n), "num_batches": float(total_batches or 0), "skipped_batches": float(skipped),
        "numerical_skipped_batches": float(numerical_skipped), "malformed_skipped_batches": float(malformed_skipped),
        "empty_supervision_skipped_batches": float(empty_supervision_skipped),
        "fp64_grad_norm_fallbacks": float(fp64_grad_norm_fallbacks),
        "max_preclip_grad_norm": float(max_preclip_grad_norm),
        "skip_fraction": skip_fraction, "seconds": float(elapsed),
    }
    _log(f"{phase} {args.baseline} epoch={epoch} done samples={n} skipped={skipped} seconds={elapsed:.1f} loss={out.get('loss', float('nan')):.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train source-faithful external planning adaptations on COWP/WOMD tensor-cache data.")
    ap.add_argument("--baseline", choices=["gameformer", "dtpp", "pluto", "plant2"], required=True)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--val-cache-dir", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-neighbors", type=int, default=10)
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--future-len", type=int, default=80)
    ap.add_argument("--gameformer-modalities", type=int, default=6)
    ap.add_argument("--gameformer-encoder-layers", type=int, default=6)
    ap.add_argument("--gameformer-decoder-levels", type=int, default=4)
    cost_group = ap.add_mutually_exclusive_group()
    cost_group.add_argument(
        "--dtpp-variable-cost", action="store_true",
        help="Opt in to DTPP scene-conditioned cost weights. Official/public DTPP training defaults to global/fixed cost weights.",
    )
    cost_group.add_argument(
        "--dtpp-fixed-cost", action="store_true",
        help="Deprecated compatibility flag; fixed/global cost is now the default.",
    )
    ap.add_argument("--vector-d-model", type=int, default=128)
    ap.add_argument("--vector-heads", type=int, default=8)
    ap.add_argument("--vector-layers", type=int, default=4)
    ap.add_argument("--vector-dropout", type=float, default=0.1)
    ap.add_argument("--pluto-lateral-queries", type=int, default=4)
    ap.add_argument("--pluto-longitudinal-queries", type=int, default=6)
    ap.add_argument("--pluto-contrast-weight", type=float, default=0.05)
    ap.add_argument("--pluto-aux-weight", type=float, default=0.20)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training/validation.")
    ap.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16"], default="auto", help="AMP dtype; auto prefers BF16 to avoid FP16 overflow in trajectory/GMM losses.")
    ap.add_argument("--max-skip-fraction", type=float, default=0.02, help="Fail an epoch when all excluded batches exceed this fraction.")
    ap.add_argument("--max-numerical-skip-fraction", type=float, default=0.0, help="Allowed fraction of non-finite loss/gradient batches. Default 0 makes the first numerical failure fatal and preserves its traceback.")
    ap.add_argument("--max-consecutive-numerical-skips", type=int, default=1, help="Compatibility guard when a nonzero numerical skip fraction is explicitly allowed.")
    ap.add_argument("--prefetch-factor", type=int, default=int(os.environ.get("PREFETCH_FACTOR", "2")))
    ap.add_argument("--val-prefetch-factor", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Save numbered epoch checkpoints every N epochs (best/final are always saved). Set >1 to reduce filesystem I/O without changing optimization.")
    ap.add_argument("--sharing-strategy", choices=["auto", "current", "file_descriptor", "file_system"], default=None)
    ap.add_argument("--no-persistent-workers", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "25")), help="Emit one line per N batches even when tqdm is disabled. Use 0 to disable heartbeat lines.")
    args = ap.parse_args()
    loader_runtime = configure_dataloader_runtime(args.sharing_strategy)

    _log(f"external training entry baseline={args.baseline} pid={os.getpid()} python={sys.executable} torch={torch.__version__}")
    _log(f"DataLoader IPC runtime={json.dumps(loader_runtime, sort_keys=True)}")
    _log(f"args={json.dumps(vars(args), sort_keys=True)}")
    cfg = load_config(args.label_config, args.data_config, args.train_config)
    device = _device(args.device)
    _seed(args.seed, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # This trainer has no resume semantics: every invocation starts from random
    # initialization.  Remove stale success markers/checkpoints first so an
    # interrupted V5 retrain cannot be mistaken for a valid older experiment.
    stale = [
        out_dir / f"external_{args.baseline}_training_complete.json",
        out_dir / f"external_{args.baseline}_history.json",
        out_dir / f"external_{args.baseline}_best.pt",
    ] + list(out_dir.glob(f"external_{args.baseline}_epoch*.pt"))
    for path in stale:
        try:
            path.unlink(missing_ok=True)
        except TypeError:  # pragma: no cover - Python <3.8 compatibility
            if path.exists():
                path.unlink()
    _log(f"device={device} cuda_available={torch.cuda.is_available()} output_dir={out_dir} contract={EXTERNAL_TRAINING_CONTRACT_VERSION}")

    _log(f"loading train dataset from {args.cache_dir}")
    train_ds = ExternalCOWPDataset(args.cache_dir, include_waymax_outcomes=False, baseline=args.baseline, purpose="train")
    _log(f"train dataset ready scenes={len(train_ds)}")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_torch,
        "pin_memory": (device.type == "cuda" and args.num_workers > 0),
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers
        loader_kwargs["prefetch_factor"] = max(int(args.prefetch_factor), 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    _log(f"train loader ready batches={len(train_loader)}")
    val_loader = None
    if args.val_cache_dir:
        _log(f"loading val dataset from {args.val_cache_dir}")
        val_ds = ExternalCOWPDataset(args.val_cache_dir, include_waymax_outcomes=False, baseline=args.baseline, purpose="train")
        _log(f"val dataset ready scenes={len(val_ds)}")
        val_loader_kwargs = dict(loader_kwargs)
        val_workers = max(int(args.val_num_workers), 0)
        val_loader_kwargs["num_workers"] = val_workers
        val_loader_kwargs["pin_memory"] = (device.type == "cuda" and val_workers > 0)
        if val_workers > 0:
            # Validation is repeated every epoch; keeping workers alive avoids
            # process/spawn + NPZ-reader warmup while preserving exact samples.
            val_loader_kwargs["persistent_workers"] = not args.no_persistent_workers
            val_loader_kwargs["prefetch_factor"] = max(int(args.val_prefetch_factor), 1)
        else:
            val_loader_kwargs.pop("persistent_workers", None)
            val_loader_kwargs.pop("prefetch_factor", None)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **val_loader_kwargs)
        _log(f"val loader ready batches={len(val_loader)}")

    model = _build_model(args).to(device)
    total_params, trainable_params = _num_parameters(model)
    _log(f"model built baseline={args.baseline} total_params={total_params} trainable_params={trainable_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args)
    resolved_amp_dtype = _resolved_amp_dtype(device, bool(args.amp), str(args.amp_dtype))
    scaler = _make_grad_scaler(bool(args.amp and device.type == "cuda" and resolved_amp_dtype == torch.float16))
    if scaler is not None:
        _log("AMP GradScaler enabled")
    best_metric = math.inf
    history = []
    best_path = out_dir / f"external_{args.baseline}_best.pt"

    for epoch in range(1, args.epochs + 1):
        _log(f"epoch {epoch}/{args.epochs} begin baseline={args.baseline}")
        train_metrics = _run_epoch(model, train_loader, cfg, args, device, optimizer, epoch, scaler=scaler)
        val_metrics = _run_epoch(model, val_loader, cfg, args, device, None, epoch, scaler=None) if val_loader is not None else {}
        # Select a finite validation metric.  A sparse/empty metric must never
        # make best-checkpoint selection silently stop for the rest of training.
        metric_candidates = [
            val_metrics.get("plannerADE"), val_metrics.get("loss"),
            train_metrics.get("plannerADE"), train_metrics.get("loss"),
        ]
        metric = math.inf
        for candidate_metric in metric_candidates:
            if candidate_metric is None:
                continue
            try:
                candidate_metric = float(candidate_metric)
            except Exception:
                continue
            if math.isfinite(candidate_metric):
                metric = candidate_metric
                break
        if not math.isfinite(metric):
            raise RuntimeError(f"No finite checkpoint-selection metric at epoch={epoch}: train={train_metrics} val={val_metrics}")
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        _log("epoch summary " + json.dumps(record, sort_keys=True))
        bad_param = _first_nonfinite_parameter(model)
        if bad_param is not None:
            raise _FatalModelStateError(f"Refusing to save checkpoint with non-finite parameter: {bad_param}")
        ckpt = {
            "baseline": args.baseline,
            "training_contract_version": EXTERNAL_TRAINING_CONTRACT_VERSION,
            "model": _state_dict(model),
            "cfg": cfg,
            "args": vars(args),
            "epoch": epoch,
            "metrics": record,
            "reference_metadata": baseline_reference_metadata(args.baseline),
        }
        save_numbered = bool(epoch == args.epochs or args.checkpoint_every <= 1 or epoch % max(int(args.checkpoint_every), 1) == 0)
        if save_numbered:
            epoch_path = out_dir / f"external_{args.baseline}_epoch{epoch}.pt"
            torch.save(ckpt, epoch_path)
            _log(f"saved checkpoint {epoch_path}")
        if metric < best_metric:
            best_metric = metric
            torch.save(ckpt, best_path)
            _log(f"updated best checkpoint {best_path} best_metric={best_metric:.6f}")
        with (out_dir / f"external_{args.baseline}_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if scheduler is not None:
            scheduler.step()
            _log(f"scheduler step baseline={args.baseline} lr={optimizer.param_groups[0]['lr']:.8g}")
    completion = {
        "baseline": args.baseline, "epochs": int(args.epochs),
        "best_checkpoint": str(best_path), "best_metric": float(best_metric),
        "completed": True,
        "training_signature": {
            "contract_version": EXTERNAL_TRAINING_CONTRACT_VERSION,
            "explicit_validity_masks": True,
            "lr": float(args.lr), "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size), "seed": int(args.seed),
            "future_len": int(args.future_len), "max_neighbors": int(args.max_neighbors),
            "max_candidates": int(args.max_candidates),
            "amp": bool(args.amp), "amp_dtype": str(args.amp_dtype),
            "grad_clip": float(args.grad_clip),
            "dtpp_variable_cost": bool(getattr(args, "dtpp_variable_cost", False)),
        },
    }
    with (out_dir / f"external_{args.baseline}_training_complete.json").open("w", encoding="utf-8") as f:
        json.dump(completion, f, indent=2)
    _log(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
