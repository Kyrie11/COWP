from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _sanitize_cuda_alloc_conf_before_torch_import() -> None:
    """Avoid known PyTorch expandable-segment allocator crashes on some CUDA stacks.

    This must run before the first CUDA allocation.  ``expandable_segments`` is
    optional and experimental; if a launcher exports it, keep the stable knobs
    such as ``max_split_size_mb`` but remove expandable-segment mode.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if not conf or "expandable_segments" not in conf:
        return
    kept = []
    for item in conf.split(","):
        item = item.strip()
        if not item:
            continue
        key = item.split(":", 1)[0].strip().lower()
        if key == "expandable_segments":
            continue
        kept.append(item)
    if kept:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(kept)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


_sanitize_cuda_alloc_conf_before_torch_import()

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, Sampler

from cowp.core.config import load_config
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime
from cowp.models.losses import candidate_certificate_loss, candidate_classification_loss, natural_loss, paper_aligned_supervision_batch, planner_imitation_loss, planner_outcome_loss, planner_outcome_supervision, planner_ranking_loss, priority_claim_loss, response_loss, set_transport_loss, witness_loss


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _setup_distributed(device_arg: str) -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = _distributed_env()
    distributed = world_size > 1
    if not distributed:
        return False, 0, 1, 0
    use_cuda = str(device_arg).lower() != "cpu" and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _rank0_print(*args, **kwargs) -> None:
    if _is_main_process():
        print(*args, **kwargs)


def _distributed_mean_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return metrics
    if not metrics:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {k: float(v) for k, v in zip(keys, values.detach().cpu().tolist())}


def _set_sampler_epoch(dl: DataLoader | None, epoch: int) -> None:
    if dl is None:
        return
    sampler = getattr(dl, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)


def _set_runtime_defaults(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass




def _cuda_pin_memory_works(device: torch.device) -> bool:
    """Best-effort guard for explicit DataLoader pin_memory requests.

    Pinning is deliberately *not* enabled by default in this project.  Stage-B
    response training can load hundreds of MB per batch because
    ``cowp/response/traj`` is a dense [K,A,R,T,7] target.  With multi-worker
    prefetching, the pin-memory thread may try to page-lock several GB of CPU
    tensors and fail with ``CUDA error: invalid argument`` before the next batch
    reaches the model.  Pinning changes transfer mechanics only; it does not
    change the model, labels, loss, or predictions.  Therefore the stable default
    is no pinning, while ``--pin-memory`` remains available after this sanity
    check.
    """
    if device.type != "cuda":
        return False
    try:
        x = torch.empty(1)
        try:
            _ = x.pin_memory(device)
        except TypeError:
            _ = x.pin_memory()
        return True
    except Exception as exc:
        print(f"Warning: requested DataLoader pin_memory is unavailable; disabling it: {exc}")
        return False


def _heavy_label_stage(stage: str) -> bool:
    """Stages whose batches may contain very large dense supervision tensors."""
    return stage in {"response", "planner", "all"}

def _make_grad_scaler(enabled: bool):
    """Return a CUDA GradScaler that works across old and new PyTorch APIs.

    Some PyTorch versions expose AMP as ``torch.cuda.amp`` only, while newer
    versions prefer ``torch.amp``.  Importantly, when AMP is disabled we return
    ``None`` instead of constructing a scaler; this avoids crashes like
    ``AttributeError: module 'torch.amp' has no attribute 'GradScaler'`` even for
    non-AMP runs.
    """
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:  # older signature: GradScaler(enabled=True)
            return torch.amp.GradScaler(enabled=True)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


def _use_grad_scaler(stage_amp: bool, is_train: bool, amp_dtype: torch.dtype) -> bool:
    """Use dynamic loss scaling only for true fp16 training.

    bfloat16 already has fp32-like exponent range.  Scaling a bf16 forward is
    unnecessary and, for cuDNN recurrent kernels, can amplify first-step
    input-weight gradients without providing fp16 underflow protection.
    """
    return bool(stage_amp and is_train and amp_dtype == torch.float16)


def _update_grad_scaler_scale(scaler, new_scale: float) -> None:
    """Synchronously set a lower scale after a globally detected DDP overflow."""
    value = float(max(new_scale, 1.0))
    try:
        scaler.update(new_scale=value)
    except TypeError:
        scaler.update(value)


def _resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype:
    """Resolve a numerically safe CUDA autocast dtype.

    v16.2 used PyTorch's implicit CUDA AMP default (fp16).  On A30-class GPUs
    the graph encoder can overflow before the zero-initialized natural head.  A
    zero linear applied to an infinite activation produces NaN (``0 * Inf``),
    which was later hidden by loss-side ``nan_to_num``.  Prefer bfloat16 because
    it keeps fp32's exponent range; retain fp16 only as an explicit/legacy
    fallback.
    """
    name = str(requested).strip().lower()
    if name not in {"auto", "bfloat16", "float16"}:
        raise ValueError(f"Unknown AMP dtype {requested!r}; choose auto, bfloat16, or float16")
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype = torch.bfloat16):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast("cuda", enabled=True, dtype=dtype)
            except TypeError:
                return torch.amp.autocast(device_type="cuda", enabled=True, dtype=dtype)
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
    return nullcontext()


def _dataset_signature(paths: list[Path]) -> dict[str, object]:
    if not paths:
        return {"num_files": 0, "mtime_ns_sum": 0, "size_sum": 0}
    mtime_ns_sum = 0
    size_sum = 0
    for p in paths:
        try:
            st = p.stat()
            mtime_ns_sum += int(st.st_mtime_ns)
            size_sum += int(st.st_size)
        except OSError:
            pass
    return {"num_files": len(paths), "mtime_ns_sum": int(mtime_ns_sum), "size_sum": int(size_sum)}


def _read_sample_weight_from_npz(path: Path, *, positive_weight: float, stress_weight: float) -> float:
    with np.load(path, allow_pickle=True) as data:
        keys = {k.replace("__", "/"): k for k in data.files}
        w = 1.0
        if "cowp/witness/exists" in keys and np.asarray(data[keys["cowp/witness/exists"]]).any():
            w *= positive_weight
        if "cowp/candidates/false_safe" in keys and np.asarray(data[keys["cowp/candidates/false_safe"]]).any():
            w *= stress_weight
        if "cowp/candidates/noncoercive_feasible" in keys and np.asarray(data[keys["cowp/candidates/noncoercive_feasible"]]).any():
            w *= 1.25
        return float(w)


def _sample_weights(
    ds: TorchCOWPDataset,
    *,
    positive_weight: float = 3.0,
    stress_weight: float = 2.0,
    progress: bool = True,
    cache: bool = True,
) -> torch.DoubleTensor:
    paths = [Path(p) for p in ds.base.paths]
    sig = _dataset_signature(paths)
    meta = {**sig, "positive_weight": float(positive_weight), "stress_weight": float(stress_weight), "version": 2}
    cache_path = Path(ds.base.cache_dir) / f".cowp_sampler_weights_pw{positive_weight:g}_sw{stress_weight:g}_v2.npz"
    if cache and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as cached:
                cached_meta = json.loads(str(cached["metadata"].item()))
                weights = cached["weights"].astype(np.float64)
            if cached_meta == meta and len(weights) == len(paths):
                print(f"Loaded sampler weights from {cache_path}")
                return torch.as_tensor(weights, dtype=torch.double)
        except Exception:
            pass

    print("Building sampler weights once. This scans cache metadata; pass --no-positive-oversampling to skip it.")
    weights: list[float] = []
    iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc="Build sampler weights", unit="file")
    for p in iterator:
        weights.append(_read_sample_weight_from_npz(p, positive_weight=positive_weight, stress_weight=stress_weight))
    arr = np.asarray(weights, dtype=np.float64)
    if cache:
        try:
            np.savez(cache_path, weights=arr, metadata=json.dumps(meta, ensure_ascii=False))
            print(f"Saved sampler weights to {cache_path}")
        except Exception as exc:
            print(f"Warning: failed to save sampler weights cache {cache_path}: {exc}")
    return torch.as_tensor(arr, dtype=torch.double)


class DistributedWeightedRandomSampler(Sampler[int]):
    """Weighted replacement sampler split deterministically across DDP ranks."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 2026,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / max(self.num_replicas, 1)))
        self.total_size = int(self.num_samples * max(self.num_replicas, 1))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _make_loader(
    ds: TorchCOWPDataset,
    cfg: dict,
    batch_size: int,
    *,
    shuffle: bool,
    oversample: bool,
    use_cuda: bool,
    progress: bool,
    num_workers: int | None = None,
    sampler_cache: bool = True,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 2026,
) -> DataLoader:
    sampler = None
    if distributed:
        if shuffle and oversample:
            sampler = DistributedWeightedRandomSampler(
                _sample_weights(ds, progress=progress, cache=sampler_cache),
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=seed,
            )
        else:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)
        shuffle = False
    elif shuffle and oversample:
        sampler = WeightedRandomSampler(_sample_weights(ds, progress=progress, cache=sampler_cache), num_samples=len(ds), replacement=True)
        shuffle = False
    nw = int(cfg["train"].get("num_workers", 0) if num_workers is None else num_workers)
    kwargs: dict[str, Any] = {}
    if nw > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        pf = int(cfg["train"].get("prefetch_factor", 2) if prefetch_factor is None else prefetch_factor)
        kwargs["prefetch_factor"] = max(1, pf)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate_torch,
        pin_memory=bool(use_cuda and pin_memory),
        **kwargs,
    )


def _to_device(batch: dict[str, Any], device: torch.device, *, non_blocking: bool = False) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=bool(non_blocking)) for k, v in batch.items() if torch.is_tensor(v)}


def _validate_stage_supervision(dataset, stage: str, *, samples: int = 32) -> dict[str, float]:
    """Fail fast when a staged objective has no labels.

    v11 trained ``stage=witness`` without loading candidate-level NCF and
    false-safe labels.  The candidate-budget objective then returned zero for
    every batch while checkpoint selection treated that zero as an excellent
    loss.  Inspect a small deterministic prefix before allocating GPUs so this
    failure mode cannot recur silently.
    """
    required: dict[str, tuple[str, ...]] = {
        "witness": (
            "cowp/transport/mode_valid",
            "cowp/transport/mode_conflict",
            "cowp/transport/mode_retained_low_safe",
            "cowp/candidates/valid",
            "cowp/candidates/noncoercive_feasible",
            "cowp/candidates/false_safe",
        ),
        "planner": (
            "cowp/candidates/valid",
            "cowp/candidates/noncoercive_feasible",
            "cowp/candidates/false_safe",
        ),
    }
    keys = required.get(str(stage), ())
    if not keys or len(dataset) == 0:
        return {}
    n = min(max(int(samples), 1), len(dataset))
    missing: set[str] = set()
    valid_count = disc_count = ncf_count = fs_count = 0.0
    # Probe the full cache deterministically instead of only its prefix.  WOMD
    # caches are commonly written in shard/scenario order, so the first few files
    # may not represent rare false-safe/NCF supervision.
    if n == 1:
        probe_indices = [0]
    else:
        probe_indices = sorted({round(i * (len(dataset) - 1) / (n - 1)) for i in range(n)})
    for i in probe_indices:
        item = dataset[i]
        missing.update(k for k in keys if k not in item)
        if all(k in item for k in (
            "cowp/candidates/valid",
            "cowp/candidates/noncoercive_feasible",
            "cowp/candidates/false_safe",
        )):
            valid = item["cowp/candidates/valid"].bool()
            ncf = item["cowp/candidates/noncoercive_feasible"].bool() & valid
            fs = item["cowp/candidates/false_safe"].bool() & valid
            disc = (ncf & ~fs) | fs
            valid_count += float(valid.float().sum().item())
            disc_count += float(disc.float().sum().item())
            ncf_count += float((ncf & ~fs).float().sum().item())
            fs_count += float(fs.float().sum().item())
    if missing:
        raise RuntimeError(
            f"stage={stage} is missing required supervision keys: {sorted(missing)}. "
            "Rebuild/augment the cache or use the corrected stage key loader before training."
        )
    if stage in {"witness", "planner"} and disc_count <= 0:
        raise RuntimeError(
            f"stage={stage} found no discriminative NCF/false-safe candidates in {len(probe_indices)} evenly spaced samples. "
            "Candidate-budget training would be inactive; run label/cache diagnostics before training."
        )
    return {
        "candidate_valid": valid_count,
        "candidate_discriminative": disc_count,
        "candidate_ncf": ncf_count,
        "candidate_false_safe": fs_count,
        "candidate_budget_coverage": disc_count / max(valid_count, 1.0),
    }


def _compute_losses(pred: dict[str, Any], batch: dict[str, torch.Tensor], stage: str, loss_weights: dict[str, float]) -> dict[str, torch.Tensor]:
    if stage in ("witness", "planner", "all"):
        batch = paper_aligned_supervision_batch(batch, loss_weights)
    out: dict[str, torch.Tensor] = {}
    losses: list[torch.Tensor] = []
    if stage in ("natural", "representation", "all"):
        nl = natural_loss(pred["natural"], batch, loss_weights)
        out.update({f"natural/{k}": v for k, v in nl.items() if k != "loss"})
        losses.append(nl["loss"])
    if stage in ("response", "all"):
        rl = response_loss(pred["response"], batch, loss_weights)
        out.update({f"response/{k}": v for k, v in rl.items() if k != "loss"})
        losses.append(rl["loss"])
    if stage in ("witness", "planner", "all"):
        # Keep natural primitive identity trainable during transport learning.
        # v9 froze randomly initialized/new mode-identity layers for all 10 epochs,
        # so direct transport labels could not repair the primitive basis.
        natural_scale_key = "planner_natural_scale" if stage == "planner" else "witness_natural_scale"
        natural_scale = float(loss_weights.get(natural_scale_key, 0.0))
        if natural_scale > 0.0 and isinstance(pred.get("natural"), dict) and "traj" in pred["natural"]:
            nl_aux = natural_loss(pred["natural"], batch, loss_weights)
            out.update({f"natural_aux/{k}": v for k, v in nl_aux.items() if k != "loss"})
            losses.append(natural_scale * nl_aux["loss"])
        # Keep the proxy witness for token/interval explanation, while training the
        # mechanistic Set-Transport certificate as the decision certificate.
        witness_pred = pred.get("witness_proxy", pred["witness"])
        wl = witness_loss(witness_pred, batch, loss_weights)
        out.update({f"witness/{k}": v for k, v in wl.items() if k != "loss"})
        # Planner fine-tuning should not erase witness calibration learned in the
        # dedicated stage.  Keep a small consistency gradient instead of adding
        # the full witness objective a second time.
        witness_scale = 1.0 if stage in {"witness", "all"} else float(loss_weights.get("planner_witness_scale", 0.20))
        losses.append(witness_scale * wl["loss"])
        st = pred.get("set_certificate")
        natural_pred = pred.get("natural")
        natural_pred_traj = natural_pred.get("traj") if isinstance(natural_pred, dict) else None
        natural_pred_source_logits = natural_pred.get("source_logits") if isinstance(natural_pred, dict) else None
        if isinstance(st, dict):
            st_for_loss = dict(st)
            if torch.is_tensor(natural_pred_traj):
                st_for_loss["_natural_pred_traj"] = natural_pred_traj
            if torch.is_tensor(natural_pred_source_logits):
                st_for_loss["_natural_pred_source_logits"] = natural_pred_source_logits
            stl = set_transport_loss(st_for_loss, batch, loss_weights)
            out.update({f"set_transport/{k}": v for k, v in stl.items() if k != "loss"})
            losses.append(float(loss_weights.get("set_transport", 1.0)) * stl["loss"])
        response_targets_available = all(
            key in batch for key in (
                "cowp/response/valid", "cowp/response/is_safe",
                "cowp/response/is_low_burden", "cowp/response/burden_total",
            )
        )
        if (
            stage in {"witness", "planner", "all"}
            and isinstance(pred.get("response"), dict)
            and response_targets_available
        ):
            # Keep each response branch physically and semantically identifiable
            # while the aggregate Set-Transport objective is optimized.  Without
            # this auxiliary supervision the response bank can satisfy only the
            # aggregate existence/burden targets and drift into arbitrary slots.
            scale_key = "planner_response_scale" if stage == "planner" else "witness_response_scale"
            response_scale = float(loss_weights.get(scale_key, 0.25))
            if response_scale > 0.0:
                response_for_loss = dict(pred["response"])
                if torch.is_tensor(natural_pred_traj):
                    response_for_loss["_natural_pred_traj"] = natural_pred_traj
                if torch.is_tensor(natural_pred_source_logits):
                    response_for_loss["_natural_pred_source_logits"] = natural_pred_source_logits
                rl = response_loss(response_for_loss, batch, loss_weights)
                out.update({f"response_aux/{k}": v for k, v in rl.items() if k != "loss"})
                losses.append(response_scale * rl["loss"])
    if stage in ("planner", "all"):
        rank = planner_ranking_loss(
            pred["planner_score"],
            batch["cowp/candidates/noncoercive_feasible"].bool(),
            batch["cowp/candidates/false_safe"].bool(),
            batch["cowp/candidates/valid"].bool(),
        )
        imitation = planner_imitation_loss(pred["planner_score"], batch)
        outcome_legacy = planner_outcome_loss(pred["planner_score"], batch)
        outcome = planner_outcome_supervision(pred.get("outcome"), pred["planner_score"], batch, loss_weights)
        priority_claim = priority_claim_loss(pred.get("priority_claim_logits"), batch, loss_weights)
        cls = candidate_classification_loss(pred["planner_score"], batch, loss_weights)
        cert = candidate_certificate_loss(pred, batch, loss_weights)
        out["planner/ranking"] = rank
        out["planner/imitation"] = imitation
        out["planner/outcome"] = outcome["loss"]
        out["planner/outcome_cls"] = outcome["cls"]
        out["planner/outcome_logdiv"] = outcome["logdiv"]
        out["planner/outcome_rank"] = outcome["rank"]
        out["planner/outcome_expected"] = outcome["expected_cost"]
        out["planner/outcome_legacy"] = outcome_legacy
        out["planner/priority_claim"] = priority_claim
        out.update({f"planner/{k}": v for k, v in cls.items() if k != "loss"})
        out.update({f"candidate_cert/{k}": v for k, v in cert.items() if k != "loss"})
        losses.append(
            loss_weights.get("ranking", 1.0) * rank
            + loss_weights.get("imitation", 1.0) * imitation
            + loss_weights.get("closed_loop", 0.0) * outcome["loss"]
            + loss_weights.get("closed_loop_legacy", 0.0) * outcome_legacy
            + loss_weights.get("priority_claim", 0.5) * priority_claim
            + cls["loss"]
            + float(loss_weights.get("candidate_certificate", 1.0)) * cert["loss"]
        )
    if not losses:
        # Representation fallback: keep graph/planner path differentiable.
        losses.append(pred["planner_score"].mean() * 0.0 + torch.relu(pred["witness"]["opr"].mean() - 0.5))
    out["loss"] = sum(losses)
    return out


def _masked_batch_with_pred_critical(batch: dict[str, torch.Tensor], pred: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Make loss masks agree with model-visible critical-agent slots."""
    crit_mask = pred.get("critical_mask")
    if not torch.is_tensor(crit_mask) or "cowp/critical/valid" not in batch:
        return batch
    if torch.equal(batch["cowp/critical/valid"].bool(), crit_mask.bool()):
        return batch
    out = dict(batch)
    out["cowp/critical/valid"] = crit_mask.bool()
    return out



def _finite_scalar(x: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(x.detach()).all().item())
    except Exception:
        return False


def _nonfinite_tensor_paths(value: Any, prefix: str = "pred", limit: int = 16) -> list[str]:
    """Return model-output paths containing NaN/Inf with one host sync.

    The previous implementation called ``.item()`` once for every output tensor.
    Planner forward exposes many diagnostic aliases and large mode tensors, so
    those repeated CUDA synchronizations materially reduced throughput.  We now
    deduplicate tensor objects, launch all finite reductions, and synchronize once
    per device.  Only the exceptional path performs per-tensor synchronization to
    report precise names.  The validation contract is unchanged: every unique
    floating model output is still checked before loss construction.
    """
    entries: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()

    def collect(x: Any, path: str) -> None:
        if torch.is_tensor(x):
            if x.is_floating_point() or x.is_complex():
                ident = id(x)
                if ident not in seen:
                    seen.add(ident)
                    entries.append((path, x.detach()))
            return
        if isinstance(x, dict):
            for key, item in x.items():
                collect(item, f"{path}.{key}")
        elif isinstance(x, (list, tuple)):
            for i, item in enumerate(x):
                collect(item, f"{path}[{i}]")

    collect(value, prefix)
    if not entries:
        return []

    checks: list[tuple[str, torch.Tensor]] = [
        (path, torch.isfinite(tensor).all()) for path, tensor in entries
    ]
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for _, flag in checks:
        by_device.setdefault(flag.device, []).append(flag)
    all_ok = True
    for flags in by_device.values():
        if not bool(torch.stack(flags).all().item()):
            all_ok = False
    if all_ok:
        return []

    bad: list[str] = []
    for path, flag in checks:
        if not bool(flag.item()):
            bad.append(path)
            if len(bad) >= limit:
                break
    return bad

def _distributed_any(flag: bool, device: torch.device) -> bool:
    """Synchronize a fatal numeric condition so every DDP rank exits together."""
    if not (dist.is_available() and dist.is_initialized()):
        return bool(flag)
    tensor = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()))


def _nonfinite_gradient_paths(model: torch.nn.Module, limit: int = 16) -> list[str]:
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
        if len(bad) >= limit:
            break
    return bad


def _clip_grad_norm_stable(
    model: torch.nn.Module, max_norm: float
) -> tuple[torch.Tensor, list[str]]:
    """Clip exactly as global L2 clipping, with a float64 overflow fallback.

    ``clip_grad_norm_`` normally reduces in the gradient dtype.  A very large but
    finite fp32 gradient can therefore overflow only the norm accumulator.  Use
    ``error_if_nonfinite=True`` so gradients are not modified before diagnosis;
    true NaN/Inf entries are reported, while finite gradients are clipped from a
    float64 norm.
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        ref = next(model.parameters(), None)
        device = ref.device if ref is not None else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=device), []
    max_norm = max(float(max_norm), 1.0e-6)
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            params, max_norm, error_if_nonfinite=True
        )
        return norm, []
    except TypeError:
        # ``error_if_nonfinite`` exists in all supported production versions,
        # but retain compatibility with older local test environments.
        bad = _nonfinite_gradient_paths(model)
        if bad:
            return torch.full((), float("nan"), device=params[0].grad.device), bad
    except RuntimeError:
        bad = _nonfinite_gradient_paths(model)
        if bad:
            return torch.full((), float("nan"), device=params[0].grad.device), bad

    # Every gradient entry is finite; only the fp32 norm reduction overflowed.
    device = params[0].grad.device
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for param in params:
        total_sq = total_sq + param.grad.detach().to(torch.float64).square().sum()
    norm64 = total_sq.sqrt()
    if not bool(torch.isfinite(norm64).item()):
        return norm64, ["<finite entries but non-finite float64 norm>"]
    clip_coef = min(1.0, max_norm / (float(norm64.item()) + 1.0e-6))
    if clip_coef < 1.0:
        for param in params:
            param.grad.mul_(clip_coef)
    return norm64, []

def _set_fully_frozen_modules_eval(model: torch.nn.Module) -> None:
    """Keep frozen top-level modules deterministic during a training epoch.

    v16.4 changed ``requires_grad`` but ``model.train(True)`` re-enabled dropout
    in the graph, so component-ablation arms did not receive a deterministic
    fixed representation.  This also removes needless training-mode overhead.
    """
    core = model.module if hasattr(model, "module") else model
    for module in core.children():
        params = tuple(module.parameters())
        if params and all(not p.requires_grad for p in params):
            module.eval()


def _run_epoch(
    model: COWPModel,
    dl: DataLoader,
    device: torch.device,
    stage: str,
    loss_weights: dict[str, float],
    opt: torch.optim.Optimizer | None = None,
    *,
    epoch: int = 0,
    progress: bool = True,
    amp: bool = False,
    amp_dtype: str = "auto",
    non_blocking_transfer: bool = False,
    decode_response_traj: bool = True,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    is_train = opt is not None
    model.train(is_train)
    if is_train:
        _set_fully_frozen_modules_eval(model)
    # Keep metric accumulation on-device and transfer once per epoch.  The old
    # loop converted every individual loss term to a Python float each batch,
    # forcing dozens of CUDA synchronizations in planner training.
    sums: dict[str, torch.Tensor] = {}
    count = 0
    resolved_amp_dtype = _resolve_amp_dtype(device, amp_dtype)
    # Natural-basis learning is the highest-risk stage because its zero-initialized
    # residual head is fed by the scene graph.  If bf16 is unavailable, execute
    # the whole natural/representation forward in fp32 rather than accepting the
    # fp16 overflow mode observed in v16.2.
    stage_amp = bool(amp and device.type == "cuda")
    if stage in {"natural", "representation"} and resolved_amp_dtype == torch.float16:
        stage_amp = False
    # Dynamic loss scaling is an fp16 remedy.  BF16 has fp32-like exponent
    # range and is intentionally trained without GradScaler.  The old code
    # scaled both dtypes and then raised before GradScaler could perform its
    # documented skip-and-downscale recovery.
    scaler_enabled = _use_grad_scaler(stage_amp, is_train, resolved_amp_dtype)
    scaler = _make_grad_scaler(scaler_enabled)
    optimizer_steps = 0
    amp_skipped_steps = 0
    consecutive_amp_overflows = 0
    # inference_mode is a strict no-autograd superset that also disables version
    # counter/view bookkeeping, reducing validation overhead without changing any
    # model outputs or losses.
    context = torch.enable_grad() if is_train else torch.inference_mode()
    desc = f"{'train' if is_train else 'val'} {stage} epoch {epoch}"
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc=desc, unit="batch")
    with context:
        for step, batch in enumerate(iterator, start=1):
            batch = _to_device(batch, device, non_blocking=non_blocking_transfer)
            if is_train:
                opt.zero_grad(set_to_none=True)
            autocast_enabled = stage_amp
            # Only the model forward runs under AMP.  Losses are intentionally
            # computed in fp32 outside autocast: this prevents CUDA AMP from
            # rejecting unsafe probability-space BCE kernels and keeps scatter /
            # masked reductions numerically stable.
            with _autocast_context(device, autocast_enabled, resolved_amp_dtype):
                pred = model(batch, stage=stage, decode_response_traj=decode_response_traj)
            local_bad_paths = _nonfinite_tensor_paths(pred)
            if _distributed_any(bool(local_bad_paths), device):
                raise FloatingPointError(
                    "Non-finite model output before loss construction at "
                    f"stage={stage} epoch={epoch} step={step}; "
                    f"local_bad_paths={local_bad_paths or ['reported_on_another_ddp_rank']}. "
                    "The batch/checkpoint must be fixed; predictions are never nan_to_num-sanitized."
                )
            batch_for_loss = _masked_batch_with_pred_critical(batch, pred)
            losses = _compute_losses(pred, batch_for_loss, stage, loss_weights)
            loss = losses["loss"]
            if not _finite_scalar(loss):
                bad = []
                for k, v in losses.items():
                    if torch.is_tensor(v) and not _finite_scalar(v):
                        bad.append(k)
                print(f"Warning: non-finite loss at stage={stage} epoch={epoch} step={step}; skipped batch; bad_terms={bad[:8]}")
                if is_train:
                    opt.zero_grad(set_to_none=True)
                continue
            if is_train:
                if scaler is not None:
                    scale_before = float(scaler.get_scale())
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)

                    # Check entries before clipping.  In fp16, a non-finite
                    # unscaled gradient can be a normal dynamic-loss-scaling
                    # overflow.  All DDP ranks must skip together and use the
                    # same lower scale; abort only if it persists at a tiny scale.
                    bad_grad_paths = _nonfinite_gradient_paths(model)
                    global_amp_overflow = _distributed_any(bool(bad_grad_paths), device)
                    if global_amp_overflow:
                        amp_skipped_steps += 1
                        consecutive_amp_overflows += 1
                        opt.zero_grad(set_to_none=True)
                        scale_after = max(scale_before * 0.5, 1.0)
                        _update_grad_scaler_scale(scaler, scale_after)
                        if amp_skipped_steps <= 3 or (amp_skipped_steps & (amp_skipped_steps - 1)) == 0:
                            _rank0_print(
                                "AMP fp16 overflow: skipped optimizer step and reduced "
                                f"scale {scale_before:g}->{scale_after:g} at "
                                f"stage={stage} epoch={epoch} step={step}; "
                                f"local_bad_grad_paths={bad_grad_paths or ['reported_on_another_ddp_rank']}"
                            )
                        if consecutive_amp_overflows >= 8 or scale_before <= 1.0:
                            raise FloatingPointError(
                                "Persistent non-finite fp16 gradient after dynamic-scale recovery at "
                                f"stage={stage} epoch={epoch} step={step}; "
                                f"scale={scale_before:g}; consecutive_overflows={consecutive_amp_overflows}; "
                                f"local_bad_grad_paths={bad_grad_paths or ['reported_on_another_ddp_rank']}. "
                                "This is a structural numeric/data error, not a recoverable AMP overflow."
                            )
                    else:
                        grad_norm, bad_grad_paths = _clip_grad_norm_stable(model, grad_clip)
                        local_bad_grad = bool(bad_grad_paths) or not _finite_scalar(grad_norm)
                        if _distributed_any(local_bad_grad, device):
                            raise FloatingPointError(
                                "Non-finite gradient norm after finite-entry fp16 backward at "
                                f"stage={stage} epoch={epoch} step={step}; "
                                f"local_bad_grad_paths={bad_grad_paths or ['reported_on_another_ddp_rank']}."
                            )
                        scaler.step(opt)
                        scaler.update()
                        optimizer_steps += 1
                        consecutive_amp_overflows = 0
                else:
                    loss.backward()
                    grad_norm, bad_grad_paths = _clip_grad_norm_stable(model, grad_clip)
                    local_bad_grad = bool(bad_grad_paths) or not _finite_scalar(grad_norm)
                    if _distributed_any(local_bad_grad, device):
                        raise FloatingPointError(
                            "Non-finite gradient at "
                            f"stage={stage} epoch={epoch} step={step}; "
                            f"local_bad_grad_paths={bad_grad_paths or ['reported_on_another_ddp_rank']}."
                        )
                    opt.step()
                    optimizer_steps += 1
            bs = int(next(iter(batch.values())).shape[0]) if batch else 1
            count += bs
            for k, v in losses.items():
                term = v.detach().to(dtype=torch.float64) * bs
                sums[k] = term if k not in sums else sums[k] + term
            if hasattr(iterator, "set_postfix") and (step == 1 or step % 10 == 0):
                iterator.set_postfix(
                    loss=f"{float(loss.detach().item()):.4f}",
                    seen=count,
                    refresh=True,
                )
    if sums:
        metric_keys = sorted(sums)
        metric_values = torch.stack([sums[k] for k in metric_keys]).detach().cpu().tolist()
        result = {k: float(v) / max(count, 1) for k, v in zip(metric_keys, metric_values)}
    else:
        result = {}
    if is_train:
        result["runtime/optimizer_steps"] = float(optimizer_steps)
        result["runtime/amp_skipped_steps"] = float(amp_skipped_steps)
        result["runtime/amp_bfloat16"] = float(stage_amp and resolved_amp_dtype == torch.bfloat16)
        result["runtime/amp_float16"] = float(stage_amp and resolved_amp_dtype == torch.float16)
        result["runtime/amp_scaler_enabled"] = float(scaler_enabled)
        if optimizer_steps <= 0:
            raise RuntimeError(
                f"Stage {stage} completed no optimizer steps at epoch {epoch}; "
                f"AMP skipped={amp_skipped_steps}. Refusing to write a misleading checkpoint."
            )
        attempted = optimizer_steps + amp_skipped_steps
        if attempted > 0 and amp_skipped_steps / attempted > 0.02:
            raise RuntimeError(
                f"Stage {stage} AMP skipped {amp_skipped_steps}/{attempted} optimizer steps (>2%). "
                "Use --amp-dtype bfloat16 or disable AMP and inspect the first non-finite batch."
            )
    return result


def _model_state_dict_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a checkpoint state_dict that can be loaded without DDP/torch.compile wrappers."""
    inner = getattr(model, "module", model)
    inner = getattr(inner, "_orig_mod", inner)
    if isinstance(inner, torch.nn.Module):
        return inner.state_dict()
    return model.state_dict()


def _load_model_state_robust(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    reset_prefixes: tuple[str, ...] = (),
) -> None:
    """Load compatible weights while explicitly reinitializing selected modules.

    Cross-version natural warm starts are unsafe when mode identities or decoder
    parameterization changed.  ``reset_prefixes=("natural_decoder",)`` keeps the
    pretrained scene graph but guarantees a fresh typed option basis.
    """
    candidates = [state]
    if state and all(k.startswith("_orig_mod.") for k in state.keys()):
        candidates.insert(0, {k[len("_orig_mod."):]: v for k, v in state.items()})
    prefixes = tuple(p.strip().rstrip(".") for p in reset_prefixes if p and p.strip())
    last_error: RuntimeError | None = None
    model_state = model.state_dict()
    for cand0 in candidates:
        cand = {
            k: v for k, v in cand0.items()
            if not any(k == p or k.startswith(p + ".") for p in prefixes)
        }
        if not prefixes:
            try:
                model.load_state_dict(cand)
                return
            except RuntimeError as exc:
                last_error = exc
        compatible = {k: v for k, v in cand.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing = sorted(set(model_state) - set(compatible))
        unexpected = sorted(set(cand) - set(compatible))
        if compatible:
            model.load_state_dict(compatible, strict=False)
            if prefixes:
                reset_keys = [k for k in model_state if any(k == p or k.startswith(p + ".") for p in prefixes)]
                print(f"Explicitly reinitialized {len(reset_keys)} checkpoint keys under prefixes={prefixes}")
            if missing:
                print(f"Loaded checkpoint with {len(missing)} initialized/missing keys, e.g. {missing[:6]}")
            if unexpected:
                print(f"Ignored {len(unexpected)} incompatible/unexpected keys, e.g. {unexpected[:6]}")
            return
    if last_error is not None:
        raise last_error
    raise RuntimeError("Checkpoint contains no compatible model parameters")



def _score_from_history_row(row: dict[str, Any]) -> float:
    """Return the scalar used for best-checkpoint selection for a history row."""
    if "checkpoint/score" in row:
        return float(row.get("checkpoint/score", float("inf")))
    if "val/loss" in row:
        return float(row.get("val/loss", float("inf")))
    if "train/loss" in row:
        return float(row.get("train/loss", float("inf")))
    return float("inf")


def _checkpoint_selection_score(metrics: dict[str, float], stage: str) -> tuple[float, str]:
    """Choose planner checkpoints by certificate separation, not only total loss.

    v5's total loss was dominated by large certificate coefficients and the launch
    script then evaluated the final epoch.  A collapsed constant head can still have
    a deceptively ordinary BCE.  The composite below heavily penalizes ranking and
    risk-calibration collapse while retaining a small total/outcome term.
    """
    total = float(metrics.get("loss", float("inf")))
    if stage in {"natural", "representation"}:
        def gn(name: str, default: float = 100.0) -> float:
            try:
                value = float(metrics.get(name, default))
                return value if math.isfinite(value) else default
            except Exception:
                return default
        # Component-neutral checkpoint score.  v16.5 included total loss and
        # regularizers that are intentionally absent in an ablation, so every arm
        # selected a different training time for partly tautological reasons.  The
        # score below uses only prediction quantities shared by every arm; the
        # attribution script still enforces the exact same checkpoint epoch.
        score = (
            0.55 * gn("natural/traj")
            + 0.30 * gn("natural/obs_minade")
            + 0.15 * gn("natural/branch_minade")
        )
        return float(score), "natural_common_prediction_composite"
    if stage == "witness":
        def gw(name: str, default: float = 1.0) -> float:
            try:
                value = float(metrics.get(name, default))
                return value if math.isfinite(value) else default
            except Exception:
                return default
        budget_coverage = gw("set_transport/candidate_budget_coverage", 0.0)
        # Coverage is a diagnostic, not a loss.  A checkpoint with absent budget
        # supervision is invalid rather than optimal.
        inactive_penalty = 5.0 if budget_coverage < 1.0e-4 else 0.0
        score = (
            inactive_penalty
            + 0.10 * total
            + 0.30 * gw("set_transport/witness")
            + 0.20 * gw("set_transport/opr")
            + 0.15 * gw("set_transport/burden")
            + 0.35 * gw("set_transport/mode_conflict")
            + 0.35 * gw("set_transport/mode_retain")
            + 0.30 * gw("set_transport/mode_recovery")
            + 0.15 * gw("set_transport/root_recovery")
            + 0.45 * gw("set_transport/candidate_budget")
            + 0.10 * gw("response_aux/root")
            + 0.05 * gw("response_aux/min_burden")
        )
        return float(score), "transport_composite"
    if stage != "planner":
        return total, "loss"
    def g(name: str, default: float = 0.0) -> float:
        try:
            value = float(metrics.get(name, default))
            return value if math.isfinite(value) else default
        except Exception:
            return default
    score = (
        0.08 * total
        + 0.20 * g("candidate_cert/risk_rank", 2.0)
        + 0.15 * g("candidate_cert/risk_bce", 2.0)
        + 0.10 * g("candidate_cert/rank", 2.0)
        + 0.45 * g("set_transport/witness", 1.0)
        + 0.35 * g("set_transport/opr", 1.0)
        + 0.30 * g("set_transport/conflict", 1.0)
        + 0.20 * g("set_transport/burden", 1.0)
        + 0.35 * g("set_transport/mode_conflict", 1.0)
        + 0.35 * g("set_transport/mode_retain", 1.0)
        + 0.30 * g("set_transport/mode_recovery", 1.0)
        + 0.20 * g("set_transport/root_recovery", 1.0)
        + 0.55 * g("set_transport/candidate_budget", 1.0)
        + 0.08 * g("planner/outcome_cls", 1.0)
        + 0.15 * g("planner/ranking", 1.0)
    )
    return float(score), "planner_certificate_composite"


def _set_stage_freeze(
    model: torch.nn.Module,
    stage: str,
    warmup_frozen: bool,
    *,
    freeze_natural_during_witness: bool = False,
    freeze_graph_during_natural: bool = False,
) -> None:
    """Granular v11 freeze policy.

    Transport learning always updates candidate, natural and witness identity heads;
    only the expensive graph encoder is protected during warm-up.  Planner learning
    keeps natural/witness identities fixed, while allowing candidate features to
    adapt after warm-up.  Set-transport and response modules always remain trainable.
    """
    core = model.module if hasattr(model, "module") else model
    if stage in {"natural", "representation"}:
        # Natural-basis repair should not rewrite the scene encoder inherited from
        # a strong planner checkpoint.  Training only the natural decoder yields a
        # stable primitive basis that later transport stages can treat as an
        # identifiable coordinate system.
        policy = {
            "graph": bool(freeze_graph_during_natural),
            "candidate_encoder": True,
            "natural_decoder": False,
            "witness_decoder": True,
        }
    elif stage == "witness":
        policy = {
            "graph": bool(warmup_frozen),
            "candidate_encoder": False,
            # v11 jointly moved the natural basis while learning root-indexed
            # transport labels.  Validation natural minADE drifted from ~48 m to
            # ~67 m, so mode identities used by the certificate were nonstationary.
            # v12 pretrains/repairs the basis, then freezes it during transport.
            "natural_decoder": bool(freeze_natural_during_witness),
            "witness_decoder": False,
        }
    elif stage == "planner":
        policy = {
            "graph": True,
            "candidate_encoder": bool(warmup_frozen),
            "natural_decoder": True,
            "witness_decoder": True,
        }
    else:
        policy = {"graph": False, "candidate_encoder": False, "natural_decoder": False, "witness_decoder": False}
    for module_name, frozen in policy.items():
        module = getattr(core, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(not frozen)


def _freeze_inactive_architecture_branches(model: torch.nn.Module) -> list[str]:
    """Freeze checkpoint-compatible branches unused by the selected architecture.

    The typed natural decoders retain the legacy dense trajectory head only so old
    checkpoints can be loaded.  Keeping those parameters trainable forces DDP to
    search for unused parameters every iteration and makes the AdamW parameter
    groups larger without changing a single output.
    """
    core = model.module if hasattr(model, "module") else model
    frozen: list[str] = []
    decoder = getattr(core, "natural_decoder", None)
    if decoder is not None and bool(getattr(decoder, "uses_typed_basis", False)):
        legacy_head = getattr(decoder, "head", None)
        if legacy_head is not None:
            for param in legacy_head.parameters():
                param.requires_grad_(False)
            frozen.append("natural_decoder.head")
    return frozen


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return trainable, total


def _load_existing_history(path: Path, *, before_epoch: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read existing history {path}: {exc}")
        return []
    if not isinstance(data, list):
        return []
    rows = [r for r in data if isinstance(r, dict)]
    if before_epoch is not None:
        rows = [r for r in rows if int(r.get("epoch", -1)) < int(before_epoch)]
    return rows


def _atomic_write_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Persist history after every completed epoch without partial JSON files."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write checkpoints atomically so an interruption cannot corrupt resume state."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _best_score_from_history(history: list[dict[str, Any]]) -> float:
    best = float("inf")
    for row in history:
        score = _score_from_history_row(row)
        if math.isfinite(score) and score < best:
            best = score
    return best


def _checkpoint_stage(ckpt: dict[str, Any]) -> str | None:
    stage = ckpt.get("stage")
    return str(stage) if stage is not None else None


def _checkpoint_epoch(ckpt: dict[str, Any]) -> int | None:
    epoch = ckpt.get("epoch")
    if epoch is None:
        return None
    try:
        return int(epoch)
    except Exception:
        return None


def _make_checkpoint_payload(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    opt: torch.optim.Optimizer,
    *,
    epoch: int,
    stage: str,
    best_val: float,
    save_optimizer: bool,
    extra: dict[str, Any] | None = None,
    scheduler: Any | None = None,
    no_improve_checks: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": _model_state_dict_for_save(model),
        "cfg": cfg,
        "epoch": int(epoch),
        "stage": stage,
        "best_val": float(best_val),
    }
    if save_optimizer:
        payload["optimizer"] = opt.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload["no_improve_checks"] = int(no_improve_checks)
    if extra:
        payload.update(extra)
    return payload


def _make_adamw_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float, fused: bool = False) -> torch.optim.Optimizer:
    """Create AdamW over trainable parameters only.

    Natural-stage v16.5 froze the graph after constructing AdamW, so optimizer
    state and foreach/fused bookkeeping still included millions of permanently
    frozen parameters.  Filtering here is safe for the natural stage because its
    permanent freeze is now applied before optimizer construction.  Witness and
    planner warm-up parameters remain trainable at construction and therefore stay
    in the optimizer for their later unfreeze.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Cannot create AdamW: model has no trainable parameters")
    kwargs: dict[str, Any] = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except TypeError:
            pass
        except RuntimeError as exc:
            print(f"Warning: fused AdamW unavailable, falling back to standard AdamW: {exc}")
    return torch.optim.AdamW(params, **kwargs)




def _make_lr_scheduler(
    opt: torch.optim.Optimizer,
    *,
    mode: str,
    epochs: int,
    early_stop_patience: int,
    min_lr: float,
    min_delta: float,
) -> Any | None:
    mode = str(mode).lower()
    if mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(1, int(early_stop_patience) // 2),
            min_lr=float(min_lr), threshold=float(min_delta),
        )
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(int(epochs), 1), eta_min=float(min_lr)
        )
    return None


def _reconstruct_scheduler_from_history(
    scheduler: Any,
    opt: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    *,
    mode: str,
    base_lr: float,
    epochs: int,
    early_stop_patience: int,
    min_lr: float,
    min_delta: float,
) -> int:
    """Rebuild legacy scheduler state when an older checkpoint lacks it.

    Historical checkpoints were written before ``scheduler.step`` and did not
    contain scheduler state.  Replaying completed epoch scores on a shadow
    optimizer reconstructs the exact post-epoch LR/scheduler counters without
    touching AdamW moments or rerunning model updates.
    """
    rows = sorted(
        (row for row in history if isinstance(row.get("epoch"), int) and int(row["epoch"]) >= 0),
        key=lambda row: int(row["epoch"]),
    )
    if not rows:
        return 0
    dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    shadow_opt = torch.optim.SGD([dummy], lr=float(base_lr))
    shadow = _make_lr_scheduler(
        shadow_opt, mode=mode, epochs=epochs, early_stop_patience=early_stop_patience,
        min_lr=min_lr, min_delta=min_delta,
    )
    if shadow is None:
        return 0
    best_seen = float("inf")
    replayed = 0
    for row in rows:
        if str(mode).lower() == "plateau":
            try:
                score = float(row.get("checkpoint/score", float("inf")))
            except Exception:
                score = float("inf")
            if math.isfinite(score):
                best_seen = min(best_seen, score)
            shadow.step(score if math.isfinite(score) else best_seen)
        else:
            shadow.step()
        replayed += 1
    scheduler.load_state_dict(shadow.state_dict())
    if len(opt.param_groups) != len(shadow_opt.param_groups):
        raise RuntimeError("cannot reconstruct scheduler: optimizer parameter-group count changed")
    for actual, rebuilt in zip(opt.param_groups, shadow_opt.param_groups):
        actual["lr"] = float(rebuilt["lr"])
    return replayed


def _maybe_compile_model(model: COWPModel, enabled: bool, backend: str | None = None) -> torch.nn.Module:
    """Compile the model in a failure-tolerant way.

    torch.compile performs most real compilation at first forward.  Setting
    Dynamo suppress_errors lets unsupported Inductor/Triton subgraphs fall back
    to eager instead of aborting the training run.
    """
    if not enabled:
        return model
    import torch
    if not hasattr(torch, "compile"):
        print("Warning: torch.compile is not available in this PyTorch version; continuing without compile.")
        return model
    try:
        import torch._dynamo  # type: ignore[attr-defined]
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass
    try:
        backend_msg = f", backend={backend}" if backend else ""
        print(f"Compiling model with torch.compile(mode='reduce-overhead'{backend_msg}, suppress_errors=True) ...")
        kwargs: dict[str, Any] = {"mode": "reduce-overhead"}
        if backend:
            kwargs["backend"] = backend
        return torch.compile(model, **kwargs)
    except Exception as exc:
        print(f"Warning: torch.compile setup failed, continuing without compile: {exc}")
        return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train COWP model stages on COWP tensor cache.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml", help="Label/planning config. Loaded into the training checkpoint so witness/planner semantics match evaluation.")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--val-cache-dir", default=None)
    ap.add_argument("--stage", choices=["representation", "natural", "response", "witness", "planner", "all"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-training", action="store_true", help="When --resume points to a checkpoint from the same stage, continue epoch numbering/history instead of treating it as a warm start.")
    ap.add_argument("--reset-checkpoint-prefix", action="append", default=[], help="Reinitialize a module prefix instead of loading it from --resume. Repeatable; v14 natural training should pass natural_decoder.")
    ap.add_argument("--eval-before-train", action="store_true", help="Evaluate/save the initialized or warm-started model as epoch -1 before optimization. Prevents training from silently degrading a strong analytic basis.")
    ap.add_argument("--no-save-optimizer", action="store_true", help="Do not store optimizer state in epoch/best checkpoints. This saves disk but weakens exact crash resume.")
    ap.add_argument("--output-dir", default="outputs/checkpoints")
    ap.add_argument("--device", default=None, help="Training device: auto, cuda, cuda:0, cpu. Overrides configs/train.yaml.")
    ap.add_argument("--num-workers", type=int, default=None, help="Override training DataLoader num_workers.")
    ap.add_argument("--val-num-workers", type=int, default=None, help="Override validation DataLoader num_workers independently. Keeping validation smaller reduces DDP shared-storage pressure without changing metrics.")
    ap.add_argument("--prefetch-factor", type=int, default=None, help="Override training DataLoader prefetch_factor when num_workers > 0. For response/planner/all stages, default is capped to 1 to avoid huge queued batches.")
    ap.add_argument("--val-prefetch-factor", type=int, default=None, help="Override validation DataLoader prefetch_factor independently.")
    ap.add_argument("--sharing-strategy", choices=["auto", "current", "file_descriptor", "file_system"], default=None, help="PyTorch CPU tensor sharing strategy. auto prefers file_system on Linux to prevent 'received 0 items of ancdata'.")
    ap.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent DataLoader workers for both train and validation.")
    ap.add_argument("--persistent-val-workers", action="store_true", help="Keep validation workers alive across validation passes. Disabled by default to release IPC resources after each pass.")
    ap.add_argument("--no-positive-oversampling", action="store_true")
    ap.add_argument("--force-positive-oversampling", action="store_true", help="Force witness/candidate positive oversampling even for representation/natural stages.")
    ap.add_argument("--no-sampler-cache", action="store_true", help="Do not read/write cached sampler weights for positive oversampling.")
    ap.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for lower memory and faster training.")
    ap.add_argument(
        "--amp-dtype", choices=["auto", "bfloat16", "float16"], default="auto",
        help="CUDA autocast dtype. auto prefers bfloat16 to avoid fp16 overflow in the graph/natural basis.",
    )
    ap.add_argument("--pin-memory", action="store_true", help="Force DataLoader pin_memory on when using CUDA.")
    ap.add_argument("--no-pin-memory", action="store_true", help="Force DataLoader pin_memory off. Useful for CUDA invalid-argument pinning errors.")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-epoch tqdm progress bars.")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile for faster repeated training on PyTorch 2.x. Unsupported subgraphs fall back to eager.")
    ap.add_argument("--compile-backend", default=None, help="Optional torch.compile backend, e.g. aot_eager for safer fallback when Inductor/Triton is unstable.")
    ap.add_argument("--fused-adamw", action="store_true", help="Use fused CUDA AdamW when supported.")
    ap.add_argument("--response-traj-weight", type=float, default=None, help="Override loss_weights.response_traj_l1. Set to 0 for fast response/witness/planner smoke training without dense response trajectory labels.")
    ap.add_argument("--no-response-traj", action="store_true", help="Shortcut for --response-traj-weight 0. Avoids loading cowp/response/traj and skips the response trajectory head.")
    ap.add_argument("--no-response-components", action="store_true", help="Do not load/supervise cowp/response/burden_components during response training.")
    ap.add_argument("--with-waymax-outcome-labels", action="store_true", help="For planner training, load optional waymax/candidate_{collision,offroad,log_divergence} labels if present. Default keeps broad waymax tensors out of batches.")
    ap.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs. Use 0 to disable validation during quick smoke training.")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=0, help="For planner training, freeze graph/candidate/witness backbones for the first N epochs so new certificate/outcome heads warm up without erasing pretrained representations.")
    ap.add_argument("--natural-graph-warmup-epochs", type=int, default=None, help="Deprecated v16.4 alias; ignored unless explicitly mapped by an old launcher.")
    ap.add_argument("--natural-graph-unfreeze-epoch", type=int, default=None, help="Explicit epoch at which to unfreeze the graph in natural training. Negative/default keeps it frozen for the whole stage.")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Global gradient-norm clip. Typed residual training uses 1.0 by default.")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="Stop after this many validation checks without composite-score improvement. 0 disables early stopping.")
    ap.add_argument("--early-stop-min-delta", type=float, default=1.0e-4, help="Minimum checkpoint-score decrease counted as an improvement.")
    ap.add_argument("--lr-scheduler", choices=["none", "plateau", "cosine"], default="none", help="Learning-rate scheduler for stable planner fine-tuning.")
    ap.add_argument("--min-lr", type=float, default=1.0e-6, help="Minimum learning rate used by plateau/cosine scheduling.")
    ap.add_argument("--save-every", type=int, default=1, help="Save epoch checkpoints every N epochs. Best checkpoint is always saved.")
    args = ap.parse_args()

    cfg = load_config(args.model_config, args.label_config, args.train_config, args.data_config)
    tcfg = cfg["train"]
    stage = args.stage or tcfg.get("stage", "witness")
    loss_weights = dict(cfg.get("loss_weights", {}))
    if args.no_response_traj:
        loss_weights["response_traj_l1"] = 0.0
    if args.response_traj_weight is not None:
        loss_weights["response_traj_l1"] = float(args.response_traj_weight)
    if args.no_response_components:
        loss_weights["response_components_l1"] = 0.0

    device_arg = args.device or tcfg.get("device", "auto")
    loader_runtime = configure_dataloader_runtime(args.sharing_strategy)
    distributed, rank, world_size, local_rank = _setup_distributed(str(device_arg))
    if distributed and str(device_arg).lower() != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _device(device_arg)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device.type == "cpu" and str(device_arg) == "auto" and visible_devices in {"-1", "none", "None", ""}:
        print(
            "Warning: CUDA is hidden by CUDA_VISIBLE_DEVICES="
            f"{visible_devices!r}; training will run on CPU. "
            "Run `unset CUDA_VISIBLE_DEVICES` or `export CUDA_VISIBLE_DEVICES=0` before training if a GPU is available."
        )
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False. "
            "Check CUDA_VISIBLE_DEVICES and the PyTorch CUDA build."
        )
    _set_runtime_defaults(int(tcfg.get("seed", 2026)), device)
    batch_size = args.batch_size or int(tcfg.get("batch_size", 8))
    if args.no_pin_memory:
        pin_memory = False
    elif args.pin_memory:
        pin_memory = _cuda_pin_memory_works(device)
        if pin_memory and _heavy_label_stage(stage):
            print(
                f"Warning: pin_memory was forced on for heavy stage={stage}. "
                "If CUDA invalid-argument errors recur, rerun with --no-pin-memory."
            )
    else:
        # Stable default: keep pinning off.  This avoids Stage-B crashes caused by
        # page-locking dense response supervision batches.  Use --pin-memory only
        # after confirming the host/CUDA setup can sustain the memory pressure.
        pin_memory = False

    user_prefetch = args.prefetch_factor
    effective_prefetch = user_prefetch
    if effective_prefetch is None and _heavy_label_stage(stage):
        effective_prefetch = 1
    elif effective_prefetch is not None and effective_prefetch > 1 and _heavy_label_stage(stage):
        print(
            f"Warning: stage={stage} has large dense label tensors; "
            f"prefetch_factor={effective_prefetch} can queue very large CPU batches. "
            "Use --prefetch-factor 1 if host RAM or pinned-memory errors appear."
        )

    train_num_workers = int(tcfg.get("num_workers", 0) if args.num_workers is None else args.num_workers)
    val_num_workers = int(train_num_workers if args.val_num_workers is None else args.val_num_workers)
    if train_num_workers < 0 or val_num_workers < 0:
        raise ValueError(f"DataLoader worker counts must be >= 0, got train={train_num_workers}, val={val_num_workers}")
    val_prefetch = effective_prefetch if args.val_prefetch_factor is None else max(1, int(args.val_prefetch_factor))
    train_persistent_workers = bool(train_num_workers > 0 and not args.no_persistent_workers)
    val_persistent_workers = bool(
        val_num_workers > 0 and not args.no_persistent_workers and args.persistent_val_workers
    )

    compile_enabled = bool(args.compile or tcfg.get("compile", False))
    # Respect response-head disabling flags in both response-only and all-head
    # training.  Without this, ``--stage all --response-traj-weight 0`` still
    # loaded ``cowp/response/traj`` and decoded the giant trajectory head.
    # All stages that instantiate the response bank obey the dense-label flags.
    # v8 restricted this check to response/all, which would make planner loading
    # unexpectedly pull the huge trajectory tensor once response labels were
    # correctly enabled for planner.
    response_stage = stage in {"response", "witness", "planner", "all"}
    include_response_traj = bool(response_stage and float(loss_weights.get("response_traj_l1", 0.1)) != 0.0)
    include_response_components = bool(response_stage and float(loss_weights.get("response_components_l1", 0.25)) != 0.0)
    if distributed and compile_enabled:
        _rank0_print("Warning: disabling torch.compile under DDP for staged training stability.")
        compile_enabled = False
    _rank0_print(f"COWP train startup: stage={stage}, device={device}, distributed={distributed}, rank={rank}/{world_size}, cuda_available={torch.cuda.is_available()}, amp={args.amp}, compile={compile_enabled}, batch_size_per_gpu={batch_size}, global_batch_size={batch_size * world_size if distributed else batch_size}, pin_memory={pin_memory}, train_workers={train_num_workers}, val_workers={val_num_workers}, train_prefetch={effective_prefetch if effective_prefetch is not None else tcfg.get('prefetch_factor', 2)}, val_prefetch={val_prefetch if val_prefetch is not None else tcfg.get('prefetch_factor', 2)}, train_persistent={train_persistent_workers}, val_persistent={val_persistent_workers}, sharing_strategy={loader_runtime['selected']}, response_traj_l1={float(loss_weights.get('response_traj_l1', 0.0))}, load_response_traj={include_response_traj}, load_waymax_outcomes={bool(args.with_waymax_outcome_labels)}, cuda_alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}")
    _rank0_print("DataLoader IPC runtime: " + json.dumps(loader_runtime, ensure_ascii=False, sort_keys=True))
    if device.type == "cuda":
        try:
            _rank0_print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        except Exception:
            pass

    train_ds = TorchCOWPDataset(
        args.cache_dir or cfg["outputs"]["tensor_cache_dir"],
        stage=stage,
        include_response_traj=include_response_traj,
        include_response_components=include_response_components,
        include_waymax_outcomes=bool(args.with_waymax_outcome_labels),
    )
    val_ds = (
        TorchCOWPDataset(
            args.val_cache_dir,
            stage=stage,
            include_response_traj=include_response_traj,
            include_response_components=include_response_components,
            include_waymax_outcomes=bool(args.with_waymax_outcome_labels),
        )
        if args.val_cache_dir
        else None
    )
    _rank0_print(f"Loaded datasets: train={len(train_ds)}" + (f", val={len(val_ds)}" if val_ds is not None else ""))
    train_supervision = _validate_stage_supervision(train_ds, stage)
    if train_supervision:
        _rank0_print(f"Stage supervision check (train prefix): {train_supervision}")
    if val_ds is not None:
        val_supervision = _validate_stage_supervision(val_ds, stage)
        if val_supervision:
            _rank0_print(f"Stage supervision check (val prefix): {val_supervision}")
    oversample_enabled = bool(tcfg.get("positive_pair_oversampling", True)) and not args.no_positive_oversampling
    # Representation/natural stages do not consume witness/candidate labels.
    # Scanning every npz for those labels can add many minutes before the first
    # batch and does not change the Stage-A objective.
    if stage in {"representation", "natural"} and not args.force_positive_oversampling:
        if oversample_enabled:
            _rank0_print(f"Skipping positive oversampling for stage={stage}; use --force-positive-oversampling to enable it.")
        oversample_enabled = False

    seed = int(tcfg.get("seed", 2026))
    if distributed and oversample_enabled and not args.no_sampler_cache and rank != 0:
        dist.barrier()
    train_dl = _make_loader(
        train_ds,
        cfg,
        batch_size,
        shuffle=True,
        oversample=oversample_enabled,
        use_cuda=device.type == "cuda",
        progress=(not args.no_progress) and _is_main_process(),
        num_workers=train_num_workers,
        sampler_cache=not args.no_sampler_cache,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=train_persistent_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    if distributed and oversample_enabled and not args.no_sampler_cache and rank == 0:
        dist.barrier()
    val_dl = _make_loader(
        val_ds, cfg, batch_size, shuffle=False, oversample=False, use_cuda=device.type == "cuda",
        progress=(not args.no_progress) and _is_main_process(), num_workers=val_num_workers, pin_memory=pin_memory,
        prefetch_factor=val_prefetch, persistent_workers=val_persistent_workers,
        distributed=distributed, rank=rank, world_size=world_size, seed=seed,
    ) if val_ds is not None else None

    _rank0_print(
        f"DataLoader: train_workers={train_dl.num_workers}, train_persistent={getattr(train_dl, 'persistent_workers', False)}, "
        f"train_pin_memory={train_dl.pin_memory}, train_prefetch={getattr(train_dl, 'prefetch_factor', None)}, "
        f"train_batches_per_rank={len(train_dl)}"
        + (
            f", val_workers={val_dl.num_workers}, val_persistent={getattr(val_dl, 'persistent_workers', False)}, "
            f"val_prefetch={getattr(val_dl, 'prefetch_factor', None)}, val_batches_per_rank={len(val_dl)}"
            if val_dl is not None else ""
        )
    )
    model = COWPModel(cfg).to(device)
    resume_ckpt: dict[str, Any] | None = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        _load_model_state_robust(model, resume_ckpt["model"], reset_prefixes=tuple(args.reset_checkpoint_prefix))

    # Apply architecture and permanent stage freezes *before* DDP and AdamW.
    # v16.5 did this only inside the epoch loop, so DDP searched for unused
    # parameters every batch and AdamW still tracked permanently frozen modules.
    inactive_branches = _freeze_inactive_architecture_branches(model)
    natural_unfreeze_epoch = int(
        args.natural_graph_unfreeze_epoch
        if args.natural_graph_unfreeze_epoch is not None
        else tcfg.get("natural_graph_unfreeze_epoch", -1)
    )
    permanent_natural_freeze = bool(
        stage in {"natural", "representation"}
        and bool(tcfg.get("freeze_graph_during_natural", False))
        and natural_unfreeze_epoch < 0
    )
    if stage in {"natural", "representation"}:
        _set_stage_freeze(
            model,
            stage,
            False,
            freeze_natural_during_witness=bool(tcfg.get("freeze_natural_during_witness", False)),
            freeze_graph_during_natural=permanent_natural_freeze,
        )
    trainable_params, total_params = _parameter_counts(model)
    _rank0_print(
        f"Parameter policy before DDP: trainable={trainable_params:,}/{total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.2f}%), "
        f"permanent_natural_freeze={permanent_natural_freeze}, "
        f"inactive_branches={inactive_branches}"
    )

    model = _maybe_compile_model(model, compile_enabled, backend=args.compile_backend)
    if distributed:
        # Static DDP is valid only when the set of parameters participating in
        # backward is invariant for the entire run.  Natural/representation
        # stages satisfy this after their permanent graph/branch freezes.
        #
        # Witness/transport and planner stages do not: their losses contain
        # data-dependent branches such as ``mask.any()``, positive/negative pair
        # mining, candidate-budget supervision, and optional response/root
        # targets.  Consequently, two successive batches can legitimately use
        # different parameter subsets even when ``freeze_backbone_epochs=0``.
        # Enabling ``static_graph=True`` for those stages causes the reducer to
        # fail at the next forward with "Expected to have finished reduction".
        # Keep ordinary unused-parameter discovery for these stages.  This is a
        # DDP execution-policy fix only; model outputs, losses, optimizer
        # parameters, and evaluation behavior are unchanged.
        static_stage_ddp = bool(
            permanent_natural_freeze and stage in {"natural", "representation"}
        )
        fully_used_static_natural = bool(
            permanent_natural_freeze and stage in {"natural", "representation"}
        )
        ddp_kwargs: dict[str, Any] = {
            "find_unused_parameters": not fully_used_static_natural,
        }
        if static_stage_ddp:
            ddp_kwargs.update({"static_graph": True, "gradient_as_bucket_view": True})
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        try:
            model = DDP(model, **ddp_kwargs)
        except TypeError:
            # Compatibility with older PyTorch versions lacking static_graph or
            # gradient_as_bucket_view.  Keep the safe no-unused traversal choice.
            ddp_kwargs.pop("static_graph", None)
            ddp_kwargs.pop("gradient_as_bucket_view", None)
            model = DDP(model, **ddp_kwargs)
        _rank0_print(
            f"DDP policy: find_unused_parameters={ddp_kwargs.get('find_unused_parameters')}, "
            f"static_graph={ddp_kwargs.get('static_graph', False)}, "
            f"gradient_as_bucket_view={ddp_kwargs.get('gradient_as_bucket_view', False)}"
        )
    epochs = args.epochs or int(tcfg.get("epochs", 10))
    optimizer_lr = float(args.lr if args.lr is not None else tcfg.get("lr", 3e-4))
    opt = _make_adamw_optimizer(
        model,
        lr=optimizer_lr,
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        fused=bool(args.fused_adamw or tcfg.get("fused_adamw", False)),
    )
    scheduler_mode = str(args.lr_scheduler).lower()
    scheduler = _make_lr_scheduler(
        opt, mode=scheduler_mode, epochs=epochs,
        early_stop_patience=int(args.early_stop_patience), min_lr=float(args.min_lr),
        min_delta=float(args.early_stop_min_delta),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"history_{stage}.json"
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    start_epoch = 0
    resume_no_improve_checks = 0
    if args.resume_training:
        if not args.resume or resume_ckpt is None:
            raise ValueError("--resume-training requires --resume to point to a same-stage checkpoint")
        ckpt_stage = _checkpoint_stage(resume_ckpt)
        ckpt_epoch = _checkpoint_epoch(resume_ckpt)
        if ckpt_epoch is None:
            raise ValueError(f"Checkpoint {args.resume} has no integer epoch; cannot use --resume-training")
        if ckpt_stage is not None and ckpt_stage != stage:
            raise ValueError(f"--resume-training expected stage={stage}, but checkpoint stage={ckpt_stage}; use plain --resume for cross-stage warm start")
        start_epoch = ckpt_epoch + 1
        history = _load_existing_history(history_path, before_epoch=start_epoch)
        best_val = _best_score_from_history(history)
        ckpt_best = resume_ckpt.get("best_val")
        if ckpt_best is not None:
            try:
                best_val = min(best_val, float(ckpt_best))
            except Exception:
                pass
        if not math.isfinite(best_val):
            best_val = float("inf")
        if "optimizer" in resume_ckpt:
            try:
                opt.load_state_dict(resume_ckpt["optimizer"])
                _rank0_print(f"Resumed optimizer state from {args.resume}")
            except Exception as exc:
                _rank0_print(f"Warning: failed to load optimizer state from {args.resume}; continuing with a fresh optimizer: {exc}")
        else:
            _rank0_print(f"Warning: checkpoint {args.resume} has no optimizer state; continuing model weights from epoch {ckpt_epoch} with a fresh optimizer.")
        if scheduler is not None and "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
                _rank0_print(f"Resumed LR scheduler state from {args.resume}")
            except Exception as exc:
                _rank0_print(f"Warning: failed to load scheduler state from {args.resume}: {exc}")
        elif scheduler is not None and history:
            try:
                replayed = _reconstruct_scheduler_from_history(
                    scheduler, opt, history, mode=scheduler_mode, base_lr=optimizer_lr,
                    epochs=epochs, early_stop_patience=int(args.early_stop_patience),
                    min_lr=float(args.min_lr), min_delta=float(args.early_stop_min_delta),
                )
                _rank0_print(
                    f"Reconstructed legacy LR scheduler from {replayed} completed history rows; "
                    f"resume_lr={opt.param_groups[0]['lr']:.8g}"
                )
            except Exception as exc:
                _rank0_print(f"Warning: failed to reconstruct legacy scheduler state: {exc}")
        if "no_improve_checks" in resume_ckpt:
            resume_no_improve_checks = max(int(resume_ckpt.get("no_improve_checks", 0)), 0)
        else:
            historical_counts = [
                int(row["checkpoint/no_improve_checks"])
                for row in history
                if isinstance(row.get("checkpoint/no_improve_checks"), int)
            ]
            resume_no_improve_checks = max(historical_counts[-1] if historical_counts else 0, 0)
            if historical_counts:
                _rank0_print(
                    f"Reconstructed legacy early-stop counter from history: "
                    f"no_improve_checks={resume_no_improve_checks}"
                )
        _rank0_print(f"Resume-training stage={stage}: checkpoint_epoch={ckpt_epoch}, next_epoch={start_epoch}, target_epochs={epochs}, previous_history_rows={len(history)}, no_improve_checks={resume_no_improve_checks}")
    if args.eval_before_train and val_dl is not None and not args.resume_training and start_epoch == 0:
        _set_sampler_epoch(val_dl, -1)
        init_metrics = _run_epoch(
            model, val_dl, device, stage, loss_weights, None, epoch=-1,
            progress=(not args.no_progress) and _is_main_process(),
            amp=args.amp, amp_dtype=args.amp_dtype, non_blocking_transfer=pin_memory,
            decode_response_traj=include_response_traj, grad_clip=args.grad_clip,
        )
        init_metrics = _distributed_mean_metrics(init_metrics, device)
        init_score, init_kind = _checkpoint_selection_score(init_metrics, stage)
        if _is_main_process():
            row0: dict[str, Any] = {
                "epoch": -1,
                **{f"val/{k}": v for k, v in init_metrics.items()},
                "checkpoint/score": float(init_score),
                "checkpoint/kind": init_kind,
                "checkpoint/improved": True,
                "initial_basis_evaluation": True,
            }
            history.append(row0)
            best_val = float(init_score)
            _atomic_torch_save(
                _make_checkpoint_payload(
                    model, cfg, opt, epoch=-1, stage=stage, best_val=best_val,
                    save_optimizer=not args.no_save_optimizer,
                    extra={"val_" + init_kind: init_score, "initial_basis_evaluation": True},
                ),
                output_dir / f"cowp_{stage}_best.pt",
            )
            _rank0_print("Initial model-facing basis: " + json.dumps(row0, ensure_ascii=False))

    if start_epoch >= epochs:
        _rank0_print(f"Stage {stage} already reached target epochs: checkpoint next_epoch={start_epoch}, target_epochs={epochs}")
        _cleanup_distributed()
        return
    no_improve_checks = int(resume_no_improve_checks)
    early_stop_patience = max(int(args.early_stop_patience), 0)
    min_delta = max(float(args.early_stop_min_delta), 0.0)
    try:
        for epoch in range(start_epoch, epochs):
            freeze_now = stage in {"witness", "planner"} and epoch < max(int(args.freeze_backbone_epochs), 0)
            _set_stage_freeze(
                model,
                stage,
                freeze_now,
                freeze_natural_during_witness=bool(
                    tcfg.get("freeze_natural_during_witness", False)
                ),
                freeze_graph_during_natural=(
                    bool(tcfg.get("freeze_graph_during_natural", False))
                    and (
                        int(
                            args.natural_graph_unfreeze_epoch
                            if args.natural_graph_unfreeze_epoch is not None
                            else tcfg.get("natural_graph_unfreeze_epoch", -1)
                        ) < 0
                        or epoch < int(
                            args.natural_graph_unfreeze_epoch
                            if args.natural_graph_unfreeze_epoch is not None
                            else tcfg.get("natural_graph_unfreeze_epoch", -1)
                        )
                    )
                ),
            )
            if _is_main_process() and stage in {"witness", "planner"} and (epoch == start_epoch or epoch == int(args.freeze_backbone_epochs)):
                _rank0_print(f"{stage} warmup_frozen={freeze_now} at epoch={epoch}")
            if _is_main_process() and stage in {"natural", "representation"}:
                unfreeze_epoch = int(
                    args.natural_graph_unfreeze_epoch
                    if args.natural_graph_unfreeze_epoch is not None
                    else tcfg.get("natural_graph_unfreeze_epoch", -1)
                )
                graph_frozen = bool(tcfg.get("freeze_graph_during_natural", False)) and (
                    unfreeze_epoch < 0 or epoch < unfreeze_epoch
                )
                if epoch == start_epoch or epoch == unfreeze_epoch:
                    _rank0_print(
                        f"natural graph_frozen={graph_frozen} at epoch={epoch}; "
                        f"explicit_unfreeze_epoch={unfreeze_epoch}"
                    )
            _set_sampler_epoch(train_dl, epoch)
            train_metrics = _run_epoch(
                model,
                train_dl,
                device,
                stage,
                loss_weights,
                opt,
                epoch=epoch,
                progress=(not args.no_progress) and _is_main_process(),
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                non_blocking_transfer=pin_memory,
                decode_response_traj=include_response_traj,
                grad_clip=args.grad_clip,
            )
            train_metrics = _distributed_mean_metrics(train_metrics, device)
            row: dict[str, Any] = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}
            do_val = val_dl is not None and int(args.val_every) > 0 and ((epoch + 1) % int(args.val_every) == 0 or epoch == epochs - 1)
            if do_val:
                _set_sampler_epoch(val_dl, epoch)
                val_metrics = _run_epoch(
                    model,
                    val_dl,
                    device,
                    stage,
                    loss_weights,
                    None,
                    epoch=epoch,
                    progress=(not args.no_progress) and _is_main_process(),
                    amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    non_blocking_transfer=pin_memory,
                    decode_response_traj=include_response_traj,
                    grad_clip=args.grad_clip,
                )
                val_metrics = _distributed_mean_metrics(val_metrics, device)
                row.update({f"val/{k}": v for k, v in val_metrics.items()})
                score_loss, score_kind = _checkpoint_selection_score(val_metrics, stage)
                score_name = "val_" + score_kind
                row["checkpoint/score"] = float(score_loss)
                row["checkpoint/kind"] = score_kind
            else:
                # If validation is intentionally disabled for smoke training, still
                # produce cowp_<stage>_best.pt so downstream stage commands can resume.
                if val_dl is None or int(args.val_every) == 0:
                    score_loss, score_kind = _checkpoint_selection_score(train_metrics, stage)
                    score_name = "train_" + score_kind
                    row["checkpoint/score"] = float(score_loss)
                    row["checkpoint/kind"] = score_kind
                else:
                    score_loss = float("inf")
                    score_name = "train_unavailable"
            improved = False
            if _is_main_process():
                improved = math.isfinite(score_loss) and score_loss < (best_val - min_delta)
                if improved:
                    best_val = score_loss
                    no_improve_checks = 0
                    _atomic_torch_save(
                        _make_checkpoint_payload(
                            model,
                            cfg,
                            opt,
                            epoch=epoch,
                            stage=stage,
                            best_val=best_val,
                            save_optimizer=not args.no_save_optimizer,
                            extra={score_name: score_loss},
                        ),
                        output_dir / f"cowp_{stage}_best.pt",
                    )
                elif math.isfinite(score_loss):
                    no_improve_checks += 1
                row["train/lr"] = float(opt.param_groups[0]["lr"])
                row["checkpoint/improved"] = bool(improved)
                row["checkpoint/no_improve_checks"] = int(no_improve_checks)
                history.append(row)
                print(json.dumps(row, ensure_ascii=False))
                save_every = max(int(args.save_every), 1)
                if ((epoch + 1) % save_every == 0) or epoch == epochs - 1:
                    _atomic_torch_save(
                        _make_checkpoint_payload(
                            model,
                            cfg,
                            opt,
                            epoch=epoch,
                            stage=stage,
                            best_val=best_val,
                            save_optimizer=not args.no_save_optimizer,
                        ),
                        output_dir / f"cowp_{stage}_epoch{epoch:03d}.pt",
                    )
            if scheduler is not None:
                if scheduler_mode == "plateau":
                    scheduler.step(float(score_loss) if math.isfinite(score_loss) else float(best_val))
                else:
                    scheduler.step()
            if _is_main_process():
                # Persist a post-scheduler checkpoint every completed epoch.  The
                # launcher prefers this file for interruption recovery, so model,
                # optimizer, LR scheduler, epoch numbering, and early-stop state
                # continue together instead of silently warm-starting from scratch.
                _atomic_torch_save(
                    _make_checkpoint_payload(
                        model, cfg, opt, epoch=epoch, stage=stage, best_val=best_val,
                        save_optimizer=not args.no_save_optimizer, scheduler=scheduler,
                        no_improve_checks=no_improve_checks,
                        extra={score_name: score_loss},
                    ),
                    output_dir / f"cowp_{stage}_last.pt",
                )
                _atomic_write_history(history_path, history)
            stop_now = bool(early_stop_patience > 0 and no_improve_checks >= early_stop_patience) if _is_main_process() else False
            if dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(int(stop_tensor.item()))
            if stop_now:
                _rank0_print(
                    f"Early stopping stage={stage} at epoch={epoch}: "
                    f"no improvement for {no_improve_checks} validation checks; best={best_val:.6f}"
                )
                break
        if _is_main_process():
            _atomic_write_history(history_path, history)
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
