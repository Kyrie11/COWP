from __future__ import annotations

import os
from typing import Any

import torch


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_or_cfg(env_name: str, cfg: dict[str, Any], key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    return cfg.get(key, default) if value is None or value == "" else value


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


def resolve_amp_dtype(training_cfg: dict[str, Any], device: torch.device) -> torch.dtype:
    """Resolve the effective AMP dtype without assuming native BF16 support.

    ``auto`` prefers BF16 on GPUs that report native support and otherwise uses
    FP16.  An explicit BF16 request is also downgraded to FP16 when the current
    CUDA device cannot execute BF16 kernels.  This avoids opaque cuDNN/SDPA
    failures on pre-Ampere or mismatched CUDA stacks.
    """

    requested = str(_env_or_cfg("OCRAP_AMP_DTYPE", training_cfg, "amp_dtype", "auto")).strip().lower()
    if device.type != "cuda":
        return torch.bfloat16 if requested in {"bf16", "bfloat16"} else torch.float16

    bf16_supported = False
    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False

    if requested in {"auto", "native"}:
        return torch.bfloat16 if bf16_supported else torch.float16
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16 if bf16_supported else torch.float16
    if requested in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(
        f"Unsupported external_baselines.training.amp_dtype={requested!r}; "
        "expected auto, bfloat16/bf16, or float16/fp16"
    )


def configure_cuda_runtime(
    training_cfg: dict[str, Any],
    device: torch.device,
    *,
    log: bool = False,
) -> dict[str, Any]:
    """Configure CUDA matmul and SDPA backends for external baseline models.

    PyTorch enables multiple scaled-dot-product-attention implementations and
    chooses one at runtime.  Some PyTorch/cuDNN/GPU combinations incorrectly
    select cuDNN attention for Transformer masks/shapes for which cuDNN cannot
    build an execution plan.  ``safe`` therefore disables only cuDNN SDPA while
    retaining FlashAttention, memory-efficient attention, and the math fallback.

    Modes can be selected with ``external_baselines.training.sdpa_backend`` or
    ``OCRAP_SDPA_BACKEND``:
      * safe (default): flash + memory-efficient + math; cuDNN disabled
      * auto: all available backends enabled
      * math: deterministic compatibility fallback; fused SDPA disabled
      * flash: flash + math fallback; cuDNN/memory-efficient disabled
    """

    if device.type != "cuda":
        return {
            "device": device.type,
            "sdpa_backend": "cpu",
            "amp_dtype": str(resolve_amp_dtype(training_cfg, device)).replace("torch.", ""),
        }

    allow_tf32 = _as_bool(_env_or_cfg("OCRAP_ALLOW_TF32", training_cfg, "allow_tf32", True), True)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = _as_bool(training_cfg.get("cudnn_benchmark", True), True)
    try:
        torch.set_float32_matmul_precision(str(training_cfg.get("matmul_precision", "high")))
    except Exception:
        pass

    mode = str(_env_or_cfg("OCRAP_SDPA_BACKEND", training_cfg, "sdpa_backend", "safe")).strip().lower()
    aliases = {"compat": "safe", "auto_safe": "safe", "pytorch": "safe", "math_only": "math"}
    mode = aliases.get(mode, mode)
    if mode not in {"safe", "auto", "math", "flash"}:
        raise ValueError(
            f"Unsupported external_baselines.training.sdpa_backend={mode!r}; "
            "expected safe, auto, math, or flash"
        )

    desired = {
        "safe": {"cudnn": False, "flash": True, "mem_efficient": True, "math": True},
        "auto": {"cudnn": True, "flash": True, "mem_efficient": True, "math": True},
        "math": {"cudnn": False, "flash": False, "mem_efficient": False, "math": True},
        "flash": {"cudnn": False, "flash": True, "mem_efficient": False, "math": True},
    }[mode]

    setters = {
        "cudnn": "enable_cudnn_sdp",
        "flash": "enable_flash_sdp",
        "mem_efficient": "enable_mem_efficient_sdp",
        "math": "enable_math_sdp",
    }
    for name, attr in setters.items():
        setter = getattr(torch.backends.cuda, attr, None)
        if callable(setter):
            setter(bool(desired[name]))

    enabled: dict[str, bool | None] = {}
    getters = {
        "cudnn": "cudnn_sdp_enabled",
        "flash": "flash_sdp_enabled",
        "mem_efficient": "mem_efficient_sdp_enabled",
        "math": "math_sdp_enabled",
    }
    for name, attr in getters.items():
        getter = getattr(torch.backends.cuda, attr, None)
        try:
            enabled[name] = bool(getter()) if callable(getter) else None
        except Exception:
            enabled[name] = None

    amp_dtype = resolve_amp_dtype(training_cfg, device)
    result = {
        "device": str(device),
        "sdpa_backend": mode,
        "cudnn_sdp": enabled["cudnn"],
        "flash_sdp": enabled["flash"],
        "mem_efficient_sdp": enabled["mem_efficient"],
        "math_sdp": enabled["math"],
        "allow_tf32": allow_tf32,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
    }
    if log:
        print({"event": "external_baseline_cuda_runtime", **result}, flush=True)
    return result
