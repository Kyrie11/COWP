from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import torch
import yaml

from cowp.models.losses import natural_loss
from cowp.models.natural_decoder import NaturalDecoder
from cowp.core.constants import NaturalSource

PIPELINE_MODULES = (
    "cowp.scripts.03_train",
    "cowp.scripts.04_eval_closed_loop",
    "cowp.scripts.17_merge_waymax_shards",
    "cowp.scripts.24_summarize_planner_delta",
    "cowp.scripts.25_verify_mechanism_effect",
    "cowp.scripts.30_diagnose_bcot_result",
    "cowp.scripts.31_calibrate_bcot_budget",
    "cowp.scripts.32_gate_natural_basis",
    "cowp.scripts.35_diagnose_model_anchor",
    "cowp.scripts.39_diagnose_learned_natural",
    "cowp.scripts.40_gate_natural_effectiveness",
)


def _realistic_natural_smoke() -> dict[str, object]:
    torch.manual_seed(2026)
    b, a, m, r, t = 2, 6, 24, 24, 8
    decoder = NaturalDecoder(d_model=16, modes=m, future_steps=t, decoder_type="typed_causal_dynamics")
    z = torch.randn(b, a + 1, 16)
    idx = torch.arange(a).view(1, a).expand(b, -1)
    anchor = torch.zeros(b, a, 7)
    anchor[..., 3] = 5.0
    pred = decoder(z, idx, anchor7=anchor, dt=0.1)
    source = torch.tensor([0] * 8 + [1] * 8 + [2] * 8).view(1, 1, r).expand(b, a, -1)
    batch = {
        "cowp/natural/traj": pred["base_traj"].detach().clone(),
        "cowp/natural/valid": torch.ones(b, a, r, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(b, a, r),
        "cowp/natural/source": source,
        "cowp/natural/priority_preserved": (source == int(NaturalSource.PRIO)).float(),
        "cowp/critical/valid": torch.ones(b, a, dtype=torch.bool),
    }
    losses = natural_loss(pred, batch, {"natural_dt": 0.1, "natural_mode_usage": 0.03})
    losses["loss"].backward()
    finite = all(bool(torch.isfinite(v).all()) for v in losses.values() if torch.is_tensor(v))
    return {
        "pass": finite,
        "shape": {"batch": b, "critical_agents": a, "modes": m, "roots": r, "steps": t},
        "loss": float(losses["loss"].detach()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-fast environment and stage-contract preflight for COWP.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--require-waymax", action="store_true")
    ap.add_argument("--model-config", default="configs/model_cowp_v16.yaml")
    ap.add_argument("--label-config", default="configs/label_cowp_v16.yaml")
    ap.add_argument("--train-config", default="configs/train_cowp_v16.yaml")
    ap.add_argument("--eval-config", default="configs/eval_cowp_v16.yaml")
    args = ap.parse_args()

    errors: list[str] = []
    configs: dict[str, object] = {}
    for name, path_s in {
        "model": args.model_config,
        "label": args.label_config,
        "train": args.train_config,
        "eval": args.eval_config,
    }.items():
        path = Path(path_s)
        try:
            configs[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"config {name} unreadable: {path}: {exc}")

    imports: dict[str, str] = {}
    for module in PIPELINE_MODULES:
        try:
            importlib.import_module(module)
            imports[module] = "ok"
        except Exception as exc:  # noqa: BLE001
            imports[module] = f"error: {exc}"
            errors.append(f"import failed: {module}: {exc}")

    optional = {name: importlib.util.find_spec(name) is not None for name in ("jax", "tensorflow", "waymax")}
    if args.require_waymax:
        for name, present in optional.items():
            if not present:
                errors.append(f"required Waymax dependency is missing: {name}")

    cuda_ok = bool(torch.cuda.is_available())
    if args.require_cuda and not cuda_ok:
        errors.append("CUDA is required by the selected pipeline but torch.cuda.is_available() is false")
    if shutil.which("torchrun") is None:
        errors.append("torchrun executable is not available on PATH")

    try:
        natural_smoke = _realistic_natural_smoke()
        if not natural_smoke["pass"]:
            errors.append("realistic natural forward/loss/backward produced non-finite values")
    except Exception as exc:  # noqa: BLE001
        natural_smoke = {"pass": False, "error": repr(exc)}
        errors.append(f"realistic natural forward/loss/backward failed: {exc}")

    report = {
        "pass": not errors,
        "errors": errors,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": cuda_ok,
            "cuda_device_count": int(torch.cuda.device_count()),
            "torchrun": shutil.which("torchrun"),
            "dependencies": optional,
        },
        "imports": imports,
        "configs_loaded": sorted(configs),
        "realistic_natural_smoke": natural_smoke,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
