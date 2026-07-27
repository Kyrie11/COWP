from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cowp.core.config import load_config
from cowp.models.cowp_model import COWPModel


def _normalize(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(str(k).startswith("_orig_mod.") for k in state):
        return {str(k)[len("_orig_mod."):]: v for k, v in state.items()}
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail fast unless a checkpoint exactly matches the current COWP model.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--expected-stage", default=None)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model_cowp_v16.yaml")
    ap.add_argument("--label-config", default="configs/label_cowp_v16.yaml")
    ap.add_argument("--train-config", default="configs/train_cowp_v16.yaml")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise TypeError(f"Checkpoint {args.checkpoint} does not contain a model state_dict")
    cfg = load_config(args.model_config, args.label_config, args.train_config, args.data_config)
    model = COWPModel(cfg)
    model_state = model.state_dict()
    state = _normalize(ckpt["model"])
    missing = sorted(k for k in model_state if k not in state and not k.endswith("num_batches_tracked"))
    unexpected = sorted(k for k in state if k not in model_state)
    shape_mismatch = sorted(
        k for k in state
        if k in model_state and tuple(state[k].shape) != tuple(model_state[k].shape)
    )
    actual_stage = ckpt.get("stage")
    stage_ok = args.expected_stage is None or actual_stage == args.expected_stage
    report = {
        "pass": not missing and not unexpected and not shape_mismatch and stage_ok,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_stage": actual_stage,
        "expected_stage": args.expected_stage,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
        "model_parameter_tensors": len(model_state),
        "checkpoint_parameter_tensors": len(state),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
