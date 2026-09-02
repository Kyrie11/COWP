from __future__ import annotations

import argparse
from typing import Any

from ocrap.config import apply_overrides, load_config
from ocrap.data.build.builder import build_dataset
from ocrap.data.build.diagnose import diagnose_dataset
from ocrap.data.build.papercheck import papercheck_dataset
from ocrap.evaluation.evaluator import evaluate
from ocrap.simulation.closed_loop_runner import closed_loop_evaluate
from ocrap.analysis.dataset_report import analyze_dataset

from .calibrate import calibrate
from .deploy import deploy
from .train import train
from ocrap.external_baselines import train_external_baseline, evaluate_external_baselines


def add_common(p: argparse.ArgumentParser) -> None:
    # Use SUPPRESS so options supplied before the subcommand are not overwritten
    # by the subparser defaults, while options after the subcommand still work.
    p.add_argument("--config", default=argparse.SUPPRESS, help="YAML config path. Defaults to configs/default.yaml.")
    p.add_argument("--set", action="append", default=argparse.SUPPRESS, help="Override config with dotted.path=value. Can be repeated.")
    p.add_argument("--without-observation-kernel", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--without-lower-tail", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--without-calibration", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--without-anti-oracle", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--full-future-roots", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--no-occlusion-bev", action="store_true", default=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocrap", description="OC-RAP dataset/model/evaluation pipeline")
    add_common(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-dataset")
    add_common(p)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--skip-existing",
        "--resume",
        dest="skip_existing",
        action="store_true",
        help="Skip already materialized sample .npz files in the output directory and resume safely.",
    )
    p.add_argument(
        "--adopt-resume-contract",
        action="store_true",
        help=(
            "Explicitly adopt a legacy partial output that predates resume_contract.json. "
            "Use once with --resume and the original semantic build parameters."
        ),
    )
    p = sub.add_parser("diagnose")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p = sub.add_parser("papercheck")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p = sub.add_parser("train")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-dataset", default=None, help="Optional explicit validation/calibration dataset root(s) for best checkpoint selection.")
    p = sub.add_parser("calibrate")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=False, default=None)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=False, default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p = sub.add_parser("train-baseline")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-dataset", default=None)
    p.add_argument("--baseline", default=None, help="External baseline name, e.g. route_bc_lite or gameformer_lite.")
    p = sub.add_parser("evaluate-baseline")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=False, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--baselines", default=None, help="Comma-separated external baselines to evaluate.")
    p = sub.add_parser("closed-loop")
    add_common(p)
    p.add_argument("--dataset", required=True, help="WOMD TFRecord path/pattern for Waymax closed-loop evaluation.")
    p.add_argument("--checkpoint", required=False, default=None)
    p.add_argument("--output", required=True)
    p = sub.add_parser("analyze-dataset")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p = sub.add_parser("deploy")
    add_common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=False, default=None)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--time-index", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", default=None)
    p.add_argument("--delta", type=float, default=None)
    return parser


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(getattr(args, "config", None))
    cfg = apply_overrides(cfg, getattr(args, "set", None))
    ab = cfg.setdefault("ablation", {})
    for attr, key in [
        ("without_observation_kernel", "without_observation_kernel"),
        ("without_lower_tail", "without_lower_tail"),
        ("without_calibration", "without_calibration"),
        ("without_anti_oracle", "without_anti_oracle"),
        ("full_future_roots", "full_future_roots"),
        ("no_occlusion_bev", "no_occlusion_bev"),
    ]:
        if getattr(args, attr, False):
            ab[key] = True
    if ab.get("no_occlusion_bev", False):
        cfg.setdefault("model", {})["no_occlusion_bev"] = True
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    cfg = build_cfg(args)
    if args.cmd == "build-dataset":
        result = build_dataset(
            args.output,
            cfg,
            skip_existing=args.skip_existing,
            adopt_legacy_resume=bool(getattr(args, "adopt_resume_contract", False)),
        )
    elif args.cmd == "diagnose":
        result = diagnose_dataset(args.dataset, args.output, args.max_samples, cfg=cfg)
    elif args.cmd == "papercheck":
        result = papercheck_dataset(args.dataset, args.output, args.max_samples)
    elif args.cmd == "train":
        result = train(args.dataset, args.output, cfg, val_dataset=getattr(args, "val_dataset", None))
    elif args.cmd == "calibrate":
        result = calibrate(args.dataset, args.checkpoint, args.output, cfg)
    elif args.cmd == "train-baseline":
        result = train_external_baseline(args.dataset, args.output, cfg, val_dataset=getattr(args, "val_dataset", None), baseline=getattr(args, "baseline", None))
    elif args.cmd == "evaluate-baseline":
        result = evaluate_external_baselines(args.dataset, args.output, cfg, split=getattr(args, "split", "test"), checkpoint=getattr(args, "checkpoint", None), baselines=getattr(args, "baselines", None))
    elif args.cmd == "evaluate":
        result = evaluate(args.dataset, args.checkpoint, args.output, split=args.split, calibration_json=args.calibration, cfg=cfg)
    elif args.cmd == "closed-loop":
        result = closed_loop_evaluate(args.dataset, args.checkpoint, args.output, cfg)
    elif args.cmd == "analyze-dataset":
        result = analyze_dataset(args.dataset, args.output, max_samples=args.max_samples, plots=not args.no_plots)
    elif args.cmd == "deploy":
        result = deploy(args.dataset, args.checkpoint, args.scene_id, args.time_index, args.output, calibration_json=args.calibration, delta=args.delta, cfg=cfg)
    else:
        raise AssertionError(args.cmd)
    if args.cmd == "closed-loop" and isinstance(result, dict):
        # The full scene records are already persisted to RESULT.json and the
        # append-only scene journal. Printing them again can create multi-GB
        # launcher logs and can make a completed metric-only run look hung.
        print({
            "event": "closed_loop_complete",
            "output": getattr(args, "output", None),
            "method": result.get("method"),
            "source": result.get("source"),
            "num_scenes": result.get("num_scenes"),
            "num_decisions": result.get("num_decisions"),
            "bucket_target_count": result.get("bucket_target_count"),
            "run_fingerprint": result.get("run_fingerprint"),
            "metrics_valid": result.get("metrics_valid"),
            "scene_storage_detail": result.get("scene_storage_detail"),
            "scene_journal_detail": result.get("scene_journal_detail"),
        })
    else:
        print(result)


if __name__ == "__main__":
    main()
