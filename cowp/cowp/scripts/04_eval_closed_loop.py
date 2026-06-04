from __future__ import annotations

import argparse
import json
from pathlib import Path

from cowp.core.config import load_config
from cowp.waymax_eval.rollout import offline_candidate_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate COWP labels/planner and ablations; Waymax rollout can be plugged via waymax_eval.rollout.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--method", default="cowp")
    ap.add_argument("--output", default="outputs/eval_metrics.json")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    metrics = offline_candidate_eval(args.labels_dir or cfg["outputs"]["labels_dir"], cfg, method=args.method)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({args.method: metrics}, f, indent=2)
    print(json.dumps({args.method: metrics}, indent=2))


if __name__ == "__main__":
    main()
