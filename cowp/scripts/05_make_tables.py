from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cowp.core.config import load_config
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import module_effect_metrics, stress_acceptance_metrics, witness_table_from_labels
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate main/stress/ablation/witness-quality tables from COWP labels or eval outputs.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/tables")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    labels_dir = args.labels_dir or cfg["outputs"]["labels_dir"]
    methods = cfg.get("eval", {}).get("methods", ["cowp"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_paths = sorted(Path(labels_dir).glob("*.npz"))
    label_dicts = []
    for p in tqdm_iter(label_paths, enabled=True, total=len(label_paths), desc="Load labels for tables", unit="file"):
        with np.load(p, allow_pickle=True) as data:
            label_dicts.append({k: data[k] for k in data.files})
    # v16.8.5: labels are already resident in memory.  The old table script
    # called offline_candidate_eval once per method, re-reading every NPZ from
    # disk N times, then ran the planners again for stress metrics.  Reuse one
    # in-memory label load and one selection pass per method instead.
    rows = []
    stress_rows = []
    method_decisions: dict[str, list[tuple[int, np.ndarray]]] = {}
    for method in methods:
        planner = planner_for_method(method, cfg)
        decisions = []
        selected = []
        for label in label_dicts:
            dec = planner.select_from_labels(label)
            decisions.append((dec.candidate_index, dec.accepted_mask))
            selected.append(int(dec.candidate_index))
        method_decisions[method] = decisions
        from cowp.waymax_eval.metrics_cowp import metrics_from_labels
        rows.append({"Method": method, **metrics_from_labels(selected, label_dicts)})
        stress_rows.append({"Method": method, **stress_acceptance_metrics(decisions, label_dicts)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "main_results.csv", index=False)
    df[["Method", "FSR", "CBS", "OPR", "EP"]].to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(out / "stress_test.csv", index=False)
    witness_rows = [{"Method": "COWP rule certificate", **witness_table_from_labels(label_dicts)}]
    pd.DataFrame(witness_rows).to_csv(out / "witness_quality.csv", index=False)
    module_metrics = module_effect_metrics(label_dicts, cfg, methods=list(methods))
    pd.DataFrame([{"Method": k, **v} for k, v in module_metrics.items()]).to_csv(out / "module_effects.csv", index=False)
    print(df.to_string(index=False))
    print(f"Wrote module-effect diagnostics to {out / 'module_effects.csv'}")


if __name__ == "__main__":
    main()
