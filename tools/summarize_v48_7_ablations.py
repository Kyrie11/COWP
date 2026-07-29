#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows: list[dict] = []
    for exp in sorted(x for x in args.root.iterdir() if x.is_dir()):
        audit = _read(exp / "completion_audit.json")
        for variant in ("balanced", "precision"):
            base = exp / "candidates" / variant
            train = _read(base / "model_v48_trac_sr" / "train_summary.json")
            row: dict = {
                "experiment": exp.name,
                "variant": variant,
                "comparable": bool(audit.get("comparable", False)),
                "best_epoch": train.get("best_epoch"),
                "epochs_completed": train.get("epochs_completed"),
                "best_metric": train.get("best_metric"),
                "best_val_loss": train.get("best_val_loss"),
            }
            for bucket in ("near", "contact"):
                d = _read(base / "calibration" / f"direct_value_risk_{bucket}_v48.json")
                verify = d.get("verify") or {}
                prefix = f"{bucket}_"
                row[prefix + "valid"] = d.get("valid_for_deployment")
                row[prefix + "auc"] = d.get("candidate_positive_auc")
                row[prefix + "risk_harm_auc"] = d.get("candidate_risk_harm_auc", d.get("candidate_harm_auc"))
                row[prefix + "head_harm_auc"] = d.get("candidate_head_harm_auc")
                row[prefix + "rank_corr"] = d.get("candidate_rank_teacher_correlation")
                row[prefix + "top1_corr"] = d.get("unconstrained_group_top1_correlation")
                row[prefix + "top1_acc"] = d.get("positive_group_top1_accuracy")
                row[prefix + "top1_regret"] = d.get("positive_group_top1_regret_mean")
                row[prefix + "rank_margin_auc"] = d.get("top1_correctness_rank_margin_auc")
                row[prefix + "selected"] = verify.get("num_selected")
                row[prefix + "precision_lcb90"] = verify.get("precision_wilson_lcb90")
                row[prefix + "harm_ucb90"] = verify.get("harmful_group_exposure_ucb90")
                row[prefix + "recall"] = verify.get("positive_recall")
            rows.append(row)
    payload = {"root": str(args.root), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
