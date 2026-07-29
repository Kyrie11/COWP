#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    for seed_dir in sorted(args.root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        for variant in ("balanced", "precision"):
            for bucket in ("near", "contact"):
                path = seed_dir / "candidates" / variant / "calibration" / f"direct_value_risk_{bucket}_v48.json"
                d = _read(path)
                verify = d.get("verify") or {}
                rows.append({
                    "seed": seed,
                    "variant": variant,
                    "bucket": bucket,
                    "missing": not bool(d),
                    "valid": bool(d.get("valid_for_deployment", False)),
                    "risk_source": d.get("risk_source"),
                    "candidate_positive_auc": d.get("candidate_positive_auc"),
                    "candidate_risk_harm_auc": d.get("candidate_risk_harm_auc", d.get("candidate_harm_auc")),
                    "candidate_head_harm_auc": d.get("candidate_head_harm_auc"),
                    "rank_correlation": d.get("candidate_rank_teacher_correlation"),
                    "top1_correlation": d.get("unconstrained_group_top1_correlation"),
                    "top1_accuracy": d.get("positive_group_top1_accuracy"),
                    "top1_regret": d.get("positive_group_top1_regret_mean"),
                    "rank_margin_correctness_auc": d.get("top1_correctness_rank_margin_auc"),
                    "verify_selected": verify.get("num_selected"),
                    "verify_precision_lcb90": verify.get("precision_wilson_lcb90"),
                    "verify_harmful_exposure_ucb90": verify.get("harmful_group_exposure_ucb90"),
                    "verify_harmful_selected_ucb90": verify.get("harmful_selected_ucb90"),
                    "verify_positive_recall": verify.get("positive_recall"),
                    "verify_macro_share": verify.get("max_selected_macro_share"),
                })

    aggregate: list[dict] = []
    numeric_fields = (
        "candidate_positive_auc", "candidate_risk_harm_auc", "candidate_head_harm_auc",
        "rank_correlation", "top1_correlation", "top1_accuracy", "top1_regret",
        "rank_margin_correctness_auc", "verify_selected", "verify_precision_lcb90",
        "verify_harmful_exposure_ucb90", "verify_harmful_selected_ucb90",
        "verify_positive_recall", "verify_macro_share",
    )
    for variant in ("balanced", "precision"):
        for bucket in ("near", "contact"):
            subset = [r for r in rows if r["variant"] == variant and r["bucket"] == bucket and not r["missing"]]
            item: dict = {
                "variant": variant,
                "bucket": bucket,
                "num_seeds": len(subset),
                "valid_seed_count": sum(bool(r["valid"]) for r in subset),
            }
            for field in numeric_fields:
                vals = [float(r[field]) for r in subset if r.get(field) is not None]
                if vals:
                    item[field] = _stats(vals)
            aggregate.append(item)

    payload = {"root": str(args.root), "rows": rows, "aggregate": aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
