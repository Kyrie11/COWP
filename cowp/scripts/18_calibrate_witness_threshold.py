from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a COWP witness threshold on a held-out calibration cache.")
    ap.add_argument("--input", required=True, help="learned_offline JSON containing witness_threshold_sweep")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-ncf-recall", type=float, default=0.90)
    ap.add_argument("--max-fallback", type=float, default=0.25)
    ap.add_argument("--method", default="cowp", help="Method key when the input was produced by --methods shared evaluation.")
    args = ap.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload.get("witness_threshold_sweep", [])
    if isinstance(rows, dict):
        rows = rows.get(args.method, [])
    if not rows:
        raise ValueError("No witness_threshold_sweep was found")

    feasible = [
        r for r in rows
        if float(r.get("LearnedAcceptNCFRecall", 0.0)) >= args.min_ncf_recall
        and float(r.get("FallbackRate", 1.0)) <= args.max_fallback
    ]
    if feasible:
        # Risk first, then burden, then progress and fallback.
        selected = min(
            feasible,
            key=lambda r: (
                float(r.get("FSR", 1.0)),
                float(r.get("HBCR", 1.0)),
                -float(r.get("EP", 0.0)),
                float(r.get("FallbackRate", 1.0)),
            ),
        )
        status = "constraints_satisfied"
    else:
        def score(r: dict) -> float:
            recall_gap = max(0.0, args.min_ncf_recall - float(r.get("LearnedAcceptNCFRecall", 0.0)))
            fallback_gap = max(0.0, float(r.get("FallbackRate", 1.0)) - args.max_fallback)
            return (
                float(r.get("FSR", 1.0))
                + 0.40 * float(r.get("HBCR", 1.0))
                + 2.0 * recall_gap
                + 1.0 * fallback_gap
                - 0.10 * float(r.get("EP", 0.0))
            )
        selected = min(rows, key=score)
        status = "least_violation"
    out = {
        "witness_threshold": float(selected["witness_threshold"]),
        "status": status,
        "selection_metrics": selected,
        "constraints": {"min_ncf_recall": args.min_ncf_recall, "max_fallback": args.max_fallback},
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
