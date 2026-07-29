#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def gamma(path: str, delta: str | None = None) -> float:
    data = json.loads(Path(path).read_text())
    if delta is not None:
        thresholds = data.get("thresholds", {})
        key = str(delta)
        if key in thresholds:
            return float(thresholds[key])
        # tolerate CLI values such as 0.050 while JSON stores 0.05
        try:
            target = float(delta)
            for k, v in thresholds.items():
                if abs(float(k) - target) < 1e-12:
                    return float(v)
        except Exception:
            pass
    return float(data["gamma_rec"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Create gamma_rec_by_bucket JSON from per-regime calibration files.")
    ap.add_argument("--safe", required=True)
    ap.add_argument("--near", required=True)
    ap.add_argument("--contact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--delta", default=None, help="Optional calibration delta key, e.g. 0.01, 0.05, or 0.1. Defaults to each file's gamma_rec.")
    args = ap.parse_args()
    mapping = {
        "test_safe": gamma(args.safe, args.delta),
        "safe": gamma(args.safe, args.delta),
        "test_near_contact": gamma(args.near, args.delta),
        "near_contact": gamma(args.near, args.delta),
        "test_contact": gamma(args.contact, args.delta),
        "contact": gamma(args.contact, args.delta),
    }
    out = {"gamma_rec_by_bucket": mapping}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
