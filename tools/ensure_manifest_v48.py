#!/usr/bin/env python3
"""Create/reconcile OC-RAP manifest.csv metadata from existing NPZ samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifest_repair_v48 import ensure_many


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", action="append", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rebuild-if-stale", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    results = ensure_many(
        args.dataset_root,
        workers=args.workers,
        rebuild_if_stale=args.rebuild_if_stale,
        force=args.force,
    )
    print(json.dumps({"event": "manifest_preflight_complete", "datasets": [r.as_dict() for r in results]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
