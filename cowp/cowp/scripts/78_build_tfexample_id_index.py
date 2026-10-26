from __future__ import annotations

import argparse
import json
from pathlib import Path

from cowp.core.config import load_config
from cowp.data.build_cache import build_tfexample_id_index


def _glob(cfg: dict, split: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    womd = cfg.get("womd", {})
    s = split.lower()
    if s in {"train", "training"}:
        return womd["tfexample_glob"]
    if s in {"val", "valid", "validation"}:
        return womd.get("validation_tfexample_glob") or womd["tfexample_glob"]
    if s in {"test", "testing"}:
        return womd.get("test_tfexample_glob") or womd["tfexample_glob"]
    raise ValueError(split)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/reuse a WOMD tf.Example scenario-id index for sparse exact-ID Waymax evaluation.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--split", default="validation", choices=["training", "validation", "testing"])
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reuse-if-exists", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    if args.reuse_if_exists and out.is_file() and out.stat().st_size > 0:
        print(json.dumps({"status": "reused", "output": str(out), "bytes": out.stat().st_size}, indent=2))
        return
    cfg = load_config(args.data_config)
    path = _glob(cfg, args.split, args.tfexample_glob)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = build_tfexample_id_index(path, out, progress=not args.no_progress)
    print(json.dumps({"status": "built", "output": str(out), "summary": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
