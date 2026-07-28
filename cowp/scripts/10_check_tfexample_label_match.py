from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.data.build_cache import build_tfexample_id_index


def _glob_for_split(cfg: dict, split: str, explicit_glob: str | None) -> str:
    if explicit_glob:
        return explicit_glob
    womd = cfg.get("womd", {})
    s = split.lower()
    if s in ("train", "training"):
        return womd["tfexample_glob"]
    if s in ("val", "valid", "validation"):
        return womd.get("validation_tfexample_glob") or womd["tfexample_glob"]
    if s == "test":
        return womd["test_tfexample_glob"]
    raise ValueError(f"Unknown split: {split}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check overlap between COWP label npz ids and WOMD tf.Example scenario ids.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--split", default="validation", choices=["train", "training", "val", "valid", "validation", "test"])
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--tfexample-index-jsonl", required=True)
    ap.add_argument("--build-tfexample-index", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cfg = load_config(args.data_config)
    tfexample_glob = _glob_for_split(cfg, args.split, args.tfexample_glob)
    labels_dir = Path(args.labels_dir)
    label_ids = {p.stem for p in labels_dir.glob("*.npz")}
    if not label_ids:
        raise FileNotFoundError(f"No label npz files found in {labels_dir}")
    index_path = Path(args.tfexample_index_jsonl)
    if args.build_tfexample_index or not index_path.exists():
        summary = build_tfexample_id_index(tfexample_glob, index_path, progress=not args.no_progress)
        print("Built tf.Example id index:", json.dumps(summary, ensure_ascii=False))
    indexed_ids: set[str] = set()
    sample_indexed: list[str] = []
    sample_files: list[str] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("scenario_id")
            if sid is not None:
                sid = str(sid)
                indexed_ids.add(sid)
                if len(sample_indexed) < 5:
                    sample_indexed.append(sid)
            fp = row.get("file")
            if fp and len(sample_files) < 3 and str(fp) not in sample_files:
                sample_files.append(str(fp))
    overlap = label_ids & indexed_ids
    missing = sorted(label_ids - indexed_ids)[:20]
    payload = {
        "split": args.split,
        "tfexample_glob": tfexample_glob,
        "labels_dir": str(labels_dir),
        "num_label_ids": len(label_ids),
        "num_indexed_tfexample_ids": len(indexed_ids),
        "overlap": len(overlap),
        "overlap_ratio_labels": len(overlap) / max(len(label_ids), 1),
        "first_label_ids": sorted(label_ids)[:5],
        "first_indexed_ids": sample_indexed,
        "indexed_files_preview": [Path(x).name for x in sample_files],
        "first_missing_label_ids": missing,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not overlap:
        raise RuntimeError("Zero overlap between label ids and tf.Example ids. Use the matching WOMD split/version/glob.")


if __name__ == "__main__":
    main()
