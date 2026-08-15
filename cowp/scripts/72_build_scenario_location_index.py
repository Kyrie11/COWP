from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


def _import_tf():
    import tensorflow as tf  # type: ignore
    return tf


def _import_scenario_proto():
    from waymo_open_dataset.protos import scenario_pb2  # type: ignore
    return scenario_pb2


def _resolve(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no Scenario TFRecord files matched: {pattern}")
    return files


def _manifest(files: list[str]) -> tuple[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    h = hashlib.sha256()
    for f in files:
        p = Path(f).resolve()
        st = p.stat()
        row = {"path": str(p), "name": p.name, "size_bytes": int(st.st_size)}
        rows.append(row)
        h.update(str(p).encode("utf-8"))
        h.update(p.name.encode("utf-8"))
        h.update(str(int(st.st_size)).encode("ascii"))
    return h.hexdigest(), rows


def _meta_valid(meta_path: Path, index_path: Path, manifest_sha: str) -> bool:
    if not meta_path.is_file() or not index_path.is_file() or index_path.stat().st_size == 0:
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("schema_version") == "cowp_womd_scenario_location_index_v1"
        and meta.get("source_manifest_sha256") == manifest_sha
        and int(meta.get("records", 0)) > 0
        and int(meta.get("duplicate_scenario_ids", 0)) == 0
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a reusable WOMD Scenario location index containing scenario_id, TFRecord file, and record_index. "
            "Sparse smoke/strict/train-pilot builds can then read only the shards/records that contain requested ids."
        )
    )
    ap.add_argument("--proto-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--meta-output", default=None)
    ap.add_argument("--reuse-if-valid", action="store_true")
    args = ap.parse_args()

    files = _resolve(args.proto_glob)
    output = Path(args.output)
    meta_path = Path(args.meta_output) if args.meta_output else output.with_suffix(output.suffix + ".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha, manifest_rows = _manifest(files)

    if args.reuse_if_valid and _meta_valid(meta_path, output, manifest_sha):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(json.dumps({**meta, "reused": True}, indent=2, ensure_ascii=False))
        return

    tf = _import_tf()
    scenario_pb2 = _import_scenario_proto()
    tmp = output.with_suffix(output.suffix + ".tmp")
    records = 0
    duplicates = 0
    seen: set[str] = set()
    per_file: dict[str, int] = {}
    with tmp.open("w", encoding="utf-8") as out:
        for filename in files:
            file_count = 0
            ds = tf.data.TFRecordDataset([filename])
            for record_index, rec in enumerate(ds):
                raw = bytes(rec.numpy())
                msg = scenario_pb2.Scenario()
                msg.ParseFromString(raw)
                sid = str(msg.scenario_id)
                if sid in seen:
                    duplicates += 1
                else:
                    seen.add(sid)
                out.write(json.dumps({
                    "scenario_id": sid,
                    "file": str(Path(filename).resolve()),
                    "record_index": int(record_index),
                }, ensure_ascii=False) + "\n")
                records += 1
                file_count += 1
            per_file[Path(filename).name] = file_count
    tmp.replace(output)
    meta = {
        "schema_version": "cowp_womd_scenario_location_index_v1",
        "proto_glob": args.proto_glob,
        "output": str(output.resolve()),
        "num_files": len(files),
        "records": records,
        "unique_scenario_ids": len(seen),
        "duplicate_scenario_ids": duplicates,
        "source_manifest_sha256": manifest_sha,
        "source_files": manifest_rows,
        "records_per_file": per_file,
        "reused": False,
        "pass": records > 0 and duplicates == 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if duplicates:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
