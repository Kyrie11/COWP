from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a strict code/config provenance manifest for one experiment root.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--file", action="append", default=[], dest="files")
    ap.add_argument("--data-protocol", default="unknown")
    ap.add_argument("--raw-train-cache", default="")
    ap.add_argument("--raw-val-cache", default="")
    ap.add_argument("--train-cache", default="")
    ap.add_argument("--val-cache", default="")
    ap.add_argument("--strict-existing", action="store_true")
    args = ap.parse_args()

    file_rows: list[dict[str, str | int]] = []
    seen_names: set[str] = set()
    for raw in args.files:
        # Accept ``logical_name=path`` so a candidate config can be hashed from a
        # temporary location without making the experiment signature depend on
        # that temporary absolute path.  Plain paths keep their normalized
        # basename as the logical name for backward-compatible CLI use.
        if "=" in raw:
            logical_name, raw_path = raw.split("=", 1)
            logical_name = logical_name.strip()
        else:
            raw_path = raw
            logical_name = Path(raw).name
        if not logical_name:
            raise ValueError(f"empty logical name in --file={raw!r}")
        if logical_name in seen_names:
            raise ValueError(f"duplicate logical file name: {logical_name}")
        seen_names.add(logical_name)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        file_rows.append({"name": logical_name, "size": path.stat().st_size, "sha256": _sha256(path)})
    signature_payload = {
        "files": file_rows,
        "data_protocol": args.data_protocol,
        "raw_train_cache": args.raw_train_cache,
        "raw_val_cache": args.raw_val_cache,
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        **signature_payload,
        "signature": signature,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if args.strict_existing and old.get("signature") != signature:
            print(json.dumps({
                "pass": False,
                "reason": "experiment root already contains artifacts from a different code/config signature",
                "old_signature": old.get("signature"),
                "new_signature": signature,
                "output": str(args.output),
            }, indent=2))
            raise SystemExit(2)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pass": True, "signature": signature, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
