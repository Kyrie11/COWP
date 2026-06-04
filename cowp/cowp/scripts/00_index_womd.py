from __future__ import annotations

import argparse
from pathlib import Path

from cowp.core.config import load_config
from cowp.data.parse_scenario_proto import write_index


def main() -> None:
    ap = argparse.ArgumentParser(description="Index WOMD Scenario proto TFRecords for COWP.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--proto-glob", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.data_config)
    proto_glob = args.proto_glob or cfg["womd"]["scenario_proto_glob"]
    output = args.output or cfg["outputs"]["index_jsonl"]
    write_index(proto_glob, output, limit=args.limit)
    print(f"Wrote index to {output}")


if __name__ == "__main__":
    main()
