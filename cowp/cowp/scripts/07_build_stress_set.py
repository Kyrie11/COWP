from __future__ import annotations

import argparse

from cowp.core.config import load_config
from cowp.label.stress_set import write_stress_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the COWP false-safe stress-set manifest from label npz files.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--label-config", default="configs/label.yaml")
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.label_config, args.data_config)
    labels_dir = args.labels_dir or cfg["outputs"]["labels_dir"]
    output = args.output or cfg["outputs"].get("stress_manifest", "outputs/stress/cowp_stress_manifest.jsonl")
    write_stress_manifest(labels_dir, output)
    print(f"Wrote stress manifest to {output}")


if __name__ == "__main__":
    main()
