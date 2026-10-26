from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge V16.8.45 RCRSO sidecar shard manifests without touching NPZ payloads.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "heldout"])
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    root = Path(args.root)
    records: list[dict] = []
    summaries: list[dict] = []
    for shard in range(args.num_shards):
        mp = root / f"manifest_{args.split}_s{shard}of{args.num_shards}.jsonl"
        sp = root / f"summary_{args.split}_s{shard}of{args.num_shards}.json"
        if not mp.is_file() or not sp.is_file():
            raise SystemExit(f"missing sidecar shard metadata: {mp} or {sp}")
        summaries.append(json.loads(sp.read_text(encoding="utf-8")))
        for line in mp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    files = [str(x["file"]) for x in records]
    if len(files) != len(set(files)):
        raise SystemExit("duplicate sidecar NPZ filename across shards")
    missing = [f for f in files if not (root / args.split / f).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} sidecar payloads; first={missing[:3]}")

    # Sorting is deterministic and does not change the Dataset's payload set.
    records.sort(key=lambda x: (str(x.get("scenario_id", "")), int(x.get("candidate_index", -1)), int(x.get("agent_index", -1)), int(x.get("root_index", -1)), str(x.get("file", ""))))
    merged_manifest = root / f"manifest_{args.split}.jsonl"
    merged_manifest.write_text("\n".join(json.dumps(x, sort_keys=True) for x in records) + ("\n" if records else ""), encoding="utf-8")

    merged_counts: dict[str, int] = {}
    for summary in summaries:
        for key, value in dict(summary.get("counts", {})).items():
            if isinstance(value, (int, float)):
                merged_counts[key] = int(merged_counts.get(key, 0) + int(value))
    scenario_ids = {str(x.get("scenario_id", "")) for x in records}
    candidate_groups = {(str(x.get("scenario_id", "")), int(x.get("candidate_index", -1))) for x in records}
    merged = {
        "version": "V16.8.45",
        "split": args.split,
        "num_shards": args.num_shards,
        "records": len(records),
        "unique_files": len(set(files)),
        "unique_scenarios": len(scenario_ids),
        "complete_candidate_hypothesis_groups_by_builder_contract": len(candidate_groups),
        "counts_sum": merged_counts,
        "contract": {
            "payloads_unchanged": True,
            "complete_hypothesis_groups_required": True,
            "base_compact5k_modified": False,
        },
    }
    (root / f"summary_{args.split}.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(merged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
