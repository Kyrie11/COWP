from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = row.get("scenario_id", row.get("id"))
                if sid is not None:
                    ids.add(str(sid))
                    continue
            except Exception:
                pass
            ids.add(line.split()[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Check scenario-id leakage across train/val/test index or id files.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fail-on-overlap", action="store_true")
    args = ap.parse_args()

    groups = {"train": _read_ids(args.train), "val": _read_ids(args.val)}
    if args.test:
        groups["test"] = _read_ids(args.test)
    report = {"counts": {k: len(v) for k, v in groups.items()}, "overlaps": {}}
    names = list(groups)
    total_overlap = 0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ov = sorted(groups[a] & groups[b])
            report["overlaps"][f"{a}__{b}"] = {"count": len(ov), "preview": ov[:20]}
            total_overlap += len(ov)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_overlap and total_overlap:
        raise SystemExit(f"Found {total_overlap} scenario-id overlaps across splits")


if __name__ == "__main__":
    main()
