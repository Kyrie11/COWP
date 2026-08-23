from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read_ids(path: str | Path) -> set[str]:
    out: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
                sid = str(row.get("scenario_id", row.get("id", ""))).strip()
            except Exception:
                sid = ""
        else:
            sid = line.split()[0]
        if sid:
            out.add(sid)
    return out


def _index_ids(path: str | Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        sid = str(row.get("scenario_id", "")).strip()
        if sid and sid not in seen:
            ids.append(sid)
            seen.add(sid)
    return ids


def _digest(ids: list[str]) -> str:
    h = hashlib.sha256()
    for sid in ids:
        h.update(sid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create a deterministic content-blind final holdout from a Scenario location index. "
            "Selection uses scenario IDs only; it never decodes scenario content."
        )
    )
    ap.add_argument("--scenario-index-jsonl", required=True)
    ap.add_argument("--exclude-ids", action="append", default=[], help="txt/JSONL ID file to exclude; repeatable")
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--seed", default="cowp-v16.8.25-final-blind-20260823")
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    if int(args.count) <= 0:
        raise ValueError("--count must be positive")
    all_ids = _index_ids(args.scenario_index_jsonl)
    excluded: set[str] = set()
    for path in args.exclude_ids:
        excluded.update(_read_ids(path))
    eligible = [sid for sid in all_ids if sid not in excluded]
    ranked = sorted(
        eligible,
        key=lambda sid: (hashlib.sha256(f"{args.seed}|{sid}".encode("utf-8")).hexdigest(), sid),
    )
    if len(ranked) < int(args.count):
        raise ValueError(f"only {len(ranked)} eligible IDs remain; requested {args.count}")
    selected = ranked[: int(args.count)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "cowp_v16_8_25_final_blind_split_v1",
        "selection_policy": "scenario_id_hash_only_no_scenario_content_decoding",
        "scenario_index_jsonl": str(Path(args.scenario_index_jsonl).resolve()),
        "exclude_id_files": [str(Path(p).resolve()) for p in args.exclude_ids],
        "seed": str(args.seed),
        "requested_count": int(args.count),
        "index_unique_ids": len(all_ids),
        "excluded_unique_ids": len(excluded),
        "eligible_unique_ids": len(eligible),
        "selected_count": len(selected),
        "selected_ids_sha256": _digest(selected),
        "freeze_contract": "Do not decode/evaluate these scenarios until algorithm, training recipe, thresholds, and checkpoint-selection policy are frozen.",
    }
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
