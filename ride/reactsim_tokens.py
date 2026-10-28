"""Read-only and auditable ReactSim scenario-token / nuPlan sqlite consistency check.

No dataset changes; no pickle loading. A raw lidar_pc token is not necessarily a
valid nuPlan scenario: nuPlan's get_scenarios_from_db filters boundary scenes and
requires joins to sensor/log/map metadata. We distinguish these cases explicitly.

Scientific rule: never silently discard missing tokens from the official test list.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

HEX_TOKEN = re.compile(r"^[0-9a-fA-F]{2,64}$")


def _as_blob(token: str) -> bytes:
    if len(token) % 2 or not HEX_TOKEN.fullmatch(token):
        raise ValueError("Token must be an even-length hexadecimal lidar_pc token")
    return bytes.fromhex(token)


def _open_readonly(db: Path) -> sqlite3.Connection:
    if not db.is_file():
        raise FileNotFoundError(str(db))
    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _query(con: sqlite3.Connection, sql: str, args=()):
    return con.execute(sql, args).fetchall()


def check_one(con: sqlite3.Connection, token: str) -> dict[str, Any]:
    """Classify one ID against a DB without invoking heavy nuPlan dependencies."""
    blob = _as_blob(token)
    row = con.execute(
        "SELECT token, timestamp, scene_token, lidar_token FROM lidar_pc WHERE token=?",
        (blob,),
    ).fetchone()
    result: dict[str, Any] = {"token": token.lower()}
    if row is None:
        return {**result, "status": "RAW_TOKEN_ABSENT",
                "message": "The supplied sqlite lidar_pc table does not contain this token."}
    result["timestamp_us"] = int(row["timestamp"])
    scene_token = row["scene_token"]
    result["scene_token"] = scene_token.hex() if scene_token is not None else None
    if scene_token is None:
        return {**result, "status": "MISSING_SCENE_LINK"}
    scene = con.execute("SELECT name FROM scene WHERE token=?", (scene_token,)).fetchone()
    if scene is None:
        return {**result, "status": "MISSING_SCENE_ROW"}
    result["scene_name"] = str(scene["name"])
    # Same ordering and boundary predicate as the public nuPlan-devkit query:
    # ROW_NUMBER() OVER (ORDER BY scene.name ASC) >=3 AND row_num < count-1.
    idx = con.execute("SELECT COUNT(*) FROM scene WHERE name < ?", (scene["name"],)).fetchone()[0] + 1
    total = con.execute("SELECT COUNT(*) FROM scene").fetchone()[0]
    result["scene_order_1based"] = int(idx)
    result["scene_count"] = int(total)
    if idx < 3 or idx >= total - 1:
        result["status"] = "FILTERED_SCENE_BOUNDARY"
        result["message"] = "NuPlan excludes this scene near the beginning/end of this log."
        return result
    meta = con.execute(
        "SELECT l.map_version FROM lidar AS ld INNER JOIN log AS l "
        "ON ld.log_token=l.token WHERE ld.token=?", (row["lidar_token"],)
    ).fetchone()
    if meta is None:
        result["status"] = "MISSING_LIDAR_LOG_JOIN"
        return result
    result["map_version"] = meta["map_version"]
    if meta["map_version"] is None:
        return {**result, "status": "MISSING_MAP_VERSION"}
    # A near-exact minimal join sufficient to detect raw-vs-query mismatch.
    # The official query additionally LEFT JOINs scenario_tag and GROUP BYs.
    result["status"] = "ELIGIBLE_BY_SQL_JOINS"
    return result


def inspect_token(db: Path, token: str, quick_check: bool = False) -> dict[str, Any]:
    out = {"db": str(db), "db_exists": db.is_file(), "token": token.lower()}
    if not db.is_file():
        return {**out, "status": "DB_MISSING"}
    out["db_bytes"] = db.stat().st_size
    try:
        with _open_readonly(db) as con:
            if quick_check:
                res = con.execute("PRAGMA quick_check(1)").fetchone()[0]
                out["sqlite_quick_check"] = str(res)
                if res != "ok":
                    return {**out, "status": "SQLITE_INTEGRITY_FAIL"}
            return {**out, **check_one(con, token)}
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        return {**out, "status": "SQLITE_QUERY_ERROR", "error": str(exc)}
    except ValueError as exc:
        return {**out, "status": "MALFORMED_TOKEN", "error": str(exc)}


def audit_ids(db_dir: Path, ids: Path, max_tokens: int = 0, max_examples: int = 20) -> dict[str, Any]:
    """Audit authoritative scenario IDs without skipping or rewriting entries."""
    raw = ids.read_bytes()
    data = json.loads(raw)
    groups = data.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Expected official grouped ReactSim scenario IDs JSON with 'groups'")
    if max_tokens < 0:
        raise ValueError("max_tokens cannot be negative")
    count = collections.Counter()
    examples: list[dict[str, Any]] = []
    processed = 0
    total = 0
    for group in groups:
        total += len(group["scenario_tokens"])
    stop = False
    for group in groups:
        if stop:
            break
        name = group["log_name"]
        tokens = group["scenario_tokens"]
        db = db_dir / f"{name}.db"
        if not db.is_file():
            take = min(len(tokens), max_tokens-processed) if max_tokens else len(tokens)
            count["DB_MISSING"] += take
            if len(examples) < max_examples:
                examples.append({"db": str(db), "token": tokens[0], "status": "DB_MISSING"})
            processed += take
        else:
            try:
                with _open_readonly(db) as con:
                    for token in tokens:
                        if max_tokens and processed >= max_tokens:
                            stop = True
                            break
                        try:
                            detail = check_one(con, token)
                        except (sqlite3.DatabaseError, ValueError) as exc:
                            detail = {"status": "SQLITE_QUERY_ERROR", "token": token, "error": str(exc)}
                        status = detail["status"]
                        count[status] += 1
                        processed += 1
                        if status != "ELIGIBLE_BY_SQL_JOINS" and len(examples) < max_examples:
                            examples.append({"db": str(db), **detail})
            except sqlite3.DatabaseError as exc:
                take = min(len(tokens), max_tokens-processed) if max_tokens else len(tokens)
                count["SQLITE_QUERY_ERROR"] += take
                processed += take
                if len(examples) < max_examples:
                    examples.append({"db": str(db), "status": "SQLITE_QUERY_ERROR", "error": str(exc)})
        if max_tokens and processed >= max_tokens:
            stop = True
    complete = processed == total
    return {"source_ids": str(ids), "ids_sha256": hashlib.sha256(raw).hexdigest(),
            "db_dir": str(db_dir), "total_ids": total, "checked_ids": processed,
            "full_list_audited": complete, "status_counts": dict(count),
            "examples_of_failures": examples,
            "eligible_for_full_official_preprocess":
                complete and count.get("ELIGIBLE_BY_SQL_JOINS", 0) == total,
            "warning": ("SQL joins are a necessary check, not a guarantee that 91-frame tensors or "
                        "the original nuPlan query succeed. Never publish a subset as full ReactSim-Bench.")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, help="Exact problematic .db path")
    p.add_argument("--token", help="Specific lidar_pc hex token")
    p.add_argument("--ids", type=Path, help="Official .scenario_ids_val/train.json file")
    p.add_argument("--db-dir", type=Path, help="Staged nuPlan trainval directory")
    p.add_argument("--max-tokens", type=int, default=0, help="0 audits the entire ID list")
    p.add_argument("--quick-check", action="store_true", help="Check sqlite integrity for --db")
    p.add_argument("--output", type=Path)
    p.add_argument("--enforce", action="store_true", help="Exit nonzero on mismatched/unverified IDs")
    args = p.parse_args(argv)
    if args.db is not None and args.token is not None and not args.ids:
        result = inspect_token(args.db, args.token, args.quick_check)
        ok = result["status"] == "ELIGIBLE_BY_SQL_JOINS"
    elif args.ids is not None and args.db_dir is not None and not args.db:
        result = audit_ids(args.db_dir, args.ids, args.max_tokens)
        ok = result["eligible_for_full_official_preprocess"]
    else:
        p.error("Either --db PATH --token HEX or --ids PATH --db-dir PATH is required")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.enforce and not ok:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
