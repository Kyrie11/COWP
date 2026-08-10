from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-fast contract check for the v16.8.9 strict 400-hard + representative-random probe manifest.")
    ap.add_argument("--ceiling-json", required=True)
    ap.add_argument("--hard-ids", required=True)
    ap.add_argument("--random-ids", required=True)
    ap.add_argument("--union-ids", required=True)
    ap.add_argument("--expected-hard", type=int, default=400)
    ap.add_argument("--expected-random", type=int, default=800)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ceiling_path = Path(args.ceiling_json)
    hard_path = Path(args.hard_ids)
    random_path = Path(args.random_ids)
    union_path = Path(args.union_ids)
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8")) if ceiling_path.is_file() else {}
    hard = _ids(hard_path)
    random = _ids(random_path)
    union = _ids(union_path)
    hard_set, random_set, union_set = set(hard), set(random), set(union)
    expected_union = hard_set | random_set

    checks = {
        "ceiling_exists": ceiling_path.is_file(),
        "hard_file_exists": hard_path.is_file(),
        "random_file_exists": random_path.is_file(),
        "union_file_exists": union_path.is_file(),
        "hard_exact_count": len(hard) == int(args.expected_hard),
        "random_exact_count": len(random) == int(args.expected_random),
        "hard_unique": len(hard_set) == len(hard),
        "random_unique": len(random_set) == len(random),
        "hard_random_disjoint": not bool(hard_set & random_set),
        "union_unique": len(union_set) == len(union),
        "union_exact_set": union_set == expected_union,
        "union_exact_count": len(union) == len(expected_union),
        "union_target_count": len(union) == int(args.expected_hard) + int(args.expected_random),
    }
    ceiling_hard = ceiling.get("hard_scene_ids_path")
    ceiling_random = ceiling.get("representative_random_scene_ids_path")
    checks["ceiling_hard_path_matches"] = bool(ceiling_hard) and Path(str(ceiling_hard)).resolve() == hard_path.resolve()
    checks["ceiling_random_path_matches"] = bool(ceiling_random) and Path(str(ceiling_random)).resolve() == random_path.resolve()
    checks["ceiling_hard_count_matches"] = int(ceiling.get("hard_scene_probe_count", -1)) == len(hard)
    checks["ceiling_random_count_matches"] = int(ceiling.get("representative_random_scene_count", -1)) == len(random)

    # This catches the historical Bash special-variable failure explicitly:
    # $RANDOM expanded to a numeric path such as /home/.../20034.
    checks["random_path_not_numeric_basename"] = not random_path.name.isdigit()
    if ceiling_random:
        checks["ceiling_random_path_not_numeric_basename"] = not Path(str(ceiling_random)).name.isdigit()
    else:
        checks["ceiling_random_path_not_numeric_basename"] = False

    passed = all(checks.values())
    result = {
        "schema_version": "cowp_v16_8_9_strict_probe_manifest_v1",
        "pass": bool(passed),
        "counts": {
            "hard": len(hard),
            "random": len(random),
            "overlap": len(hard_set & random_set),
            "union": len(union),
            "expected_union_from_files": len(expected_union),
            "target_union": int(args.expected_hard) + int(args.expected_random),
        },
        "checks": checks,
        "paths": {
            "ceiling_json": str(ceiling_path.resolve()),
            "hard_ids": str(hard_path.resolve()),
            "random_ids": str(random_path.resolve()),
            "union_ids": str(union_path.resolve()),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
