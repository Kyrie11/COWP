from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _expand(items: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for item in items or []:
        for part in str(item).split(','):
            part = part.strip()
            if not part:
                continue
            matches = sorted(glob.glob(part)) if any(ch in part for ch in '*?[') else []
            out.extend(Path(m) for m in (matches or [part]))
    seen = set()
    uniq = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(v)


def _float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float('nan')


def _key(row: dict[str, Any]) -> tuple[str, int]:
    sid = str(row.get('scenario_id') or row.get('scenario/id') or '')
    k = int(row.get('candidate_index', row.get('candidate', row.get('k', -1))))
    return sid, k


def _read(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                k = _key(row)
                if k[0] and k[1] >= 0:
                    rows[k] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare two Waymax candidate outcome JSONL sets, e.g. step-metric vs final-metric replay.')
    ap.add_argument('--baseline-jsonl', nargs='+', required=True)
    ap.add_argument('--candidate-jsonl', nargs='+', required=True)
    ap.add_argument('--output-json', default=None)
    args = ap.parse_args()

    base = _read(_expand(args.baseline_jsonl))
    cand = _read(_expand(args.candidate_jsonl))
    common = sorted(set(base) & set(cand))
    missing_in_candidate = sorted(set(base) - set(cand))
    extra_in_candidate = sorted(set(cand) - set(base))
    fields = ['rollout_valid', 'collision', 'offroad']
    mismatches: dict[str, list[dict[str, Any]]] = {f: [] for f in fields}
    numeric_absdiff: dict[str, list[float]] = {'steps': [], 'rollout_seconds': [], 'log_divergence': []}
    for key in common:
        b = base[key]
        c = cand[key]
        for f in fields:
            if _bool(b.get(f)) != _bool(c.get(f)):
                mismatches[f].append({'scenario_id': key[0], 'candidate_index': key[1], 'baseline': b.get(f), 'candidate': c.get(f)})
        for f in numeric_absdiff:
            xb = _float(b.get(f))
            xc = _float(c.get(f))
            if np.isfinite(xb) and np.isfinite(xc):
                numeric_absdiff[f].append(abs(xb - xc))

    def stats(vals: list[float]) -> dict[str, float | int | None]:
        if not vals:
            return {'count': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None}
        arr = np.asarray(vals, dtype=float)
        return {'count': int(arr.size), 'mean': float(arr.mean()), 'p50': float(np.percentile(arr, 50)), 'p95': float(np.percentile(arr, 95)), 'max': float(arr.max())}

    scene_mismatch_counts = Counter(m['scenario_id'] for ms in mismatches.values() for m in ms)
    out = {
        'baseline_rows': len(base),
        'candidate_rows': len(cand),
        'common_rows': len(common),
        'missing_in_candidate_rows': len(missing_in_candidate),
        'extra_in_candidate_rows': len(extra_in_candidate),
        'field_mismatch_counts': {f: len(v) for f, v in mismatches.items()},
        'field_mismatch_examples': {f: v[:20] for f, v in mismatches.items()},
        'scenes_with_any_mismatch': len(scene_mismatch_counts),
        'top_mismatch_scenes': scene_mismatch_counts.most_common(20),
        'numeric_absdiff': {f: stats(v) for f, v in numeric_absdiff.items()},
        'exact_safety_label_match': all(len(v) == 0 for v in mismatches.values()) and len(missing_in_candidate) == 0 and len(extra_in_candidate) == 0,
    }
    text = json.dumps(out, indent=2, ensure_ascii=False, allow_nan=True)
    print(text)
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
