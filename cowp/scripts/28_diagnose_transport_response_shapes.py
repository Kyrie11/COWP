from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _get(z: np.lib.npyio.NpzFile, key: str):
    for candidate in (key, key.replace('/', '__')):
        if candidate in z.files:
            return z[candidate]
    raise KeyError(key)


def main() -> None:
    ap = argparse.ArgumentParser(description='Diagnose v9 response/root tensor shapes in materialized or overlay cache.')
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--max-files', type=int, default=256)
    ap.add_argument('--sidecar-subdir', default='.transport_v9')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    root = Path(args.cache_dir)
    paths = sorted(p for p in root.glob('*.npz') if not p.name.startswith('.'))[: args.max_files]
    shape_counts: dict[str, Counter] = {
        'response_valid': Counter(), 'response_root_index': Counter(),
        'response_is_min_burden': Counter(), 'natural_valid': Counter(), 'natural_traj': Counter(),
    }
    errors = []
    checked = 0
    valid_root_values = []

    for base in paths:
        side = root / args.sidecar_subdir / base.name
        try:
            with np.load(base, allow_pickle=False) as bz:
                rv = _get(bz, 'cowp/response/valid')
                nv = _get(bz, 'cowp/natural/valid')
                nt = _get(bz, 'cowp/natural/traj')
            src = side if side.exists() else base
            with np.load(src, allow_pickle=False) as tz:
                ri = _get(tz, 'cowp/transport/response_root_index')
                mb = _get(tz, 'cowp/transport/response_is_min_burden')

            shape_counts['response_valid'][tuple(rv.shape)] += 1
            shape_counts['response_root_index'][tuple(ri.shape)] += 1
            shape_counts['response_is_min_burden'][tuple(mb.shape)] += 1
            shape_counts['natural_valid'][tuple(nv.shape)] += 1
            shape_counts['natural_traj'][tuple(nt.shape)] += 1
            if tuple(ri.shape) != tuple(rv.shape):
                raise ValueError(f'root_index shape {ri.shape} != response valid shape {rv.shape}')
            if tuple(mb.shape) != tuple(rv.shape):
                raise ValueError(f'min_burden shape {mb.shape} != response valid shape {rv.shape}')
            if ri.size:
                valid_root_values.append((int(ri.min()), int(ri.max()), int(nv.shape[-1])))
                if ri.min() < 0 or ri.max() >= nv.shape[-1]:
                    raise ValueError(f'root index range [{ri.min()}, {ri.max()}] outside [0, {nv.shape[-1]-1}]')
            checked += 1
        except Exception as exc:
            errors.append({'file': base.name, 'error': repr(exc)})

    report = {
        'cache_dir': str(root),
        'files_seen': len(paths),
        'files_checked': checked,
        'error_count': len(errors),
        'shape_counts': {k: {str(shape): n for shape, n in v.items()} for k, v in shape_counts.items()},
        'root_index_ranges_sample': valid_root_values[:20],
        'errors': errors[:50],
        'complete': bool(paths) and checked == len(paths) and not errors,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
