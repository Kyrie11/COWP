"""Read-only ReactSim custom-behavior/processed-scene file-name coverage audit.

This is a FILE-presence check only, not a model-accuracy/causal validation.
Never unpickles downloaded data. Filenames are compared verbatim by stem;
metadata and nuPlan log/scenario identities require a later native audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compare(behavior_dir, processed_dir, *, sample_limit=20):
    a = Path(behavior_dir)
    b = Path(processed_dir)
    if not a.is_dir():
        raise FileNotFoundError(f'Custom behavior directory missing: {a}')
    if not b.is_dir():
        raise FileNotFoundError(f'Processed scenario directory missing: {b}')
    before = {p.stem for p in a.glob('*.pkl') if p.is_file()}
    after = {p.stem for p in b.glob('*.pkl') if p.is_file()}
    matched = before & after
    missing = before - after
    return {
        'audit_type': 'reactsim_exact_filename_stem_coverage',
        'custom_behavior_files': len(before),
        'processed_scenario_files': len(after),
        'matched_names': len(matched),
        'missing_processed_count': len(missing),
        'custom_coverage_fraction': len(matched) / len(before) if before else None,
        'missing_sample': sorted(missing)[:sample_limit],
        'cannot_claim': ['nuPlan token mapping is correct', 'checkpoints usable',
                         'two ego actions matched', 'simulator valid', 'RQ1/RQ2 supported'],
        'notes': 'Exact filename stem presence only. No pickle deserialization.'
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--behaviors', required=True)
    p.add_argument('--processed', required=True,
                   help='.../nuplan_processed/val/processed_scenarios')
    p.add_argument('--output', help='Save a JSON copy')
    a = p.parse_args(argv)
    data = compare(a.behaviors, a.processed)
    out = json.dumps(data, ensure_ascii=False, indent=2)
    if a.output:
        dst=Path(a.output)
        if dst.exists():
            raise FileExistsError(f'Will not overwrite {dst}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(out, encoding='utf8')
    print(out)


if __name__ == '__main__':
    main()
