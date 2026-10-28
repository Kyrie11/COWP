"""Create a locally patched *copy* of the author's nuPlan evaluation script.

The upstream code unconditionally invokes a graphical NuBoard after simulation.
This patch suppresses only the interactive viewer, not the simulation or metrics.
Does not overwrite author's run_nuplan_test.py; no checkpoint is fabricated.
"""
from __future__ import annotations
import argparse,hashlib
from pathlib import Path

MARK='    build_nuboard(builder, simulation_file)'

def patch(source: str) -> str:
    if source.count(MARK)!=1:
        raise ValueError('Unexpected GameFormer run_nuplan_test.py version; require manual review')
    return source.replace(MARK,
        "    # RIDE HEADLESS PATCH: only disable opening the interactive NuBoard GUI.\n"
        "    if __import__('os').environ.get('RIDE_HEADLESS', '0') != '1':\n"
        "        build_nuboard(builder, simulation_file)")

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args(argv)
    if args.source.resolve()==args.dest.resolve():raise ValueError('Source and destination must differ')
    if args.dest.exists():raise FileExistsError('Refusing to overwrite existing generated runner')
    src=args.source.read_text(encoding='utf8');output=patch(src)
    args.dest.parent.mkdir(parents=True,exist_ok=True)
    args.dest.write_text(output,encoding='utf8')
    print(f'Copied headless runner: {args.dest}\nOriginal SHA256={hashlib.sha256(src.encode()).hexdigest()}\nNo planner scoring changes')
if __name__=='__main__':main()
