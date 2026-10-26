#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
OUT="${1:-$RIDE_HOME/RIDE_SCHEMATIC_OUTPUTS_$(date +%Y%m%d).tar.gz}"
# Includes diagnostics and results, never raw WOMD/nuPlan data.
python - "$RIDE_OUTPUT_ROOT" "$OUT" <<'PY'
import pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2])
if not root.is_dir():raise SystemExit('No output directory: '+str(root))
allow={'.json','.jsonl','.log','.csv','.txt'}
files=[]
for d in ('diagnostics','carla','waymax','womd','frozen'):
    for p in (root/d).rglob('*'):
        if not p.is_file() or p.suffix not in allow:continue
        if p.stat().st_size>100*1024*1024:continue
        if p.name=='cohort_manifest.json':
            # May contain sensitive trajectories or observational future information
            continue
        files.append(p)
with tarfile.open(dst,'w:gz') as t:
    for p in files:t.add(p,arcname=p.relative_to(root))
print('Bundled',len(files),'diagnostic/output files ->',dst)
PY
