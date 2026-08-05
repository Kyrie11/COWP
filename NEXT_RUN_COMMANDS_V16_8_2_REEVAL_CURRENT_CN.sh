#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUN="${SOURCE_RUN:-outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_2_reeval_v9base_seed2026}"
MANIFEST="$SOURCE_RUN/configs/natural_attribution_transfer_manifest.json"
[[ -s "$MANIFEST" ]] || { echo "missing $MANIFEST" >&2; exit 2; }
readarray -t NATURAL_PATHS < <("$PYTHON_BIN" - "$MANIFEST" <<'PY2'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(x['natural_checkpoint'])
print(x['natural_history'])
PY2
)
export NATURAL_CKPT="${NATURAL_CKPT:-${NATURAL_PATHS[0]}}"
export NATURAL_HISTORY="${NATURAL_HISTORY:-${NATURAL_PATHS[1]}}"
export TRANSPORT_CKPT="${TRANSPORT_CKPT:-$SOURCE_RUN/checkpoints/transport/cowp_witness_best.pt}"
export CKPT="${CKPT:-$SOURCE_RUN/checkpoints/planner/cowp_planner_best.pt}"
for p in "$NATURAL_CKPT" "$NATURAL_HISTORY" "$TRANSPORT_CKPT" "$CKPT"; do
  [[ -s "$p" ]] || { echo "missing source artifact: $p" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT/configs"
"$PYTHON_BIN" - "$NATURAL_CKPT" "$NATURAL_HISTORY" "$TRANSPORT_CKPT" "$CKPT" "$SOURCE_RUN" "$OUT_ROOT/configs/reused_checkpoint_manifest.json" <<'PY2'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()
paths=list(map(Path,sys.argv[1:5]))
out=Path(sys.argv[6])
payload={'schema_version':'cowp_v16_8_2_checkpoint_reeval_v1','source_run':sys.argv[5],
         'artifacts':{k:{'path':str(p),'sha256':sha(p)} for k,p in zip(
             ['natural_checkpoint','natural_history','transport_checkpoint','planner_checkpoint'],paths)}}
out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
PY2
# This run isolates selector/metric/priority/fallback fixes from retraining.
export RUN_DIAGNOSE=0 RUN_NATURAL=0 RUN_TRANSPORT=0 RUN_PLANNER=0
export RUN_OFFLINE=1 RUN_PROBE=0 RUN_FULL=0 RUN_PAIRMAX_ABLATION=0 RUN_PARETO_ABLATION=0
export FORCE_TRAIN=0 FORCE_EVAL=1 STOP_AFTER_STAGE=offline
export REQUIRE_INIT_CKPT=0 REQUIRE_WAYMAX_PREFLIGHT=0 BACKGROUND="${BACKGROUND:-0}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_CN.sh"
