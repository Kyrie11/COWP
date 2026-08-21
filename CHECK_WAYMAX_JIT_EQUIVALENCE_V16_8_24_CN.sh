#!/usr/bin/env bash
set -euo pipefail

# Safety gate before enabling JIT in a long full replay.  It compares the exact
# candidate safety labels (rollout_valid/collision/offroad) and rollout step count
# produced by eager vs JIT execution on the same deterministic scene/candidate set.
PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
CACHE_DIR="${WAYMAX_JIT_CHECK_CACHE:-$COWP_ROOT/tensor_cache_train}"
OUT_ROOT="${WAYMAX_JIT_CHECK_ROOT:-$COWP_ROOT/waymax_jit_equivalence_v16_8_24}"
SCENES="${WAYMAX_JIT_CHECK_SCENES:-16}"
MAX_CANDIDATES="${MAX_REPLAY_CANDIDATES:-24}"
HORIZON="${REPLAY_HORIZON:-80}"
GPU="${WAYMAX_JIT_CHECK_GPU:-0}"

[[ -d "$CACHE_DIR" ]] || { echo "Missing cache: $CACHE_DIR" >&2; exit 3; }
mkdir -p "$OUT_ROOT"
EAGER="$OUT_ROOT/eager.jsonl"
JIT="$OUT_ROOT/jit.jsonl"
REPORT="$OUT_ROOT/compare.json"
rm -f "$EAGER" "$JIT" "$REPORT"

common=(
  --data-config configs/data.yaml
  --label-config configs/label_cowp_v16_8.yaml
  --eval-config configs/eval_cowp_v16_8.yaml
  --cache-dir "$CACHE_DIR"
  --state-source cache
  --candidate-selection balanced
  --max-candidates-per-scene "$MAX_CANDIDATES"
  --rollout-horizon-steps "$HORIZON"
  --waymax-device gpu
  --waymax-action-mode absolute_xy_yaw
  --metric-set safety
  --metric-eval-mode step
  --metric-eval-interval 1
  --done-check-interval 1
  --limit-scenes "$SCENES"
  --gc-every-scenes 64
  --profile-detail scene
  --no-resume
)

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "=== eager reference: $SCENES scenes ==="
"$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates "${common[@]}" --outcomes-jsonl "$EAGER" --progress-desc "JIT gate eager"

echo "=== JIT candidate: $SCENES scenes ==="
"$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates "${common[@]}" --outcomes-jsonl "$JIT" --jit-env-step --jit-env-reset --progress-desc "JIT gate jit"

"$PYTHON_BIN" - "$EAGER" "$JIT" "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path

ref_path, jit_path, report_path = map(Path, sys.argv[1:4])

def read(p):
    d={}
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        key=(str(r.get('scenario_id','')), int(r.get('candidate_index',-1)))
        d[key]=r
    return d

a=read(ref_path); b=read(jit_path)
keys=sorted(set(a)|set(b))
missing_ref=[k for k in keys if k not in a]
missing_jit=[k for k in keys if k not in b]
mismatches=[]
for k in keys:
    if k not in a or k not in b:
        continue
    ra,rb=a[k],b[k]
    fields=('rollout_valid','collision','offroad','steps')
    diff={f:(ra.get(f),rb.get(f)) for f in fields if ra.get(f)!=rb.get(f)}
    # A different error outcome also means the execution paths are not equivalent.
    if bool(ra.get('error')) != bool(rb.get('error')):
        diff['error_presence']=(ra.get('error'),rb.get('error'))
    if diff:
        mismatches.append({'key':list(k),'diff':diff})

payload={
    'schema_version':'cowp_v16_8_24_waymax_jit_equivalence_v1',
    'reference_rows':len(a),
    'jit_rows':len(b),
    'missing_in_reference':len(missing_ref),
    'missing_in_jit':len(missing_jit),
    'label_or_step_mismatches':len(mismatches),
    'mismatch_examples':mismatches[:20],
}
payload['pass']=(len(a)>0 and len(b)>0 and not missing_ref and not missing_jit and not mismatches)
report_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
if not payload['pass']:
    raise SystemExit('JIT equivalence FAILED: keep WAYMAX_JIT_ENV_STEP=0 WAYMAX_JIT_ENV_RESET=0')
PY

echo "JIT EQUIVALENCE PASS"
echo "You may enable full-run JIT with: export WAYMAX_JIT_ENV_STEP=1 WAYMAX_JIT_ENV_RESET=1"
