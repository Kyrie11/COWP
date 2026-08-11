#!/usr/bin/env bash
set -euo pipefail

# Exact-semantics Waymax candidate replay for the already-promoted v16.8.10 core cache.
# Multiple GPUs can be used by setting WAYMAX_GPUS=0,1,...; each GPU gets a deterministic shard.
PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_10_full}"
BASE_TRAIN="${BASE_TRAIN:-$COWP_ROOT/tensor_cache_train}"
BASE_VAL="${BASE_VAL:-$COWP_ROOT/tensor_cache_val}"
OUT_TRAIN="${OUT_TRAIN:-$COWP_ROOT/tensor_cache_train_waymax}"
OUT_VAL="${OUT_VAL:-$COWP_ROOT/tensor_cache_val_waymax}"
REPLAY_ROOT="${REPLAY_ROOT:-$COWP_ROOT/waymax_replay_v16_8_10}"
MAX_CANDIDATES="${MAX_REPLAY_CANDIDATES:-24}"
HORIZON="${REPLAY_HORIZON:-80}"
WAYMAX_GPUS="${WAYMAX_GPUS:-0}"
IFS=',' read -r -a GPUS <<< "$WAYMAX_GPUS"
NUM_SHARDS="${REPLAY_NUM_SHARDS:-${#GPUS[@]}}"
[[ "$NUM_SHARDS" -ge 1 ]] || NUM_SHARDS=1
[[ -d "$BASE_TRAIN" && -d "$BASE_VAL" ]] || { echo "Missing core tensor caches under $COWP_ROOT" >&2; exit 3; }
mkdir -p "$REPLAY_ROOT" "$OUT_TRAIN" "$OUT_VAL"

replay_split(){
  local split="$1" cache="$2"
  local -a pids=() files=()
  for ((s=0;s<NUM_SHARDS;s++)); do
    local gpu="${GPUS[$((s % ${#GPUS[@]}))]}"
    local out="$REPLAY_ROOT/${split}_bal${MAX_CANDIDATES}_safety_shard$(printf '%03d' "$s")_of_$(printf '%03d' "$NUM_SHARDS").jsonl"
    files+=("$out")
    echo "[$split shard $s/$NUM_SHARDS] GPU=$gpu -> $out"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates \
        --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --eval-config configs/eval_cowp_v16_8.yaml \
        --cache-dir "$cache" --state-source cache --outcomes-jsonl "$out" \
        --candidate-selection balanced --max-candidates-per-scene "$MAX_CANDIDATES" \
        --rollout-horizon-steps "$HORIZON" --waymax-device gpu \
        --waymax-action-mode absolute_xy_yaw --metric-set safety \
        --num-shards "$NUM_SHARDS" --shard-index "$s" --gc-every-scenes "${WAYMAX_GC_EVERY_SCENES:-64}" \
        --profile-replay-jsonl "$REPLAY_ROOT/${split}_profile_shard$(printf '%03d' "$s").jsonl"
    ) > >(tee "$REPLAY_ROOT/${split}_shard$(printf '%03d' "$s").log") 2>&1 &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  [[ "$rc" -eq 0 ]] || { echo "$split replay failed (rc=$rc)" >&2; return "$rc"; }
  printf '%s\n' "${files[@]}"
}

# Capture only the final JSONL paths, not progress output.
mapfile -t TRAIN_FILES < <(replay_split training "$BASE_TRAIN" | tee /dev/stderr | grep -E '^/.+\.jsonl$')
mapfile -t VAL_FILES < <(replay_split validation "$BASE_VAL" | tee /dev/stderr | grep -E '^/.+\.jsonl$')
[[ "${#TRAIN_FILES[@]}" -eq "$NUM_SHARDS" && "${#VAL_FILES[@]}" -eq "$NUM_SHARDS" ]] || {
  echo "Could not collect replay shard outputs" >&2; exit 4;
}

"$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
  --cache-dir "$BASE_TRAIN" --output-dir "$OUT_TRAIN" --outcomes-jsonl "${TRAIN_FILES[@]}" \
  --repair-outcomes-jsonl --skip-existing
"$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
  --cache-dir "$BASE_VAL" --output-dir "$OUT_VAL" --outcomes-jsonl "${VAL_FILES[@]}" \
  --repair-outcomes-jsonl --skip-existing

"$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$OUT_TRAIN"
"$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$OUT_VAL"

# Full-cache support gate for the current closed_loop objective: both classes and
# at least one within-scene safe/unsafe ranking pair must exist.
"$PYTHON_BIN" - "$OUT_TRAIN" "$OUT_VAL" <<'PY'
import json,sys
from pathlib import Path
import numpy as np

def audit(root):
    scenes=valid=safe=unsafe=mixed=0
    missing=[]
    for p in sorted(Path(root).glob('*.npz')):
        scenes+=1
        with np.load(p,allow_pickle=True) as z:
            def get(k):
                kk=k if k in z.files else k.replace('/','__')
                return np.asarray(z[kk]) if kk in z.files else None
            cv=get('cowp/candidates/valid'); rv=get('waymax/candidate_rollout_valid')
            co=get('waymax/candidate_collision'); off=get('waymax/candidate_offroad')
            if any(x is None for x in (cv,rv,co,off)):
                missing.append(p.name); continue
            mask=cv.astype(bool)&rv.astype(bool)
            bad=(co.astype(bool)|off.astype(bool))&mask
            good=(~(co.astype(bool)|off.astype(bool)))&mask
            valid+=int(mask.sum()); unsafe+=int(bad.sum()); safe+=int(good.sum())
            mixed+=int(bool(bad.any() and good.any()))
    out={'cache_dir':str(root),'scenes':scenes,'missing_key_scenes':len(missing),
         'valid_outcomes':valid,'safe_outcomes':safe,'unsafe_outcomes':unsafe,'mixed_safe_unsafe_scenes':mixed}
    out['pass']=not missing and valid>0 and safe>0 and unsafe>0 and mixed>0
    return out

reports=[audit(x) for x in sys.argv[1:]]
print(json.dumps(reports,indent=2,ensure_ascii=False))
if not all(x['pass'] for x in reports):
    raise SystemExit('Waymax outcome cache is not sufficient for the configured closed-loop planner loss')
PY

echo "WAYMAX OUTCOME ATTACH PASS"
echo "TRAIN_CACHE=$OUT_TRAIN"
echo "VAL_CACHE=$OUT_VAL"
echo "Planner/mainline training must include: --with-waymax-outcome-labels"
