#!/usr/bin/env bash
set -euo pipefail

# Three-split Waymax candidate replay for the v16.8.24 compact COWP cache.
# Label construction is CPU-only; the two A30 GPUs are used here, where JAX/
# Waymax can actually benefit from them.
PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full}"
BASE_TRAIN="${BASE_TRAIN:-$COWP_ROOT/tensor_cache_train}"
BASE_VAL="${BASE_VAL:-$COWP_ROOT/tensor_cache_val}"
BASE_TEST="${BASE_TEST:-$COWP_ROOT/tensor_cache_heldout_test}"
OUT_TRAIN="${OUT_TRAIN:-$COWP_ROOT/tensor_cache_train_waymax}"
OUT_VAL="${OUT_VAL:-$COWP_ROOT/tensor_cache_val_waymax}"
OUT_TEST="${OUT_TEST:-$COWP_ROOT/tensor_cache_heldout_test_waymax}"
REPLAY_ROOT="${REPLAY_ROOT:-$COWP_ROOT/waymax_replay_v16_8_24}"
MAX_CANDIDATES="${MAX_REPLAY_CANDIDATES:-24}"
HORIZON="${REPLAY_HORIZON:-80}"
WAYMAX_GPUS="${WAYMAX_GPUS:-0,1}"
IFS=',' read -r -a GPUS <<< "$WAYMAX_GPUS"
NUM_SHARDS="${REPLAY_NUM_SHARDS:-${#GPUS[@]}}"
[[ "$NUM_SHARDS" -ge 1 ]] || NUM_SHARDS=1

for p in "$BASE_TRAIN" "$BASE_VAL" "$BASE_TEST"; do
  [[ -d "$p" ]] || { echo "Missing core tensor cache: $p" >&2; exit 3; }
done
mkdir -p "$REPLAY_ROOT" "$OUT_TRAIN" "$OUT_VAL" "$OUT_TEST"

replay_split(){
  local split="$1" cache="$2"
  local -a pids=() files=()
  for ((s=0;s<NUM_SHARDS;s++)); do
    local gpu="${GPUS[$((s % ${#GPUS[@]}))]}"
    local out="$REPLAY_ROOT/${split}_bal${MAX_CANDIDATES}_safety_shard$(printf '%03d' "$s")_of_$(printf '%03d' "$NUM_SHARDS").jsonl"
    files+=("$out")
    echo "[$split shard $s/$NUM_SHARDS] GPU=$gpu -> $out" >&2
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
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

attach_split(){
  local split="$1" base="$2" outdir="$3"
  local -a files=()
  mapfile -t files < <(replay_split "$split" "$base" | grep -E '^/.+\.jsonl$')
  [[ "${#files[@]}" -eq "$NUM_SHARDS" ]] || { echo "Could not collect $split replay shards" >&2; return 4; }
  "$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
    --cache-dir "$base" --output-dir "$outdir" --outcomes-jsonl "${files[@]}" \
    --repair-outcomes-jsonl --skip-existing
  "$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$outdir"
}

attach_split training "$BASE_TRAIN" "$OUT_TRAIN"
attach_split validation "$BASE_VAL" "$OUT_VAL"
attach_split heldout_test "$BASE_TEST" "$OUT_TEST"

# Support gate for the current closed-loop objective/evaluation.  Train must
# support the closed-loop training term; val and heldout test must both have
# meaningful safe/unsafe variation for evaluation.
"$PYTHON_BIN" - "$OUT_TRAIN" "$OUT_VAL" "$OUT_TEST" "$COWP_ROOT/waymax_outcome_support_v16_8_24.json" <<'PY'
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
    out['pass']=not missing and scenes>0 and valid>0 and safe>0 and unsafe>0 and mixed>0
    return out

names=['train','val','heldout_test']
reports={n:audit(p) for n,p in zip(names,sys.argv[1:4])}
payload={'schema_version':'cowp_v16_8_24_waymax_outcome_support_v1','splits':reports,
         'pass':all(x['pass'] for x in reports.values())}
Path(sys.argv[4]).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(payload,indent=2,ensure_ascii=False))
if not payload['pass']:
    raise SystemExit('Waymax outcome caches are not sufficient for train/val/heldout-test')
PY

echo "WAYMAX OUTCOME ATTACH PASS (train + val + heldout_test)"
echo "TRAIN_CACHE=$OUT_TRAIN"
echo "VAL_CACHE=$OUT_VAL"
echo "HELDOUT_TEST_CACHE=$OUT_TEST"
echo "Planner/mainline training must include: --with-waymax-outcome-labels"
