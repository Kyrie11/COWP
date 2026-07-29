#!/usr/bin/env bash
set -euo pipefail

# v48.12 TRIDENT core ablations.  Eight tasks (4 designs x 2 variants) are
# submitted automatically, with at most one task per A30 and two concurrent
# tasks total.  A failed task is recorded but does not prevent the remaining
# tasks from running.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_12_ablations}"
ASSET_ROOT="${ASSET_ROOT:-runs/ocrap_v48_8_shared_assets_4801}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"

COMMON=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=$INIT_CKPT"
  "INIT_CKPT_BALANCED=${INIT_CKPT_BALANCED:-$INIT_CKPT}"
  "INIT_CKPT_PRECISION=${INIT_CKPT_PRECISION:-$INIT_CKPT}"
  "CALIBRATION_MODE=proxy_val_split"
  "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}" "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=1" "AUTO_ENSURE_MANIFESTS=0"
  "PREBUILT_SPLIT_ROOT=$ASSET_ROOT/dataset_splits" "REUSE_PREBUILT_SPLITS=1"
  "SHARED_GROUP_INDEX=$ASSET_ROOT/teacher_pcd_train_index.jsonl"
  "SHARED_GROUP_SUMMARY=$ASSET_ROOT/teacher_pcd_train_index_summary.json"
  "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${NUM_WORKERS:-6}" "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}"
  "PREFERENCE_EPOCHS=${PREFERENCE_EPOCHS:-14}" "PREFERENCE_PATIENCE=${PREFERENCE_PATIENCE:-4}"
  "EVIDENCE_EPOCHS=${EVIDENCE_EPOCHS:-12}" "EVIDENCE_PATIENCE=${EVIDENCE_PATIENCE:-4}"
  "FOREGROUND=1" "EXACT_TEACHER_PCD=true" "GROUP_DRO_WEIGHT=0"
  "TRAIN_SCRIPT=scripts/train_ocrap_v48_12_trident.sh"
  "RISK_SOURCE=ordinal_evidence" "CONDITIONAL_RECOVERY_RANKING=true"
  "POLICY_FIRST_NO_FALLBACK=true" "MACRO_CONSTRAINT_MODE=opportunity_normalized"
  "MAX_MACRO_EXCESS_SHARE=${MAX_MACRO_EXCESS_SHARE:-0.10}"
)

run_task() {
  local group="$1" variant="$2" gpu="$3"; shift 3
  local out="$ROOT/tasks/${group}_${variant}"
  local marker="$out/TASK_COMPLETE.json"
  if [[ -f "$marker" ]]; then
    echo "[skip] $group $variant already complete"
    return 0
  fi
  mkdir -p "$out"
  set +e
  env "${COMMON[@]}" OUTPUTDIR="$out" VARIANTS="$variant" GPU0="$gpu" GPU1="$gpu" "$@" \
    bash run_v48_two_gpu_fast_commands.txt >"$ROOT/logs/${group}_${variant}.log" 2>&1
  local rc=$?
  set -e
  echo "$rc" >"$out/controller.exit_code"
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    printf '{"complete":false,"group":"%s","variant":"%s","controller_exit":%s}\n' \
      "$group" "$variant" "$rc" >"$out/TASK_FAILED.json"
    return "$rc"
  fi
  python - "$out" "$group" "$variant" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
ck=out/'candidates'/variant/'model_v48_trac_sr'/'best.pt'
near=out/'candidates'/variant/'calibration'/'direct_value_risk_near_v48.json'
contact=out/'candidates'/variant/'calibration'/'direct_value_risk_contact_v48.json'
missing=[str(p) for p in (ck,near,contact) if not p.is_file()]
if missing:
    raise SystemExit('incomplete task: '+','.join(missing))
(out/'TASK_COMPLETE.json').write_text(json.dumps({
    'complete':True,'group':group,'variant':variant,'controller_exit':rc,
    'created_unix':time.time(),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
},indent=2)+'\n')
PY
}

# A: engineering-isolated CASTER reference (conditional checkpoint semantics
# fixed, but no new pairwise ranking).  B: recovery-pair tournament only.
# C: cross-group ordinal evidence only.  D: full TRIDENT.
TASKS=(
  "A_contract_fix balanced 0 0.0 0.0 0.0"
  "A_contract_fix precision 1 0.0 0.0 0.0"
  "B_recovery_pair balanced 0 1.0 0.0 0.0"
  "B_recovery_pair precision 1 1.0 0.0 0.0"
  "C_bipolar_evidence balanced 0 0.0 0.60 1.40"
  "C_bipolar_evidence precision 1 0.0 0.60 1.40"
  "D_full_trident balanced 0 1.0 0.60 1.40"
  "D_full_trident precision 1 1.0 0.60 1.40"
)

failures=0
for ((i=0; i<${#TASKS[@]}; i+=2)); do
  pids=(); labels=()
  for offset in 0 1; do
    idx=$((i+offset)); (( idx < ${#TASKS[@]} )) || continue
    read -r group variant slot rank_w benefit_w harm_w <<<"${TASKS[$idx]}"
    gpu="$GPU0"; [[ "$slot" == 1 ]] && gpu="$GPU1"
    (
      run_task "$group" "$variant" "$gpu" \
        PREFERENCE_CONDITIONAL_PAIRWISE_WEIGHT="$rank_w" \
        ORDERED_EVIDENCE_PAIRWISE_BENEFIT_WEIGHT="$benefit_w" \
        ORDERED_EVIDENCE_PAIRWISE_HARM_WEIGHT="$harm_w"
    ) &
    pids+=("$!"); labels+=("${group}_${variant}")
  done
  set +e
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"; rc=$?
    if [[ "$rc" != 0 ]]; then
      echo "[failed] ${labels[$j]} rc=$rc" >&2
      failures=$((failures+1))
    fi
  done
  set -e
done

python tools/summarize_v48_12_ablations.py \
  --root "$ROOT" --output "$ROOT/ablation_summary_v48_12.json"
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
expected=[
 'A_contract_fix_balanced','A_contract_fix_precision',
 'B_recovery_pair_balanced','B_recovery_pair_precision',
 'C_bipolar_evidence_balanced','C_bipolar_evidence_precision',
 'D_full_trident_balanced','D_full_trident_precision',
]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.12','max_parallel_gpu_jobs':2,
     'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else:
    raise SystemExit('ablation suite incomplete: '+','.join(missing))
PY
