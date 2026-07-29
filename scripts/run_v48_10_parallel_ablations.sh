#!/usr/bin/env bash
set -euo pipefail

# v48.10 COPE causal ablations. At most two jobs run concurrently, one per A30.
# Every task is resumable and must produce an immutable completion marker before
# the aggregate summary is written.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_10_ablations}"
ASSET_ROOT="${ASSET_ROOT:-runs/ocrap_v48_8_shared_assets_4801}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT to a completed checkpoint}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
[[ -f "$ASSET_ROOT/SHARED_ASSETS_COMPLETE.json" ]] || {
  echo "missing shared assets: $ASSET_ROOT/SHARED_ASSETS_COMPLETE.json" >&2; exit 2;
}

COMMON=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=$INIT_CKPT" "CALIBRATION_MODE=proxy_val_split"
  "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}"
  "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=1" "AUTO_ENSURE_MANIFESTS=0"
  "PREBUILT_SPLIT_ROOT=$ASSET_ROOT/dataset_splits" "REUSE_PREBUILT_SPLITS=1"
  "SHARED_GROUP_INDEX=$ASSET_ROOT/teacher_pcd_train_index.jsonl"
  "SHARED_GROUP_SUMMARY=$ASSET_ROOT/teacher_pcd_train_index_summary.json"
  "BATCH_SIZE=${BATCH_SIZE:-96}" "NUM_WORKERS=${NUM_WORKERS:-6}"
  "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}" "FOREGROUND=1"
  "EXACT_TEACHER_PCD=true" "GROUP_DRO_WEIGHT=0"
  "PREFERENCE_CONTEXT_HIDDEN=${PREFERENCE_CONTEXT_HIDDEN:-32}"
)

run_task() {
  local group="$1" variant="$2" gpu="$3"; shift 3
  local out="$ROOT/tasks/${group}_${variant}"
  local marker="$out/TASK_COMPLETE.json"
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] SKIP completed $group $variant" | tee -a "$ROOT/logs/scheduler.log"
    return 0
  fi
  mkdir -p "$out"
  echo "[$(date -Is)] START $group $variant GPU=$gpu" | tee -a "$ROOT/logs/scheduler.log"
  set +e
  env "${COMMON[@]}" OUTPUTDIR="$out" VARIANTS="$variant" GPU0="$gpu" GPU1="$gpu" "$@" \
    bash run_v48_two_gpu_fast_commands.txt >"$ROOT/logs/${group}_${variant}.log" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$out/controller.exit_code"
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    echo "[$(date -Is)] FAIL $group $variant rc=$rc" | tee -a "$ROOT/logs/scheduler.log"
    return "$rc"
  fi
  python - "$out" "$group" "$variant" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
ck=out/'candidates'/variant/'model_v48_trac_sr'/'best.pt'
near=out/'candidates'/variant/'calibration'/'direct_value_risk_near_v48.json'
contact=out/'candidates'/variant/'calibration'/'direct_value_risk_contact_v48.json'
missing=[str(p) for p in (ck,near,contact) if not p.is_file()]
if missing: raise SystemExit('incomplete ablation task: '+','.join(missing))
doc={'complete':True,'group':group,'variant':variant,'controller_exit':rc,
     'created_unix':time.time(),'checkpoint':str(ck),
     'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
  echo "[$(date -Is)] END $group $variant rc=$rc" | tee -a "$ROOT/logs/scheduler.log"
}

# A: v48.9-style nominal-inclusive preference + continuous delta.
# B: conditional preference only.
# C: ordinal evidence only.
# D: full COPE.
A=(TRAIN_SCRIPT=scripts/train_ocrap_v48_10_cope.sh COPE_CONDITIONAL_PREFERENCE=false COPE_ORDINAL_EVIDENCE=false RISK_SOURCE=direct_delta CONDITIONAL_RECOVERY_RANKING=false)
B=(TRAIN_SCRIPT=scripts/train_ocrap_v48_10_cope.sh COPE_CONDITIONAL_PREFERENCE=true  COPE_ORDINAL_EVIDENCE=false RISK_SOURCE=direct_delta CONDITIONAL_RECOVERY_RANKING=true)
C=(TRAIN_SCRIPT=scripts/train_ocrap_v48_10_cope.sh COPE_CONDITIONAL_PREFERENCE=false COPE_ORDINAL_EVIDENCE=true  RISK_SOURCE=ordinal_evidence CONDITIONAL_RECOVERY_RANKING=false)
D=(TRAIN_SCRIPT=scripts/train_ocrap_v48_10_cope.sh COPE_CONDITIONAL_PREFERENCE=true  COPE_ORDINAL_EVIDENCE=true  RISK_SOURCE=ordinal_evidence CONDITIONAL_RECOVERY_RANKING=true)

declare -a groups=(A_reference B_conditional_preference C_ordinal_evidence D_full_cope)
for variant in balanced precision; do
  run_task A_reference "$variant" "$GPU0" "${A[@]}" & p0=$!
  run_task B_conditional_preference "$variant" "$GPU1" "${B[@]}" & p1=$!
  wait "$p0"; wait "$p1"
  run_task C_ordinal_evidence "$variant" "$GPU0" "${C[@]}" & p0=$!
  run_task D_full_cope "$variant" "$GPU1" "${D[@]}" & p1=$!
  wait "$p0"; wait "$p1"
done

python - "$ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); missing=[]
for g in ('A_reference','B_conditional_preference','C_ordinal_evidence','D_full_cope'):
  for v in ('balanced','precision'):
    p=root/'tasks'/f'{g}_{v}'/'TASK_COMPLETE.json'
    if not p.is_file(): missing.append(str(p))
if missing: raise SystemExit('ablation suite incomplete:\n'+'\n'.join(missing))
(root/'ABLATIONS_COMPLETE.json').write_text(json.dumps({
  'complete':True,'version':'v48.10','experiments':4,'variants':2,
  'max_parallel_gpu_jobs':2,'task_markers':8},indent=2)+'\n')
PY
python tools/summarize_v48_8_parallel_ablations.py --root "$ROOT" \
  --output "$ROOT/ablation_summary_v48_10.json"
