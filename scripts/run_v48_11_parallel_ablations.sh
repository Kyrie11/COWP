#!/usr/bin/env bash
set -euo pipefail

# v48.11 ablations, max two concurrent single-GPU jobs.
# A: v48.10 COPE reference (evidence-first fallback semantics)
# B: same A checkpoint, policy-first/no-fallback recalibration only
# C: set tournament + old independent-BCE ordinal evidence
# D: full CASTER (set tournament + ordered NLL + regime experts + policy features)

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_11_ablations}"
ASSET_ROOT="${ASSET_ROOT:-runs/ocrap_v48_8_shared_assets_4801}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"

COMMON=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=$INIT_CKPT" "CALIBRATION_MODE=proxy_val_split"
  "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}" "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=1" "AUTO_ENSURE_MANIFESTS=0"
  "PREBUILT_SPLIT_ROOT=$ASSET_ROOT/dataset_splits" "REUSE_PREBUILT_SPLITS=1"
  "SHARED_GROUP_INDEX=$ASSET_ROOT/teacher_pcd_train_index.jsonl"
  "SHARED_GROUP_SUMMARY=$ASSET_ROOT/teacher_pcd_train_index_summary.json"
  "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${NUM_WORKERS:-6}" "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}"
  "FOREGROUND=1" "EXACT_TEACHER_PCD=true" "GROUP_DRO_WEIGHT=0"
)

run_train_task() {
  local group="$1" variant="$2" gpu="$3"; shift 3
  local out="$ROOT/tasks/${group}_${variant}" marker="$ROOT/tasks/${group}_${variant}/TASK_COMPLETE.json"
  [[ -f "$marker" ]] && return 0
  mkdir -p "$out"
  set +e
  env "${COMMON[@]}" OUTPUTDIR="$out" VARIANTS="$variant" GPU0="$gpu" GPU1="$gpu" "$@" \
    bash run_v48_two_gpu_fast_commands.txt >"$ROOT/logs/${group}_${variant}.log" 2>&1
  local rc=$?; set -e; echo "$rc" >"$out/controller.exit_code"
  [[ "$rc" == 0 || "$rc" == 20 ]] || return "$rc"
  python - "$out" "$group" "$variant" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); g=sys.argv[2]; v=sys.argv[3]; rc=int(sys.argv[4])
ck=out/'candidates'/v/'model_v48_trac_sr'/'best.pt'
near=out/'candidates'/v/'calibration'/'direct_value_risk_near_v48.json'
contact=out/'candidates'/v/'calibration'/'direct_value_risk_contact_v48.json'
if not all(p.is_file() for p in (ck,near,contact)): raise SystemExit('incomplete task')
(out/'TASK_COMPLETE.json').write_text(json.dumps({'complete':True,'group':g,'variant':v,'controller_exit':rc,'created_unix':time.time(),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()},indent=2)+'\n')
PY
}

# A tasks, one per GPU.
for variant in balanced precision; do
  gpu="$GPU0"; [[ "$variant" == precision ]] && gpu="$GPU1"
  run_train_task A_cope_reference "$variant" "$gpu" \
    TRAIN_SCRIPT=scripts/train_ocrap_v48_10_cope.sh COPE_CONDITIONAL_PREFERENCE=true \
    COPE_ORDINAL_EVIDENCE=true RISK_SOURCE=ordinal_evidence CONDITIONAL_RECOVERY_RANKING=true &
done
wait

# B is a calibration-only semantic ablation over immutable A checkpoints.
for variant in balanced precision; do
  gpu="$GPU0"; [[ "$variant" == precision ]] && gpu="$GPU1"
  (
    src="$ROOT/tasks/A_cope_reference_${variant}/candidates/${variant}"
    out="$ROOT/tasks/B_policy_first_${variant}"; mkdir -p "$out/candidates/$variant/calibration" "$out/logs"
    ck="$src/model_v48_trac_sr/best.pt"
    split="$ASSET_ROOT/dataset_splits"
    for regime in near contact; do
      data="$split/calibration_${regime/near/near_contact}"
      CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
        --dataset "$data" --checkpoint "$ck" --bucket "$regime" --risk-source ordinal_evidence \
        --conditional-recovery-ranking --policy-first-no-fallback \
        --output "$out/candidates/$variant/calibration/direct_value_risk_${regime}_v48.json" \
        --rows-output "$out/candidates/$variant/calibration/direct_value_risk_${regime}_v48.rows.jsonl" \
        --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-verify-selected=1 \
        >"$out/logs/calibrate_${regime}.log" 2>&1 || true
    done
    cp "$ck" "$out/candidates/$variant/best.pt"
    printf '{"complete":true,"calibration_only":true,"source":"%s"}\n' "$ck" > "$out/TASK_COMPLETE.json"
  ) &
done
wait

# C and D model ablations, two jobs at a time.
for variant in balanced precision; do
  run_train_task C_tournament_old_evidence "$variant" "$GPU0" \
    TRAIN_SCRIPT=scripts/train_ocrap_v48_11_caster.sh CASTER_ORDERED_EVIDENCE=false \
    DELTA_REGIME_EXPERTS=false DELTA_POLICY_FEATURES=false RISK_SOURCE=ordinal_evidence \
    CONDITIONAL_RECOVERY_RANKING=true POLICY_FIRST_NO_FALLBACK=true & p0=$!
  run_train_task D_full_caster "$variant" "$GPU1" \
    TRAIN_SCRIPT=scripts/train_ocrap_v48_11_caster.sh CASTER_ORDERED_EVIDENCE=true \
    DELTA_REGIME_EXPERTS=true DELTA_POLICY_FEATURES=true RISK_SOURCE=ordinal_evidence \
    CONDITIONAL_RECOVERY_RANKING=true POLICY_FIRST_NO_FALLBACK=true & p1=$!
  wait "$p0"; wait "$p1"
done

python tools/summarize_v48_11_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_11.json"
printf '{"complete":true,"version":"v48.11","max_parallel_gpu_jobs":2}\n' > "$ROOT/ABLATIONS_COMPLETE.json"
