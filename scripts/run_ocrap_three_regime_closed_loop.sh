#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${MODEL_RUN:=runs/ocrap_v48_34_barrier_crossfit_dedicated_4834}"
: "${MODEL_VARIANT:=balanced}"
# Main runs normally expose candidates/<variant>; some retained bundles only
# expose dedicated_candidates/<variant>.  Resolve either layout without asking
# the user to edit the script. Explicit CHECKPOINT/GAMMA_REC_JSON still win.
if [[ -z "${MODEL_CANDIDATE_ROOT:-}" ]]; then
  MODEL_CANDIDATE_ROOT="$MODEL_RUN/candidates/$MODEL_VARIANT"
  if [[ ! -f "$MODEL_CANDIDATE_ROOT/model_v48_trac_sr/best.pt" && -f "$MODEL_RUN/dedicated_candidates/$MODEL_VARIANT/model_v48_trac_sr/best.pt" ]]; then
    MODEL_CANDIDATE_ROOT="$MODEL_RUN/dedicated_candidates/$MODEL_VARIANT"
  fi
fi
: "${CHECKPOINT:=$MODEL_CANDIDATE_ROOT/model_v48_trac_sr/best.pt}"
: "${GAMMA_REC_JSON:=$MODEL_CANDIDATE_ROOT/calibration/gamma_rec_by_bucket_v48.json}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing OC-RAP checkpoint: $CHECKPOINT" >&2; exit 2; }

if [[ -z "${SAFE_GAMMA_REC:-}" || -z "${NEAR_GAMMA_REC:-}" || -z "${CONTACT_GAMMA_REC:-}" ]]; then
  if [[ -f "$GAMMA_REC_JSON" ]]; then
    read -r _auto_safe _auto_near _auto_contact < <(python - "$GAMMA_REC_JSON" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))['gamma_rec_by_bucket']
print(x['test_safe'],x['test_near_contact'],x['test_contact'])
PY
)
  elif [[ -n "${GAMMA_REC:-}" ]]; then
    _auto_safe="$GAMMA_REC"; _auto_near="$GAMMA_REC"; _auto_contact="$GAMMA_REC"
    echo "[WARN] Reusing one GAMMA_REC across regimes; bucket-specific calibration is preferred." >&2
  else
    echo "Missing bucket calibration JSON: $GAMMA_REC_JSON" >&2; exit 2
  fi
fi
: "${SAFE_GAMMA_REC:=${_auto_safe:-}}"
: "${NEAR_GAMMA_REC:=${_auto_near:-}}"
: "${CONTACT_GAMMA_REC:=${_auto_contact:-}}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${OUT:=runs/ocrap_three_regime_closed_loop_v50}"
: "${WOMD_NUM_SHARDS:=150}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${SAFE_WOMD:=$WOMD_VAL}"
: "${NEAR_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${CONTACT_WOMD:=$WOMD_VAL_INTERACTIVE}"
SAFE_WOMD="$(v50_normalize_womd_spec "$SAFE_WOMD" "$WOMD_NUM_SHARDS")"
NEAR_WOMD="$(v50_normalize_womd_spec "$NEAR_WOMD" "$WOMD_NUM_SHARDS")"
CONTACT_WOMD="$(v50_normalize_womd_spec "$CONTACT_WOMD" "$WOMD_NUM_SHARDS")"
: "${SAFE_BUCKET:=$OCRAP_ROOT/test_safe}"
: "${NEAR_BUCKET:=$OCRAP_ROOT/test_near_contact}"
: "${CONTACT_BUCKET:=$OCRAP_ROOT/test_contact}"
: "${BUCKET_SPLIT:=test}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"
: "${SAFE_LABEL_MODE:=fast}"
: "${NEAR_LABEL_MODE:=fast}"
: "${CONTACT_LABEL_MODE:=fast}"
: "${AUDIT_EVERY_N_STEPS:=0}"
: "${RENDER_SAFE:=false}"
: "${RENDER_NEAR:=false}"
: "${RENDER_CONTACT:=false}"
: "${SAFE_TARGET_KEYS_FILE:=}"
: "${NEAR_TARGET_KEYS_FILE:=}"
: "${CONTACT_TARGET_KEYS_FILE:=}"
: "${RESUME_FORCE:=false}"
: "${ALLOW_DIAGNOSTIC_RC20:=0}"
: "${RUN_SAFE:=1}"
: "${RUN_NEAR:=1}"
: "${RUN_CONTACT:=1}"
: "${SKIP_COMPLETE_REGIMES:=true}"
: "${FINALIZE_COMPLETE_JOURNALS:=true}"

[[ "$ALLOW_DIAGNOSTIC_RC20" == 1 ]] || { echo "Set ALLOW_DIAGNOSTIC_RC20=1: current model did not pass the Natural gate." >&2; exit 2; }
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"; ((${#GPUS[@]})) || GPUS=(0)
mkdir -p "$OUT/safe" "$OUT/near" "$OUT/contact"

refresh_index() {
  local rc="${1:-1}"
  python tools/build_ocrap_three_regime_index.py --root "$OUT" --launcher-exit-code "$rc" >/dev/null || true
}
# Make the index visible before launching any long-running worker. If the
# parent is later killed by a scheduler, users still have a recoverable status
# document instead of a missing file.
refresh_index 1

write_phase() {
  local regime="$1" status="$2" rc="$3" started="$4" ended="$5"
  python - "$OUT/$regime.phase.json" "$regime" "$status" "$rc" "$started" "$ended" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
json.dump({'regime':sys.argv[2],'status':sys.argv[3],'exit_code':int(sys.argv[4]),'started_at':sys.argv[5],'ended_at':sys.argv[6]},p.open('w'),indent=2)
PY
}

FINALIZED=0
finalize_index() {
  local rc=$?
  [[ "$FINALIZED" == 1 ]] && return 0
  FINALIZED=1; set +e
  python tools/build_ocrap_three_regime_index.py --root "$OUT" --launcher-exit-code "$rc"
  return 0
}
trap finalize_index EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

run_one() {
  local regime="$1" womd="$2" bucket="$3" label_mode="$4" render="$5" gpu="$6" gamma="$7" target_keys="${8:-}"
  local target_env=()
  [[ -n "$target_keys" ]] && target_env=(TARGET_KEYS_FILE="$target_keys" REQUIRE_TARGET_KEYS=true)
  env RUN_DIR="$OUT/$regime" OUTPUT="$OUT/$regime/closed_loop_ocrap.json" \
    WOMD_VAL="$womd" WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS" CHECKPOINT="$CHECKPOINT" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" LABEL_MODE="$label_mode" AUDIT_EVERY_N_STEPS="$AUDIT_EVERY_N_STEPS" \
    NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    BUCKET_DATASET="$bucket" BUCKET_SPLIT="$BUCKET_SPLIT" MAX_TARGETS_PER_SCENE=1 \
    RENDER_TRACE="$render" SAVE_PARTIAL=true RESUME_FORCE="$RESUME_FORCE" \
    "${target_env[@]}" bash scripts/run_ocrap_closed_loop_optimized.sh
}

run_one_status() {
  local regime="$1" enabled="$2"; shift 2
  local started rc status artifact="$OUT/$regime/closed_loop_ocrap.json"
  started="$(v50_iso_now)"
  if [[ "$enabled" != 1 ]]; then
    write_phase "$regime" skipped 0 "$started" "$(v50_iso_now)"
    return 0
  fi
  if v50_bool_true "$FINALIZE_COMPLETE_JOURNALS"; then
    python tools/finalize_closed_loop_from_journal.py --output "$artifact" >/dev/null 2>&1 || true
  fi
  if v50_bool_true "$SKIP_COMPLETE_REGIMES" && python tools/check_closed_loop_artifact.py --output "$artifact" --quiet; then
    echo "[REUSE] $regime closed-loop artifact is already complete: $artifact"
    write_phase "$regime" complete 0 "$started" "$(v50_iso_now)"
    return 0
  fi
  write_phase "$regime" running 0 "$started" ""
  if run_one "$regime" "$@"; then rc=0; status=complete; else rc=$?; status=failed; fi
  write_phase "$regime" "$status" "$rc" "$started" "$(v50_iso_now)"
  return "$rc"
}

failed=0
if ((${#GPUS[@]} >= 2)); then
  run_one_status safe "$RUN_SAFE" "$SAFE_WOMD" "$SAFE_BUCKET" "$SAFE_LABEL_MODE" "$RENDER_SAFE" "${GPUS[0]}" "$SAFE_GAMMA_REC" "$SAFE_TARGET_KEYS_FILE" & p_safe=$!
  run_one_status near "$RUN_NEAR" "$NEAR_WOMD" "$NEAR_BUCKET" "$NEAR_LABEL_MODE" "$RENDER_NEAR" "${GPUS[1]}" "$NEAR_GAMMA_REC" "$NEAR_TARGET_KEYS_FILE" & p_near=$!
  # Contact starts as soon as GPU 0 is free, even when Safe failed.
  wait "$p_safe" || failed=1
  refresh_index "$failed"
  run_one_status contact "$RUN_CONTACT" "$CONTACT_WOMD" "$CONTACT_BUCKET" "$CONTACT_LABEL_MODE" "$RENDER_CONTACT" "${GPUS[0]}" "$CONTACT_GAMMA_REC" "$CONTACT_TARGET_KEYS_FILE" & p_contact=$!
  wait "$p_near" || failed=1
  refresh_index "$failed"
  wait "$p_contact" || failed=1
  refresh_index "$failed"
else
  run_one_status safe "$RUN_SAFE" "$SAFE_WOMD" "$SAFE_BUCKET" "$SAFE_LABEL_MODE" "$RENDER_SAFE" "${GPUS[0]}" "$SAFE_GAMMA_REC" "$SAFE_TARGET_KEYS_FILE" || failed=1
  refresh_index "$failed"
  run_one_status near "$RUN_NEAR" "$NEAR_WOMD" "$NEAR_BUCKET" "$NEAR_LABEL_MODE" "$RENDER_NEAR" "${GPUS[0]}" "$NEAR_GAMMA_REC" "$NEAR_TARGET_KEYS_FILE" || failed=1
  refresh_index "$failed"
  run_one_status contact "$RUN_CONTACT" "$CONTACT_WOMD" "$CONTACT_BUCKET" "$CONTACT_LABEL_MODE" "$RENDER_CONTACT" "${GPUS[0]}" "$CONTACT_GAMMA_REC" "$CONTACT_TARGET_KEYS_FILE" || failed=1
  refresh_index "$failed"
fi

python tools/build_ocrap_three_regime_index.py --root "$OUT" --launcher-exit-code "$failed"
FINALIZED=1
((failed==0)) || exit 1
