#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export BASE_OUT GPU0 GPU1
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}" CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}" CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
export SOURCE_RUN="${SOURCE_RUN:-$BASE_OUT/ocrap_v48_45_source_rebuild_s7}"
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-3}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-3}" CACHE_SAMPLES_IN_MEMORY=false PERSISTENT_TENSOR_CACHE=true
export PERSISTENT_TENSOR_CACHE_DIR="${OCRAP_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4846}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

# The v48.56 A arm is a valid reference because v48.57 changes neither teacher
# semantics nor the protocol.  Reuse is fail-closed and is preferred to spending
# two GPUs to regenerate an identical control.  Set V4857_FORCE_FRESH_A=1 to
# deliberately rerun the control from the v48.57 checkout.
REFERENCE_A="${V4857_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
FRESH_A="$BASE_OUT/ocrap_v48_57_dcp_drfc_bcde_cmri_reference_A"
B_RUN="$BASE_OUT/ocrap_v48_57_dcp_drfc_bcde_cmri_main"
REF_CHECK="$BASE_OUT/OC-RAP-v48.57-reference-reuse-contract.json"
AUDIT_B_PREC="$BASE_OUT/OC-RAP-v48.57-root-source-decomposition-B-precision.json"
AUDIT_B_BAL="$BASE_OUT/OC-RAP-v48.57-root-source-decomposition-B-balanced.json"
AUDIT_A_PREC="$BASE_OUT/OC-RAP-v48.57-root-source-decomposition-A-precision.json"
COMPARE="$BASE_OUT/OC-RAP-v48.57-DCP-DRFC-BCDE-CMRI-comparison.json"
PERF_LOG="$BASE_OUT/OC-RAP-v48.57-runtime-telemetry.jsonl"; : > "$PERF_LOG"
perf_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  ( while true; do
      ts="$(date +%s)"
      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null |
        TS="$ts" python -c 'import json,os,sys
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 print(json.dumps({"unix":float(os.environ["TS"]),"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":None if p[5] in {"N/A","[N/A]"} else float(p[5])}))' >> "$PERF_LOG" || true
      sleep "${V4857_TELEMETRY_INTERVAL_S:-30}"
    done ) & perf_pid=$!
fi
cleanup(){
  if [[ -n "$perf_pid" ]]; then kill "$perf_pid" 2>/dev/null || true; wait "$perf_pid" 2>/dev/null || true; fi
  python tools/summarize_v48_46_runtime_telemetry.py --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.57-runtime-telemetry-summary.json" >/dev/null 2>&1 || true
}; trap cleanup EXIT
trap 'exit 130' INT; trap 'exit 143' TERM

bash scripts/prepare_v48_45_protocol.sh
export V4845_SKIP_PROTOCOL_PREPARE=1
NEAR_DEV="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; CONTACT_DEV="$PROTOCOL_ROOT/evidence_adapt_dev_contact"

accept(){ case "$1" in 0) echo "$2: RC=0";; 20) echo "$2: RC=20 algorithm rejection (valid evidence)";; *) echo "$2: RC=$1 ENGINEERING FAILURE" >&2; return 1;; esac; }
run_arm(){
  local arm="$1" out="$2" g0="$3" g1="$4" train_idx="$5" train_summary="$6" dev_idx="$7" dev_summary="$8"
  rm -rf "$out"; mkdir -p "$out/logs"
  set +e
  OUTPUTDIR="$out" GPU0="$g0" GPU1="$g1" SERIAL_VARIANTS_ON_ONE_GPU=0 \
    V4856_RAW_TEACHER_INDEX="$train_idx" V4856_RAW_TEACHER_SUMMARY="$train_summary" \
    V4856_RAW_DEV_TEACHER_INDEX="$dev_idx" V4856_RAW_DEV_TEACHER_SUMMARY="$dev_summary" \
    bash scripts/run_v48_57_dcp_drfc_bcde_cmri_arm.sh "$arm" >"$out/logs/v48_57_launcher.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$out/logs/v48_57_launcher.rc"
  accept "$rc" "$arm" || return 30
  return 0
}

if [[ "${V4857_FORCE_FRESH_A:-0}" == 1 ]]; then
  REFERENCE_A="$FRESH_A"
else
  set +e
  python tools/check_v48_57_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_CHECK"
  ref_rc=$?
  set -e
  if [[ "$ref_rc" != 0 ]]; then
    echo "v48.57: reusable v48.56-A reference unavailable/invalid; generating a fresh exact A." >&2
    REFERENCE_A="$FRESH_A"
  fi
fi

if [[ "$REFERENCE_A" == "$FRESH_A" ]]; then
  # A has identical teacher semantics, so any validated raw teacher index from
  # the old reference may be reused for preprocessing only; otherwise A builds it.
  old_ref="${V4857_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
  ti=""; ts=""; di=""; ds=""
  if [[ -f "$old_ref/evidence_adapt_teacher_pcd_index.jsonl" && -f "$old_ref/evidence_adapt_teacher_pcd_index_summary.json" ]]; then ti="$old_ref/evidence_adapt_teacher_pcd_index.jsonl"; ts="$old_ref/evidence_adapt_teacher_pcd_index_summary.json"; fi
  if [[ -f "$old_ref/evidence_adapt_dev_teacher_pcd_index.jsonl" && -f "$old_ref/evidence_adapt_dev_teacher_pcd_index_summary.json" ]]; then di="$old_ref/evidence_adapt_dev_teacher_pcd_index.jsonl"; ds="$old_ref/evidence_adapt_dev_teacher_pcd_index_summary.json"; fi
  run_arm A "$FRESH_A" "$GPU0" "$GPU1" "$ti" "$ts" "$di" "$ds"
fi
python tools/check_v48_57_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_CHECK"

# B/Main is the only fresh scientific intervention. Balanced and Precision are
# placed on different GPUs, which removes the v48.56 long-tail idle interval.
run_arm B "$B_RUN" "$GPU0" "$GPU1" \
  "$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" "$REFERENCE_A/evidence_adapt_teacher_pcd_index_summary.json" \
  "$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" "$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index_summary.json"

# Root-source decomposition is diagnostic only and reads adaptation-dev. Run the
# two selector checkpoints concurrently so the post-run audit also uses both GPUs.
run_source_audit(){
  local variant="$1" gpu="$2" out="$3" run="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/audit_v48_57_root_source_decomposition.py \
    --checkpoint "$run/candidates/$variant/model_v48_trac_sr/best.pt" --device cuda \
    --dataset "near=$NEAR_DEV" --dataset "contact=$CONTACT_DEV" --positive-gain 0.015 --output "$out"
}
set +e
run_source_audit precision "$GPU0" "$AUDIT_B_PREC" "$B_RUN" >"$B_RUN/logs/v48_57_root_audit_precision.log" 2>&1 & p0=$!
run_source_audit balanced "$GPU1" "$AUDIT_B_BAL" "$B_RUN" >"$B_RUN/logs/v48_57_root_audit_balanced.log" 2>&1 & p1=$!
wait "$p0"; ar0=$?; wait "$p1"; ar1=$?
set -e
if [[ "$ar0" != 0 || "$ar1" != 0 ]]; then echo "v48.57 root-source audit failed (precision=$ar0 balanced=$ar1)" >&2; exit 30; fi

# Optional baseline audit if the retained reference checkpoint is still present.
if [[ -f "$REFERENCE_A/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU0" python tools/audit_v48_57_root_source_decomposition.py \
    --checkpoint "$REFERENCE_A/candidates/precision/model_v48_trac_sr/best.pt" --device cuda \
    --dataset "near=$NEAR_DEV" --dataset "contact=$CONTACT_DEV" --positive-gain 0.015 --output "$AUDIT_A_PREC" \
    >"$B_RUN/logs/v48_57_root_audit_reference_A.log" 2>&1 || rm -f "$AUDIT_A_PREC"
else
  rm -f "$AUDIT_A_PREC"
fi

cmp_args=(--reference-a "$REFERENCE_A" --b "$B_RUN" --audit-b "$AUDIT_B_PREC" --output "$COMPARE")
[[ -f "$AUDIT_A_PREC" ]] && cmp_args+=(--audit-a "$AUDIT_A_PREC")
python tools/compare_v48_57_cmri.py "${cmp_args[@]}"
python tools/summarize_v48_46_runtime_telemetry.py --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.57-runtime-telemetry-summary.json" || true

# Package only the fresh v48.57 run plus compact audits. The retained A is
# referenced by cryptographic contract and need not be duplicated.
cd "$BASE_OUT"
b="$(basename "$B_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
audit_files=("$REF_CHECK" "$AUDIT_B_PREC" "$AUDIT_B_BAL" "$COMPARE" "$BASE_OUT/OC-RAP-v48.57-runtime-telemetry-summary.json")
[[ -f "$AUDIT_A_PREC" ]] && audit_files+=("$AUDIT_A_PREC")
zip -qj "OC-RAP-v48.57-CMRI-audits.zip" "${audit_files[@]}" 2>/dev/null || true

echo "v48.57 complete. Upload $b.zip + OC-RAP-v48.57-CMRI-audits.zip (or the individual JSON audits)."
