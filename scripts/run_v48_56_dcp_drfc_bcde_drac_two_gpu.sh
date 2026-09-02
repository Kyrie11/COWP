#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}" CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}" CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
SOURCE_RUN="${SOURCE_RUN:-$BASE_OUT/ocrap_v48_45_source_rebuild_s7}"
export BASE_OUT SOURCE_RUN GPU0 GPU1; export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-3}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-3}" CACHE_SAMPLES_IN_MEMORY=false PERSISTENT_TENSOR_CACHE=true
export PERSISTENT_TENSOR_CACHE_DIR="${OCRAP_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4846}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Teacher-index construction is CPU/NPZ bound and occurs before GPU training.
# Four workers are conservative on the observed host and can be overridden.
export V4856_TEACHER_INDEX_WORKERS="${V4856_TEACHER_INDEX_WORKERS:-4}"
export V4856_TEACHER_INDEX_CHUNKSIZE="${V4856_TEACHER_INDEX_CHUNKSIZE:-16}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

PERF_LOG="$BASE_OUT/OC-RAP-v48.56-runtime-telemetry.jsonl"; : > "$PERF_LOG"; perf_pid=""
GLOBAL_TIMING_LOG="$BASE_OUT/OC-RAP-v48.56-stage-timing.jsonl"; : > "$GLOBAL_TIMING_LOG"
global_timing(){
  local event="$1" stage="$2" start="${3:-}" rc="${4:-}" now duration_json rc_json
  now="$(date +%s.%N)"
  if [[ -n "$start" ]]; then duration_json="$(awk -v a="$start" -v b="$now" 'BEGIN{printf "%.6f", b-a}')"; else duration_json="null"; fi
  if [[ -n "$rc" ]]; then rc_json="$rc"; else rc_json="null"; fi
  printf '{"unix":%s,"event":"%s","stage":"%s","duration_seconds":%s,"rc":%s}\n' \
    "$now" "$event" "$stage" "$duration_json" "$rc_json" >> "$GLOBAL_TIMING_LOG"
}
if command -v nvidia-smi >/dev/null 2>&1; then
 ( while true; do ts="$(date +%s)"; load="$(awk '{print $1","$2","$3}' /proc/loadavg 2>/dev/null || true)"; mem="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || true)"; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | TS="$ts" LOAD="$load" MEMKB="$mem" python -c 'import json,sys,os
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]; print(json.dumps({"unix":float(os.environ["TS"]),"loadavg":os.environ.get("LOAD",""),"mem_available_kb":int(os.environ["MEMKB"]) if os.environ.get("MEMKB") else None,"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":float(p[5]) if p[5] not in {"N/A","[N/A]"} else None}))' >> "$PERF_LOG" || true; sleep "${V4856_TELEMETRY_INTERVAL_S:-30}"; done ) & perf_pid=$!
fi
cleanup(){
  if [[ -n "$perf_pid" ]]; then kill "$perf_pid" 2>/dev/null || true; wait "$perf_pid" 2>/dev/null || true; fi
  python tools/summarize_v48_46_runtime_telemetry.py --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.56-runtime-telemetry-summary.json" >/dev/null 2>&1 || true
  python tools/summarize_v48_56_stage_timing.py --base-out "$BASE_OUT" --global-log "$GLOBAL_TIMING_LOG" --output "$BASE_OUT/OC-RAP-v48.56-stage-timing-summary.json" >/dev/null 2>&1 || true
}; trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bash scripts/prepare_v48_45_protocol.sh; export V4845_SKIP_PROTOCOL_PREPARE=1
NEAR_CERT="$PROTOCOL_ROOT/certificate_pool_near_contact"; CONTACT_CERT="$PROTOCOL_ROOT/certificate_pool_contact"; NEAR_DEV="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; CONTACT_DEV="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
# v48.56 intentionally runs a fresh A.  Teacher/deployment semantics are the
# experimental object, and the new comparator recomputes fixed legacy+DRAC
# labels from raw teacher coordinates. Reusing a historical A calibration
# artifact would therefore move the evaluation schema across arms.
A_RUN="$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A"; B_RUN="$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_B"; C_RUN="$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_C"; D_RUN="$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_main"
# If a previous invocation was interrupted after the expensive deterministic
# teacher preprocessing, preserve those raw coordinates before run_arm cleans
# the incomplete directory.  The builder re-validates dataset/alpha/beta/top-m/
# option-semantics before reuse and falls back to a fresh build on mismatch.
RAW_CACHE_DIR="$BASE_OUT/.v48_56_raw_teacher_cache"; mkdir -p "$RAW_CACHE_DIR"
A_RAW_TRAIN_INDEX=""; A_RAW_TRAIN_SUMMARY=""; A_RAW_DEV_INDEX=""; A_RAW_DEV_SUMMARY=""
if [[ -f "$A_RUN/evidence_adapt_teacher_pcd_index.jsonl" && -f "$A_RUN/evidence_adapt_teacher_pcd_index_summary.json" ]]; then
  cp -f "$A_RUN/evidence_adapt_teacher_pcd_index.jsonl" "$RAW_CACHE_DIR/A_train.jsonl"
  cp -f "$A_RUN/evidence_adapt_teacher_pcd_index_summary.json" "$RAW_CACHE_DIR/A_train_summary.json"
fi
if [[ -f "$RAW_CACHE_DIR/A_train.jsonl" && -f "$RAW_CACHE_DIR/A_train_summary.json" ]]; then
  A_RAW_TRAIN_INDEX="$RAW_CACHE_DIR/A_train.jsonl"; A_RAW_TRAIN_SUMMARY="$RAW_CACHE_DIR/A_train_summary.json"
fi
if [[ -f "$A_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl" && -f "$A_RUN/evidence_adapt_dev_teacher_pcd_index_summary.json" ]]; then
  cp -f "$A_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl" "$RAW_CACHE_DIR/A_dev.jsonl"
  cp -f "$A_RUN/evidence_adapt_dev_teacher_pcd_index_summary.json" "$RAW_CACHE_DIR/A_dev_summary.json"
fi
if [[ -f "$RAW_CACHE_DIR/A_dev.jsonl" && -f "$RAW_CACHE_DIR/A_dev_summary.json" ]]; then
  A_RAW_DEV_INDEX="$RAW_CACHE_DIR/A_dev.jsonl"; A_RAW_DEV_SUMMARY="$RAW_CACHE_DIR/A_dev_summary.json"
fi
run_arm(){ (
  set +e
  local arm="$1" out="$2" g0="$3" g1="$4" serial="$5"
  local raw_train_index="${6:-}" raw_train_summary="${7:-}" raw_dev_index="${8:-}" raw_dev_summary="${9:-}"
  local arm_t0; arm_t0="$(date +%s.%N)"; global_timing start "arm_${arm}" "$arm_t0"
  rm -rf "$out"; mkdir -p "$out/logs"
  date +%s.%N > "$out/logs/v48_56_launcher.start_unix"
  BASE_OUT="$BASE_OUT" GPU0="$g0" GPU1="$g1" SERIAL_VARIANTS_ON_ONE_GPU="$serial" \
    V4856_RAW_TEACHER_INDEX="$raw_train_index" V4856_RAW_TEACHER_SUMMARY="$raw_train_summary" \
    V4856_RAW_DEV_TEACHER_INDEX="$raw_dev_index" V4856_RAW_DEV_TEACHER_SUMMARY="$raw_dev_summary" \
    bash scripts/run_v48_56_dcp_drfc_bcde_drac_arm.sh "$arm" >"$out/logs/v48_56_launcher.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$out/logs/v48_56_launcher.rc"
  date +%s.%N > "$out/logs/v48_56_launcher.end_unix"
  global_timing end "arm_${arm}" "$arm_t0" "$rc"
  exit "$rc"
); }

accept(){ case "$1" in 0) echo "$2: RC=0";; 20) echo "$2: RC=20 algorithm rejection";; *) echo "$2: RC=$1 ENGINEERING FAILURE" >&2; return 1;; esac; }

# Fresh exact control: same source/data/q-hard BC-FC + smooth-NAP, old component roles.
set +e; run_arm A "$A_RUN" "$GPU0" "$GPU1" 0 "$A_RAW_TRAIN_INDEX" "$A_RAW_TRAIN_SUMMARY" "$A_RAW_DEV_INDEX" "$A_RAW_DEV_SUMMARY"; ra=$?; set -e; accept "$ra" A || exit 1
python tools/audit_v48_56_teacher_component_correctness.py --run "$A_RUN" \
  --output "$BASE_OUT/OC-RAP-v48.56-teacher-component-correctness-audit-A.json" || true

# B/C are independent main effects.  Give each arm one GPU and serialize its
# Balanced/Precision variants to avoid oversubscribing the same device.
set +e
run_arm B "$B_RUN" "$GPU0" "$GPU0" 1 "$A_RUN/evidence_adapt_teacher_pcd_index.jsonl" "$A_RUN/evidence_adapt_teacher_pcd_index_summary.json" "$A_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl" "$A_RUN/evidence_adapt_dev_teacher_pcd_index_summary.json" & pb=$!
run_arm C "$C_RUN" "$GPU1" "$GPU1" 1 "$A_RUN/evidence_adapt_teacher_pcd_index.jsonl" "$A_RUN/evidence_adapt_teacher_pcd_index_summary.json" "$A_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl" "$A_RUN/evidence_adapt_dev_teacher_pcd_index_summary.json" & pc=$!
wait "$pb"; rb=$?
wait "$pc"; rc=$?
set -e
accept "$rb" B || exit 1; accept "$rc" C || exit 1

# D/Main then uses both GPUs, one variant per device.
set +e; run_arm D "$D_RUN" "$GPU0" "$GPU1" 0 "$A_RUN/evidence_adapt_teacher_pcd_index.jsonl" "$A_RUN/evidence_adapt_teacher_pcd_index_summary.json" "$A_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl" "$A_RUN/evidence_adapt_dev_teacher_pcd_index_summary.json"; rd=$?; set -e; accept "$rd" D || exit 1
python tools/summarize_v48_46_runtime_telemetry.py --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.56-runtime-telemetry-summary.json" || true
AUDIT="$BASE_OUT/OC-RAP-v48.56-DCP-DRFC-BCDE-DRAC-2x2-audit.json"; python tools/compare_v48_56_dcp_drfc_bcde_drac_2x2.py --a "$A_RUN" --b "$B_RUN" --c "$C_RUN" --d "$D_RUN" --output "$AUDIT"
python tools/audit_v48_56_teacher_component_correctness.py --run "$D_RUN" \
  --output "$BASE_OUT/OC-RAP-v48.56-teacher-component-correctness-audit-D.json" || true
D_RC="$(python - "$D_RUN/AUTHORITATIVE_RUN_STATUS.json" <<'PY'
import json,sys; print(int(json.load(open(sys.argv[1])).get('authoritative_exit_code',99)))
PY
)"
if [[ "$D_RC" == 0 ]]; then MAIN_RUN="$D_RUN" bash scripts/run_v48_56_postgate_if_authorized.sh; elif [[ "$D_RC" == 20 ]]; then echo "BLOCKED: v48.56 Main Natural gate failed (RC=20)."; else echo "ENGINEERING FAILURE: D RC=$D_RC" >&2; exit 1; fi
cd "$BASE_OUT"; for run in "$A_RUN" "$B_RUN" "$C_RUN" "$D_RUN"; do b="$(basename "$run")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
cp -f "$AUDIT" "$BASE_OUT/OC-RAP-v48.56-DCP-DRFC-BCDE-DRAC-2x2-audit.upload.json"
echo "v48.56 complete. Upload fresh A/B/C/Main ZIPs + 2x2 audit + teacher audits + runtime telemetry."
