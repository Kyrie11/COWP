#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}" CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}" CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
SOURCE_RUN="${SOURCE_RUN:-$BASE_OUT/ocrap_v48_45_source_rebuild_s7}"; REF_A="${V4854_REFERENCE_A:-$BASE_OUT/ocrap_v48_53_dcp_drfc_bcde_cse_ablation_A}"; FORCE_FRESH="${V4854_FORCE_FRESH_REFERENCE:-0}"
export BASE_OUT SOURCE_RUN GPU0 GPU1; export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-3}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-3}" CACHE_SAMPLES_IN_MEMORY=false PERSISTENT_TENSOR_CACHE=true
export PERSISTENT_TENSOR_CACHE_DIR="${OCRAP_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4846}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

PERF_LOG="$BASE_OUT/OC-RAP-v48.54-runtime-telemetry.jsonl"; : > "$PERF_LOG"; perf_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
 ( while true; do ts="$(date +%s)"; load="$(awk '{print $1","$2","$3}' /proc/loadavg 2>/dev/null || true)"; mem="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || true)"; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | TS="$ts" LOAD="$load" MEMKB="$mem" python -c 'import json,sys,os
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]; print(json.dumps({"unix":float(os.environ["TS"]),"loadavg":os.environ.get("LOAD",""),"mem_available_kb":int(os.environ["MEMKB"]) if os.environ.get("MEMKB") else None,"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":float(p[5]) if p[5] not in {"N/A","[N/A]"} else None}))' >> "$PERF_LOG" || true; sleep "${V4854_TELEMETRY_INTERVAL_S:-30}"; done ) & perf_pid=$!
fi
cleanup(){ [[ -z "$perf_pid" ]] || kill "$perf_pid" 2>/dev/null || true; }; trap cleanup EXIT INT TERM

bash scripts/prepare_v48_45_protocol.sh; export V4845_SKIP_PROTOCOL_PREPARE=1
NEAR_CERT="$PROTOCOL_ROOT/certificate_pool_near_contact"; CONTACT_CERT="$PROTOCOL_ROOT/certificate_pool_contact"; NEAR_DEV="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; CONTACT_DEV="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
REUSE_CONTRACT="$BASE_OUT/OC-RAP-v48.54-A-reference-reuse-contract.json"; reuse=0
if [[ "$FORCE_FRESH" != 1 && -d "$REF_A" ]]; then
 set +e; python tools/check_v48_54_reference_reuse.py --reference "$REF_A" --source-run "$SOURCE_RUN" --safe "$CAL_SAFE" --near-cert "$NEAR_CERT" --contact-cert "$CONTACT_CERT" --near-dev "$NEAR_DEV" --contact-dev "$CONTACT_DEV" --output "$REUSE_CONTRACT"; rr=$?; set -e
 if [[ "$rr" == 0 ]]; then reuse=1; echo "v48.54: semantically reusing validated v48.53-A reference"; else echo "v48.54: reference reuse rejected; running fresh A" >&2; fi
fi
A_NEW="$BASE_OUT/ocrap_v48_54_dcp_drfc_bcde_ipbd_ablation_A"; B_RUN="$BASE_OUT/ocrap_v48_54_dcp_drfc_bcde_ipbd_main"
run_arm(){ local arm="$1" out="$2"; rm -rf "$out"; mkdir -p "$out/logs"; date +%s.%N > "$out/logs/v48_54_launcher.start_unix"; set +e; BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" SERIAL_VARIANTS_ON_ONE_GPU=0 bash scripts/run_v48_54_dcp_drfc_bcde_ipbd_arm.sh "$arm" >"$out/logs/v48_54_launcher.log" 2>&1; rc=$?; set -e; printf '%s\n' "$rc" > "$out/logs/v48_54_launcher.rc"; date +%s.%N > "$out/logs/v48_54_launcher.end_unix"; return "$rc"; }
accept(){ case "$1" in 0) echo "$2: RC=0";; 20) echo "$2: RC=20 algorithm rejection";; *) echo "$2: RC=$1 ENGINEERING FAILURE" >&2; return 1;; esac; }
if [[ "$reuse" == 1 ]]; then A_RUN="$REF_A"; else set +e; run_arm A "$A_NEW"; ra=$?; set -e; accept "$ra" A || exit 1; A_RUN="$A_NEW"; fi
set +e; run_arm B "$B_RUN"; rb=$?; set -e; accept "$rb" B || exit 1
python tools/summarize_v48_46_runtime_telemetry.py --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.54-runtime-telemetry-summary.json" || true
AUDIT="$BASE_OUT/OC-RAP-v48.54-DCP-DRFC-BCDE-IPBD-AB-audit.json"; python tools/compare_v48_54_dcp_drfc_bcde_ipbd_ab.py --a "$A_RUN" --b "$B_RUN" --output "$AUDIT"
B_RC="$(python - "$B_RUN/AUTHORITATIVE_RUN_STATUS.json" <<'PY'
import json,sys; print(int(json.load(open(sys.argv[1])).get('authoritative_exit_code',99)))
PY
)"
if [[ "$B_RC" == 0 ]]; then MAIN_RUN="$B_RUN" bash scripts/run_v48_54_postgate_if_authorized.sh; elif [[ "$B_RC" == 20 ]]; then echo "BLOCKED: v48.54 Main Natural gate failed (RC=20)."; else echo "ENGINEERING FAILURE: B RC=$B_RC" >&2; exit 1; fi
cd "$BASE_OUT"; b="$(basename "$B_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; if [[ "$reuse" != 1 ]]; then a="$(basename "$A_NEW")"; rm -f "$a.zip"; zip -qr "$a.zip" "$a"; fi
cp -f "$AUDIT" "$BASE_OUT/OC-RAP-v48.54-DCP-DRFC-BCDE-IPBD-AB-audit.upload.json"
echo "v48.54 complete. Upload Main ZIP + AB audit + runtime telemetry; upload fresh A ZIP only if reuse=false."
