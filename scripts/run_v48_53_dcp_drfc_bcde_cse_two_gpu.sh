#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
export CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
SOURCE_RUN="${SOURCE_RUN:-$BASE_OUT/ocrap_v48_45_source_rebuild_s7}"
REF_A="${V4853_REFERENCE_A:-$BASE_OUT/ocrap_v48_52_dcp_drfc_bcde_psa_ablation_A}"
REF_B="${V4853_REFERENCE_B:-$BASE_OUT/ocrap_v48_52_dcp_drfc_bcde_psa_main}"
FORCE_FRESH="${V4853_FORCE_FRESH_REFERENCE:-0}"
export BASE_OUT SOURCE_RUN OCRAP_ROOT PROTOCOL_ROOT CAL_NEAR CAL_CONTACT CAL_SAFE
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-3}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-3}"
export CACHE_SAMPLES_IN_MEMORY=false PERSISTENT_TENSOR_CACHE=true
export PERSISTENT_TENSOR_CACHE_DIR="${OCRAP_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4846}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

GPU_WAIT_TIMEOUT_MIN="${V4853_GPU_WAIT_TIMEOUT_MIN:-240}"; GPU_POLL_SECONDS="${V4853_GPU_POLL_SECONDS:-30}"; GPU_MIN_FREE_MB="${V4853_GPU_MIN_FREE_MB:-12000}"
free_mb(){ nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9.'; }
wait_gpu(){ local g="$1" role="$2" start now free; command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi missing" >&2; return 30; }; start="$(date +%s)"; while true; do free="$(free_mb "$g")"; if [[ "$free" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v f="$free" -v m="$GPU_MIN_FREE_MB" 'BEGIN{exit !(f>=m)}'; then return 0; fi; now="$(date +%s)"; (( now-start < GPU_WAIT_TIMEOUT_MIN*60 )) || { echo "GPU lease timeout role=$role gpu=$g free=${free:-unknown}" >&2; return 30; }; sleep "$GPU_POLL_SECONDS"; done; }

PERF_LOG="$BASE_OUT/OC-RAP-v48.53-runtime-telemetry.jsonl"; : > "$PERF_LOG"; perf_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
 ( while true; do ts="$(date +%s)"; load="$(awk '{print $1","$2","$3}' /proc/loadavg 2>/dev/null || true)"; mem="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || true)"; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | TS="$ts" LOAD="$load" MEMKB="$mem" python -c 'import json,sys,os
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]; print(json.dumps({"unix":float(os.environ["TS"]),"loadavg":os.environ.get("LOAD",""),"mem_available_kb":int(os.environ["MEMKB"]) if os.environ.get("MEMKB") else None,"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":float(p[5]) if p[5] not in {"N/A","[N/A]"} else None}))' >> "$PERF_LOG" || true; sleep "${V4853_TELEMETRY_INTERVAL_S:-30}"; done ) & perf_pid=$!
fi
cleanup(){ [[ -z "$perf_pid" ]] || kill "$perf_pid" 2>/dev/null || true; }; trap cleanup EXIT INT TERM

bash scripts/prepare_v48_45_protocol.sh
export V4845_SKIP_PROTOCOL_PREPARE=1
SEAL="$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json"; [[ -s "$SEAL" ]] || { echo "missing protocol seal" >&2; exit 30; }
REUSE_CONTRACT="$BASE_OUT/OC-RAP-v48.53-AB-reference-reuse-contract.json"
reuse=0
if [[ "$FORCE_FRESH" != 1 && -d "$REF_A" && -d "$REF_B" ]]; then
  set +e; python tools/check_v48_53_ab_reference_reuse.py --a "$REF_A" --b "$REF_B" --protocol-seal "$SEAL" --source-run "$SOURCE_RUN" --output "$REUSE_CONTRACT"; rr=$?; set -e
  if [[ "$rr" == 0 ]]; then reuse=1; echo "v48.53: reusing validated v48.52 A/B references"; else echo "v48.53: A/B reuse contract rejected; running fresh four-arm fallback" >&2; fi
fi

A_NEW="$BASE_OUT/ocrap_v48_53_dcp_drfc_bcde_cse_ablation_A"; B_NEW="$BASE_OUT/ocrap_v48_53_dcp_drfc_bcde_cse_ablation_B"; C_RUN="$BASE_OUT/ocrap_v48_53_dcp_drfc_bcde_cse_ablation_C"; D_RUN="$BASE_OUT/ocrap_v48_53_dcp_drfc_bcde_cse_main"
run_arm(){ local arm="$1" gpu="$2" out="$3"; rm -rf "$out"; mkdir -p "$out/logs"; wait_gpu "$gpu" "arm-$arm" || return $?; date +%s.%N > "$out/logs/v48_53_launcher.start_unix"; set +e; BASE_OUT="$BASE_OUT" GPU0="$gpu" GPU1="$gpu" SERIAL_VARIANTS_ON_ONE_GPU=1 bash scripts/run_v48_53_dcp_drfc_bcde_cse_arm.sh "$arm" >"$out/logs/v48_53_launcher.log" 2>&1; rc=$?; set -e; printf '%s\n' "$rc" > "$out/logs/v48_53_launcher.rc"; date +%s.%N > "$out/logs/v48_53_launcher.end_unix"; python - "$out" "$arm" "$rc" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); l=out/'logs'; a=float((l/'v48_53_launcher.start_unix').read_text()); b=float((l/'v48_53_launcher.end_unix').read_text()); (out/'V48_53_RUNTIME.json').write_text(json.dumps({'arm':sys.argv[2],'launcher_start_unix':a,'launcher_end_unix':b,'wall_seconds':b-a,'exit_code':int(sys.argv[3]),'variant_parallelism':'serial_on_single_gpu'},indent=2)+'\n')
PY
 return "$rc"; }
accept(){ case "$1" in 0) echo "$2: RC=0";; 20) echo "$2: RC=20 algorithm rejection";; *) echo "$2: RC=$1 ENGINEERING FAILURE" >&2; return 1;; esac; }

if [[ "$reuse" == 1 ]]; then A_RUN="$REF_A"; B_RUN="$REF_B"; else
  run_arm A "$GPU0" "$A_NEW" & pa=$!; run_arm B "$GPU1" "$B_NEW" & pb=$!; set +e; wait "$pa"; ra=$?; wait "$pb"; rb=$?; set -e; accept "$ra" A || exit 1; accept "$rb" B || exit 1; A_RUN="$A_NEW"; B_RUN="$B_NEW";
fi
run_arm C "$GPU0" "$C_RUN" & pc=$!; run_arm D "$GPU1" "$D_RUN" & pd=$!; set +e; wait "$pc"; rc=$?; wait "$pd"; rd=$?; set -e; accept "$rc" C || exit 1; accept "$rd" D || exit 1

python tools/summarize_v48_46_runtime_telemetry.py --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.53-runtime-telemetry-summary.json" || true
AUDIT="$BASE_OUT/OC-RAP-v48.53-DCP-DRFC-BCDE-CSE-2x2-audit.json"
python tools/compare_v48_53_dcp_drfc_bcde_cse_2x2.py --a "$A_RUN" --b "$B_RUN" --c "$C_RUN" --d "$D_RUN" --output "$AUDIT"
D_RC="$(python - "$D_RUN/AUTHORITATIVE_RUN_STATUS.json" <<'PY'
import json,sys; print(int(json.load(open(sys.argv[1],encoding='utf-8')).get('authoritative_exit_code',99)))
PY
)"
if [[ "$D_RC" == 0 ]]; then MAIN_RUN="$D_RUN" bash scripts/run_v48_53_postgate_if_authorized.sh; elif [[ "$D_RC" == 20 ]]; then echo "BLOCKED: v48.53 D/Main Natural gate failed (RC=20)."; else echo "ENGINEERING FAILURE: D RC=$D_RC" >&2; exit 1; fi
cd "$BASE_OUT"
for d in "$C_RUN" "$D_RUN"; do b="$(basename "$d")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
if [[ "$reuse" != 1 ]]; then for d in "$A_NEW" "$B_NEW"; do b="$(basename "$d")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done; fi
cp -f "$AUDIT" "$BASE_OUT/OC-RAP-v48.53-DCP-DRFC-BCDE-CSE-2x2-audit.upload.json"
echo "v48.53 complete. Upload C/D ZIP + 2x2 audit + runtime telemetry; upload A/B ZIP only if reuse=false."
