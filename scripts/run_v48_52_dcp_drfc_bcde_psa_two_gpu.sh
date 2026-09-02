#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
BASE_OUT="$(python - "$BASE_OUT" <<'PY_BASE'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY_BASE
)"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
if [[ "$GPU0" == "$GPU1" && "${V4852_ALLOW_SHARED_GPU_IDS:-0}" != 1 ]]; then
  echo "v48.52 requires two distinct GPU ids (GPU0=$GPU0 GPU1=$GPU1)" >&2
  exit 2
fi
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
export CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
SOURCE_RUN="${SOURCE_RUN:-$BASE_OUT/ocrap_v48_45_source_rebuild_s7}"
REFERENCE_RUN="${V4852_REFERENCE_RUN:-$BASE_OUT/ocrap_v48_51_dcp_drfc_bcde_ablation_B}"
FORCE_FRESH_REFERENCE="${V4852_FORCE_FRESH_REFERENCE:-0}"
export BASE_OUT SOURCE_RUN OCRAP_ROOT PROTOCOL_ROOT CAL_NEAR CAL_CONTACT CAL_SAFE
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-3}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-3}"
export CACHE_SAMPLES_IN_MEMORY=false
export PERSISTENT_TENSOR_CACHE=true
export PERSISTENT_TENSOR_CACHE_DIR="${OCRAP_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4846}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

GPU_WAIT_TIMEOUT_MIN="${V4852_GPU_WAIT_TIMEOUT_MIN:-240}"
GPU_POLL_SECONDS="${V4852_GPU_POLL_SECONDS:-30}"
GPU_MIN_FREE_MB="${V4852_GPU_MIN_FREE_MB:-12000}"

free_mb() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9.'
}
compute_apps_json() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
    python -c 'import json,sys; rows=[]
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 if p and p[0].isdigit(): rows.append({"pid":int(p[0]),"used_memory_mb":float(p[1]) if len(p)>1 and p[1].replace(".","",1).isdigit() else None})
print(json.dumps(rows,separators=(",",":")))' || printf '[]\n'
}
wait_for_gpu_lease() {
  local gpu="$1" role="$2" start now free apps
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi missing; cannot establish GPU lease" >&2; return 30; }
  start="$(date +%s)"
  while true; do
    free="$(free_mb "$gpu")"; apps="$(compute_apps_json "$gpu")"
    if [[ "$free" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v f="$free" -v m="$GPU_MIN_FREE_MB" 'BEGIN{exit !(f>=m)}'; then
      GPU="$gpu" ROLE="$role" FREE="$free" APPS="$apps" MINFREE="$GPU_MIN_FREE_MB" python - <<'PY'
import json,os,time
print(json.dumps({'event':'v48_52_gpu_lease_granted','unix':time.time(),'gpu':int(os.environ['GPU']),
 'role':os.environ['ROLE'],'free_mb':float(os.environ['FREE']),'minimum_free_mb':float(os.environ['MINFREE']),
 'preexisting_compute_apps':json.loads(os.environ['APPS'])},separators=(',',':')))
PY
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= GPU_WAIT_TIMEOUT_MIN * 60 )); then
      echo "GPU lease timeout role=$role gpu=$gpu free_mb=${free:-unknown} required=$GPU_MIN_FREE_MB apps=$apps" >&2
      return 30
    fi
    echo "waiting for GPU lease: role=$role gpu=$gpu free_mb=${free:-unknown} required=$GPU_MIN_FREE_MB" >&2
    sleep "$GPU_POLL_SECONDS"
  done
}

PERF_LOG="$BASE_OUT/OC-RAP-v48.52-runtime-telemetry.jsonl"
: > "$PERF_LOG"
perf_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      ts="$(date +%s)"; load="$(awk '{print $1","$2","$3}' /proc/loadavg 2>/dev/null || true)"
      mem_kb="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || true)"
      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | \
      TS="$ts" LOAD="$load" MEMKB="$mem_kb" python -c 'import json,sys,os
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 print(json.dumps({"unix":float(os.environ["TS"]),"loadavg":os.environ.get("LOAD",""),"mem_available_kb":int(os.environ["MEMKB"]) if os.environ.get("MEMKB") else None,"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":float(p[5]) if p[5] not in {"N/A","[N/A]"} else None}))' >> "$PERF_LOG" || true
      sleep "${V4852_TELEMETRY_INTERVAL_S:-30}"
    done
  ) & perf_pid=$!
fi
cleanup_perf(){ [[ -z "$perf_pid" ]] || kill "$perf_pid" 2>/dev/null || true; }
trap cleanup_perf EXIT INT TERM

# Prepare once, then make all arm/reference checks consume the same immutable seal.
bash scripts/prepare_v48_45_protocol.sh
export V4845_SKIP_PROTOCOL_PREPARE=1
SEAL="$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json"
[[ -s "$SEAL" ]] || { echo "missing shared protocol seal: $SEAL" >&2; exit 30; }

REFERENCE_CONTRACT="$BASE_OUT/OC-RAP-v48.52-reference-reuse-contract.json"
reference_reused=0
if [[ "$FORCE_FRESH_REFERENCE" != 1 && -d "$REFERENCE_RUN" ]]; then
  set +e
  python tools/check_v48_52_reference_reuse.py \
    --run "$REFERENCE_RUN" --protocol-seal "$SEAL" --source-run "$SOURCE_RUN" \
    --output "$REFERENCE_CONTRACT"
  ref_rc=$?
  set -e
  if [[ "$ref_rc" == 0 ]]; then
    reference_reused=1
    echo "v48.52: reusing validated v48.51-B reference: $REFERENCE_RUN"
  else
    echo "v48.52: historical reference failed reuse contract; fresh A/B will be run" >&2
  fi
fi

A_FRESH="$BASE_OUT/ocrap_v48_52_dcp_drfc_bcde_psa_ablation_A"
MAIN="$BASE_OUT/ocrap_v48_52_dcp_drfc_bcde_psa_main"

run_single_gpu_arm() {
  local arm="$1" gpu="$2" out="$3" log rc
  rm -rf "$out"; mkdir -p "$out/logs"
  wait_for_gpu_lease "$gpu" "arm-$arm" >"$out/logs/v48_52_gpu_lease.json" || return $?
  date +%s.%N >"$out/logs/v48_52_launcher.start_unix"
  log="$out/logs/v48_52_launcher.log"
  set +e
  BASE_OUT="$BASE_OUT" GPU0="$gpu" GPU1="$gpu" SERIAL_VARIANTS_ON_ONE_GPU=1 \
    bash scripts/run_v48_52_dcp_drfc_bcde_psa_arm.sh "$arm" >"$log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$out/logs/v48_52_launcher.rc"; date +%s.%N >"$out/logs/v48_52_launcher.end_unix"
  python - "$out" "$arm" "$rc" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); logs=out/'logs'; start=float((logs/'v48_52_launcher.start_unix').read_text()); end=float((logs/'v48_52_launcher.end_unix').read_text())
(out/'V48_52_RUNTIME.json').write_text(json.dumps({'arm':sys.argv[2],'launcher_start_unix':start,'launcher_end_unix':end,'wall_seconds':end-start,'exit_code':int(sys.argv[3]),'variant_parallelism':'serial_on_single_gpu'},indent=2)+'\n')
PY
  return "$rc"
}

run_dual_gpu_main() {
  local log rc
  rm -rf "$MAIN"; mkdir -p "$MAIN/logs"
  wait_for_gpu_lease "$GPU0" "main-balanced" >"$MAIN/logs/v48_52_gpu0_lease.json" || return $?
  wait_for_gpu_lease "$GPU1" "main-precision" >"$MAIN/logs/v48_52_gpu1_lease.json" || return $?
  GPU0="$GPU0" GPU1="$GPU1" python - "$MAIN/V48_52_GPU_SCHEDULER_DECISION.json" <<'PY'
import json,os,pathlib,sys,time
pathlib.Path(sys.argv[1]).write_text(json.dumps({'event':'v48_52_gpu_scheduler_decision','created_unix':time.time(),
 'gpu_balanced':int(os.environ['GPU0']),'gpu_precision':int(os.environ['GPU1']),
 'parallel_balanced_precision':True,'reference_reused':True,'strategy_regime_conditioning':False},indent=2)+'\n')
PY
  date +%s.%N >"$MAIN/logs/v48_52_launcher.start_unix"; log="$MAIN/logs/v48_52_launcher.log"
  set +e
  BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" SERIAL_VARIANTS_ON_ONE_GPU=0 \
    bash scripts/run_v48_52_dcp_drfc_bcde_psa_arm.sh B >"$log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$MAIN/logs/v48_52_launcher.rc"; date +%s.%N >"$MAIN/logs/v48_52_launcher.end_unix"
  python - "$MAIN" "$rc" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); logs=out/'logs'; start=float((logs/'v48_52_launcher.start_unix').read_text()); end=float((logs/'v48_52_launcher.end_unix').read_text())
(out/'V48_52_RUNTIME.json').write_text(json.dumps({'arm':'B','launcher_start_unix':start,'launcher_end_unix':end,'wall_seconds':end-start,'exit_code':int(sys.argv[2]),'variant_parallelism':'balanced_precision_two_gpu'},indent=2)+'\n')
PY
  return "$rc"
}

accept_algorithm_rc() {
  local rc="$1" label="$2"
  case "$rc" in
    0) echo "$label: RC=0 gate passed" ;;
    20) echo "$label: RC=20 pipeline-valid algorithm rejection" ;;
    *) echo "$label: RC=$rc ENGINEERING FAILURE" >&2; return 1 ;;
  esac
}

if [[ "$reference_reused" == 1 ]]; then
  set +e; run_dual_gpu_main; main_rc=$?; set -e
  accept_algorithm_rc "$main_rc" "v48.52 Main" || exit 1
  A_RUN="$REFERENCE_RUN"
else
  # Fresh fallback remains a clean A/B experiment: each arm owns one GPU and
  # runs Balanced/Precision serially, exactly matching the v48.51 scheduler.
  run_single_gpu_arm A "$GPU0" "$A_FRESH" & pa=$!
  run_single_gpu_arm B "$GPU1" "$MAIN" & pb=$!
  set +e; wait "$pa"; a_rc=$?; wait "$pb"; main_rc=$?; set -e
  accept_algorithm_rc "$a_rc" "v48.52 fresh A" || exit 1
  accept_algorithm_rc "$main_rc" "v48.52 Main" || exit 1
  A_RUN="$A_FRESH"
fi

python tools/summarize_v48_46_runtime_telemetry.py \
  --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.52-runtime-telemetry-summary.json" || true

AUDIT="$BASE_OUT/OC-RAP-v48.52-DCP-DRFC-BCDE-PSA-AB-audit.json"
python tools/compare_v48_52_dcp_drfc_bcde_psa_ab.py \
  --a "$A_RUN" --b "$MAIN" --output "$AUDIT"

RC="$(python - "$MAIN/AUTHORITATIVE_RUN_STATUS.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1],encoding='utf-8')).get('authoritative_exit_code',99)))
PY
)"
if [[ "$RC" == 0 ]]; then
  MAIN_RUN="$MAIN" bash scripts/run_v48_52_postgate_if_authorized.sh
elif [[ "$RC" == 20 ]]; then
  echo "BLOCKED: v48.52 Main Natural gate failed (RC=20). Do not run authoritative test/closed-loop."
else
  echo "ENGINEERING FAILURE: v48.52 Main authoritative RC=$RC" >&2
  exit 1
fi

# Package only newly generated runs.  Historical v48.51-B is referenced by the
# reuse contract/audit and does not need to be duplicated.
cd "$BASE_OUT"
if [[ "$reference_reused" != 1 && -d "$(basename "$A_FRESH")" ]]; then
  rm -f "$(basename "$A_FRESH").zip"
  zip -qr "$(basename "$A_FRESH").zip" "$(basename "$A_FRESH")"
fi
if [[ -d "$(basename "$MAIN")" ]]; then
  rm -f "$(basename "$MAIN").zip"
  zip -qr "$(basename "$MAIN").zip" "$(basename "$MAIN")"
fi
cp -f "$AUDIT" "$BASE_OUT/OC-RAP-v48.52-DCP-DRFC-BCDE-PSA-AB-audit.upload.json"

echo "v48.52 complete. Upload Main ZIP + AB audit + runtime telemetry summary; upload fresh A ZIP only if reference_reused=false."
