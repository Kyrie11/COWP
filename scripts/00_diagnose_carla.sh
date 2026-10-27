#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
echo '=== RIDE/CARLA environment ==='
printf 'RIDE_HOME=%s\nCARLA_ROOT=%s\nCARLA_HOST=%s\nCARLA_RPC_PORT=%s\nCARLA_TM_PORT=%s\nCARLA_GPU_ADAPTER=%s\n' \
  "$RIDE_HOME" "$CARLA_ROOT" "$CARLA_HOST" "$CARLA_RPC_PORT" "$CARLA_TM_PORT" "${CARLA_GPU_ADAPTER:-auto}"
printf 'Python executable: '; command -v python || true
python - <<'PY'
import os, sys, importlib.util
print('sys.executable=',sys.executable)
print('PYTHONPATH=',os.environ.get('PYTHONPATH',''))
for name in ('carla','agents'):
    try:
        spec=importlib.util.find_spec(name)
        print(f'{name} module=',None if spec is None else spec.origin)
    except Exception as exc:
        print(f'{name} import probe failed: {exc}')
PY
echo '=== Listening TCP ports (status only) ==='
python -m ride.carla.ports --mode status --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
echo '=== CARLA processes (may be restricted for other users) ==='
ps -eo pid,ppid,user,comm,args | grep -E '[C]arlaUE4|[C]arlaUnreal' || true
if command -v nvidia-smi >/dev/null 2>&1; then
  echo '=== GPU process and memory snapshot ==='
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
fi
