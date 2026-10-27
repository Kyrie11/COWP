#!/usr/bin/env bash
# Read-only crash snapshot. No kill, reload, map tick, or privileged operation.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
mkdir -p "$RIDE_OUTPUT_ROOT/diagnostics"
OUT="$RIDE_OUTPUT_ROOT/diagnostics/carla_${CARLA_RPC_PORT}_$(date +%Y%m%d_%H%M%S).log"
{
  echo "========== CARLA crash snapshot =========="
  date -Is
  printf 'CARLA_ROOT=%s RPC=%s TM=%s\n' "$CARLA_ROOT" "$CARLA_RPC_PORT" "$CARLA_TM_PORT"
  echo '========== CPU / RAM =========='
  free -h || true
  echo '========== available disk =========='
  df -h "$CARLA_ROOT" "$RIDE_OUTPUT_ROOT" || true
  echo '========== ports =========='
  ss -ltnp | grep -E ":($CARLA_RPC_PORT|$((CARLA_RPC_PORT+1))|$CARLA_TM_PORT)\\b" || true
  echo '========== CARLA processes =========='
  ps -eo pid,ppid,user,etime,stat,rss,args | grep -E '[C]arlaUE4|[C]arlaUnreal' || true
  echo '========== GPU =========='
  if command -v nvidia-smi >/dev/null; then nvidia-smi || true; fi
  echo '========== CARLA server logs =========='
  find "$CARLA_ROOT" -type f \( -name 'CarlaUE4.log' -o -name 'CarlaUnreal.log' \) -path '*/Saved/Logs/*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 || true
  echo '========== newest CARLA log tail =========='
  NEWEST="$(find "$CARLA_ROOT" -type f \( -name 'CarlaUE4.log' -o -name 'CarlaUnreal.log' \) -path '*/Saved/Logs/*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [[ -n "$NEWEST" ]]; then tail -n 90 "$NEWEST" || true; fi
} 2>&1 | tee "$OUT"
printf 'Saved diagnostics: %s\n' "$OUT"
