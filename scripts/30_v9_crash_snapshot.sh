#!/usr/bin/env bash
# CARLA Signal 11 read-only forensic capture. Does not stop or restart any processes.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
OUT="${RIDE_CRASH_REPORT_DIR:-$DATASET_ROOT/RIDE/carla_crash_reports}"
mkdir -p "$OUT"
REPORT="$OUT/carla_crash_$(date -u +%Y%m%dT%H%M%SZ).log"
{
 echo '===== CARLA crash triage (UTC) =====';date -u -Is
 echo "CARLA_ROOT=$CARLA_ROOT RPC=$CARLA_RPC_PORT TM=$CARLA_TM_PORT"
 echo '===== host =====';uname -a || :; uptime || :
 echo '===== free memory / swap ====='; free -h || :
 echo '===== disks, shared-memory ====='; df -h "$CARLA_ROOT" "$OUT" /dev/shm || :
 echo '===== GPU =====';if command -v nvidia-smi >/dev/null; then nvidia-smi || :; fi
 echo '===== CARLA server processes ====='; ps -eo pid,ppid,user,stat,rss,etime,args | grep -E '[C]arlaUE4|[C]arlaUnreal' || :
 echo '===== CARLA ports ====='; ss -lntp 2>/dev/null | grep -E ":($CARLA_RPC_PORT|$((CARLA_RPC_PORT+1))|$CARLA_TM_PORT)\\b" || :
 echo '===== kernel errors, available without sudo ====='
 dmesg --ctime 2>/dev/null | grep -Ei 'nvrm.*xid|oom-kill|out of memory|segfault|vulkan|carlaue' | tail -70 || :
 echo '===== coredumps ====='
 if command -v coredumpctl >/dev/null;then coredumpctl list --no-pager 2>/dev/null | tail -20 || :;fi
 echo '===== newest CARLA and UE4 log files ====='
 dirs=("$HOME/.config/Epic/CarlaUE4/Saved/Logs" \
       "$CARLA_ROOT/CarlaUE4/Saved/Logs" \
       "$CARLA_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4/Saved/Logs")
 for d in "${dirs[@]}";do
   if [[ -d "$d" ]];then
     echo "DIRECTORY $d"
     find "$d" -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -10 || :
     recent=$(find "$d" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
     if [[ -n "$recent" ]];then echo "TAIL $recent";tail -n 160 "$recent" || :;fi
   fi
 done
 echo '===== end ====='
} 2>&1 | tee "$REPORT"
echo "RIDE_V9_CRASH_REPORT=$REPORT"
