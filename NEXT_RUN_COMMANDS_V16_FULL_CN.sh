#!/usr/bin/env bash
set -euo pipefail
# natural 阶段通过两个门禁后执行。已存在的完整 natural checkpoint/history 会被复用。
STOP_AFTER_STAGE=none RUN_DIAGNOSE=0 bash "$(cd "$(dirname "$0")" && pwd)/NEXT_RUN_COMMANDS_V16_CN.sh"
