#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment}"
export COWP_ROOT="${COWP_ROOT:-$DATA_ROOT}"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
export USE_WAYMAX_OUTCOME_LABELS="${USE_WAYMAX_OUTCOME_LABELS:-0}"
export OFFLINE_OUTCOME_RISK_PENALTY="${OFFLINE_OUTCOME_RISK_PENALTY:-0.0}"
export DATA_PROTOCOL="${DATA_PROTOCOL:-v16_8_6_priority_commitment_fresh}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_6_priority_commitment_seed2026}"
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/eval/probe"

"$PYTHON_BIN" -u -m cowp.scripts.53_gate_fresh_v16_8_6_cache_protocol \
  --cowp-root "$COWP_ROOT" \
  --raw-train "$RAW_TRAIN_CACHE" --raw-val "$RAW_VAL_CACHE" \
  --transport-train "$TRAIN_CACHE" --transport-val "$VAL_CACHE" \
  --sample-scenes "${FRESH_CACHE_GATE_SAMPLE_SCENES:-256}" \
  --output "$OUT_ROOT/configs/fresh_cache_protocol_gate_v16_8_6.json" \
  | tee "$OUT_ROOT/logs/fresh_cache_protocol_gate_v16_8_6.log"

# The full closed-loop run is expensive.  Require the paired Waymax probe itself
# to pass before expansion, not merely the existence of a probe JSON file.
PROBE_SCENARIOS="${PROBE_SCENARIOS:-100}"
PROBE_REF="$OUT_ROOT/eval/probe/conventional_safety_${PROBE_SCENARIOS}.json"
PROBE_COWP="$OUT_ROOT/eval/probe/cowp_root_transport_${PROBE_SCENARIOS}.json"
MIN_PROBE_ROLLOUTS="${WAYMAX_PROBE_MIN_ROLLOUTS:-$(( (PROBE_SCENARIOS * 8 + 9) / 10 ))}"
"$PYTHON_BIN" -u -m cowp.scripts.48_gate_waymax_probe_v16_8_4 \
  --reference "$PROBE_REF" --candidate "$PROBE_COWP" \
  --min-rollouts "$MIN_PROBE_ROLLOUTS" \
  --output "$OUT_ROOT/eval/probe/promotion_gate_v16_8_6.json" \
  | tee "$OUT_ROOT/logs/promotion_gate_v16_8_6.log"

exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_2_FULL_CN.sh"
