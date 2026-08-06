#!/usr/bin/env bash
set -euo pipefail

# v16.8.4: 先测 proposal ceiling，再决定是否支付约 4 天的完整重建成本。
# 本脚本只重建一个配对 validation 标签 probe；不训练模型、不运行 Waymax。
PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_4_bcs_rmr_bcte_proposal_probe}"
HARD_COUNT="${HARD_COUNT:-400}"
RANDOM_COUNT="${RANDOM_COUNT:-800}"
LABEL_WORKERS="${LABEL_WORKERS:-24}"
SEED="${SEED:-2026}"
FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"

SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
LABEL_CFG="configs/label_cowp_v16_8.yaml"
CURRENT_DIAG="$PROBE_ROOT/current_proposal_ceiling.json"
HARD_IDS="$PROBE_ROOT/hard_scene_ids.txt"
RANDOM_IDS="$PROBE_ROOT/representative_random_scene_ids.txt"
PROBE_IDS="$PROBE_ROOT/probe_union_scene_ids.txt"
FRESH_LABELS="$PROBE_ROOT/labels_val_bcs_rmr_bcte"
COMPARE_JSON="$PROBE_ROOT/paired_proposal_probe.json"

mkdir -p "$PROBE_ROOT/logs"
if [[ "$FORCE_REBUILD_PROBE" == "1" ]]; then
  rm -rf "$FRESH_LABELS"
fi
mkdir -p "$FRESH_LABELS"
FINGERPRINT="$($PYTHON_BIN - <<'PYHASH'
import hashlib
from pathlib import Path
files = [
    'cowp/geometry/lane_graph.py',
    'cowp/label/trajectory_primitives.py',
    'cowp/label/ego_candidates.py',
    'cowp/label/label_engine.py',
    'cowp/data/cache_schema.py',
    'configs/label_cowp_v16_8.yaml',
    'cowp/scripts/45_diagnose_proposal_ceiling.py',
    'cowp/scripts/46_compare_proposal_probe.py',
]
h = hashlib.sha256()
for name in files:
    h.update(name.encode()); h.update(Path(name).read_bytes())
print(h.hexdigest())
PYHASH
)"
printf '%s\n' "$FINGERPRINT" > "$PROBE_ROOT/code_fingerprint.sha256"
run() {
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$PROBE_ROOT/logs/${name}.log") 2> >(tee -a "$PROBE_ROOT/logs/${name}.log" >&2)
}

# A. 在旧 cache 上计算不可被 selector/threshold 突破的 proposal floor，
#    并抽取 hard scenes + 无偏随机 scenes。
run diagnose_current "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$OLD_VAL_CACHE" --output "$CURRENT_DIAG" \
  --hard-scene-ids "$HARD_IDS" --hard-count "$HARD_COUNT" \
  --random-scene-ids "$RANDOM_IDS" --random-count "$RANDOM_COUNT" \
  --control-count 0 --seed "$SEED"
cat "$HARD_IDS" "$RANDOM_IDS" | awk 'NF && !seen[$1]++ {print $1}' > "$PROBE_IDS"
PROBE_TOTAL="$(wc -l < "$PROBE_IDS" | tr -d ' ')"
echo "[probe] unique scenarios: $PROBE_TOTAL"

# B. 只为 probe 场景重建 fresh labels。这里会实际使用：
#    1) 修正后的 jerk filter；2) BCS-RMR-BCTE；3) boundary-consistent proposal provenance。
run build_fresh_probe "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH_LABELS" \
  --allow-scenario-ids "$PROBE_IDS" --limit "$PROBE_TOTAL" \
  --num-workers "$LABEL_WORKERS" --start-method forkserver \
  --max-pending-multiplier 2 --no-compress --skip-existing \
  --skip-diagnostics --cpu-only

# C. 按场景 ID 配对比较。总体覆盖只在 representative random 子集上统计，
#    hard recovery 单独在旧 bank 无 NCF 的场景上统计，避免采样偏差。
run compare_probe "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe \
  --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH_LABELS" \
  --representative-scene-ids "$RANDOM_IDS" --hard-scene-ids "$HARD_IDS" \
  --output "$COMPARE_JSON" \
  --min-overall-any-ncf 0.40 --max-false-safe-floor 0.55 \
  --min-hard-recovery 0.20 \
  --max-rmr-target-tta-error-s 0.20

"$PYTHON_BIN" - "$COMPARE_JSON" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
print("\n===== v16.8.4 BCS-RMR-BCTE proposal probe verdict =====")
print(json.dumps({
  "pairing_completeness": p["pairing_completeness"],
  "old": p["old"],
  "new": p["new"],
  "paired": p["paired"],
  "gate_checks": p["gate_checks"],
  "promote_to_full_rebuild": p["promote_to_full_rebuild"],
}, indent=2, ensure_ascii=False))
if p["promote_to_full_rebuild"]:
    print("PASS: 允许进入 PREPARE_COWP_V16_8_4_DATA_CN.sh 全量重建。")
else:
    print("STOP: 不要全量重建；先查看 proposal_source/macro 诊断并继续修改候选生成器。")
PY

echo "CURRENT_DIAG=$CURRENT_DIAG"
echo "FRESH_LABELS=$FRESH_LABELS"
echo "COMPARE_JSON=$COMPARE_JSON"
