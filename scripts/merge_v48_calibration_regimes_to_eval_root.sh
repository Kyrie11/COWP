#!/usr/bin/env bash
set -euo pipefail

# Merge any completed dedicated-calibration worker pairs directly into the
# evaluation OCRAP root. Safe/Near can be merged before Contact is finished.
# Final layout:
#   $TARGET_ROOT/calibration_safe
#   $TARGET_ROOT/calibration_near_contact
#   $TARGET_ROOT/calibration_contact

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

BUILD_ROOT="${BUILD_ROOT:-/data0/senzeyu2/dataset/OCRAP/calibration}"
SHARD_ROOT="${SHARD_ROOT:-$BUILD_ROOT/shards}"
TARGET_ROOT="${TARGET_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-$TARGET_ROOT}"
REGIMES="${REGIMES:-safe,near_contact,contact}"
OVERWRITE="${OVERWRITE:-0}"
LINK_MODE="${LINK_MODE:-hardlink}"
WORK_ROOT="${WORK_ROOT:-$BUILD_ROOT/merge_to_eval_root}"
LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$LOG_DIR" "$TARGET_ROOT"

case "$LINK_MODE" in hardlink|symlink|copy) ;; *) echo "LINK_MODE must be hardlink, symlink, or copy" >&2; exit 2;; esac
IFS=',' read -r -a REQUESTED <<< "$REGIMES"

exclude_args=()
for d in val_safe test_safe val_near_contact test_near_contact val_contact test_contact; do
  [[ -d "$EVAL_OCRAP_ROOT/$d" ]] && exclude_args+=(--exclude-root "$EVAL_OCRAP_ROOT/$d")
done

shards_for() {
  case "$1" in
    safe) echo "calibration_safe_w0 calibration_safe_w1" ;;
    near_contact|near) echo "calibration_near_w2 calibration_near_w3" ;;
    contact) echo "calibration_contact_w4 calibration_contact_w5" ;;
    *) echo "unknown regime: $1" >&2; return 2 ;;
  esac
}
final_name() {
  case "$1" in
    safe) echo calibration_safe ;;
    near_contact|near) echo calibration_near_contact ;;
    contact) echo calibration_contact ;;
  esac
}

merged_roots=()
for regime0 in "${REQUESTED[@]}"; do
  regime="${regime0// /}"
  [[ -n "$regime" ]] || continue
  read -r shard_a shard_b <<< "$(shards_for "$regime")"
  final="$(final_name "$regime")"
  for shard in "$shard_a" "$shard_b"; do
    [[ -f "$SHARD_ROOT/$shard/manifest.csv" ]] || {
      echo "missing completed shard manifest: $SHARD_ROOT/$shard/manifest.csv" >&2
      echo "Do not request regime=$regime until both workers complete." >&2
      exit 3
    }
  done
  raw="$WORK_ROOT/raw_${final}"
  tmp="$TARGET_ROOT/.${final}.v48_7_tmp"
  rm -rf "$raw" "$tmp"
  merge_args=()
  [[ "$LINK_MODE" == hardlink ]] && merge_args+=(--hardlink)
  python tools/merge_dataset_roots.py "${merge_args[@]}" --output "$raw" \
    "$SHARD_ROOT/$shard_a" "$SHARD_ROOT/$shard_b" \
    2>&1 | tee "$LOG_DIR/merge_${final}.log"
  python tools/filter_dataset_scenes_v48.py --overwrite --input "$raw" --output "$tmp" \
    --link-mode "$LINK_MODE" "${exclude_args[@]}" \
    2>&1 | tee "$LOG_DIR/filter_${final}.log"
  [[ -s "$tmp/manifest.csv" ]] || { echo "empty merged manifest for $final" >&2; exit 4; }
  if [[ -e "$TARGET_ROOT/$final" ]]; then
    [[ "$OVERWRITE" == 1 ]] || {
      echo "$TARGET_ROOT/$final already exists; set OVERWRITE=1 after verifying the destination" >&2
      exit 5
    }
    rm -rf "$TARGET_ROOT/$final"
  fi
  mv "$tmp" "$TARGET_ROOT/$final"
  merged_roots+=("$TARGET_ROOT/$final")
  python - "$TARGET_ROOT/$final" "$shard_a" "$shard_b" <<'PY'
import csv,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
with (root/'manifest.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
scenes={str(r.get('original_scenario_id') or r.get('scene_id') or '') for r in rows}; scenes.discard('')
doc={'event':'dedicated_calibration_regime_merged','created_unix':time.time(),
     'root':str(root),'source_shards':sys.argv[2:],'samples':len(rows),'scenes':len(scenes)}
(root/'MERGE_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
print(doc)
PY
done

audit_args=()
for d in val_safe val_near_contact val_contact test_safe test_near_contact test_contact; do
  [[ -d "$EVAL_OCRAP_ROOT/$d" ]] && audit_args+=(--development-root "$EVAL_OCRAP_ROOT/$d")
done
for root in "${merged_roots[@]}"; do audit_args+=(--test-root "$root"); done
python tools/check_scene_overlap_v48.py "${audit_args[@]}" \
  --output "$WORK_ROOT/calibration_overlap_audit.json" --fail-on-development-test-overlap \
  2>&1 | tee "$LOG_DIR/calibration_overlap_audit.log"

echo "Merged dedicated calibration roots:"
printf '  %s\n' "${merged_roots[@]}"
