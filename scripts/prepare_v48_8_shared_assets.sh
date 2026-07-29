#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_OCRAP_ROOT="${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
ASSET_ROOT="${ASSET_ROOT:-runs/ocrap_v48_8_shared_assets_4801}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.50}"
CALIBRATION_SEED="${CALIBRATION_SEED:-4801}"
SPLIT_ROOT="$ASSET_ROOT/dataset_splits"
GROUP_INDEX="$ASSET_ROOT/teacher_pcd_train_index.jsonl"
GROUP_SUMMARY="$ASSET_ROOT/teacher_pcd_train_index_summary.json"
mkdir -p "$ASSET_ROOT/logs" "$SPLIT_ROOT"

python tools/ensure_manifest_v48.py \
  --dataset-root "$TRAIN_OCRAP_ROOT/train_near_contact" \
  --dataset-root "$TRAIN_OCRAP_ROOT/train_contact" \
  --dataset-root "$EVAL_OCRAP_ROOT/val_safe" \
  --dataset-root "$EVAL_OCRAP_ROOT/val_near_contact" \
  --dataset-root "$EVAL_OCRAP_ROOT/val_contact" \
  --workers="${MANIFEST_WORKERS:-6}" --rebuild-if-stale \
  2>&1 | tee "$ASSET_ROOT/logs/ensure_manifests.log"

split_one() {
  local input="$1" cal="$2" dev="$3" name="$4"
  python tools/split_calibration_by_scene_v48.py \
    --input "$input" --calibration-output "$cal" --validation-output "$dev" \
    --calibration-fraction="$CALIBRATION_FRACTION" --seed="$CALIBRATION_SEED" --overwrite \
    2>&1 | tee "$ASSET_ROOT/logs/split_${name}.log"
}
split_one "$EVAL_OCRAP_ROOT/val_safe" "$SPLIT_ROOT/calibration_safe" "$SPLIT_ROOT/dev_val_safe" safe
split_one "$EVAL_OCRAP_ROOT/val_near_contact" "$SPLIT_ROOT/calibration_near_contact" "$SPLIT_ROOT/dev_val_near_contact" near
split_one "$EVAL_OCRAP_ROOT/val_contact" "$SPLIT_ROOT/calibration_contact" "$SPLIT_ROOT/dev_val_contact" contact

python tools/check_scene_overlap_v48.py \
  --development-root "$SPLIT_ROOT/dev_val_safe" \
  --development-root "$SPLIT_ROOT/dev_val_near_contact" \
  --development-root "$SPLIT_ROOT/dev_val_contact" \
  --test-root "$SPLIT_ROOT/calibration_safe" \
  --test-root "$SPLIT_ROOT/calibration_near_contact" \
  --test-root "$SPLIT_ROOT/calibration_contact" \
  --output "$ASSET_ROOT/development_calibration_overlap_audit.json" \
  --fail-on-development-test-overlap

python tools/build_teacher_pcd_index_v48.py \
  --dataset "$TRAIN_OCRAP_ROOT/train_near_contact,$TRAIN_OCRAP_ROOT/train_contact" \
  --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" \
  --alpha="${OCMERO_ALPHA:-0.2}" --beta="${OCMERO_BETA:-0.2}" --top-m="${OCMERO_TOP_M:-8}" \
  --positive-gain="${POSITIVE_GAIN:-0.015}" \
  --min-positive-groups-near="${MIN_POSITIVE_GROUPS_NEAR:-200}" \
  --min-positive-groups-contact="${MIN_POSITIVE_GROUPS_CONTACT:-120}" \
  --min-positive-scenes-near="${MIN_POSITIVE_SCENES_NEAR:-80}" \
  --min-positive-scenes-contact="${MIN_POSITIVE_SCENES_CONTACT:-60}" \
  --quality-mode=warn 2>&1 | tee "$ASSET_ROOT/logs/build_teacher_pcd_index.log"

python - "$ASSET_ROOT" "$SPLIT_ROOT" "$GROUP_INDEX" "$GROUP_SUMMARY" <<'PY'
import hashlib,json,pathlib,sys,time
root,split,index,summary=map(pathlib.Path,sys.argv[1:])
for p in (index,summary):
    if not p.is_file(): raise SystemExit(f'missing shared asset: {p}')
doc={
 'event':'v48_8_shared_assets_complete','created_unix':time.time(),
 'split_root':str(split),'group_index':str(index),'group_summary':str(summary),
 'group_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
}
(root/'SHARED_ASSETS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY

echo "Shared assets ready: $ASSET_ROOT"
