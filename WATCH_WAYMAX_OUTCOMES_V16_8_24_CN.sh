#!/usr/bin/env bash
set -euo pipefail
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
REPLAY_ROOT="${REPLAY_ROOT:-$COWP_ROOT/waymax_replay_v16_8_24}"
for split in training validation heldout_test; do
  case "$split" in
    training) out="$COWP_ROOT/tensor_cache_train_waymax";;
    validation) out="$COWP_ROOT/tensor_cache_val_waymax";;
    heldout_test) out="$COWP_ROOT/tensor_cache_heldout_test_waymax";;
  esac
  rows=0
  shopt -s nullglob
  files=("$REPLAY_ROOT/${split}_"*.jsonl)
  for f in "${files[@]}"; do
    [[ "$f" == *"_profile_"* ]] && continue
    n=$(wc -l < "$f" || echo 0)
    rows=$((rows+n))
  done
  npz=0
  [[ -d "$out" ]] && npz=$(find "$out" -maxdepth 1 -type f -name '*.npz' -printf . 2>/dev/null | wc -c)
  printf '%-13s replay_rows=%-9d attached_npz=%d\n' "$split" "$rows" "$npz"
done

echo "--- latest replay logs ---"
find "$REPLAY_ROOT" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -4 | cut -d' ' -f2- | while read -r f; do
  echo "### $f"
  tail -n 3 "$f" || true
done
