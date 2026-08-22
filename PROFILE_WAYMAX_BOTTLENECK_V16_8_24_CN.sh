#!/usr/bin/env bash
set -euo pipefail

# Non-destructive timing probe for the current v16.8.24 Waymax candidate replay.
# It NEVER writes to the authoritative waymax_replay_v16_8_24 shard JSONLs and
# NEVER writes tensor_cache_*_waymax NPZs.  It replays a tiny number of candidates
# into a timestamped diagnostic directory so a stopped full run can be profiled
# safely before resuming from the original JSONLs.

PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
PROFILE_CACHE="${WAYMAX_PROFILE_CACHE:-$COWP_ROOT/tensor_cache_train}"
WAYMAX_GPUS="${WAYMAX_GPUS:-0,1}"
PROFILE_NUM_SHARDS="${WAYMAX_PROFILE_NUM_SHARDS:-2}"
PROFILE_CANDIDATES="${WAYMAX_PROFILE_CANDIDATES:-4}"
PROFILE_HORIZON="${WAYMAX_PROFILE_HORIZON:-80}"
PROFILE_SCENES="${WAYMAX_PROFILE_SCENES_PER_WORKER:-1}"
RUN_TAG="${WAYMAX_PROFILE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PROBE_ROOT="${WAYMAX_PROFILE_ROOT:-$COWP_ROOT/waymax_bottleneck_probe_v16_8_24/$RUN_TAG}"

IFS=',' read -r -a GPUS <<< "$WAYMAX_GPUS"
[[ -d "$PROFILE_CACHE" ]] || { echo "Missing cache: $PROFILE_CACHE" >&2; exit 2; }
[[ "${#GPUS[@]}" -ge "$PROFILE_NUM_SHARDS" ]] || {
  echo "Need at least $PROFILE_NUM_SHARDS GPU ids in WAYMAX_GPUS; got $WAYMAX_GPUS" >&2
  exit 2
}
mkdir -p "$PROBE_ROOT"

echo "============================================================"
echo "Waymax bottleneck probe (NON-DESTRUCTIVE)"
echo "cache=$PROFILE_CACHE"
echo "gpus=$WAYMAX_GPUS shards=$PROFILE_NUM_SHARDS"
echo "scenes_per_worker=$PROFILE_SCENES candidates_per_scene=$PROFILE_CANDIDATES horizon=$PROFILE_HORIZON"
echo "output=$PROBE_ROOT"
echo "The first half of candidates use normal dispatch timing; the second half use synchronization-aware stage timing."
echo "============================================================"

pids=()
profiles=()
for ((s=0; s<PROFILE_NUM_SHARDS; s++)); do
  gpu="${GPUS[$s]}"
  tag="shard$(printf '%03d' "$s")_of_$(printf '%03d' "$PROFILE_NUM_SHARDS")"
  outcomes="$PROBE_ROOT/training_probe_${tag}.jsonl"
  profile="$PROBE_ROOT/training_profile_${tag}.jsonl"
  log="$PROBE_ROOT/training_probe_${tag}.log"
  profiles+=("$profile")
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export COWP_TQDM_POSITION="$s"
    "$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates \
      --data-config configs/data.yaml \
      --label-config configs/label_cowp_v16_8.yaml \
      --eval-config configs/eval_cowp_v16_8.yaml \
      --cache-dir "$PROFILE_CACHE" \
      --state-source cache \
      --outcomes-jsonl "$outcomes" \
      --candidate-selection balanced \
      --max-candidates-per-scene "$PROFILE_CANDIDATES" \
      --rollout-horizon-steps "$PROFILE_HORIZON" \
      --limit-scenes "$PROFILE_SCENES" \
      --waymax-device gpu \
      --waymax-action-mode absolute_xy_yaw \
      --metric-set safety \
      --metric-eval-mode step \
      --metric-eval-interval 1 \
      --done-check-interval 1 \
      --num-shards "$PROFILE_NUM_SHARDS" \
      --shard-index "$s" \
      --gc-every-scenes 0 \
      --profile-replay-jsonl "$profile" \
      --profile-detail probe \
      --profile-probe-candidates "$PROFILE_CANDIDATES" \
      --progress-desc "PROFILE GPU=$gpu ${s}/${PROFILE_NUM_SHARDS}" \
      --overwrite
  ) > >(tee "$log") 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  one_rc=0
  wait "$pid" || one_rc=$?
  [[ "$one_rc" -eq 0 ]] || rc="$one_rc"
done
[[ "$rc" -eq 0 ]] || { echo "One or more profiling workers failed (rc=$rc)" >&2; exit "$rc"; }

summary="$PROBE_ROOT/timing_summary.json"
"$PYTHON_BIN" -m cowp.scripts.80_summarize_waymax_replay_profile \
  "${profiles[@]}" \
  --output "$summary"

echo
echo "PROFILE COMPLETE"
echo "Please upload these files for bottleneck analysis:"
echo "  $summary"
for p in "${profiles[@]}"; do echo "  $p"; done
for ((s=0; s<PROFILE_NUM_SHARDS; s++)); do
  tag="shard$(printf '%03d' "$s")_of_$(printf '%03d' "$PROFILE_NUM_SHARDS")"
  echo "  $PROBE_ROOT/training_probe_${tag}.log"
done
