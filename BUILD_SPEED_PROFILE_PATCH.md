# Build speed / profiling patch

This patch adds low-overhead timing outputs for WOMD/Waymax dataset construction and a small I/O speed switch.

New outputs when `--set profiling.enabled=true`:

- `build_profile.csv`: per saved sample timing, including future generation, teacher margins, root clustering, observation, OC-MERO, and Waymax teacher rollout/cache counters.
- `build_scene_time_profile.csv`: per scenario-time group timing, including `construct_history_s`, `build_samples_s`, sample timing sums, `npz_serialize_s`, `npz_write_s`, and manifest checkpoint time.
- `build_stage_profile.json`: running cumulative stage totals and throughput.
- `dataset_status.json`: now includes profile file paths and current stage totals while a long build is running.

New I/O config:

- `io.compress_npz`: default `true`; set `false` for local speed profiling to avoid CPU compression bottlenecks.
- `io.fsync_npz`: default `true`; set `false` for local smoke/profiling runs to avoid a synchronous disk flush after every sample.

A helper script was added:

```bash
python tools/merge_dataset_shards.py --output <merged_dataset> <shard0> <shard1> [more_shards]
```

This is intended for two-GPU construction with existing `scenario_stride` / `scenario_worker_index` config knobs.
