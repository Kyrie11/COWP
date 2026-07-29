# OC-RAP dataset smoke analysis and acceleration patch

This patch adds optional dataset-regime quality gates and a dataset-root merge utility.

## Added quality gates

The builder now supports these optional `dataset_quality` keys:

- `require_nonartifact_sample`
- `require_feasible_prefix`
- `require_deployable_recoverable_sample`
- `min_deployable_margin`
- `require_oracle_recoverable_sample`
- `min_oracle_margin`
- `max_prefix_hard_violation`
- `max_prefix_harm_proxy`
- `max_oracle_gap`

They are disabled by default. Use them mainly for the safe regime to avoid saving low-headroom or hard-violating candidates.

## Two-GPU sharded generation

The Waymax loader already supports:

- `scenario_stride`
- `scenario_worker_index`
- `scenario_start_index`

Run one process per GPU with the same command plus different worker index and output root. Then merge with:

```bash
python tools/merge_dataset_roots.py --output <merged_root> <root_gpu0> <root_gpu1>
```

This avoids concurrent writes to the same manifest and lets each A30 run an independent Waymax/JAX process.

## Speed-critical knobs

Profiling from the uploaded smoke build shows `build_samples_s` dominates wall time, and within samples the teacher rollout dominates. For smoke/debug builds, use screened hybrid:

```bash
--set waymax.teacher_rollout_top_k_options=4 \
--set 'waymax.teacher_rollout_option_modes=[stop,brake_lane,yield_rejoin,pull_over,avoid_secondary]'
```

For final strict paper-quality labels, keep `teacher_rollout_top_k_options=0` unless the paper reports screened hybrid.

Disable per-sample compression/fsync for local dataset construction:

```bash
--set io.compress_npz=false --set io.fsync_npz=false
```

This is smaller than the teacher bottleneck but avoids unnecessary synchronous disk stalls.
