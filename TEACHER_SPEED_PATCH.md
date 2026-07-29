# OC-RAP Waymax teacher speed patch

## Bottleneck diagnosed from live profile

The posted profile lines show that one sample spends ~351-355 seconds inside `teacher_margins`, while future generation is ~13-25 seconds and observation is ~4 seconds. With `futures=7`, `opts=12`, and a 4-second recovery horizon at 10 Hz, the legacy hybrid teacher runs roughly:

```text
7 futures × 12 options × 40 recovery steps
```

The previous default `waymax.teacher_metrics_stride=1` also calls `waymax_env.metrics(...)` at every recovery step. This creates thousands of small Python/JAX/Waymax dispatches per sample and leaves the GPU under-utilized.

## Changes in this patch

1. Changed the default Waymax teacher recovery metric mode:

```yaml
waymax:
  teacher_metrics_stride: 0
```

This records only the final recovery metric by default. It avoids per-step metric evaluation and enables one-scan rollout.

2. Enabled JAX scan recovery rollouts by default:

```yaml
waymax:
  use_jit_scan_rollouts: true
```

This moves the recovery horizon from many Python `env.step` calls into one `jax.lax.scan` dispatch when Waymax/JAX supports it. If a Waymax/JAX version cannot trace the scan, the code falls back to the Python loop while still using final-step-only metrics when `teacher_metrics_stride=0`.

3. Added live per-sample profile output:

```text
build_profile_live.csv
```

This file is written immediately after each sample materializes, so you no longer need to wait for a full scene-time group to finish before seeing machine-readable timing.

4. Extended slow-sample stdout log with:

```text
wx_rollouts=<count> wx_cache_hits=<count> screened=<0/1>
```

This makes it clear whether Waymax teacher metric rollout caching and screened hybrid are actually being used.

## Recommended command addition

Your old command will pick up the faster defaults after this patch. For clarity, you can also make the speed mode explicit:

```bash
--set waymax.teacher_metrics_stride=0 \
--set waymax.use_jit_scan_rollouts=true \
--set profiling.enabled=true
```

For a stricter audit run, restore legacy per-step metrics:

```bash
--set waymax.teacher_metrics_stride=1 \
--set waymax.use_jit_scan_rollouts=false
```

That strict setting will be much slower and should only be used on a small subset.
