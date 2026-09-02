# External baseline v52: GameFormer gradient, dataset exposure, and two-A30 scheduler audit

## Scope

This revision is based on the user-uploaded `OC-RAP-external-baselines-v51-nan-speed(1).zip`.
The existing top-level launcher and its command-line interface are preserved.

## 1. Why GameFormer had a finite loss but a non-finite gradient

`gameformer_lite` is a learned model. The observed failure happened after `loss.backward()`:

```
FloatingPointError: Non-finite gradient norm at epoch=1; optimizer step aborted
```

That distinction matters: the v51 finite-loss checks were already passing, so the remaining defect had to be in a backward path.

The strongest code-level root cause was in `GameFormerFutureEncoder`. The decoder trajectory is a cumulative displacement from the current ego origin, but v51 constructed its temporal difference as:

```
prev = cat([traj[..., :1, :], traj[..., :-1, :]])
dxy = traj - prev
heading = atan2(dy, dx)
```

The first `dxy` was therefore forced to `(0, 0)` by construction. `atan2(0, 0)` has a finite forward value but an undefined mathematical derivative. CUDA/BF16 backward can consequently produce NaN/Inf gradients while the forward loss remains completely finite.

v52 fixes both the numerical and kinematic problem:

- first-step previous position is the current ego origin `(0, 0)`, which is the correct reference for a cumulative future displacement;
- genuinely stationary/near-stationary steps are detected before `atan2` and are given a neutral +x heading input, so `atan2` is never evaluated at the singular origin;
- finite-difference and angle geometry are evaluated in FP32 even when the surrounding model uses BF16 AMP;
- modal softmax in the FutureEncoder is evaluated in bounded FP32.

The expensive LSTM/attention/MLP portions remain under AMP, so this fix is intentionally narrow rather than globally disabling BF16.

## 2. Additional loss-path hardening

The previous revision already skipped zero-weight auxiliary losses and ran the GameFormer Gaussian trajectory NLL in FP32. v52 also fixes the remaining differentiable-zero corner cases:

- `_zero_loss()` no longer performs `NaN.sum() * 0`;
- `_masked_mse()` no-valid-target branch no longer performs `NaN.sum() * 0`;
- `_focal_topk_loss()` no-valid-target branch is hardened in the same way.

All graph-zero tensors are sanitized before multiplication by zero.

If any non-finite gradient remains on the user's A30/data, fail-fast now includes the exact parameter names, number of non-finite gradient elements, and maximum finite gradient magnitude. It does not silently replace a real training gradient with zero.

## 3. Dataset exposure policy: no cross-regime purity filtering

The v51 loader inferred a purity contract from directory names and removed groups whose metadata contained another regime. That behavior has been removed.

`group_sample_paths()` now preserves **every group present in the requested split**. Therefore:

- `train_safe` means all groups actually stored in `train_safe`;
- `train_near_contact` means all groups actually stored in `train_near_contact`, including cross-regime/hard examples intentionally present there;
- `train_contact` means all groups actually stored in `train_contact`.

Only the ordinary `split_id` selection (train/val/test/calibration) remains. No `regime_label`, post-contact, prefix-contact, or nominal-regime purity filter is applied.

This makes external-baseline data exposure symmetric with the user's OC-RAP training policy. The previously reported near-contact mixture is therefore treated as a property of the constructed training split rather than something the baseline loader should repair.

## 4. Existing top-level command compatibility

The unchanged command remains valid:

```
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
OUT=/home/senzeyu2/code/OC-RAP/runs/external_baselines_v50 \
MAX_SCENARIOS=0 \
DO_TRAIN_SAFE=true \
DO_TRAIN_NEAR=true \
DO_TRAIN_CONTACT=true \
DO_CALIBRATE_NEAR=true \
DO_OFFLINE=false \
DO_CLOSED_LOOP=true \
bash scripts/run_all_regime_external_baselines_optimized.sh
```

The output path can continue to be named `external_baselines_v50`; it is only a directory name and does not select old code.

## 5. Two-A30 scheduling behavior

Every GPU worker is launched with a single physical device in `CUDA_VISIBLE_DEVICES`. Thus one baseline process cannot see or use both A30s.

Safe learned training is now a dynamic two-slot queue instead of a fixed pair barrier. For example, if GameFormer is training on GPU0 and PlanTF is immediately reused on GPU1, GPU1 immediately receives PLUTO rather than waiting for GameFormer. The same dynamic scheduler is used for learned offline evaluation when enabled.

Safe/Near/Contact closed-loop main-table evaluation already uses the same work-conserving rule: whichever A30 becomes free first receives the next baseline.

The top-level launcher now propagates `USE_DYNAMIC_SCHEDULER` consistently to every regime.

Two caveats are intrinsic, not scheduler bugs:

1. Near/Contact main-table methods are non-learning controllers/optimizers, so their `DO_TRAIN_*` phase is CPU registration/calibration rather than fake GPU training.
2. If fewer than two runnable GPU jobs remain (for example only GameFormer is missing while every other checkpoint/result is already reusable), no scheduler can keep both cards occupied without launching redundant work. The code guarantees **at most one baseline per physical GPU and maximal occupancy while at least two runnable jobs exist**.

## 6. Validation in this environment

- `bash -n` passed for Safe, Near, Contact, and all-regime launchers.
- `py_compile` passed for modified external-baseline data/model/train and registration code.
- Relevant regression suite: **49 passed**.
- Dedicated tests cover:
  - zero-motion GameFormer FutureEncoder gradients;
  - `atan2` never receiving `(0,0)` in the stable geometry path;
  - correct first-step displacement relative to the current origin;
  - BF16/large-coordinate trajectory-loss stability;
  - NaN/Inf padded candidates;
  - inactive NaN auxiliary heads and differentiable-zero paths;
  - no regime-purity filtering of named regime datasets;
  - observation-only/teacher-leakage contracts;
  - CUDA runtime configuration and closed-loop hot paths.

Physical A30 execution and the user's `/data0` dataset are not mounted in this environment, so no claim is made that a full server epoch was run here. The reported source-level gradient singularity has been removed, and any remaining server-only issue will now identify the offending parameter in the exception.
