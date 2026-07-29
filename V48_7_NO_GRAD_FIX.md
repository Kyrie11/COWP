# v48.7 SPIRE staged-training no-grad fix

## Symptom

Stage P aborts on `total.backward()` with:

`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

## Root cause

Stage P freezes the encoder/value/opportunity/harm/certificate paths and trains only:

- `direct_preference_adapter`
- `direct_preference_context_adapter`

The SPIRE preference objectives are activated only for scene-time groups that contain a positive candidate-vs-nominal exact-PCD opportunity. A valid batch can contain only negative/tied groups. In that case all Stage-P active preference terms are absent, while the remaining configured terms have zero weights and depend only on frozen value outputs. The resulting scalar has value zero and `requires_grad=False`, so PyTorch cannot execute `backward()`.

The checkpoint compatibility messages are expected and unrelated:

- missing `direct_preference_context_adapter.*`: this is the new v48.6/v48.7 context head and is zero-initialized;
- unexpected `direct_set_context_adapter.*` and `direct_set_context_gate`: the v48.5 checkpoint contains the legacy set-context branch, while SPIRE explicitly runs with `SET_CONTEXT_ENABLED=false`;
- no shape mismatches were reported.

## Fix

`direct_uncertainty_recovery_value_loss` now adds an exactly zero-valued autograd anchor over all optional direct outputs. Therefore a batch with no active supervision remains a zero-loss/zero-gradient optimizer step, but retains a valid computation graph.

This does not change:

- dataset or sampler behavior;
- model forward values;
- loss values or weights;
- trainable/frozen parameter prefixes;
- Stage P / Stage C separation;
- checkpoint loading;
- early-stopping metrics;
- calibration or deployment logic.

## Regression tests

Added tests cover:

1. preference-only Stage-P batch with no positive recovery group;
2. certificate-only Stage-C batch with no allowed recovery macro.

Validation performed:

- complete test suite: `133 passed`;
- v48.5-v48.7 focused suite: `31 passed`;
- all Python files compiled successfully;
- all shell and command scripts passed `bash -n`;
- original/fixed active-loss values were bitwise identical in representative positive-ranking cases.
