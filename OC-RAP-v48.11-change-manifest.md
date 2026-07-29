# v48.11 CASTER Change Manifest

## Algorithm

- Added `RecoverySetTournament`, a recovery-only permutation-equivariant self-attention ranker.
- Added `model.direct_recovery_set_tournament*` controls and checkpoint/inference plumbing.
- Added optional regime-specific evidence adapters through `model.direct_recovery_delta_regime_experts`.
- Added frozen-policy score/gap inputs through `model.direct_recovery_delta_policy_features`.
- Added class-weighted proper ordered three-state NLL for harmful/dead/beneficial evidence.
- Added policy-first/no-fallback candidate semantics to calibration and runtime selection.

## Engineering

- Calibration JSON records `direct_value_policy_first_no_fallback`.
- The fast two-GPU controller forwards the policy-first option.
- Added strict staged v48.11 training script and immutable architecture/checkpoint markers.
- Added v48.11 multi-seed and dedicated-calibration scripts.
- Added a two-GPU ablation scheduler; the policy-semantics ablation reuses the reference checkpoint.
- Added dynamic v48.11 ablation summarization.
- Added v48.11 staged learning-gate audit.

## Tests

- Set-tournament permutation equivariance and nominal pinning.
- Policy-first no-fallback abstention.
- Regime-specific ordered-evidence simplex validity.
