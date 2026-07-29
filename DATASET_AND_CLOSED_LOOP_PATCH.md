# OC-RAP dataset-regime and closed-loop patch

This patch focuses on two issues:

1. The old README commands produced too many partially incompatible dataset folders. The training mix excluded the only set that produced oracle/deployable artifacts, while the proof set used a different root/option geometry.
2. Evaluation was offline candidate-selection evaluation, not a full Waymax receding-horizon loop.

## Main code changes

- Adds branch-intent recovery margins for hidden-yield / hidden-accelerate futures without using the hard proof margin override.
- Adds dataset quality gates for per-sample observation-negative pairs and optional negative-deployable filtering.
- Supports a unified recommended geometry: `num_roots=8`, `num_recovery_options=12`.
- Allows current Waymax simulator state to be exported as a spliced RawScenario so candidate generation replans from the current closed-loop state.
- Patches Waymax rollout so closed-loop branches use the current SimulatorState instead of resetting back to log.
- Adds a `closed-loop` CLI command that follows the Waymax protocol: `reset -> build candidates from current state -> select prefix -> step -> replan`.

## New CLI

```bash
PYTHONPATH=src python -m ocrap.cli closed-loop \
  --dataset "$WOMD_VAL" \
  --checkpoint <run>/best.pt \
  --output <run>/closed_loop_ocrap.json \
  --set closed_loop.max_scenarios=8 \
  --set closed_loop.max_steps=40 \
  --set closed_loop.method=ocrap \
  --set closed_loop.replan_interval_steps=1 \
  --set training.device=cuda:0
```

The command reports OC-RAP closed-loop metrics and Waymax metric summaries in the output JSON.
