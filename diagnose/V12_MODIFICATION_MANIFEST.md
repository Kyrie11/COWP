# v12-RIOT Modification Manifest

This manifest lists the files changed relative to the uploaded COWP package.
The detailed rationale is in `COWP_V11_DIAGNOSIS_AND_V12_RIOT_PLAN_CN.md` and
`ALGORITHM_CHANGELOG.md`.

## Algorithm and training

- `cowp/data/dataset.py`: witness-stage candidate labels; compact planner-eval response/root labels.
- `cowp/models/losses.py`: root-indexed target, direct recovery loss, candidate-budget coverage, no silent negative fallback.
- `cowp/models/set_transport_head.py`: conditional retention factorization, direct root recovery, corrected existential burden, separated transport/generic outputs.
- `cowp/models/cowp_model.py`: actual invocation of generic certificate as diagnostic; response decoding stage wiring.
- `cowp/scripts/03_train.py`: supervision validation, freeze policy, checkpoint selection/gates.

## Evaluation

- `cowp/waymax_eval/policy_wrapper.py`: separate pair witness threshold and BCOT budget; transport-pure selector.
- `cowp/waymax_eval/rollout.py`: BCOT budget sweep, direct root-transport metrics, and metric plumbing.
- `cowp/scripts/04_eval_closed_loop.py`: CLI and output schema for separate thresholds/budget sweeps.
- `cowp/scripts/25_verify_mechanism_effect.py`: calibrated BCOT and direct root-transport verification.
- `cowp/scripts/30_diagnose_bcot_result.py`: budget-sweep readiness diagnosis.
- `cowp/scripts/31_calibrate_bcot_budget.py`: new candidate-budget calibration.
- `cowp/scripts/32_gate_natural_basis.py`: new natural-option hard gate.

## Configuration and launcher

- `configs/train_cowp_v12.yaml`
- `configs/label_cowp_v12.yaml`
- `configs/eval_cowp_v12.yaml`
- `configs/label_cowp_v12_pairmax_ablation.yaml`
- `configs/label_cowp_v12_pareto_ablation.yaml`
- `configs/label_cowp_v11.yaml` (backward-compatible separate budget/selector defaults)
- `configs/label_cowp_v11_pairmax_ablation.yaml`
- `configs/label_cowp_v11_pareto_ablation.yaml`
- `run_cowp_v12_dual_gpu.sh`

## Tests and documentation

- `tests/test_v9_transport_supervision.py`: direct root-target, unordered alignment, and evaluation-metric checks.
- `ALGORITHM_CHANGELOG.md`: v11 postmortem and v12 registered changes.
- `COWP_V11_DIAGNOSIS_AND_V12_RIOT_PLAN_CN.md`: complete analysis and run plan.
- `COWP_V11_TO_CCFA_GAP.{json,csv}`: metric gap table.
