# v47 OC-TRAC Change Manifest

## Modified source

- `src/ocrap/models/ocrap.py`: benefit/opportunity/harm heads, risk-attitude experts, robust disagreement aggregation.
- `src/ocrap/models/losses.py`: tri-state, class-balanced, nominal-relative, setwise abstention/admission losses.
- `src/ocrap/cli/train.py`: exact teacher-PCD group sampler, asymmetric all-stress expert supervision, robust checkpoint diagnostics.
- `src/ocrap/models/inference.py`: v47 head loading and harm prediction.
- `src/ocrap/planning/selector.py`: predicted-harm veto.
- `src/ocrap/planning/prefix_generation.py`: configurable stress macro schedule and non-duplicate recovery variants.
- `src/ocrap/evaluation/baselines.py`, `evaluator.py`: paired harm propagation.
- `src/ocrap/simulation/closed_loop_runner.py`: all-regime route-free physical metrics and explicit collision/offroad aliases.
- `src/ocrap/config/defaults.py`: v47 defaults.

## New tools/scripts

- `tools/calibrate_direct_value_risk_v47.py`
- `tools/check_closed_loop_dataset_support.py`
- `tools/check_v47_quick_gate.py`
- `tools/select_v47_candidate.py`
- `scripts/train_ocrap_v47_trac.sh`
- `scripts/calibrate_ocrap_v47_trac.sh`
- `scripts/run_ocrap_v47_trac.sh`
- `scripts/run_v47_all_regime_reference_closed_loop.sh`
- `run_v47_two_gpu_fast_commands.txt`
- `tests/test_v47_trac.py`

## Key engineering fixes

- enforce current clean-base initialization;
- exact PCD positive-group sampling with replacement;
- separate `predicted_harm` from physical `harm_proxy`;
- explicit train/evaluation dataset-root separation and val/test bucket split propagation;
- align stress bucket raw source through `WOMD_STRESS` (standard validation by default);
- restore sequential `wait_pair` barriers in stress dataset rebuild;
- front-load diverse merge/brake/stabilize/yield candidates before quality pruning;
- closed-loop reference outputs even when learned certificate fails;
- Safe/Near/Contact physical metrics in a common schema.

## Not modified

- `post-collision.tex`;
- existing datasets and experiment artifacts;
- existing external baseline implementations/results;
- OC-MERO equations and original teacher labels.
