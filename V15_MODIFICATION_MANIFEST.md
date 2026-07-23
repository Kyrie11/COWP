# COWP v15 Modification Manifest

## Release identity

- Release: **v15-CNOB**
- Full name: **Causal Natural Option Basis, Protocol Integrity, and OBS Decontamination**
- Base package: uploaded COWP v14 code
- Seed convention: 2026
- New default output: `outputs/cowp_v15_causal_natural_seed2026`
- New data root: `/data0/senzeyu2/dataset/COWP/formal_v15`

## Modified implementation files

### Causal data and coordinates

- `cowp/models/cowp_model.py`
  - disables future-label reconstruction of encoder state by default;
  - requires real history/current input in reported runs;
  - wires the strict SDC identity option.
- `cowp/models/coordinate.py`
  - adds `require_explicit` SDC validation;
  - refuses silent row-zero ego assumptions.
- `cowp/data/cache_schema.py`
  - registers v15 map-verification and OBS-contamination fields.

### Natural-option construction

- `cowp/label/trajectory_primitives.py`
  - current-state anchored resampling;
  - smooth zero-origin lateral perturbation;
  - velocity/yaw reconstruction and finite-motion checks.
- `cowp/label/natural_alternatives.py`
  - map-aware filtering;
  - OBS pressure-contamination scoring, downweighting and rejection;
  - causal metadata output.
- `cowp/label/label_engine.py`
  - stores the new natural-option diagnostics in labels.

### Natural decoder and losses

- `cowp/models/natural_decoder.py`
  - adds `typed_causal_residual` alias/profile;
  - source-specific bounded residual capacity;
  - higher OBS correction capacity and stronger NEU/PRIO priors;
  - source identity treated as structural.
- `cowp/models/losses.py`
  - source-specific base-deviation penalties and metrics;
  - branch geometric losses remain source-restricted.

### Waymax evaluation and metrics

- `cowp/waymax_eval/policy_wrapper.py`
  - disables implicit `log_trajectory` fallback;
  - allows privileged future only in explicit oracle ablations;
  - uses causal state/history prediction in the main path.
- `cowp/waymax_eval/metrics_cowp.py`
  - separates label-space proxy safety from simulator closed-loop CR;
  - emits protocol provenance fields.

### Gates and audits

- `cowp/scripts/32_gate_natural_basis.py`
  - adds OBS absolute quality and branch-spread hard gates;
  - stops treating source CE as learned evidence for typed roots.
- `cowp/scripts/36_audit_causal_protocol.py` (new)
  - audits causality, SDC identity, metric provenance, map/OBS label policy,
    mapping, root range and log-divergence handling.

## New configurations

- `configs/model_cowp_v15.yaml`
- `configs/label_cowp_v15.yaml`
- `configs/train_cowp_v15.yaml`
- `configs/eval_cowp_v15.yaml`
- `configs/label_cowp_v15_pareto_ablation.yaml`
- `configs/label_cowp_v15_pairmax_ablation.yaml`

## New executable entry points

- `prepare_cowp_v15_data.sh`
- `run_cowp_v15_dual_gpu.sh`
- `NEXT_RUN_COMMANDS_V15_CN.sh`

All three pass `bash -n`. The former prose-plus-command file is no longer used as
an executable.

## Tests and local evidence

- `tests/test_v15_causal_integrity.py` (new)
- Full local suite: **81 passed**
- `V15_CAUSAL_AUDIT_SAMPLE.json`: pass
- Reconstructed v14 original gate:
  `/mnt/data/results_v14/eval/reconstructed_natural_basis_gate.json`
- v15 strict gate applied to v14:
  `/mnt/data/results_v14/eval/v15_strict_gate_on_v14.json` (expected fail on OBS
  quality and branch spread)

## Data compatibility

v15 changes natural labels, so old `labels_*`, tensor caches and transport overlay
must not be silently reused for paper-facing results. Build them in the new
`formal_v15` root. Waymax attached outcomes are replayed/reattached to avoid
mixing old label semantics with new tensors.

## Known remaining work

- Train and verify v15 on full data.
- Add an independently reactive non-ego protocol (at minimum Waymax IDM and one
  learned sim-agent model) before claiming causal burden reduction.
- Align the paper text with the actual decoder: the current implementation is a
  typed analytic basis with learned residuals, not the transformer/diffusion
  architecture currently described in parts of the TeX.
- Run full-validation, at least three seeds, paired bootstrap confidence
  intervals, and external baseline comparisons before an SOTA claim.
