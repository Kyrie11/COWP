# v40 OC-UVRA

## Purpose

v39 OC-RAC attempted to fix near/contact candidate ordering by back-propagating a counterfactual PCD advantage through the existing OC-MERO components (`R_dep`, shared-option DRS, and oracle gap). The uploaded v39 results show that this changes a few audit selections but does not improve near-contact physical margins and degrades contact PCD/FRA/NUP. v40 separates two questions that should not share one prediction head:

1. **Certificate:** is the candidate observation-consistently recoverable and admissible?
2. **Preference:** among already certified candidates, is one candidate's deployable counterfactual value materially better than nominal?

OC-MERO and the calibrated CRISP admission path continue to answer the first question. OC-UVRA adds a direct value/uncertainty head for the second question.

## Method

For candidate prefix `a`, the model predicts

- `V_hat(a) in [0,1]`: direct observation-consistent recovery value;
- `sigma_hat(a)`: learned candidate-value uncertainty.

The teacher target is the same observation-consistent PCD used by the audit. Training combines heteroscedastic point regression, listwise scene-time ranking, and asymmetric nominal-versus-recovery advantage supervision.

At inference, a recovery candidate can challenge nominal only when:

- it is already admitted by the existing OC-MERO/shared-option/protective certificate;
- it satisfies macro, feasibility, hard-rule, harm, budget, and consecutive-run guards;
- its calibrated advantage lower bound is positive:

```text
A_LCB(a, a_nom) = V_hat(a) - V_hat(a_nom)
                  - z_cal * sqrt(sigma_hat(a)^2 + sigma_hat(a_nom)^2)
```

`z_cal` is not a fixed Gaussian constant. `tools/calibrate_direct_value_advantage.py` calibrates it on held-out scene-time groups using the maximum standardized over-estimation across all recovery candidates in each group. This max-over-candidates score protects the lower bound after candidate selection.

## Important fixes beyond the new head

- Validation now keeps complete scene-time candidate groups and uses deterministic group order without replacement.
- The scalar paired control explicitly disables the v40 direct-value channel.
- Scalar and v40 publication audits use identical target counts, avoiding incomplete paired comparisons.
- Intervention run-length aggregation now reports the true global maximum and pooled mean.
- Physical metrics include paired scene-wise comparisons and bootstrap confidence intervals.
- Stable-stop is conditioned on not already being stopped at rollout start.

## Main commands

```bash
bash run_v40_two_gpu_commands.txt
```

The command file first trains `head_only` and `adapter_light` concurrently, evaluates the safer `head_only` candidate, performs a medium confirmation gate, and only then launches three publication-scale seeds.

Ablations:

```bash
bash run_v40_ablation_commands.txt
```

## Validation performed in this package

```text
python -m compileall -q src tools tests
bash -n scripts/train_ocrap_v40_ocuvra.sh
bash -n scripts/run_ocrap_v40_ocuvra.sh
bash -n run_v40_two_gpu_commands.txt
PYTHONPATH=src pytest -q
```

All 57 tests pass in the packaging environment. GPU training and Waymax closed-loop evaluation were not executed here.
