# External Baseline NaN, Dataset-Contract, and Runtime Audit (v51)

This audit is based on the uploaded `OC-RAP-external-baselines-v50(2).zip` and
`reports.zip`.  The optimized code preserves the existing top-level training and
testing commands.

## 1. GameFormer-lite is a learning baseline

`gameformer_lite` is not a rule/non-learning planner in this repository.  Its
`GameFormerLevelK` module contains learned LSTM history encoders, a Transformer
fusion encoder, multi-modal trajectory/score heads, level-k interaction blocks,
policy heads, and scalar heads.  The shipped Safe config trains it for 30 epochs
with policy, level-k, response-consistency, and multi-modal trajectory losses.

Therefore a NaN training loss is an implementation/numerical problem, not an
expected consequence of using a non-learning method.

### NaN sources fixed

1. **Zero-weight losses were still evaluated.**  The old `_loss_dict` computed
   every auxiliary loss and multiplied it by its configured weight afterwards.
   IEEE floating point defines `0 * NaN = NaN`; an inactive auxiliary head could
   therefore poison the total objective.  The new implementation computes only
   nonzero-weight losses.  This also removes wasted work for policy-only PlanTF
   and PDM-Hybrid.
2. **GameFormer Gaussian trajectory NLL ran inside AMP.**  On the user's A30,
   `amp_dtype=auto` selects BF16.  Squared residuals, exponentials, and masked
   reductions do not benefit from low precision and are more numerically fragile.
   The trajectory NLL and score CE now run in FP32 while the model forward remains
   under AMP.
3. **Masked invalid/padded values could still create NaN.**  The old expression
   formed the NLL first and multiplied by a zero mask afterwards, so an invalid
   `Inf` could become `0 * Inf -> NaN`.  Prediction, target, and log-sigma are now
   sanitized with `where` before arithmetic; padded candidates are excluded from
   best-mode mining.
4. **Non-finite gradients could reach `optimizer.step()`.**  Training now aborts
   immediately on a non-finite loss component or gradient norm, clears gradients,
   and reports the offending loss names.  A corrupted checkpoint is not written.
5. Cross-entropy/KL/topology/PLUTO auxiliary reductions are evaluated in FP32 at
   the loss boundary.  Inactive losses are not evaluated at all.

The actual first non-finite tensor from the user's server run cannot be recovered
without that run log and the `/data0/...` samples.  The bugs above are concrete
correctness defects in the uploaded code and can produce or propagate the
observed NaN.  If a data-specific non-finite remains, the new fail-fast error will
identify the component instead of silently continuing.

## 2. Dataset report audit

### Summary

| dataset | samples | groups | scenes | candidates/group mean | feasible | hard mean | harm mean | nominal expected regime | post-contact candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| train_safe | 20,000 | 2,500 | 1,171 | 8.000 | 0.9135 | 0.0063 | 0.0000 | 2,500 normal | 0 |
| val_safe | 2,328 | 291 | 132 | 8.000 | 0.9240 | 0.0112 | 0.0000 | 291 normal | 0 |
| test_safe | 3,216 | 402 | 175 | 8.000 | 0.9275 | 0.0093 | 0.0000 | 402 normal | 0 |
| train_near_contact | 13,324 | 1,800 | 600 | 7.402 | 0.8786 | 0.0894 | 0.0289 | **1,403 / 1,800 near-contact** | **251** |
| val_near_contact | 3,445 | 433 | 176 | 7.956 | 0.9350 | 0.0087 | 0.0000 | 433 near-contact | 0 |
| test_near_contact | 4,723 | 595 | 250 | 7.938 | 0.8994 | 0.0161 | 0.0000 | 595 near-contact | 0 |
| calibration_near_contact | 6,039 | 765 | 316 | 7.894 | 0.8619 | 0.0348 | 0.0000 | 765 near-contact | 0 |
| train_contact | 16,790 | 2,000 | 500 | 8.395 | 0.8699 | 0.0936 | 0.0283 | 2,000 post-contact | 16,790 |
| val_contact | 6,477 | 723 | 211 | 8.959 | 0.9177 | 0.0148 | 0.0000 | 723 post-contact | 6,477 |
| test_contact | 6,687 | 747 | 209 | 8.952 | 0.8965 | 0.0212 | 0.0000 | 747 post-contact | 6,687 |

All uploaded reports list an empty `failures` array.

### Safe

Safe data usage is correct for the four learned Safe planners: training/validation
use only `train_safe`/`val_safe`, testing uses `test_safe`, and supervision stays
on the logged nominal candidate.  The Safe reports contain exactly one nominal
candidate and exactly eight candidates per group.  The many report warnings about
missing a targeted future reflect the Safe dataset design: Safe samples have three
futures consisting of replay/reactive coverage, not the 11/13 stress-future design
used by Near/Contact.  The external Safe learned adapters do not consume teacher
future branches, so these warnings do not invalidate their training.

### Near-contact

`val_near_contact` and `test_near_contact` have the intended strict contract:
nominal near-contact and no post-contact/prefix-contact group.  The content of
`calibration_near_contact` is also clean and remains the only calibration source
for Conformal-PSF.

`train_near_contact` is not equally clean: only 1,403 of 1,800 nominal groups are
reported as near-contact, and 251 candidates carry `post_contact`.  Its report
also has an empty regime contract.  The new loader therefore enforces the contract
again from the directory name and manifest metadata: a `*_near_contact` group is
kept only when its nominal sample is near-contact and no candidate is post-contact,
prefix-collision, or prefix-contact.  This lets the current dataset be used safely
without rebuilding it immediately.  Rebuilding `train_near_contact` with the same
strict construction rules as validation/test is still recommended for the final
paper artifact.

The Near train split is deliberately/historically much harder than val/test
(hard mean 0.0894 versus 0.0087/0.0161; harm mean 0.0289 versus 0/0).  None of the
six current main-table Near planners fits neural weights on this split, so this
shift does not currently enter learned parameters.  A future learned Near planner
should document this hard-mining distribution and should not use test data to
rebalance it.

### Contact

Although the Contact reports carry a lax/empty `dataset_contract` field, their
content is internally consistent: all train/val/test candidates are post-contact,
and every nominal group is post-contact.  The new loader additionally requires a
post-contact nominal for every `*_contact` dataset, making the intended contract
explicit in code.

The Contact train split is also considerably harder than val/test (hard mean
0.0936 versus 0.0148/0.0212; harm 0.0283 versus 0/0), but all six main Contact
baselines are controllers/optimizers with no neural training.  Their train stage
is only a data-contract check.

`traincontact.json` appears to be a stale/alternate report for the same
`train_contact` path (16,306 samples / 1,943 groups / 486 scenes versus the
`train_contact.json` report's 16,790 / 2,000 / 500).  Do not use `traincontact.json`
for paper statistics; use `train_contact.json`.

## 3. Removed repeated computation

1. **Selective NPZ loading.** External baseline train/eval/calibration now loads
   only fields the adapters consume instead of decompressing all teacher-future
   and debug arrays.  This is especially relevant to Near/Contact, where each
   sample stores 11/13 futures according to the reports.
2. **Shared candidate-group features.** Scene/time-invariant feature components
   are built once per candidate group rather than once for every 6--9 candidate.
3. **Shared history transformation.** Ego/neighbor history is transformed once per
   scene/time group.  Only the candidate-specific prefix trajectory is transformed
   per candidate.
4. **Non-learning train-contract scans.** Near six methods: six repeated scans ->
   one scan. Contact six methods: six -> one. Safe PDM-Closed + IDM: two -> one.
   These validation scans run on CPU and no longer occupy an A30.
5. **Offline learned inference uses AMP.** Model forward uses the configured AMP
   dtype while result conversion is safely promoted to FP32.
6. **Observed-risk profiles in multi-method offline evaluation remain shared once
   per candidate group**, rather than recomputed per hand-designed method.
7. **Closed-loop scheduling is work-conserving.** On Bash versions supporting
   `wait -n -p`, the next planner is assigned to whichever A30 becomes free first;
   fixed-pair scheduling remains the fallback.

These are structural reductions in repeated work.  A trustworthy wall-clock
speedup percentage requires running on the user's `/data0` storage and A30s, which
are not mounted in this environment.

## 4. Two-A30 training schedule

The Safe launcher now separates training from offline evaluation.  With the
unchanged default `CUDA_DEVICES=0,1` it trains:

- pair 1: `gameformer_lite` on GPU 0 + `plantf` on GPU 1;
- pair 2: `pluto` on GPU 0 + `pdm_hybrid` on GPU 1.

Only after the learning phase completes does offline evaluation begin.  This
prevents a short model's evaluation from occupying a GPU while the next training
pair is waiting.  Near/Contact main-table methods are non-learning and therefore
do not pretend to consume GPU training time.

## 5. Validation performed in this environment

- shell syntax: Safe / Near / Contact / all-regime launchers passed;
- Python compilation: modified training/data/evaluation/calibration/registration
  modules passed;
- targeted and existing external/closed-loop regression tests: **34 passed**;
- explicit numerical cases include BF16 GameFormer outputs, large trajectory
  coordinates, non-finite padded candidates, and NaN inactive auxiliary heads.

The existing top-level train/test commands are unchanged.
