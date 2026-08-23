# v16.8.25 — Certificate-Then-Utility probe, immutable planner repair, and exact-ID Waymax evaluation

## Scope and evidence contract

This revision is based only on the user's original v16.8.24 code, original `formal_v16_8_24_compact_full_5k` caches/statistics, original training command, and original `v16_8_24_compact5k_all` results. No previously proposed assistant-side MCFC/proposal modification was run by the user and **MCFC is not part of v16.8.25**. This revision deliberately performs **no label, natural-root, RCOT target, proposal-bank, split, cache-schema, or dataset reconstruction change**.

The purpose is to isolate the largest model-side question that remains answerable with the existing cache: after a protected-priority BCOT certificate has accepted a candidate, does the current selector gain or lose by applying BCOT risk a second time through the set-preservation frontier instead of ranking the certified set by planner/ego utility?

## Evidence from the existing v16.8.24 run

### 1. The primary global ceiling is proposal support, not BCOT discrimination

On held-out data:

- `ProposalCoverage/AnyConventionalSafeSceneRate = 0.9550`;
- `ProposalCoverage/AnyNCFSceneRate = 0.36346`;
- fixed-bank global false-safe lower bound = `0.59475`;
- priority-eligible-without-priority-NCF PBTR lower bound = `0.41322`.

The validation calibration report is explicitly `status=proposal_infeasible`: its requested maximum selected global false-safe rate is 0.55 while the fixed-bank lower bound is 0.60371. Therefore a threshold-only repair cannot satisfy the requested operating-point contract.

The held-out fallback rate is `0.32667`, fallback PBTR is `0.82414`, and only `0.07908` of fallback-selected scenes contain any NCF proposal. Thus most fallback burden is downstream of candidate-bank insufficiency rather than a recoverable selector mistake.

Proposal-bank redesign is still an algorithmic bottleneck, but it is intentionally deferred in this revision because changing proposals would require label/cache regeneration and would confound the present selector/model diagnosis.

### 2. RCOT/BCOT is currently the strongest learned mechanism

Held-out diagnostics:

- `RootTransport/LowSafeExist_AUPRC = 0.89743`;
- `RootTransport/ConflictConditioned_AUPRC = 0.80326`;
- `RootTransport/PriorityConflict_AUPRC = 0.78196`;
- `BCOT/PriorityFalseSafe_AUPRC = 0.83736`;
- `BCOT/GlobalFalseSafe_AUPRC = 0.92806`;
- `BCOT/PriorityRiskRankingPairAccuracy = 0.83837`.

By contrast, the generic candidate classifier has `NCF_AUPRC = 0.17558`, `FalseSafe_AUPRC = 0.35444`, and risk-ranking pair accuracy `0.53660`. It remains a diagnostic/ablation head and should not be promoted to the paper's mechanism.

### 3. Protected-priority feasibility is supported; universal hard NCF is over-conservative

At the same checkpoint/candidate bank:

- COWP: `PBTR=0.47107`, `EP=0.61547`, `Fallback=0.32667`;
- soft-burden-only: `PBTR=0.51033`, `EP=0.44307`, `Fallback=0.045`;
- conventional safety: `PBTR=0.51963`, `EP=0.44424`, `Fallback=0.045`;
- universal NCF: `PBTR=0.48347`, `EP=0.52334`, `Fallback=0.56250`.

These aggregate results directionally support a hard protected-priority feasibility layer over a finite soft burden cost, and reject universal all-critical hard veto as the default. Publication claims still require paired confidence intervals / multi-seed evidence.

### 4. There is a real but secondary post-certificate selector gap

Held-out:

- `Selector/NCFSelectionRecallGivenAvailable = 0.78877`;
- selected false-safe excess above the fixed-bank oracle floor = `0.07677` absolute;
- `PlannerRankingPairAccuracy = 0.61321`.

Therefore proposal support dominates globally, but approximately 7.7 percentage points of selected false-safe rate and about 21% of NCF-available scenes remain selector-side room that can be studied without regenerating data.

### 5. Current physical-safety evidence is a warning, not a final conclusion

The cached candidate Waymax outcome subset covers only `623/1200 = 0.51917` held-out selected scenes. On this partial subset COWP has collision `0.02729`, offroad `0.11236`, unsafe union `0.13965`, versus conventional-safety unsafe `~0.07465`. Because coverage is incomplete and these are attached candidate replay outcomes rather than the strict online exact-ID rollout, this is not a publication-level regression claim. It is a strong reason to expose outcome-head discrimination and run strict Waymax before adding more mechanism complexity.

### 6. Current 5k dataset is sufficient for another model/selector iteration

Train/val/held-out mechanism prevalence is stable: relevant-pair rate is about 42.9%, witness given relevant is about 68--69%, root recovery positives are about 40%, and intervention-induced viability/recovery switches cover >93% of scenes. The current data therefore supports another selector/planner experiment without rebuilding.

One exception is **burden-only affected-root supervision**: `affected_root == unsafe_root` on the current train/val/held-out caches and burden-only affected-root prevalence is zero. Consequently the affected-root extension has no independently identifiable signal in this dataset. It may remain in the current checkpoint for compatibility, but no paper claim should state that the current experiment proves an additional non-collision/high-burden affected-root mechanism. A future clean conflict-only retraining ablation is required before retaining that component as a claimed contribution.

## Training-protocol defect discovered from the original command

The original published run uses `--stage all`. In `03_train.py`, stage `all` optimizes natural/response/witness/transport/planner objectives jointly and checkpoint selection falls back to total loss. The existing `history_all.json` confirms every checkpoint used `checkpoint/kind=loss`.

This is inconsistent with the v16.8 changelog's intended **immutable mechanism checkpoint** protocol. A planner-only stage already has a planner-specific checkpoint composite and zero planner-side RCOT/response auxiliary scales, but the old freeze policy allowed `CandidateEncoder` to leave the frozen state after warmup. Because RCOT/BCOT consumes that representation, this also re-enabled dropout in a mechanism input during planner training even though planner features are detached.

v16.8.25 repairs planner-only freezing:

- graph frozen;
- candidate encoder frozen for the full planner stage;
- natural decoder frozen;
- witness decoder frozen;
- SetTransport/BCOT frozen;
- response decoder frozen;
- learned priority-claim gate frozen;
- downstream planner/outcome-side heads remain trainable according to the existing objective.

This repair does not change the existing v16.8.24 checkpoint. It only makes a later `--stage planner --resume cowp_all_best.pt` experiment scientifically interpretable.

## Algorithm change: Certificate-Then-Utility (CTU)

A new internal evaluation method `cowp_cert_utility` is added. It is deliberately small and does not replace `cowp` unless the paired probe passes.

CTU keeps exactly the same:

1. conventional candidate validity/safety;
2. protected-priority relation semantics;
3. BCOT budget gate and uncertainty UCB;
4. localized severe protected-pair veto;
5. action/rule/outcome physical shield;
6. fallback implementation.

It changes only the post-certificate ordering:

- **COWP v16.8.24:** BCOT defines hard admissibility and is then reused in `select_set_preservation_frontier_*` to construct/rank a second non-coercion frontier.
- **CTU:** BCOT defines hard admissibility once; after the same physical shield, the learned planner score chooses among admissible candidates.

This matches the manuscript's intended decomposition: mechanism evidence defines feasibility; ego utility ranks the feasible set. CTU is a one-factor diagnostic/repair, not a new CCF-A novelty claim.

## Outcome-head diagnostics

The existing checkpoint already trains collision/offroad outcome heads because the run used `--with-waymax-outcome-labels`, while current selection has `candidate_selection_outcome_weight=0` and evaluation uses zero outcome-risk penalty. v16.8.25 therefore reports, without using them for selection:

- `OutcomeHead/Collision_AUPRC`;
- `OutcomeHead/Offroad_AUPRC`;
- `OutcomeHead/UnsafeUnion_AUPRC`;
- `OutcomeHead/EvaluatedCandidates`;
- `OutcomeHead/UsedForSelection`.

Do **not** add an outcome shield/penalty solely because the head exists. A later intervention is justified only if these diagnostics are strong and strict Waymax confirms a physical-safety bottleneck.

## Exact-ID Waymax repair

The original command supplied `--scenario-ids-file`, but the original evaluator did not parse or propagate that argument. v16.8.25 adds a real exact-ID path:

- CLI accepts `--scenario-ids-file`;
- rollout resolves only requested IDs;
- duplicate IDs are rejected;
- `--num-scenarios` cannot silently truncate an exact-ID run;
- missing requested IDs are a hard error;
- output records resolved scenario IDs and manifest SHA256.

This is an evaluation-integrity repair, not an algorithm change.

## Preregistered next experiment sequence (no data rebuild)

1. **Validation CTU budget sweep:** same checkpoint/cache, CTU only, budgets 0.30--0.50.
2. **Validation paired shared-pass comparison:** COWP vs CTU plus existing baselines at budget 0.50. Certificate metrics must be numerically invariant. If CTU worsens PBTR or selected false-safe by >1 pp, or retains <95% EP, do not promote it.
3. **Held-out paired learned-offline comparison:** only if validation is non-inferior. Since the existing held-out result has already informed algorithm design, treat it as developer held-out evidence, not a final blind test.
4. **200-scenario paired exact-ID Waymax probe:** identical SHA-selected held-out IDs, horizon, action mode, metrics, checkpoint and budget for COWP/CTU.
5. **Planner-only repair:** only if the previous probes show remaining planner-head room. Warm-start the current checkpoint for 6 epochs at `1e-5` on existing train/val caches. No data reconstruction.
6. **Mechanism ablations after selector lock:** clean affected-root-vs-conflict-only and causal-relevance ablations, retrained under identical seeds/checkpoint-selection protocol. Do not mix these with the selector repair.

## Decision rules

- If CTU preserves certificate invariants and improves/non-inferior PBTR/FSR/EP plus strict Waymax: promote CTU as the paper-aligned selector implementation.
- If CTU degrades burden metrics: retain v16.8.24 frontier; conclude that BCOT has useful post-certificate ranking information and do not force the manuscript abstraction onto the implementation.
- If learned-offline improves but strict Waymax degrades: the bottleneck is physical/online ranking, not RCOT/BCOT; inspect outcome-head AUPRC and online/offline candidate/selector parity before changing the certificate.
- If CTU is neutral and planner-only repair is neutral: stop optimizing selector/model heads. The dominant remaining algorithmic ceiling is the proposal bank; only then reconsider proposal redesign/cache regeneration.

## Local regression

- Focused CTU + exact-ID + immutable-planner tests: **9 passed**.
- Full v16.8.25 repository suite: **252 passed, 5 skipped, 8 failed**.
- The same eight failure classes are present in the original uploaded repository: six tests reference archived launcher files absent from the uploaded zip, and two tests hard-code an older semantic fingerprint. They are not modified to manufacture a green suite.
- Python `compileall` and the new execution launcher `bash -n` checks pass.
