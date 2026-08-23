# v16.8.25 — Exact-ID Waymax repair + experimental Multi-Conflict Feasibility Corridor (MCFC)

Date: 2026-08-23

## Why this revision exists

The uploaded v16.8.24 compact-full data/results are sufficient to attribute the dominant learned-offline limitation, but they are **not sufficient to claim a new algorithmic improvement yet**.

Observed evidence on the frozen v16.8.24 candidate bank:

- validation: `AnyNCFSceneRate=0.35110`, oracle best-case selected false-safe floor `0.60371`;
- current diagnostic held-out: `AnyNCFSceneRate=0.36346`, oracle floor `0.59475`;
- held-out BCOT false-safe discrimination is strong (`Priority AUPRC=0.83736`, `Global AUPRC=0.92806`), while the generic candidate certificate is weak (`NCF AUPRC=0.17558`, `FalseSafe AUPRC=0.35444`);
- when an NCF candidate exists, the learned selector chooses one in `0.78877` of held-out eligible scenes, so selection is a secondary but non-negligible error, not the first ceiling;
- JR-NCF is comparatively high-yield conditional on being conventionally safe (about `0.301` NCF/conv-safe on validation and `0.350` on the inspected held-out), but it contributes only a small number of candidates per scene.

Therefore the first bottleneck is **proposal-support insufficiency**: most certificate-complete scenes contain no NCF candidate for any selector to recover. Merely retuning the BCOT risk budget cannot cross this oracle proposal floor and must not be treated as an algorithm fix.

A second execution issue was found in the supplied strict Waymax command. The command passed `--scenario-ids-file`, but `cowp.scripts.04_eval_closed_loop` and `waymax_closed_loop_rollout` did not implement that argument/path. The intended exact-ID held-out evaluation therefore could not have completed under the uploaded code. The uploaded result archive indeed contains learned-offline outputs but no exact-ID Waymax held-out result.

## Scientific status of the existing held-out split

The current 1,200-scene `heldout_test_scene_ids.txt` has now been inspected and used for diagnosis. It must therefore be treated as a **development/diagnostic held-out split**, not as the untouched final paper test split for v16.8.25 and later algorithm selection. A new deterministic final-blind scenario set must be sampled from unused official WOMD validation IDs, excluding train, validation, and the already-inspected 1,200 scenes, and must remain unevaluated until the algorithm/hyperparameters are frozen.

## Exact-ID Waymax evaluation repair

`cowp/scripts/04_eval_closed_loop.py` now supports:

- `--scenario-ids-file`: ordered txt/JSONL exact allowlist;
- `--tfexample-index-jsonl`: optional scenario-ID -> tf.Example shard index;
- `--allow-missing-scenario-ids`: diagnostic-only escape hatch; strict is the default.

`cowp/waymax_eval/rollout.py` now:

- decodes only requested IDs through `waymax_state_generator_for_sids`;
- deterministically shards an exact allowlist without changing global scenario indices;
- refuses `--num-scenarios` truncation under strict exact-ID mode;
- records `scenario_id` in every rollout item;
- fails if any requested scenario is unresolved.

The JSON output records the source allowlist, requested/evaluated counts, SHA256 digest, exact coverage, and missing IDs. This closes the previous failure mode where a nominal “held-out Waymax” command could not actually enforce the held-out ID set.

## Experimental algorithm: MCFC

### Motivation

The current JR-NCF family maps protected conflicts onto a causal lane route but uses one constant longitudinal acceleration to satisfy multiple pass-after constraints. This is low-dimensional: a single acceleration can be forced by an early binding conflict and remain unnecessarily slow after the conflict, or fail to satisfy several separated timing constraints simultaneously.

### Mechanism

MCFC (Multi-Conflict Feasibility Corridor) keeps the same causal current-state + HD-map contract and constructs a **piecewise progress-time corridor** over multiple protected conflict regions:

1. construct causal route polylines from the current SDC pose and map topology;
2. identify protected `AGENT_PRIORITY` / `EQUAL_OR_NEGOTIATED` conflict constraints without using future ego ground truth;
3. convert the protected agents' late-arrival envelopes plus a gap into ordered pass-after timing knots along route progress;
4. fit a piecewise quintic-Hermite longitudinal progress profile with zero endpoint acceleration at each knot;
5. keep a non-zero low conflict-entry speed to avoid repeating PCHR's stop/hold/release feasibility collapse;
6. when horizon and route length permit, add a post-binding-conflict recovery knot so the proposal does not remain slow for the rest of the horizon;
7. project the progress profile onto the causal route and hard-check that every protected conflict is entered no earlier than its required target, within tolerance;
8. pass MCFC proposals through the **unchanged** conventional validity/map filters and the **unchanged** RCOT/BCOT non-coercive certificate.

New source ID: `ProposalSource.MULTI_CONFLICT_CORRIDOR = 15`.

The reusable primitive `piecewise_quintic_progress_trajectory` supports multi-knot distance/time/speed interpolation with zero endpoint accelerations and monotone non-negative progress repair.

### Scope / novelty discipline

MCFC is **not promoted as a paper contribution in this revision**. It is an experimental proposal-support repair designed to test the attribution above. The paper's main mechanism remains root-conditioned counterfactual option transport (RCOT) plus the protected-priority BCOT feasibility certificate. If MCFC passes the paired proposal probe, it may be described as a certificate-compatible causal proposal family or become the initialization for a later certificate-guided proposal optimizer; it should not create a second competing safety score/head.

The legacy `configs/label_cowp_v16_8.yaml` remains MCFC-free. MCFC is opt-in via `configs/label_cowp_v16_8_25_mcfc.yaml`, and the code default is disabled unless this experimental config is explicitly loaded. This prevents accidental semantic drift of old v16.8 experiments.

## Required promotion sequence

Do **not** rebuild/retrain the full dataset first.

1. Run exact-ID Waymax on the current v16.8.24 diagnostic held-out set to establish whether the current online policy wrapper has a physical-safety/offroad problem and to quantify the offline/online proposal mismatch.
2. On validation only, run a paired MCFC fresh-label proposal probe using old hard scenes plus an unbiased random subset. Compare old-vs-new banks on exactly the same scene IDs.
3. Require the proposal gate to pass before full rebuild. The default strict targets remain approximately: overall AnyNCF >= 0.40, false-safe oracle floor <= 0.55, PBTR oracle floor <= 0.45, hard-scene recovery >= 0.20, plus proposal validity/integrity.
4. Run source ablation and build-profile diagnostics. MCFC must generate actual valid candidates and contribute incremental NCF/protected-NCF coverage; a pass caused only by unrelated source displacement or missing-scene filtering is invalid.
5. Only after (1)-(4) pass, perform the full fresh v16.8.25-MCFC label/tensor/Waymax-outcome rebuild and retrain.
6. Hyperparameters and BCOT budget are selected on validation only. The current inspected held-out remains diagnostic. Final paper metrics are computed once on a newly sampled blind final split after model/config freeze.

If MCFC fails the paired proposal gate, disable it and do **not** spend a full retraining cycle. The next non-redundant direction is certificate-guided continuous proposal refinement over route-progress knots (using frozen BCOT/UCB plus physical constraints), not another stop/hold heuristic, another BCOT threshold sweep, or another flat candidate classifier.

## Important evidence limitation retained

The present data have zero `burden_only_affected_roots` on train/val/current held-out even though some affected roots cross the burden budget. Thus the dataset does not empirically identify the distinction between “conflict-only affectedness” and the more general affected-root formulation. Do not headline that sub-claim without a dedicated burden-only stress set or additional counterfactual supervision. Recovery/OPR switching and BCOT discrimination are supported; burden-only affected-root superiority is not yet supported.

Also, strict Waymax SDC closed-loop with logged-replay non-ego agents is necessary for benchmark physical metrics but is not by itself causal proof of transferred burden. A final CCF-A evidence package should add a reactive-agent stress protocol and a human-audited false-safe set.

## Local validation

Before this revision, the repository's full test run had 8 pre-existing failures (missing archived historical shell wrappers and stale fixed semantic-fingerprint expectations). After the changes, the failure classes/count remain unchanged, while new focused tests pass.

- exact-ID Waymax rollout focused tests: pass;
- MCFC multi-knot trajectory primitive regression: pass;
- focused modified-suite run: `11 passed`;
- full post-change run observed earlier: `251 passed, 5 skipped, 8 failed`, with the same pre-existing failure classes as baseline;
- modified Python modules compile;
- new v16.8.25 build/Waymax-attach shell scripts pass `bash -n`.

No MCFC performance claim is made before the user runs the validation proposal probe on the full WOMD/cache installation.

## 8. Evidence-first execution / source-attributed promotion gate

Added `cowp/scripts/85_screen_v16_8_25_mcfc_probe.py` and
`NEXT_RUN_COMMANDS_V16_8_25_MCFC_CN.sh`.

The MCFC branch is **not** promoted merely because the aggregate fresh proposal
bank passes the historical v16.8 proposal gate.  The new source-attributed
screen additionally requires, by default, all of the following on the paired
validation probe:

- the general paired proposal-bank gate passes;
- MCFC itself emits at least 5 global-NCF candidates and 3 priority-NCF
  candidates;
- removing MCFC lowers scene-level any-NCF by at least 3 percentage points;
- removing MCFC lowers scene-level priority-NCF by at least 2 percentage points;
- including MCFC lowers the global oracle false-safe floor by at least 3
  percentage points and the protected-priority PBTR floor by at least 2 points;
- the fresh bank loses old NCF support on at most 2% of paired scenes.

These thresholds are an **engineering promotion rule**, not a statistical
significance claim.  Passing them authorizes the expensive full rebuild and
retrain; it does not establish final closed-loop benefit.

The execution launcher enforces the intended evidence order:

1. content-blind final-holdout ID freeze, before inspecting any new experiment output;
2. exact-ID strict-Waymax diagnosis of the current v16.8 checkpoint;
3. wider validation-only budget/witness diagnostics;
4. validation-only 600-scene MCFC proposal probe;
5. full rebuild/retrain only after the MCFC source gate passes;
6. development held-out diagnosis;
7. one-shot final-blind evaluation only after algorithm, hyperparameters,
   calibration rule, and checkpoint-selection rule are frozen.

The previously inspected 1,200-scene held-out set is explicitly treated as a
development/diagnostic set from v16.8.25 onward; it must not be presented as a
new untouched final test after it has already influenced mechanism design.
