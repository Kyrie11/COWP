# COWP Algorithm Changelog — v16.8.27

## Scope

v16.8.27 is a **strict-Waymax semantic-integrity repair**. It does not change the trained checkpoint, dataset, cache, natural-root construction, RCOT/BCOT certificate, proposal-family definitions, learned heads, or paper-level mechanism. No algorithm promotion is authorized from the v16.8.26 strict physical attribution until the repaired exact-ID evaluation is rerun.

## Triggering evidence from v16.8.26

The v16.8.26 learned-offline diagnostics completed normally and remain usable. The outcome head has stable but modest physical discrimination (held-out collision/offroad/unsafe-union AUPRC about 0.598/0.436/0.733), while the RCOT/BCOT mechanism remains substantially stronger for its intended false-safe task.

The exact 200-ID Waymax runs also completed, and the physical events themselves are real simulator outcomes. However, first-event localization showed that 44/45 COWP collision episodes had a fallback action immediately before first overlap. Code review then found a semantic-integrity bug in the online candidate generator that directly contaminates this attribution path.

## Critical bug: NEUTRAL_EGO was falsely promoted to conventional-safe

In v16.8.26, `_add_candidate(..., conventional_check=False)` was used for the reserved and final `NEUTRAL_EGO` smooth-stop candidates. `_add_candidate` initialized `conv=True` and skipped both the roadgraph drivable screen and causal constant-velocity collision screen when this flag was false. The resulting boolean was then written to `cowp/candidates/conventional_safe`.

Consequently a neutral/smooth-stop candidate could enter the `no_certificate_use_least_coercive_conventional` fallback pool **without ever being conventionally safety-screened**. Smooth stopping is not automatically safe: it may leave the drivable corridor or create/retain a collision with a close rear, crossing, or merging actor.

This is an implementation/logic error, not an algorithmic result. It invalidates the v16.8.26 claim that the observed first-event concentration proves the fallback objective itself is the dominant physical bottleneck. The Waymax CR/collision/offroad/kinematics numbers are still factual outcomes for that buggy controller, but the fallback-mechanism attribution must be rerun after repair.

## Repair 1: no conventional-safety bypass exists anymore

The `conventional_check` argument is removed from `_add_candidate` entirely. Every generated candidate, including `NEUTRAL_EGO`, is always evaluated by:

1. the same kinematic validity check;
2. the same causal roadgraph drivable screen; and
3. the same causal constant-velocity collision screen.

A candidate that fails the conventional screen is **not deleted**: it remains a dynamically valid last-resort candidate, so emergency fallback coverage is preserved. It simply cannot masquerade as `conventional_safe`.

This establishes the invariant:

> membership in the conventional-safe fallback pool always implies that the online conventional screen actually ran and passed.

A runtime integrity assertion now aborts if `no_certificate_use_least_coercive_conventional` ever selects a candidate whose `conventional_safe` flag is false.

## Repair 2: selected-candidate semantic provenance

Every online policy diagnostic now records:
- selected macro id/name;
- selected-valid flag;
- selected-conventional-safe flag;
- fallback reason.

The compact episode diagnostics additionally record, for the action immediately before the first collision/offroad/kinematics event:
- whether the action was fallback;
- whether the selected candidate was truly conventional-safe;
- selected macro;
- exact fallback reason.

The physical comparison script reports these distributions. This is required before making any fallback-side mechanism claim.

## Repair 3: outcome-head metadata bug

The learned-offline multi-method result writer used a stale `method_name` variable while annotating `OutcomeHead/UsedForSelection` and `OutcomeHead/SelectionScope`. When `cowp_fallback_outcome` was the last method, the ordinary `cowp` row could be mislabeled as if the outcome head had participated in selection.

The selection implementation itself was not affected: ordinary COWP still used zero outcome risk when `--outcome-risk-penalty 0`. v16.8.27 makes the metadata method-local through `_outcome_head_selection_metadata()`.

## Repair 4: fine-grained policy runtime profiler

The v16.8.26 outer profiler shows that Waymax `env.step` is not the dominant cost; most wall time is inside the COWP policy. v16.8.27 adds an opt-in `--profile-policy-runtime` profiler that separates:
- state/map extraction;
- CPU online candidate construction;
- host-to-device transfer;
- model forward;
- selection/certificate logic;
- action projection.

With `--profile-policy-sync`, PyTorch CUDA synchronization is used around these fine-grained sections for profiling accuracy; `--profile-waymax-sync` remains the outer Waymax timer. This mode is diagnostic only and should not be used for publication timing.

## Scientific disposition

The following v16.8.26 conclusions remain valid because they come from learned-offline caches and are independent of the online conventional-safety bug:
- CTU remains rejected as a replacement for the certificate-compatible frontier;
- RCOT/BCOT remains the strongest learned mechanism signal;
- outcome head remains diagnostic-quality, not a hard certificate;
- fixed-bank proposal sufficiency remains a global ceiling in the existing cached dataset.

The following conclusion is **withdrawn pending rerun**:
- “strict physical failure is dominantly caused by the fallback objective/recovery policy.”

No new recovery algorithm, proposal family, or certificate component is added in v16.8.27. First restore online semantic integrity, rerun the same exact 200 IDs, then decide the next algorithmic question.

## Regression contract

New focused tests require:
- no `conventional_check` bypass in `_add_candidate` or online neutral generation;
- a neutral candidate that fails the drivable screen is retained as a candidate but marked non-conventional;
- outcome-head selection metadata is method-local;
- first-event diagnostics retain conventional-safe/macro/fallback provenance.

Reconstructed artifact regression (from the byte-identical uploaded v16.8.26 package): **261 passed / 5 skipped / 8 historical failures**. The 8 failures are unchanged repository-history issues (six tests reference launchers absent from the supplied archive; two tests hard-code an old semantic fingerprint). The v16.8.27 integrity + v16.8.26 fallback + Waymax diagnostic + CTU focused set is **13/13 passed**; the packaged `sanity` command is **11/11 passed**.
