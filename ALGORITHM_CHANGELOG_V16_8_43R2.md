# V16.8.43R2 — BC-IARE Runtime-Fidelity Repair

**Classification:** engineering-only repair. **No new scientific algorithm version.**

This repair exists because the uploaded V16.8.43 result package contains only
`equivalence16` and `profile8`; the preregistered `counterfactual48` Stage-1 result is
absent. The available profile8 is internally consistent but exposes a practical
execution blocker: BC-IARE selection consumes about 99.5% of policy time and the 8-scene
parallel profile took 26,242 s (~7.29 h). Under the project rule, no V43 algorithm
attribution or new mechanism is permitted until the unchanged Stage-1 can be run.

## Frozen scientific method

The method ID remains:

```text
cowp_blocker_conditioned_interaction_aware_reachable_response_envelope
```

The following are unchanged:

- exact V42-first nesting;
- exact-blocker-conditioned late-bound recovery support semantics;
- social RCOT/BCOT critical/protected set;
- NaturalDecoder weights and root-scene latent;
- canonical root probability measure (`p_min=0.03`, floor `0.02`, dedup `0.10 m`,
  minimum roots `2`, retained mass `>=0.75`);
- same-root low-burden response bank and adaptive AGENT_PRIORITY beta;
- responder↔environment bidirectional safety;
- multi-blocker exact CSP;
- V39 physical tube, 8 s horizon, shift closure, roadgraph and Waymax kinematics;
- common controller limits, selector ordering and execution override;
- compact-5k data, split, checkpoint, loss and proposal geometry;
- the six preregistered Stage-1 GO conditions.

## Engineering root cause

The V43 scientific hypothesis is **exact blocker late binding**, but the released runtime
implementation did more work than that hypothesis requires:

1. it built a query list from nearly every model-visible non-critical actor in the frozen
   collision context;
2. it decoded natural roots for that full list before knowing which actor actually caused
   an unsupported-blocker failure;
3. after exact V42 returned no certificate, it reran the full V42 interaction hypothesis
   family with the enlarged support domain.

A support extension can only change a V42 hypothesis that was rejected specifically as
`unsupported_collision_blocker`. A hypothesis already rejected for no blocker, residual
physical failure, root unrecoverability or joint incompatibility cannot become feasible
merely because an unrelated actor receives natural-response support.

## Semantics-preserving repair

V43R2 therefore adds an internal trace to the unchanged V42 constructor:

```text
unsupported_hypothesis_indices
unsupported_blocker_union
```

The second pass now:

1. runs the exact V42 pass first;
2. returns the exact V42 selection unchanged if it exists;
3. if V42 is empty, keeps only hypotheses whose original failure reason was
   `unsupported_collision_blocker`;
4. intersects the late-bound query pool with the **exact unsupported blocker union**;
5. defers NaturalDecoder execution until that exact blocker set is known;
6. skips re-running nested V39 in the second pass because the identical first pass has
   already proved the V39 hard set empty;
7. replays only the repairable unsupported hypotheses under the otherwise unchanged V42
   hard certificate;
8. retains the existing policy-step-local environment/joint/successor caches.

This is a work-elimination transform, not a relaxation or a new selector.

## New diagnostics

The runtime repair records:

```text
blocker_conditioned_query_candidate_agents_before_exact_filter
blocker_conditioned_query_exact_blocker_agent_count
blocker_conditioned_query_replayed_hypothesis_count
```

These are exported into scenario metrics to quantify how much irrelevant work was removed.

## Validation

- V42 + V43 dedicated tests after repair: **21/21 passed**.
- V16.8.25→V16.8.43 focused semantic/integrity launcher sanity: **107/107 passed**.
- Python compile: PASS.
- launcher shell syntax: PASS.
- exact200/equivalence16/counterfactual48/fresh37/profile8 manifest hashes: PASS.
- new tests cover exact-blocker filtering, second-pass skip when no unsupported failure,
  and deferred NaturalDecoder invocation only on exact blockers.

No end-to-end Waymax speedup is claimed locally because the server dataset/checkpoint/GPU
runtime is not available here. The repaired `profile8_parallel2` plus
`verify_profile8_repair` must establish behavioral fidelity and actual wall-clock speed on
the server before Stage-1 is rerun.

## Frozen experiment gate

The Stage-1 conjunction remains exactly:

```text
historical RVR rescues retained >= 5/10
historical RVR induced avoided >= 7/9
net COWP collision failures removed >= 3
Kinematics net regression <= 1 scene
paired mean EP delta >= -0.05
nonzero action-changing intervention
```

No threshold may be changed after seeing V43R2 results.

## Prohibited scientific changes during this repair

All V43 prohibitions remain active. In particular, do not:

- tune `p_min`, probability floor, root mass, root count, dedup or beta;
- enlarge the social protected/critical set;
- relax environment safety, joint CSP, roadgraph, kinematics, shift closure or 8 s horizon;
- add first-action grids, schedules, switch times, horizon stacking or scalar weights;
- enable dense response heads or train a new viability model;
- rebuild compact-5k;
- mix the `fccd...` long-horizon mismatch or accepted-path kinematics into this rerun.

Only after a reliable counterfactual48 result exists may the scientific branch decision be
made from the already frozen V43 interpretation rules.
