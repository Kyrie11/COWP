# V16.8.45R3 — Stage-0 Runtime / Observability Fidelity Repair

Classification: **engineering-only**.

Scientific algorithm: **V16.8.45 RCRSO unchanged**.

Scientific verdict from uploaded R2 package: **NOT AVAILABLE** because the mandatory validation `stage0_val_support_audit.json` was never completed. No GO/STOP, no new bottleneck, no promotion/closure, and no V46 algorithm is inferred.

## Repairs

- live Stage-0 progress and phase timing;
- scenario-disjoint `parallel2` execution;
- exact raw-count shard merge/finalization with provenance checks;
- streaming group metrics;
- duplicate fixed-static replay removal;
- exact memoization of repeated CSP trajectory-pair predicates;
- semantic cache namespace includes candidate-local roadgraph/environment/root-state inputs.

## Frozen scientific contracts

- K `{2,4,8,16}` and 95%-plateau validation selection;
- Stage-0 FullHypothesisRootCoverage lift `>=3 pp` and VerifiedRootRecall `>0`;
- hard verifier and exact CSP;
- compact-5k / base checkpoint / NaturalDecoder / RCOT / BCOT / controller / 8 s conventional contract;
- lost7 / rescue10 / induced9 / CF48 / fresh37 gates.

## Validation

- R2→R3 legacy smoke: 17/17 scientific fields exact;
- cross-hypothesis 24-example legacy comparison: 17/17 exact;
- two-shard exact merge smoke: 17/17 exact;
- focused semantic/integrity sanity: 143/143 passed prior to packaging.
