# V16.8.45R1 — RCRSO Schema / Support-Semantics Fidelity Repair

**Classification:** engineering + semantic-fidelity repair of V16.8.45. **Not V16.8.46 and not a new scientific branch.** No V16.8.45 scientific result existed before this repair because the first `sidecar_smoke` failed before training/Stage-0/closed-loop evidence.

## Trigger

The original V16.8.45 `sidecar_smoke` crashed in `_roadgraph` because formal compact tensor caches preserve raw WOMD tf.Example roadgraph vector fields as flat arrays (for example `roadgraph_samples/dir` with shape `[P*3]`), while the new sidecar builder indexed them as if they were `[P,3]`. The Waymax dataloader already restores these raw arrays to `[P,3]`; the sidecar builder had bypassed that reshape contract.

## Correctness / fidelity repairs

1. **Raw WOMD roadgraph reshape** — restore flat `xyz/dir` to point-major matrices and validate valid/type lengths; malformed fields fail loudly instead of being silently reinterpreted.
2. **Critical actor row identity** — prefer `cowp/critical/input_index` over Scenario-proto `track_index`; dataset alignment preserves authoritative cached input rows when object-ID remapping is unavailable.
3. **Blocker heading** — RCRSO blocker heading now uses the project-wide state layout yaw slot `state[6]`, not `state[2]`.
4. **Shifted environment fidelity** — sidecar shifted non-ego context advances by one causal constant-velocity step, matching the online current/shift certificate surrogate.
5. **Roadgraph subset fidelity** — sidecar map cropping keeps the relevant local roadgraph and a nearest valid lane fallback, preventing empty-map vacuous drivable truth.
6. **Teacher-label fidelity** — cached/V44 controls are proposals only; every training positive is replayed through the frozen hard verifier. Verifier rejects are hard negatives, never positives.
7. **Canonical root mass** — sidecar uses cached canonical root weights when available or the shared canonical-root helper; retained-root count/mass follows the frozen `min roots=2`, `mass>=0.75` contract.
8. **Complete retained-root groups** — sidecar limits only between complete hypothesis/root groups; it never truncates a universal-root certificate halfway through one group.
9. **Transformer padding** — root/ego/environment/global memory padding is masked in query cross-attention, so padded environment tokens cannot change proposals.
10. **Stage-0 support semantics** — compare historical frozen fixed bank and V44 analytic behavior against the actual V45 extension `fixed ∪ hard-verified RCRSO proposals`; Stage-0 no longer evaluates RCRSO as a replacement-only operator.
11. **Online proposal-hole fidelity** — V45R1 first exact-nests frozen V43. Only after that hard set is empty, retained roots with empty frozen static response domains may reach RCRSO as *proposal holes*. They are not marked feasible; only RCRSO profiles that pass the unchanged burden/roadgraph/Waymax/current+shift/environment verifier can populate the domain.
12. **Joint-CSP diversity fidelity** — in the RCRSO-only pass, hard-verified learned profiles may augment an already nonempty frozen root domain, because set-valued proposal completeness can matter for exact multi-root/multi-blocker joint compatibility. No verifier predicate is relaxed.
13. **RCRSO cache identity** — dynamic response compatibility identity is tied to the actual knot/acceleration sequence (or trajectory fallback), preventing stale-cache aliasing across different learned response geometries.
14. **Fail-closed execution** — new R1 sidecar/checkpoint/closed-loop paths are distinct from original V45; closed-loop commands require both Stage-0 PASS and base-equivalence PASS.

## Scientific status after repair

- V44R1 reliable scientific conclusion remains unchanged: analytic scalar-residual responder completion is closed; the broader root-conditioned recourse-set-completeness question remains open.
- Original V16.8.45 scientific status is **UNRESOLVED / NOT RUN**, because it crashed at sidecar construction and, on audit, its online path under-implemented the intended proposal-completeness semantics.
- V16.8.45R1 is the first implementation allowed to generate RCRSO evidence.
- Do **not** design V16.8.46 from the crash or from any original V45 partial outputs.

## Frozen preregistration

1. **Stage-0A validation support gate:** K is selected only on validation from `K∈{2,4,8,16}` as the smallest K reaching 95% of the maximum observed FullHypothesisRootCoverage plateau. GO to Waymax only if selected RCRSO FullHypothesisRootCoverage improves by at least **3 percentage points** over the best historical frozen fixed/V44-analytic baseline and VerifiedRootRecall is nonzero.
2. **Common-path equivalence:** frozen reference must pass (`16 scenes / 1120 fields / 0 mismatch` contract).
3. **lost7:** at least **2/7 newly rescued**; otherwise current RCRSO architecture STOP.
4. **rescue10:** after retained3, historical RVR rescues retained **>=5/10**.
5. **induced9:** historical RVR-induced collisions avoided **>=7/9**.
6. **counterfactual48 six-item conjunction:** rescue retention >=5/10; induced avoided >=7/9; net COWP collisions removed >=3; kinematics net regression <=1 scene; paired mean EP delta >=-0.05; nonzero intervention.
7. **fresh37 no-harm:** no net collision harm; no net CR harm; offroad regression <=1; kinematics regression <=1; paired mean EP delta >=-0.03; nonzero intervention.
8. **exact200:** development confirmation only, not publication holdout.

## Learned-family convergence / closure contract

- A new preregistered architecture may continue only if it yields either `>=3 pp` FullHypothesisRootCoverage gain over its predecessor or at least `+1` lost7 rescue, while hard-verifier semantics and common path have zero regression and beta/root/CSP/horizon remain frozen.
- Two consecutive architecture iterations with neither improvement close the learned-recourse proposal family.
- Three total preregistered architectures without lost7 `>=2/7` close the family and force a branch change to natural-root validity / interaction-model uncertainty.
- Any improvement that requires relaxing the hard verifier causes immediate STOP.

## Runtime policy

Keep 80-step closed-loop semantics. Optimize number of architectures/scenes and repeated verifier work, not rollout horizon. Stage-0 snapshots/sidecar should reject weak architectures before Waymax. `profile4_parallel2` is engineering evidence only.
