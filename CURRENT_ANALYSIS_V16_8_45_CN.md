# V16.8.45 — Verified Root-Conditioned Recourse Set Operator (RCRSO)

## 1. Scientific decision inherited from V16.8.44R1

V16.8.44R1 is a reliable negative Stage-A result: frozen lost7 newly rescued = 0/7 < 2/7. The analytic same-root scalar-residual responder-completion family is therefore closed. The negative result is not caused by a dead mechanism: R1 executed 83,263 completion attempts, 1,350,867 profile evaluations and produced 26,485 hard-checked dynamic profiles, but universal retained-root recourse coverage barely changed and no V44-exclusive certificate/action appeared.

The actionable P0 is therefore **Root-Conditioned Recourse-Set Completeness under Exact-Blocker Hard Feasibility**, not another acceleration/duration grid and not a relaxation of root/burden/environment/CSP semantics.

## 2. Frozen layers

V16.8.45 does **not** modify compact-5k split/cache semantics, NaturalDecoder, natural-root probability semantics, same-root RCOT, BCOT, protected-priority hard certificate, certificate-compatible set preservation, the 8 s conventional contract, V39 controller/shift-closed ego-tube certificate, controller limits, Waymax-aligned kinematics adapter, exact blocker discovery, responder-environment bidirectional safety, exact multi-root/multi-blocker CSP, selector/fallback score, base COWP checkpoint or its training loss.

The base COWP/NaturalDecoder/RCOT/BCOT parameters remain frozen. V45 trains a separate recourse-proposal operator checkpoint.

## 3. V45 method: RCRSO

For one exact blocker and one retained natural root, RCRSO receives the root-local state, root mass/source, blocker state/control context, current+one-step-shifted ego recovery tube, frozen environment set and causal conflict-event support. A Root-Conditioned Recourse Set Transformer uses K learned query tokens to emit K bounded, time-varying longitudinal control-residual knot sequences on the **fixed root geometry**.

Each learned output is only a proposal. It is converted to a same-root response trajectory and then re-evaluated by the unchanged hard verifier:

1. adaptive burden beta;
2. roadgraph drivable screen;
3. current and shifted Waymax-aligned responder kinematics;
4. current ego-response safety;
5. shifted ego-response safety;
6. current/shift responder-environment bidirectional safety;
7. exact multi-root/multi-blocker CSP.

Therefore the learned model can improve proposal completeness but cannot turn a verifier-invalid response into a hard-feasible certificate.

### Output parameterization

- K set queries, `max_queries=16` in the first architecture.
- Each query predicts 8 normalized longitudinal control knots in [-1,1].
- Knots are expanded over the horizon and mapped into inherited accel/decel bounds while keeping root topology fixed.
- Feasibility/burden heads are auxiliary proposal-ordering/training heads only; all K proposals are still verified online.

### Inputs

- root token sequence: root-local xy, yaw tangent, velocity, normalized time, canonical mass, source;
- exact blocker state: local position/yaw/velocity, geometry/type slots and root initial acceleration context;
- current and one-step-shifted ego-tube tokens;
- permutation-invariant environment tokens;
- causal unsafe-event intervals for root↔ego current, root↔ego shifted and root↔environment support.

No logged future or outcome label is available online.

## 4. Sidecar supervision without rebuilding compact-5k

`recourse_sidecar_v16_8_45/` is independent of the base tensor cache. Train/val splits are inherited exactly; exact200 IDs are explicitly forbidden during sidecar construction.

Positive proposal supervision is pooled from:

- existing response labels;
- V44 analytic completion positives (teacher/counterexample source only);
- deterministic Sobol time-varying control proposals that pass the same hard verifier.

Verifier-rejected Sobol proposals closest to verified support are retained as hard negatives for proposal ordering. The sidecar never changes base labels or the online certificate.

The training objective contains root-mass-weighted set coverage/best-of-K matching, proposal-to-set matching, verifier-derived feasibility ordering, low-burden preference, diversity, hard-negative contrast and temporal smoothness. Same-root identity is architectural: controls are integrated on the frozen root geometry rather than generated as unrestricted xy trajectories.

## 5. Stage-0 support gate

Before any Waymax closed-loop RCRSO run, `106_eval_rcrso_support.py` replays the frozen hard verifier on a validation sidecar and compares:

- frozen fixed response bank;
- V44 analytic completion;
- RCRSO@K.

It reports `VerifiedRootRecall@K`, `FullHypothesisRootCoverage`, `ExactCSPCompletionRate`, verified burden, verifier calls and wall time. A complete hypothesis group is never truncated midway by sidecar example caps.

K is **not** selected on lost7. Among K={2,4,8,16}, the smallest K reaching 95% of the observed validation FullHypothesisRootCoverage plateau is frozen into the selected RCRSO checkpoint. Closed-loop loading fails closed unless the checkpoint contains a passing Stage-0 audit and this frozen K.

The default Stage-0 admission requires at least +3 percentage-point `FullHypothesisRootCoverage` over the best frozen fixed/analytic baseline. This is an internal mechanism-compute gate; it does not replace the paper outcome gate.

## 6. Closed-loop preregistration

The V44/V42 development outcome gates are not relaxed.

1. `lost7`: new rescues >=2/7; otherwise STOP learned-recourse family.
2. `retained3`: total historical rescue retention >=5/10.
3. `induced9`: historical RVR-induced avoided >=7/9.
4. `remaining29`: stitch the original frozen counterfactual48.
5. Counterfactual48 original six-item conjunction gate unchanged.
6. Only then fresh37 development generalization.
7. Only then historical exact200 development confirmation.

Exact200 is development-selected; publication evidence requires a new untouched holdout, >=3 independent seeds and paired scenario uncertainty. Strong social-burden claims still require reactive-agent and human-audited stress evaluation.

## 7. Runtime changes that do not weaken experiment semantics

V44R1 lost7 required 18,740 s and selection remained ~99.33% of policy runtime. V45 therefore changes computation shape rather than the 80-step closed-loop semantics:

- removes online V44 dyadic/bisection completion from the V45 method;
- one neural forward emits K proposals for a root context;
- static burden/roadgraph/kinematics verification happens before ego/environment checks;
- immutable response geometry/static checks and responder-environment compatibility reuse semantic cache identities;
- RCRSO dynamic compatibility identity is tied to the actual knot/acceleration sequence, preventing the V44 stale-profile-index bug;
- exact-root scarcity is evaluated first for fail-fast universal coverage while successful hard-set membership/CSP ordering remains unchanged;
- Stage-0 sidecar allows architecture/K screening without 80-step Waymax;
- lost7 supports a frozen `2+2+3` progressive fail-fast order;
- immutable historical baselines are reused rather than rerun.

The 80-step rollout horizon is intentionally unchanged because historical first failures can occur near the end of the rollout.

A server-side `profile4_parallel2` remains included because local code validation cannot claim end-to-end Waymax speedup without the user's data/checkpoint/runtime.

## 8. Failure taxonomy

V45 records per-root proposal failure stages without changing predicates:

- `root_intrinsic_invalid` / no proposal;
- `no_low_burden_static_control`;
- `roadgraph_or_waymax_kinematic_reject`;
- `ego_current_reject`;
- `ego_shift_reject`;
- `environment_current_or_shift_reject`;
- `root_verified_set_nonempty`;
- `joint_csp_incompatible` when verified learned profiles reach but fail the exact CSP.

These diagnostics decide what scientific layer to test next; they are not thresholds that can be tuned after seeing lost7.

## 9. Dataset conclusion

The compact-5k dataset remains frozen. Train/val/heldout sizes are 5000/1000/1200. Across splits, audit-relevant pair rate is ~0.429, protected-priority root coverage ~99.4%, rootless rate=0, and `<2 low-burden roots` rate=0. Mechanism-unauditable rate is ~4.1–4.5%. This does not support a current data-reconstruction hypothesis.

Long-term watch items are the critical-agent cap=6 with ~95.4–95.8% selected-cap saturation and weaker proposal acceptance for `PRIORITY_SMOOTH_YIELD` (~20–22%) and `TERMINAL` (~54–55%). These may become a later support ceiling but are not used to explain V44R1.

Publication provenance caveat: the archived `verify_cache_train.json` says `pass=false` due to `irrelevant_pair_blockers=58243`, although runtime supervision re-masks irrelevant pairs by `audit_target`. Before release, either regenerate/fix that serialization artifact or provide a strict cache→runtime semantic-equivalence proof.

## 10. CCF-A positioning

RCRSO itself should remain a mechanism probe until Stage-0 and closed-loop evidence support it. The stronger paper-level structure is still **Orthogonal Option-Set Feasibility**:

- social axis: collision-free safety must not rely on collapse of protected agents' natural low-burden options;
- physical-interactive axis: uncertified ego recovery must not rely on collapse/missing completeness of exact blockers' natural-root-consistent verified recourse sets.

The unified statement is: **Safety must not be obtained through critical option-set collapse.** Learned RCRSO is useful only insofar as it improves proposal completeness while the hard verifier preserves soundness.

## 11. Local code validation

- V45 dedicated tests: 6/6 PASS.
- V16.8.25→V45 focused semantic/integrity launcher: 120/120 PASS (one non-blocking PyTorch nested-tensor warning).
- Python compile: PASS.
- synthetic one-epoch RCRSO training + Stage-0 fail-closed smoke: PASS; a deliberately non-improving synthetic operator wrote its audit and exited code 4 as designed.
- full repository `pytest -q` still stops at the same historical collection errors already reproducible on the untouched V44R1 tree (`ocrap` environment/import lineage and removed historical `_recovery_bridge_viability_mask` API). These are not V45 regressions.
- Full server Waymax V45 outcome: NOT RUN locally.
