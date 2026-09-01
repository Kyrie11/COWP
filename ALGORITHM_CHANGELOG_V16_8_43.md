# V16.8.43 — Blocker-Conditioned Interaction-Aware Reachable-Response Envelope (BC-IARE)

Online method:

```text
cowp_blocker_conditioned_interaction_aware_reachable_response_envelope
```

Release role:

```text
NEW STAGE-1 SCIENTIFIC BRANCH
EXACTLY NESTS V42 RC-IARE
CHANGES ONLY RECOVERY-SUPPORT INDEXING/COVERAGE
PRESERVES SOCIAL CRITICAL SET AND ALL V42 ROOT/RESPONSE/CSP THRESHOLDS
ADDS SEMANTICS-PRESERVING PERFORMANCE CACHES
```

## V42 evidence selecting this branch

V42 result reliability independently passes 63/63 blocking checks. V42 nevertheless
fails the unchanged Stage-1 conjunction because historical RVR rescue retention is
3/10 < 5/10. Therefore V42 policy is archived and fresh37 is prohibited.

The physical outcome signal is nevertheless strong on the development panel:

```text
COWP collision = 34/48
V42 collision  = 24/48
paired = 10 rescue / 0 induced
Kinematics net regression = 0
Offroad net regression = 0
paired mean EP delta = +0.000219
```

Only 3 of those 10 rescues are historical RVR rescues; 7 are new COWP rescues.
Thus RC-IARE is retained as a mechanism object, not promoted as a completed policy.

V42 pooled reject decomposition over 495,101 interaction hypotheses:

```text
unsupported blocker        282,610 = 57.08%
root unrecoverable          173,029 = 34.95%
residual physical            38,600 =  7.80%
joint incompatibility           142 =  0.029%
```

For the seven historical RVR rescues missed by V42, unsupported blockers rise to
75,470 / 112,707 = 66.96% of hypotheses.

The code has a structural support-index mismatch:

```text
scene-level online critical support: default max 4 agents
causal collision context:             default max 24 agents
```

V42 prepares natural response support for the former but discovers exact collision
blockers in the latter. A blocker outside the active critical set therefore fails as
`unsupported_collision_blocker` even though the frozen NaturalDecoder can be queried
for that model-visible agent.

## Scientific hypothesis

P0 is narrowed to:

```text
Blocker-Conditioned Natural-Option Support Indexing / Coverage
under Interaction-Aware Uncertified Recovery
```

V43 tests only whether this support-index mismatch is suppressing useful RC-IARE
certificates. It does NOT globally enlarge the social NCF critical/protected set.

## Algorithm

1. Run the exact V42 RC-IARE constructor first.
2. If V42 returns a certificate, return that selection unchanged and do not let the
   new extension replace it.
3. Only if V42's hard set is empty, form a late-bound query pool from the frozen
   causal collision context: model-visible, current-valid, non-SDC, non-original-
   critical nearby agents.
4. Decode their natural alternatives using the frozen NaturalDecoder and the
   **root-scene** graph latent. Candidate-conditioned planner latent is forbidden.
5. Concatenate the late-bound root queries with the original critical root support
   only for the recovery certificate, not the social NCF certificate.
6. Re-run the exact V42 hard interaction certificate with unchanged canonical root
   measure, same-root low-burden response bank, environment safety, residual physical
   certificate, multi-blocker CSP, V39 schedule family, shift closure and selector.
7. Any selected action must still differ from the ordinary COWP fallback first action.

No model parameter, loss, checkpoint, data split, proposal geometry, root threshold,
burden budget, controller limit, horizon or scalar selection weight is changed.

## Frozen probability/response semantics

Exactly inherited from V42:

```text
p_min = 0.03
probability floor epsilon_p = 0.02
mean-path root dedup = 0.10 m
minimum roots = 2
retained canonical probability mass >= 0.75
AGENT_PRIORITY adaptive low-burden beta
same-root analytic recovery bank
current + shifted roadgraph / Waymax kinematics
responder-environment bidirectional safety
multi-blocker exact joint CSP
```

## Causality / leakage boundary

V43 uses only current simulator state, map, frozen causal collision context, frozen
checkpoint natural roots/logits and analytic dynamics. It does not use logged future,
future Waymax state/outcome, online mechanism GT, dense learned response trajectories,
scenario-ID special cases or counterfactual48-specific parameters.

## Performance optimization

V42 counterfactual48_parallel2 took 55,440 seconds (~15.4 h) and evaluated roughly:

```text
495,101 interaction hypotheses
7,813,492 responder-environment compatibility checks
76,616 responder-responder joint checks
```

V43 adds policy-step-local memoization for predicates whose inputs are invariant
across hypotheses:

- responder/root/profile ↔ frozen environment actor current+shift compatibility;
- unordered responder-profile pair current+shift compatibility;
- actual emitted first target + acceleration → successor collision context.

The logical check/reject counters are deliberately preserved as if uncached. Cache
hits are separate diagnostics. A synthetic repeated-compatibility microbenchmark
reduced unsafe-predicate calls 209,920→7,680 (~27.3x) and reduced that subcomponent
wall-clock by ~11.1x. This is not an end-to-end Waymax speed claim; `profile8_parallel2`
is provided for server measurement.

## Preregistered Stage-1 gate

The six inherited conditions are unchanged:

```text
historical RVR rescues retained >= 5/10
historical RVR induced avoided >= 7/9
net COWP collision failures removed >= 3
Kinematics net regression <= 1 scene
paired mean EP delta >= -0.05
nonzero action-changing intervention
```

Any failure archives V43 and prohibits fresh37.

## Preregistered interpretation after V43

- Gate pass + blocker-query selections >0: support-indexing hypothesis supported;
  enter fresh37.
- Gate pass + query selections =0: cannot credit V43; outcome is nested V42.
- Query selected and unsupported rejects fall, but root-unrecoverable dominates and
  Gate fails: stop indexing patches and move to root-conditioned control-reachable
  responder support; do not tune root/burden thresholds.
- Query roots not ready: audit late-bound natural-root indexing/calibration; do not
  retrain or tune on counterfactual48.
- Query selected but new physical failures appear: archive; do not weaken hard
  environment/joint constraints.
- Certificate→late collision (especially `fccd9a25a2a57a73`) persists: separate P1
  multi-step invariance/interaction-model uncertainty branch.
- Overall fail: stop selector/grid/schedule family and move to genuinely
  interaction-conditioned reachable response construction.

## Newly prohibited directions

- do not globally increase `max_online_critical_agents` to fix recovery support;
- do not add all nearby collision actors to the protected social NCF set;
- do not tune p_min/floor/mass/root-count/dedup/beta/response primitives on CF48;
- do not weaken universal retained-root support;
- do not remove nonblocking environment checks;
- do not weaken exact multi-agent joint compatibility;
- do not add first-action grids, interpolation points, schedules or switch times;
- do not shrink 8 s conventional safety or change common controller limits;
- do not mix the `fccd...` Type-B long-horizon repair or accepted-path Kinematics
  repair into V43;
- do not use logged-replay improvement as causal burden evidence;
- do not treat counterfactual48/fresh37/exact200 as final publication holdouts.
