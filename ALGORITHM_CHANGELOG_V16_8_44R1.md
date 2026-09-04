# V16.8.44R1 — Dynamic-Profile Cache Fidelity Repair + Exact Runtime Work Reuse

Classification: **engineering-only reliability repair**. Scientific algorithm remains V16.8.44 RC-CRRS.

## Blocking bug

V44 dynamic responder completions are ego-hypothesis-conditioned, but the shared joint-CSP cache inherited from V42/V43 used `(agent_index, root_ordinal, profile_index)` as identity. Because V44 reuses dynamic `profile_index` values across hypotheses, different trajectories could alias in the cache. A regression test proves the pre-repair false reuse and the repaired fail-closed behavior.

## Repair

- Dynamic compatibility identity = exact residual acceleration + exact residual duration inside the fixed `(agent, root)` namespace.
- Environment and joint-CSP caches use semantic response identity.
- Candidate-independent dynamic trajectory/burden/roadgraph/kinematic work is reused within one policy step.
- Root↔environment event support and dynamic responder↔environment checks are reused exactly where their inputs are invariant.
- Ego-conditioned interaction checks are not cached away.
- New diagnostics expose static/event/environment cache hits and dynamic selected-response parameters.

## Scientific status

Uploaded V44 lost7 `0/7` is **not attributable**. No GO/STOP and no algorithm-family closure is allowed until the repaired exact lost7 rerun. The preregistered gate remains `>=2/7` newly rescued.
