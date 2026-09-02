# V16.8.43R3 — BC-IARE Runtime Work-Reuse Repair (engineering-only)

**Scientific algorithm remains V16.8.43 BC-IARE. This is not V16.8.44 and does not alter the frozen six-item Stage-1 gate.**

## Why this repair exists

The uploaded result package contains reliable `equivalence16` and `profile8` engineering evidence, but no `counterfactual48` Stage-1 result. V43R2 is behaviorally equivalent to pre-R2 V43 on the same 8 profile scenes and improves wall-clock from 26,242 s to 20,649 s (1.2709x), yet selection still occupies about 99.2% of policy time. Therefore the only justified work before Stage-1 is semantics-preserving runtime elimination.

## Exact work reuse

V43R3 preserves the V42-first / V43-late-bound scientific semantics while removing duplicate work between the first RC-IARE pass and the blocker-conditioned repair pass:

1. Reuse the first pass' prepared natural-root/response support for the original critical agents; prepare support only for newly late-bound exact blockers and merge the two disjoint support maps.
2. Reuse the immutable ego hypothesis workspace from the first pass: semantic representatives, schedule family, controller-projected ego hypotheses, kinematics masks and first-action targets.
3. For hypotheses whose only failure was `unsupported_collision_blocker`, reuse the first-pass successor/shift projection record and re-evaluate only the interaction support that became newly available.
4. Preserve the logical reject/check diagnostics as scientific counters; implementation reuse is not allowed to change hard certificate membership or selection.

No p_min/floor/mass/root-count/dedup/beta, response bank, social protected set, V39 schedule family, physical certificate, shift closure, controller limit, 8 s conventional horizon, fallback score, selector, dataset, loss or checkpoint is changed.

## Lightweight fail-fast Stage-1 protocol

The frozen six-item Gate contains two conditions that depend only on the historical V29 discordant set: retain at least 5/10 RVR rescues and avoid at least 7/9 RVR-induced collisions. V43R3 therefore adds a **mandatory 19-scene early falsification stage** containing exactly those 10+9 scenes.

- If either of those two conditions fails on the 19 scenes, the full conjunction Gate is mathematically impossible to pass; stop and do not run the other 29 counterfactual48 scenes.
- If both pass, run the remaining disjoint 29 scenes exactly once, stitch 19+29 back into the frozen 48-scene manifest, and run the unchanged V43 analyzer and all six Gate checks.
- The 19-scene stage can only falsify. It never promotes a method and is not a substitute for counterfactual48.

This protocol reduces failed-branch Stage-1 rollouts from 48 to 19 scenes (60.4% fewer) without changing any promotion criterion. It also avoids rerunning the 19 scenes if the branch passes, because the final 48 result is constructed from the exact disjoint union.

## Engineering fidelity protocol

A 4-scene profile subset is added for quick R3-vs-R2 behavioral fidelity. It is not promotion evidence. Full profile8 remains optional only when an end-to-end wall-clock number is desired.

## Validation

- V16.8.25→43R3 focused semantic/integrity suite: **109/109 passed**.
- exact200/equivalence16/counterfactual48/fresh37/profile8/gate19/remaining29/profile4 manifest hashes: **PASS**.
- Python compile and launcher syntax: **PASS**.
- R3 end-to-end behavior and speed still require the server `profile4`/optional `profile8` rerun; no speedup beyond the measured R2 result is claimed locally.

## Scientific interpretation is deliberately frozen

Because the uploaded result package has no Stage-1 `counterfactual48`, V43R3 does **not**:

- issue a V43 GO/STOP verdict;
- promote or close a scientific algorithm family;
- change the dominant scientific bottleneck beyond the V42→V43 preregistration;
- design a V16.8.44 mechanism;
- tune root/response thresholds on the profile scenes.

The next scientific branch remains conditional on the original V43 preregistered interpretation after a reliable full Stage-1 result exists.
