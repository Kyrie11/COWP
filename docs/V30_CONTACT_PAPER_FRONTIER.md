# v30 contact paper-frontier correction

v29 successfully restored the closed-loop top-k audit, but contact audit still reports paper-PCD misses where the selected nominal prefix is preserved while the paper-best candidate is a hard/harm-safe brake or yield recovery.  Inspecting the audit rows shows that the learned PCD proxy is inverted on these misses: nominal has higher predicted PCD than the paper-best recovery candidate, even when the audited paper PCD strongly favors brake/yield.

v30 therefore keeps the v28 checkpoint and v29 safe/near-contact guards, but changes only the contact rescue frontier:

- Safe remains hard-locked to nominal.
- Near-contact remains nominal-preserving by default; no extra near-contact brake challenge is enabled.
- Contact PCD rescue is widened from brake-only to a macro frontier over brake/yield/merge/stabilize.
- Contact rescue/challenge no longer requires positive learned PCD gain against nominal, because the v29 audit proves that this learned gain is exactly the failing signal.
- Contact rescue uses bounded absolute evidence instead: hard/harm feasibility, macro allowlist, DRS floor, r_dep floor, gap ceiling, cooldown, and intervention budget.

The expected effect is to test whether contact nominal-preserved paper-PCD misses can be reduced without regressing safe and near-contact.
