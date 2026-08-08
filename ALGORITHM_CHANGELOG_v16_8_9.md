## v16.8.9 — Candidate-Conditioned Causal Audit and Affected-Root Transport (2026-08-07)

### Triggering evidence from the v16.8.8 96-scene smoke

The stable-critical repair itself succeeded: all 96 smoke scenes used `fixed_anchor_v1`, all proposal-union monotonicity checks passed, `AnyValid=1.0`, and there were no build/filter errors.  The remaining failure was highly structured rather than a generic proposal-quality failure.  On the paired 48-scene representative subset, the fresh bank improved protected-priority burden transfer from `0.5814` to `0.4419` and slightly improved conventional-safe coverage (`0.8958 -> 0.9167`), yet universal `AnyNCF` fell from `0.3750` to `0.2083` and the false-safe lower bound rose from `0.5208` to `0.7083`.  PSY was physically valid (`64/64` accepted in the complete smoke profile) and produced protected priority-NCF pairs, but its scene-level `AnyNCF`, false-safe and PBTR increments were exactly zero.  Removing RMR/PSY/legacy timing changed candidate counts but not the 96-scene scene-level ceiling.

This combination shows that candidate generation was no longer the sole bottleneck.  A candidate could solve one protected pair but remain universally non-NCF because every globally critical agent was still audited, even when that particular ego candidate did not causally perturb the agent's low-burden natural options.  The old data contract also represented only geometric `mode_conflict`; a collision-free natural root that was forced above the adaptive burden budget could reduce OPR/tail feasibility without a corresponding transport-support label.  These cases created the possibility of a non-NCF "silent blocker" with no coherent relevance/affected-root supervision for the model.

### Data/certificate semantic repair

1. Added a candidate-conditioned causal-audit layer with one shared floor-smoothed canonical natural-root probability measure.  For every `(candidate, critical-agent, natural-root)` tuple the cache now records whether the root is geometrically unsafe, its direct burden under the ego candidate, and whether it is **affected** (`unsafe OR direct_burden > beta`).
2. Added `cowp/audit/pair_relevant` and `cowp/audit/relevance_mass`.  A globally critical agent is audited for a candidate only when enough neutral low-burden natural-root mass is causally affected.  The support threshold reuses the existing 0.10 witness-support semantics; it is not a relaxed NCF/burden threshold.
3. Irrelevant global-critical pairs are vacuously non-coercive for that intervention (`OPR=1`, no blocker, no response search).  Relevant pairs retain the original burden/OPR thresholds.  Every relevant non-NCF pair is required to have a witness; schema/integrity gates reject silent blockers.
4. Generalized RootTransport support from `mode_conflict` to `mode_affected`.  Burden-only affected roots now receive same-root recovery, root minimum safe burden, tail-CVaR and OPR supervision exactly like unsafe roots.  Geometric conflict remains a separately reported physical-safety subset.
5. Safe-response generation skips audit-irrelevant pairs.  This is both semantically correct and targets the measured build bottleneck: v16.8.8 spent the overwhelming majority of label time in safe-response and witness search, not in candidate generation.
6. Added candidate-level audited-pair/blocker counts, pair-level NCF/blocker codes, causal relevance mass, affected-root tensors, and complete inline transport fields to the fresh-cache contract.

### Learned-model alignment

1. `WitnessDecoder` now predicts an explicit pair relevance logit.  Witness/OPR/burden losses operate on relevant pairs while relevance itself is supervised on every valid candidate/global-critical pair.
2. `SetTransportCertificateHead` adds a burden-only affected channel and enforces `P(affected) >= P(conflict)`.  Transport uses `retain + affected * q`, not `retain + conflict * q`.
3. Set-Transport loss supervises affected mass, per-root affectedness, affected-root recovery, root burden consistency and uncertainty under the same canonical root measure used by labels.
4. Planner features use relevance as a causal support gate: irrelevant pair witnesses contribute zero and their OPR contribution is neutral (`1`).
5. `paper_aligned_supervision_batch` was updated so fresh v16.8.9 caches cannot be silently interpreted with the old all-critical/conflict-only certificate.  Legacy caches remain readable only under explicit legacy semantics.

### Real learned ablations

Added forward/loss switches and config variants for two paper-critical learned ablations that use the same rich v16.8.9 dataset:

- `w/o candidate-conditioned causal relevance`: all global-critical pairs are again consumed by the learned witness/transport/planner path and the relevance head/loss is disabled.
- `conflict-only RootTransport`: burden-only affected roots are removed from the learned transport support while geometric conflict supervision is retained.

These are independent retraining experiments, not shared-checkpoint aliases.

### Promotion and rebuild policy

- **Do not full-rebuild from v16.8.8.** The next mandatory step is a fresh 48-hard + 48-random v16.8.9 causal-audit smoke because label semantics changed.
- Smoke additionally requires non-degenerate audit relevance, a measurable burden-only affected-root signal, zero silent/irrelevant blockers, zero responses on irrelevant pairs, exact audit/transport affected-root agreement, stable critical selection and proposal-union monotonicity.
- Only a subsequent 400-hard + 800-random strict probe with `AnyNCF>=0.40`, false-safe floor `<=0.55`, PBTR floor `<=0.45`, hard recovery `>=0.20` and all causal-audit integrity checks may set `recommend_full_rebuild=true`.
- Full rebuild uses the old tensor-cache directories only as scenario-ID allowlists.  All candidate/natural/audit/response/witness/transport labels are rebuilt from WOMD Scenario proto and serialized into self-contained NPZ files.
- The complete validation cache is re-gated before any GPU training.  Failing proposal or audit integrity at this stage is an explicit `DO NOT TRAIN` condition.

### Engineering and validation

- Added scripts `57_diagnose_causal_audit.py`, `58_screen_v16_8_9_causal_audit_probe.py`, `59_gate_fresh_v16_8_9_cache_protocol.py`, and `60_verify_fresh_v16_8_9_cache.py`.
- Added v16.8.9 smoke, strict-probe, self-contained full-build, mechanism, Waymax probe/full and experiment-ablation wrappers.
- Training launchers now accept explicit model/train config sources, allowing genuine independently trained ablations with strict provenance.
- Local Python regression after the complete v16.8.9 change set: **175 passed**; `compileall` and all v16.8.9/core launchers pass `bash -n`.


## v16.8.9 engineering repair — complete smoke/protocol tooling and numeric-module import fix (2026-08-08)

This repair does **not** change the v16.8.9 algorithm, labels, thresholds, output roots, or JSON output locations.

1. Restored the complete v16.8.9 tooling set in the delivered package: `57_diagnose_causal_audit.py`, `58_screen_v16_8_9_causal_audit_probe.py`, `59_gate_fresh_v16_8_9_cache_protocol.py`, and the matching v16.8.8+/v16.8.9 versions of `46_compare_proposal_probe.py` and `50_ablate_proposal_sources.py`.
2. Replaced invalid source-level imports of the numeric-prefixed module `cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol` with `importlib.import_module(...)`. Python accepts the numeric module name through importlib / `python -m`, but not in a `from ... import ...` statement because `59_...` is not a valid Python identifier token.
3. Applied the import repair consistently to the smoke, strict-probe, and fast-full-build wrappers.
4. Added `NEXT_RUN_COMMANDS_V16_8_9_CASUAL_AUDIT_SMOKE_CN.sh` as a typo-compatible alias to the canonical `...CAUSAL_AUDIT...` entrypoint; both preserve the same output paths and behavior.
5. Verified every `python -m cowp.scripts.*` reference in the v16.8.9 shell entrypoints resolves to a real module.
6. Regression: `python -m compileall -q cowp` PASS; full `pytest -q`: **175 passed**; core v16.8.9 shell `bash -n` PASS; CLI `--help` smoke for scripts 46/50/57/58/59 PASS; fingerprint import smoke PASS.

## v16.8.9 data-contract repair — exact audit/transport semantics and supervision sufficiency (2026-08-08)

This revision is an engineering/data-contract correction triggered by the completed 96-scene causal-audit smoke. It does **not** relax the NCF, false-safe, PBTR, hard-recovery, relevance, burden, or proposal thresholds, and it does not alter the v16.8.9 candidate geometry.

### Evidence from the completed smoke

- On the 48-scene unbiased representative subset the fresh bank reached `AnyNCF=0.4167`, false-safe floor `0.5000`, PBTR floor `0.4419`, and hard-scene recovery `0.2083`; the paired proposal gate itself therefore passed all point-estimate proposal thresholds.
- Relative to the v16.8.8 fresh bank on the same representative IDs, `AnyNCF` improved from `0.2083` to `0.4167`, all 10 old NCF scenes were retained, 10 new NCF scenes were gained, and PBTR remained below the target floor. This is a positive signal for candidate-conditioned relevance rather than evidence for another proposal-grid expansion.
- Final smoke promotion nevertheless failed only because `burden_only_affected_signal_present=false` and `transport_affected_matches_audit=false`.
- The latter failure was structural: 1,258 root entries had `audit/root_affected != transport/mode_affected`, caused by the witness builder skipping audit-irrelevant pairs before serializing root-level transport support.
- The former was an ill-posed prevalence gate, not a consistency failure. Under the existing physical-safety semantics, `unsafe_between` already covers collision, near-miss, TTC and RSS-gap violations, so an unchanged natural trajectory that is burden-affected without also becoming unsafe is legitimately rare. The smoke contained only 40 such roots out of more than 112k affected roots. The signal remains observable but is no longer required to exceed an arbitrary dataset prevalence.

### Contract fixes

1. Root-level audit tensors now explicitly serialize `root_budget_crossed`, `root_burden_only_affected`, and `canonical_root_weight` in addition to `root_unsafe`, `root_direct_burden`, and `root_affected`.
2. `root_affected` is defined and verified exactly as `root_unsafe OR root_budget_crossed`; `root_burden_only_affected` is exactly `root_budget_crossed AND NOT root_unsafe`.
3. Root-level transport tensors are populated **before** the pair-relevance early exit. Consequently `mode_conflict`, `mode_affected`, and `mode_retained` describe the same root support for relevant and irrelevant global-critical pairs. Pair relevance only decides whether response/witness search and learned pair losses are active.
4. Diagnostics/full-cache verification now hard-check exact equality of audit/transport conflict support, affected support, retained support and canonical root weights, rather than checking only one affected tensor.
5. The burden-only affected-root prevalence is retained as an advisory statistic and ablation signal. Promotion no longer manufactures or demands a fixed positive prevalence; hard gates instead enforce the root identities and cross-module consistency.
6. Added `61_repair_v16_8_9_audit_transport_contract.py` and `RECOVER_V16_8_9_CAUSAL_AUDIT_SMOKE_CONTRACT_CN.sh`. Existing v16.8.9 smoke NPZ files can be repaired in place without recomputing Scenario-proto labels because the repair changes only root-level serialization and does not change candidate trajectories, relevance, responses, witnesses, pair NCF or candidate NCF.
7. Added `62_audit_training_supervision.py`. Smoke/strict/full data now report whether every core learned head actually receives non-degenerate positive/negative supervision (candidate NCF, pair relevance, witness, pair NCF, conflict/affected transport, affected-root recovery and protected-priority relations). Full train/val caches are blocked before GPU training if a required head is degenerate.
8. The smoke screen now reports Wilson 95% intervals for the 48-scene representative proposal estimates. This makes explicit that the current point estimates justify a larger strict probe but are not sufficiently precise to justify a four-day full rebuild.

### Promotion consequence

Do not rebuild the 96-scene smoke from WOMD merely to repair the 1,258 transport mismatches. Repair the existing NPZs, rerun diagnostics/screen at the same output paths, and proceed to the 400-hard + 800-random strict probe only if the repaired smoke passes. A full rebuild remains forbidden until the strict probe passes under the exact repaired code/config fingerprint.

### Validation

- `python -m compileall -q cowp`: PASS.
- Full regression suite: **178 passed**.
- Core v16.8.9 smoke/strict/full-build/mechanism/Waymax/causal-ablation shell syntax: PASS.
- Repair and training-supervision audit CLI smoke: PASS.
