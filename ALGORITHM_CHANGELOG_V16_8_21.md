# COWP v16.8.21 changelog

- Fixed strict promotion semantics: protected-scene prevalence is advisory, not a proposal-quality hard gate.
- Replaced marginal protected rate gates with minimum protected evidence counts plus paper-aligned PBTR-on-eligible support.
- Added `protected_partition_consistent` to catch drift between eligible/NCF/PBTR labels.
- Added provenance-verified v16.8.20 strict policy re-audit (`cowp.scripts.73_reaudit_v16_8_20_strict_policy`).
- Added v16.8.21 execution wrapper and commands.
- **No Scenario->label semantic change.** v16.8.20 label semantic fingerprint remains `c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51`.
- Clarified research contract: WOMD supplies factual futures and HD-map evidence; COWP offline counterfactual labels are constructive feasibility/viability targets, not identified human counterfactual response ground truth.
