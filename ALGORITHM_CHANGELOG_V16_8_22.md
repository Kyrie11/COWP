# COWP v16.8.22 mechanism-support changelog

- Propagated evidence-aligned protected-primary semantics from strict promotion into train-pilot promotion.
- Removed the obsolete train-pilot hard gate `all-critical any-NCF scene prevalence >= 0.30`; replaced with absolute primary/auxiliary supervision support plus within-scene/root mechanism contrast.
- Added `cowp/scripts/74_audit_mechanism_contrast.py` for Layer 2-5 intervention identifiability.
- Added `cowp/scripts/75_reaudit_v16_8_21_train_pilot.py` and `76_reaudit_v16_8_21_strict_for_v16_8_22.py`.
- Planner primary ranking/classification now uses explicit protected-priority candidate labels; all-critical ranking is a low-weight auxiliary.
- Primary BCOT no longer receives a non-protected 5% floor or an all-critical max-deficit feature; the global BCOT head retains all-critical stress diagnostics.
- Primary protected certificate is vacuously feasible when the protected set is empty; global burden transfer remains observable through the global head.
- Fresh v16.8.22 pilot hard sampling can use `--hard-definition protected`.
- Added self-contained `PREPARE_COWP_V16_8_22_DATA_CN.sh`; full-core no longer depends on the missing historical `PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh`.
- Full-core requires WOMD-1.3.1 Waymax readiness, SDC paths, complete label/TFExample matching, and train/val six-layer mechanism-support audits before outcomes/training.
- Scenario->label semantic fingerprint remains unchanged from v16.8.21: `c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51`.
