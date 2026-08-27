# V6 External Baseline Root-Cause Fix — 2026-08-27

- External training contract upgraded to `v6_womd_map_topology_source_fidelity_20260827`.
- Replaced V5 flat roadgraph slicing with WOMD feature-id/type/direction-aware map elements for GameFormer and DTPP.
- GameFormer map context is now actor-local; crosswalks are separated from lane centers.
- Restored public GameFormer CrossTransformer and interaction modal aggregation semantics.
- Preserved WOMD roadgraph ids/types/directions through Waymax online batches.
- Added train/eval startup map-topology contract validation.
- Added bounded skip for pathological finite pre-clip gradients and isolated true numerical failures; bad batches never step the optimizer.
- Preserved V5 FP64 global-L2 overflow fallback and non-finite gradient-path diagnostics.
- Original `RUN_5_SOTA_BASELINES_COWP.sh` commands and output root remain unchanged.
