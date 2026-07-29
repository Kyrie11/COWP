# External baselines for OC-RAP safe-regime comparison

This package provides external baseline adapters that consume the same OC-RAP `.npz` candidate-prefix dataset and report the same regime-level metrics. The adapters are intentionally kept outside the OC-RAP model and selector so that they do not use OC-RAP's observation-consistent recovery admission logic.

The GameFormer and BeTop adapters in this revision were rebuilt after inspecting the uploaded `GameFormer-main.zip` and `BeTop-main.zip` source trees. They are not verbatim vendored copies of those repositories, because the original projects use their own WOMD preprocessing/cache schemas. Instead, their core model designs and losses were mapped onto OC-RAP's grouped candidate-prefix dataset interface.

## Implemented baselines

### 1. `nominal_replay` / `log_replay`

No training is required. The policy selects the logged nominal candidate (`is_nominal=1`) and falls back to the highest-utility feasible candidate only if the nominal candidate is infeasible. This is the clean safe-regime lower-bound baseline: it represents log playback / nominal planning without any learned safety intervention.

Config: `configs/external_baselines/nominal_log_replay.yaml`

### 2. `wayformer_bc` / `route_bc_lite`

This is the Waymax route-conditioned BC / Wayformer-style BC adapter. It uses:

- early-fusion attention over OC-RAP candidate-prefix tokens;
- optional latent-query attention for Wayformer-like scene compression;
- a residual MLP action head;
- cross-entropy supervision on the logged / nominal candidate in the OC-RAP prefix lattice.

Config: `configs/external_baselines/wayformer_bc.yaml`. The old `route_bc_lite.yaml` remains as an alias for compatibility.

### 3. `gameformer_lite`

`gameformer_lite` now follows the uploaded GameFormer source design much more closely. The adapter preserves the important GameFormer components:

- LSTM-style agent history encoders for ego and neighbor histories;
- Transformer scene fusion encoder;
- learned multimodal candidate queries, analogous to GameFormer's multi-modal trajectory decoder;
- level-0 trajectory decoder;
- iterative level-k interaction decoder, where each level encodes the previous level's predicted futures and re-attends to the scene context;
- deep supervision across all reasoning levels, following GameFormer's `level_k_loss` idea.

OC-RAP's dataset is candidate-prefix based rather than GameFormer's native full-agent future-cache format. The adapter therefore supervises each mode against the OC-RAP executable prefix trajectory and uses OC-RAP branch/root tensors as interaction/future proxies. This preserves the level-k game-theoretic reasoning structure while keeping the comparison fair: the model only sees the same train/val/test OC-RAP safe-regime samples as the other baselines.

Config: `configs/external_baselines/gameformer_lite.yaml`

### 4. `betopnet_lite`

`betopnet_lite` now follows the uploaded BeTop / BeTopNet source design more closely. The adapter preserves the important BeTopNet components:

- behavior-topology labels derived from candidate ego prefix geometry and nearby-agent motion, matching BeTop's braid/topology supervision role;
- map-topology labels derived from prefix-to-polyline relations, matching BeTop's map-topology branch;
- `TopoFuser`-style fusion for actor and map topology features;
- separate actor-topology and map-topology binary decoders;
- top-k topology indexing, analogous to BeTopNet's `agent_topo_indexing` and `map_topo_indexing`;
- topology-aware attention over selected actor and map topology memories;
- binary focal top-k topology loss, matching the BeTop source loss structure.

The adapter is named `BeTopNet-lite` because it avoids BeTop's original cache format, MTR-specific data loader, and large OpenPCDet-style training stack. The algorithmic comparison point is preserved: topology is explicitly predicted and then used to condition planning scores; OC-RAP's deployable-recovery certificate is not used for the baseline decision rule.

Config: `configs/external_baselines/betopnet_lite.yaml`

## Safe-regime training and testing

Run all requested safe-regime baselines with:

```bash
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export TRAIN_SAFE=$OCRAP_ROOT/train_safe
export VAL_SAFE=$OCRAP_ROOT/val_safe
export TEST_SAFE=$OCRAP_ROOT/test_safe
export RUN=runs/safe_external_baselines
export NUM_GPUS=2
bash scripts/run_safe_regime_external_baselines.sh
```

The script writes:

- `eval_safe_nominal_log_replay.json`
- `eval_safe_wayformer_bc.json`
- `eval_safe_gameformer_lite.json`
- `eval_safe_betopnet_lite.json`

## Individual commands

### Nominal / log replay

```bash
python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/nominal_log_replay.yaml \
  --dataset "$OCRAP_ROOT/test_safe" \
  --split test \
  --output runs/safe_external_baselines/eval_safe_nominal_log_replay.json \
  --baselines nominal_replay,log_replay
```

### Wayformer-style BC

```bash
torchrun --standalone --nproc_per_node=2 -m ocrap.cli train-baseline \
  --config configs/external_baselines/wayformer_bc.yaml \
  --dataset "$OCRAP_ROOT/train_safe" \
  --val-dataset "$OCRAP_ROOT/val_safe" \
  --baseline wayformer_bc \
  --output runs/safe_external_baselines/wayformer_bc

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/wayformer_bc.yaml \
  --dataset "$OCRAP_ROOT/test_safe" \
  --checkpoint runs/safe_external_baselines/wayformer_bc/best.pt \
  --split test \
  --output runs/safe_external_baselines/eval_safe_wayformer_bc.json \
  --baselines wayformer_bc
```

### GameFormer

```bash
torchrun --standalone --nproc_per_node=2 -m ocrap.cli train-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$OCRAP_ROOT/train_safe" \
  --val-dataset "$OCRAP_ROOT/val_safe" \
  --baseline gameformer_lite \
  --output runs/safe_external_baselines/gameformer_lite

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$OCRAP_ROOT/test_safe" \
  --checkpoint runs/safe_external_baselines/gameformer_lite/best.pt \
  --split test \
  --output runs/safe_external_baselines/eval_safe_gameformer_lite.json \
  --baselines gameformer_lite
```

### BeTopNet-lite

```bash
torchrun --standalone --nproc_per_node=2 -m ocrap.cli train-baseline \
  --config configs/external_baselines/betopnet_lite.yaml \
  --dataset "$OCRAP_ROOT/train_safe" \
  --val-dataset "$OCRAP_ROOT/val_safe" \
  --baseline betopnet_lite \
  --output runs/safe_external_baselines/betopnet_lite

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/betopnet_lite.yaml \
  --dataset "$OCRAP_ROOT/test_safe" \
  --checkpoint runs/safe_external_baselines/betopnet_lite/best.pt \
  --split test \
  --output runs/safe_external_baselines/eval_safe_betopnet_lite.json \
  --baselines betopnet_lite
```
