# COWP v16.8.9 smoke contract repair — 下一步执行

## A. 现在先修复现有 96-scene smoke，不重算标签

```bash
cd /path/to/COWP_v16_8_9_contract_repaired

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export SMOKE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_smoke

bash RECOVER_V16_8_9_CAUSAL_AUDIT_SMOKE_CONTRACT_CN.sh
```

该脚本直接原位修复：

```text
$SMOKE_ROOT/labels_val_v16_8_9/*.npz
```

不重新运行 `01_build_labels_from_proto`，不改变已有 candidate/relevance/response/witness/NCF，只补齐并校正 root-level audit/transport contract，然后在**原输出位置**刷新：

```text
$SMOKE_ROOT/v16_8_9_contract_repair.json
$SMOKE_ROOT/paired_probe.json
$SMOKE_ROOT/proposal_source_ablation.json
$SMOKE_ROOT/causal_audit_diagnostic.json
$SMOKE_ROOT/training_supervision_audit.json
$SMOKE_ROOT/v16_8_9_smoke_verdict.json
```

若 repair 工具发现旧 NPZ 中 `root_affected != root_unsafe OR root_budget_crossed` 等根本语义不一致，会拒绝原位修复；这时才需要重新构建 96 scenes。

## B. repaired smoke 通过后运行 strict probe

仅当：

```text
recommend_strict_probe=true
```

执行：

```bash
cd /path/to/COWP_v16_8_9_contract_repaired

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val

export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe_contract_fixed
export HARD_COUNT=400
export RANDOM_COUNT=800
export LABEL_WORKERS=24
export SEED=2026
export FORCE_REBUILD_PROBE=1

bash NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh
```

只有：

```text
$PROBE_ROOT/v16_8_9_strict_verdict.json
recommend_full_rebuild=true
```

才允许 full rebuild。

## C. strict 通过后的 full rebuild

```bash
cd /path/to/COWP_v16_8_9_contract_repaired

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_contract_fixed

export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE=$SOURCE_DATA_ROOT/tensor_cache_train
export OLD_SCENESET_VAL_CACHE=$SOURCE_DATA_ROOT/tensor_cache_val

export STRICT_VERDICT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe_contract_fixed/v16_8_9_strict_verdict.json

export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0
export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8

bash PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
```

输出仍为 self-contained：

```text
$COWP_ROOT/tensor_cache_train
$COWP_ROOT/tensor_cache_val
```

不需要 transport overlay，也不要求提前恢复/生成 train/val Waymax-outcome cache。

## D. full data 通过后主机制实验

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_contract_fixed
export OUT_ROOT=outputs/cowp_v16_8_9_contract_fixed_seed2026
export TRAIN_SEED=2026
export CUDA_VISIBLE_DEVICES=0,1
export BACKGROUND=0

bash NEXT_RUN_COMMANDS_V16_8_9_MECHANISM_CN.sh
```

先让 seed 2026 整条 mechanism gate 通过，再进入 Waymax。

## E. 必要消融

低成本/共享推理消融：

```bash
bash RUN_OFFLINE_SELECTION_ABLATIONS_V16_8_9_CN.sh
bash RUN_LABEL_AND_PROPOSAL_ABLATIONS_V16_8_9_CN.sh
```

需要独立重训、最关键的 causal ablations：

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_contract_fixed
export CUDA_VISIBLE_DEVICES=0,1
bash RUN_LEARNED_CAUSAL_ABLATIONS_V16_8_9_CN.sh
```

重点对比：

- Full COWP
- w/o candidate-conditioned causal relevance
- conflict-only RootTransport

不要把它们做成共享 checkpoint 的伪消融。

## F. Waymax

mechanism PASS 后：

```bash
export PROBE_SCENARIOS=100
bash NEXT_RUN_COMMANDS_V16_8_9_PROBE_CN.sh
```

Waymax probe promotion PASS 后：

```bash
export FULL_SCENARIOS=1000
bash NEXT_RUN_COMMANDS_V16_8_9_FULL_CN.sh
```

## G. multi-seed / baseline

seed 2026 全流程稳定后：

```bash
export SEEDS="2026 2027 2028"
bash RUN_MULTI_SEED_MECHANISM_V16_8_9_CN.sh
```

external baseline 必须在同一个 fresh proposal/data bank 上重新训练。先做 numeric smoke：

```bash
MODE=smoke RUN_ONLINE_EVAL=0 bash RUN_EXTERNAL_BASELINES_V16_8_9_CN.sh
```

确认 skipped-batch fraction <= 0.02 后再：

```bash
MODE=full RUN_ONLINE_EVAL=1 bash RUN_EXTERNAL_BASELINES_V16_8_9_CN.sh
```
