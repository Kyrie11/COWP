# COWP v16.6 执行说明

## 一、解压并进入目录

```bash
unzip COWP_v16_6_optimized.zip
cd COWP_v16_6_optimized
```

## 二、优先复用服务器已有 v16.5 checkpoint，重做协议正确的 attribution

```bash
MAIN_OUT_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
SOURCE_ABL_ROOT=outputs/cowp_v16_5_natural_ablations_v9labels_seed2026 \
ATTR_OUT_ROOT=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026 \
BACKGROUND=0 \
bash RUN_NATURAL_ATTRIBUTION_V16_6_CN.sh
```

脚本会：

1. 读取 main report 中真正选中的 epoch；
2. 查找三组相同 epoch checkpoint；
3. 如果某个消融缺少精确 checkpoint，只重训该消融到 main epoch；
4. 在完全相同的 2,000 个场景上重新诊断三组模型；
5. 计算 exact identity objective、violation mass 和 paired bootstrap CI；
6. 写出：

```text
outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/
  natural_component_attribution_gate.json
```

查看结果：

```bash
python -m json.tool \
outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json
```

需要同时关注：

```text
pass=true
paper_claim_ready=false
```

前者允许继续收集闭环证据；后者在单 seed 下应保持 false。

## 三、若服务器未保留 v16.5 main 选中 epoch checkpoint

重新运行 v16.6 strict natural recovery，并每 epoch 保存：

```bash
OUT_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
BACKGROUND=0 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
NATURAL_AMP=1 \
AMP_DTYPE=auto \
bash NEXT_RUN_COMMANDS_V16_6_RECOVERY_CN.sh
```

然后把 attribution 的 `MAIN_OUT_ROOT` 指向该目录。两个 ablation 可由 attribution 脚本自动补训到 main-selected epoch。

## 四、attribution development gate 通过后运行完整 pipeline

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
OUT_ROOT=outputs/cowp_v16_6_full_pipeline_v9labels_seed2026 \
BACKGROUND=1 \
RUN_FULL=1 \
bash NEXT_RUN_COMMANDS_V16_6_FULL_CN.sh
```

若 natural checkpoint 来自新的 v16.6 recovery，则相应修改 `SOURCE_NATURAL_ROOT`。

完整 pipeline 中 learned-offline protocol 已改为：

```text
calibration partition: dataset_index % 2 == 0
held-out evaluation:  dataset_index % 2 == 1
```

不要把两个 partition 合并，也不要用 calibration JSON 的 `selection_metrics` 作为最终结果。

## 五、查看状态

```bash
OUT_ROOT=outputs/cowp_v16_6_full_pipeline_v9labels_seed2026 \
bash CHECK_RUN_STATUS_V16_6.sh
```

重点检查：

```text
eval/learned_offline/bcot_calibration.json
eval/learned_offline/_shared_model_pass.json
eval/learned_offline/mechanism_verification.json
eval/learned_offline/bcot_readiness.json
eval/waymax/
```

## 六、禁止事项

论文正式实验中不要设置：

```bash
ALLOW_QUALITY_GATE_FAILURE=1
```

不要将：

- `least_violation` calibration；
- calibration partition 指标；
- single-seed continuation gate；
- logged replay 的 model-based proxy；

写成最终机制证明或 SOTA 结果。
