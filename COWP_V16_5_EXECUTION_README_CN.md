# COWP v16.5 执行说明

## 0. 解压与环境

```bash
unzip COWP_v16_5_optimized.zip
cd COWP_v16_5_optimized
```

默认数据目录仍为：

```bash
/data0/senzeyu2/dataset/COWP/formal
```

不同服务器路径可先设置：

```bash
export COWP_ROOT=/你的路径/COWP/formal
```

## 1. 先运行 strict natural recovery

首次运行建议前台执行，以便立即发现环境或数据问题：

```bash
OUT_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
BACKGROUND=0 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
NATURAL_AMP=1 \
AMP_DTYPE=auto \
bash NEXT_RUN_COMMANDS_V16_5_RECOVERY_CN.sh
```

说明：

- `AMP_DTYPE=auto` 在支持 BF16 的 GPU 上优先 BF16；
- graph 在 natural 阶段默认全程冻结；
- 默认 validation 每 2 epoch；
- 不会继续 transport/planner/Waymax。

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
bash CHECK_RUN_STATUS_V16_5.sh
```

必须看到：

```text
natural_basis_gate: pass=true
natural_effectiveness_gate: pass=true
optimizer_steps > 0
amp_skips = 0
```

并确认不存在：

```text
eval/QUALITY_GATES_BYPASSED.txt
```

关键报告：

```bash
python -m json.tool \
outputs/cowp_v16_5_natural_recovery_v9labels_seed2026/eval/learned_offline/learned_natural_effectiveness.json

python -m json.tool \
outputs/cowp_v16_5_natural_recovery_v9labels_seed2026/eval/learned_offline/natural_effectiveness_gate.json
```

## 2. natural gate 通过后运行严格归因

```bash
MAIN_OUT_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
ABL_ROOT=outputs/cowp_v16_5_natural_ablations_v9labels_seed2026 \
FORCE_TRAIN=1 \
NATURAL_AMP=1 \
AMP_DTYPE=auto \
bash RUN_NATURAL_ABLATIONS_V16_5_CN.sh
```

v16.5 只运行两个单因素 control：

- `no_obs_capacity_boost`；
- `no_mass_aware_root_envelope`。

检查 attribution：

```bash
python -m json.tool \
outputs/cowp_v16_5_natural_ablations_v9labels_seed2026/natural_component_attribution_gate.json
```

只有 `pass=true` 才继续完整 pipeline。

## 3. attribution 通过后运行完整 pipeline

```bash
OUT_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_5_natural_ablations_v9labels_seed2026/natural_component_attribution_gate.json \
BACKGROUND=1 \
RUN_FULL=1 \
bash NEXT_RUN_COMMANDS_V16_5_FULL_CN.sh
```

状态：

```bash
OUT_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
bash CHECK_RUN_STATUS_V16_5.sh
```

日志：

```bash
tail -f outputs/cowp_v16_5_natural_recovery_v9labels_seed2026/logs/driver.nohup.log
```

## 4. 训练速度记录建议

在 v16.5 natural 主训练和两个 ablation 中分别记录：

```bash
nvidia-smi dmon -s pucm -d 5 > gpu_profile_v16_5.log &
PROFILE_PID=$!
```

训练结束后：

```bash
kill "$PROFILE_PID"
```

建议比较：

- 每 epoch wall time；
- train samples/s；
- validation wall time；
- GPU utilization；
- peak memory；
- data wait/CPU utilization。

必须在同一服务器、同一 GPU、同一 batch、同一缓存状态下比较 v16.4 和 v16.5，不能从不同时间的 tqdm 瞬时速率推断加速倍数。

## 5. 三 seed 正式实验

单 seed 完成工程验证后，再分别设置：

```bash
SEED=2026
SEED=2027
SEED=2028
```

当前脚本/配置仍以 seed 2026 为默认。正式多 seed 前，应将 seed 通过环境变量或复制的 frozen config 显式写入 provenance，且禁止在 test 结果上重新调 gate 阈值。

## 6. 禁止事项

正式论文实验不要使用：

```bash
ALLOW_QUALITY_GATE_FAILURE=1
```

不要：

- 用 v16.4 attribution 代替 v16.5 证据；
- gate fail 后直接运行 full pipeline 并据此写论文；
- 在看到 test 结果后调整 soft/emergency envelope；
- 把 logged-replay proxy 描述为已识别的真实因果效应。
