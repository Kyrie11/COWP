# v16.7 自动续训与后台运行说明

## 1. 自动续训行为

`run_cowp_v16_7_dual_gpu.sh` 现在默认启用：

```bash
AUTO_RESUME=1
```

对三个训练阶段分别扫描：

- natural：`$OUT_ROOT/checkpoints/natural/`
- witness：`$OUT_ROOT/checkpoints/transport/`
- planner：`$OUT_ROOT/checkpoints/planner/`

每个阶段优先选择最新有效的同阶段 checkpoint：

1. `cowp_<stage>_last.pt`
2. 最新有效的 `cowp_<stage>_epochXXX.pt`
3. `cowp_<stage>_best.pt`

解析器会读取 checkpoint 内部的 `stage` 和 `epoch`，忽略损坏文件或阶段不匹配文件。若发现未完成 checkpoint，启动参数会自动切换为同阶段：

```bash
--resume <latest-checkpoint> --resume-training
```

因此会恢复模型、optimizer、LR scheduler、epoch 编号、history 和 early-stop 计数。

训练 epoch 使用零基编号。例如：

- `TRANSPORT_EPOCHS=14`
- 最终完成 checkpoint 的 `epoch=13`
- `next_epoch=14`

此时 witness train 阶段会直接跳过。

`FORCE_TRAIN=1` 在 `AUTO_RESUME=1` 下表示“运行所有尚未完成的训练阶段”，不会再无视 checkpoint 从头覆盖。

## 2. 当前 mechanism 流程说明

`NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh` 按原设计复用已经通过 gate 的 v16.6 natural checkpoint，因此该入口中：

```bash
RUN_NATURAL=0
RUN_TRANSPORT=1
RUN_PLANNER=1
```

也就是说，你当前的 mechanism 指令会自动续训 witness 和 planner。natural 自动续训逻辑已经在公共训练驱动中实现，在运行 `RUN_NATURAL=1` 的完整 v16.7 流程时生效。

## 3. 推荐后台启动命令

把原命令中的 `BACKGROUND=0` 改为 `BACKGROUND=1`：

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
BACKGROUND=1 \
AUTO_RESUME=1 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
TRANSPORT_AMP=1 \
PLANNER_AMP=1 \
bash NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```

脚本会立即返回后台 PID。整个 mechanism 流程从 gate 检查开始就处于 `nohup` 后台进程中，SSH 或远程终端断开不会终止训练。

## 4. 日志与状态

总日志：

```bash
$OUT_ROOT/logs/driver.nohup.log
```

PID：

```bash
$OUT_ROOT/logs/driver.pid
```

各阶段日志：

```bash
$OUT_ROOT/logs/train_natural_ddp.log
$OUT_ROOT/logs/train_transport_ddp.log
$OUT_ROOT/logs/train_planner_ddp.log
```

查看总日志：

```bash
tail -f outputs/cowp_v16_7_mechanism_v9labels_seed2026/logs/driver.nohup.log
```

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
bash CHECK_RUN_STATUS_V16_7.sh
```

重复执行后台启动命令时，如果 `driver.pid` 对应进程仍存活，脚本会提示已有任务运行，避免重复启动。

## 5. 前台调试

需要前台运行时显式使用：

```bash
BACKGROUND=0 bash NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```
