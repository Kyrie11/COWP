# COWP v16.7 learned-natural 兼容与断点续训修复说明

## 1. 审阅范围

本次对论文 `interactive_planning_v16_7_revised(1).tex` 全文和 `COWP(2).zip` 中全部源码做了静态解析，并重点逐行审阅了实际执行链：

```text
NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
  -> NEXT_RUN_COMMANDS_V16_7_CN.sh
     -> run_cowp_v16_7_dual_gpu.sh
        -> cowp.scripts.39_diagnose_learned_natural
        -> cowp.scripts.03_train (natural / witness / planner)
        -> cowp.scripts.04_eval_closed_loop
```

代码库存：233 个 Python 文件、58 个 shell 脚本、56 个 YAML 文件。论文共 1347 行。

## 2. 论文思想与当前代码实现

论文把 **false-safe planning** 定义为：ego 轨迹本身无碰撞，但安全性依赖其他道路参与者采取高负担让行动作。COWP 不只把这种影响作为一个 soft courtesy cost，而是作为 **non-coercive feasibility** 缺陷处理。

当前代码中对应的核心结构是：

1. `natural_decoder.py` 构造有固定语义身份的 OBS / NEU / PRIO natural roots；
2. `set_transport_head.py` 对每个 ego candidate、critical agent 和 natural root 建模冲突、低负担安全响应保留、同 root 恢复与不确定性；
3. protected-priority 风险用于默认 hard certificate，全 critical 风险作为更严格诊断；
4. `policy_wrapper.py` 先做传统物理可行性，再做 priority NCF gate，最后执行保守 fallback；
5. 实际训练阶段名称依次是 `natural`、`witness`、`planner`。启动脚本虽然把第二阶段称为 transport，但 checkpoint 的真实 stage 和文件前缀是 **witness**。

## 3. 原始报错的直接原因

报错缺失的 7 个参数全部来自 v16.7 的 `SetTransportCertificateHead`：

```text
set_transport.candidate_risk_raw_weight
set_transport.candidate_risk_threshold_logit
set_transport.candidate_risk_log_scale
set_transport.global_risk_raw_weight
set_transport.global_risk_threshold_logit
set_transport.global_risk_log_scale
set_transport.pair_deficit_raw_weight
```

这些参数是 v16.7 新增的单调、可解释 transport-risk calibration 参数。你的 mechanism 脚本有意复用已经通过 gate 的 v16.6 natural checkpoint，因此旧 checkpoint 中本来就不可能包含这些参数。

`39_diagnose_learned_natural.py` 的问题是：

- 它实例化了完整 v16.7 `COWPModel`；
- 用 `strict=False` 加载 v16.6 checkpoint；
- 随后又手工把 **任何** missing/unexpected key 都当作致命错误；
- 但该诊断实际执行 `model(batch, stage="natural")`，只依赖 `graph.*` 和 `natural_decoder.*`，完全不会执行 `set_transport`、witness、candidate certificate 或 planner。

所以这是 **诊断加载器的兼容性校验范围错误**，不是 checkpoint 损坏，也不是 DataLoader IPC 错误。日志中的：

```text
DataLoader IPC runtime: ... selected=file_system
```

只是运行时信息，和本次异常没有因果关系。

## 4. learned-natural 修复

修改文件：

```text
cowp/scripts/39_diagnose_learned_natural.py
```

新行为：

- 对 `graph.*` 和 `natural_decoder.*` 继续严格校验；任一缺失或多余仍立即报错；
- 只对不参与 natural 诊断的下游模块允许跨版本缺失，并使用当前 v16.7 配置初始化；
- 支持 `_orig_mod.` 前缀 checkpoint；
- 显式使用 `weights_only=False`，并兼容旧版 PyTorch。

因此修复不会降低 natural 诊断本身的正确性，也不会改变模型前向、loss、训练阶段或测试指标。

## 5. 精确的 checkpoint 路径与 stage 映射

| 逻辑阶段 | 实际输出目录 | checkpoint 前缀 | checkpoint `stage` | 无同阶段 checkpoint 时的上游 warm start |
|---|---|---|---|---|
| natural | `$OUT_ROOT/checkpoints/natural` | `cowp_natural_*` | `natural` | `$INIT_CKPT` |
| transport | `$OUT_ROOT/checkpoints/transport` | `cowp_witness_*` | `witness` | `$NATURAL_CKPT` |
| planner | `$OUT_ROOT/checkpoints/planner` | `cowp_planner_*` | `planner` | `$TRANSPORT_CKPT` |

特别注意：不能在 transport 目录中匹配 `cowp_transport_*`，因为训练命令真实使用的是 `--stage witness`。

## 6. 自动跳过与断点续训规则

修改文件：

```text
run_cowp_v16_7_dual_gpu.sh
NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```

每个阶段启动前会：

1. 只扫描该阶段的精确目录和精确前缀；
2. 检查 `*_last.pt`、`*_epochNNN.pt`、`*_best.pt`；
3. 实际用 `torch.load` 验证文件可读、`model` 非空、stage 匹配、epoch 合法；
4. 忽略损坏、空文件或 stage 不匹配的 checkpoint；
5. 同 epoch 时优先 `*_last.pt`，因为它在 `scheduler.step()` 后保存，并包含最完整的 optimizer / scheduler / early-stop 状态。

决策规则：

```text
checkpoint_epoch >= target_epochs - 1
    -> 阶段已完成，跳过训练

checkpoint_epoch < target_epochs - 1
    -> 同阶段 --resume <checkpoint> --resume-training

没有同阶段 checkpoint
    -> 从上游 checkpoint 做跨阶段 warm start
```

`03_train.py` 内部 epoch 是从 0 开始计数。因此：

- `PLANNER_EPOCHS=30` 时，完成标志是 checkpoint 内 `epoch >= 29`；
- checkpoint 内 `epoch=15` 表示第 15 号 epoch 已完成，续训从 `epoch=16` 开始；
- 不会重复训练已经完整写入 checkpoint 的 epoch。

同阶段 `--resume-training` 会恢复：

- 模型参数；
- optimizer state；
- LR scheduler state；
- early-stop 计数；
- 原 history，并从 `checkpoint_epoch + 1` 继续编号。

`FORCE_TRAIN=1` 现在表示“执行所请求的训练流程”，不会再抛弃同阶段 checkpoint。确实要从头重训某阶段时显式设置：

```bash
FORCE_RESTART_TRAIN=1
```

关闭自动续训可设置：

```bash
AUTO_RESUME_TRAIN=0
```

## 7. provenance 处理

你当前的 `OUT_ROOT` 已经在报错前写入旧代码签名。直接替换脚本后，严格 provenance 会检测到代码签名变化并拒绝继续。

mechanism wrapper 现在只在检测到“旧 provenance 已存在且本 hotfix 尚未登记”时，允许一次可审计的 compatible-resume amendment。成功写入新 provenance 后生成一次性 marker；今后无关代码变化仍会被 strict provenance 阻止，不会永久降低实验审计强度。

同时把 `39_diagnose_learned_natural.py` 加入 provenance 文件清单，因为它是正式 gate 的组成部分。

## 8. 重新运行

替换代码后，原命令可直接使用：

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
BACKGROUND=1 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
TRANSPORT_AMP=1 \
PLANNER_AMP=1 \
bash NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```

后台日志：

```bash
tail -f outputs/cowp_v16_7_mechanism_v9labels_seed2026/logs/driver.nohup.log
```

真实续训时应看到类似：

```text
[transport/witness] resume same stage: checkpoint=.../cowp_witness_last.pt completed_epoch=... next_epoch=...
[planner] target already complete: checkpoint=... epoch=...; skip training
```

## 9. 验证结果

已完成：

- 233 个 Python 文件 AST 解析通过；
- 58 个 shell 脚本 `bash -n` 通过；
- 56 个 YAML 文件解析通过；
- 全部 pytest：`141 passed`；
- 使用完整当前 `COWPModel` 删除上述 7 个 v16.7 参数，构造与报错一致的 v16.6-like natural checkpoint，加载测试通过；
- synthetic checkpoint 测试验证了 natural / witness / planner 的精确路径、stage、完成跳过和 epoch 15 续训判断。

本地环境不包含你服务器 `/data0/...` 下的真实 tensor cache、现有 checkpoint 和训练 GPU 环境，因此没有实际启动完整多 GPU 数据训练；模型加载、调度决策、语法和全部仓库测试均已验证。
