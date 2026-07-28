# COWP v16.6 Witness 首步非有限梯度修复与论文一致性报告

## 1. 结论

本次错误不是单纯的“学习率过大”或“缺少梯度裁剪”。报错发生在 witness 阶段第 1 个 batch 的反向传播，并同时落在双向 GRU 的：

- `module.candidate_encoder.temporal.weight_ih_l0`
- `module.candidate_encoder.temporal.weight_ih_l0_reverse`

根因是三类问题叠加：

1. **无效候选的全局坐标零填充在 ego-centric 变换后被破坏。** 原始候选库以全零表示 padding；旧实现对全部候选直接减去 ego 的全局坐标，因此无效候选会变成约 `(-ego_x, -ego_y)` 的大常量轨迹。
2. **候选编码器未使用 candidate-valid mask，也未做物理量尺度归一。** 大坐标、速度、角度、车身尺寸混合后直接进入两层双向 GRU；在混合精度和多项 witness/certificate 损失共同反传时，GRU 第一层输入权重最先溢出。
3. **旧 AMP 处理把可恢复的 FP16 动态缩放溢出当作致命错误。** 旧代码在 `GradScaler.step()/update()` 有机会跳过本次 optimizer step 并降低 scale 之前就抛异常；在 BF16 下还不必要地启用了 GradScaler。

因此，**只增加 `clip_grad_norm_` 无法解决本问题**：梯度已经是 NaN/Inf 时再裁剪为时已晚。

补丁同时修复了数值问题和论文/训练标签之间的关键语义偏差。完整单元与回归测试结果：

```text
126 passed, 1 warning
```

该 warning 来自 PyTorch NestedTensor prototype，与本修复无关。

---

## 2. 为什么恰好是双向 GRU 的两个 `weight_ih_l0`

候选轨迹编码器接收 `[B,K,T,7]`，7 个特征为：

```text
[x, y, heading, vx, vy, length, width]
```

GRU 输入权重的梯度直接依赖每个时间步的输入特征。旧坐标变换使同一无效候选的整段序列都变成大常数；双向 GRU 的正向和反向分支都读取这段异常序列，所以：

- 正向 `weight_ih_l0` 首先收到非有限梯度；
- 反向 `weight_ih_l0_reverse` 同时收到同源非有限梯度；
- 隐藏层权重不一定是日志中最先被检测到的路径。

这与用户日志中两个输入权重对称报错完全一致。

---

## 3. 数值修复

### 3.1 恢复候选 padding 语义

文件：`cowp/models/coordinate.py`

`ego_centric_inputs(...)` 新增 `candidate_valid`，并执行：

1. 检查 mask 形状必须是 `[B,K]`；
2. 坐标变换前将无效候选置零；
3. 仅对有效候选做平移、旋转和航向角归一；
4. 变换后再次把无效候选严格置零；
5. 有效候选含 NaN/Inf 时，直接报告精确索引 `[batch,candidate,time,feature]`。

这保证“全局坐标中的零 padding”不会被错误解释为“世界原点处的一条真实轨迹”。

### 3.2 候选编码器增加 mask、尺度归一和 FP32 精度岛

文件：`cowp/models/candidate_encoder.py`

主要修改：

- `forward(..., valid_mask=None)`；
- 无效候选在进入 GRU 前严格清零；
- 有效候选的 NaN/Inf 不再被静默修复，而是立即失败并报告索引；
- 采用固定物理尺度将输入变成无量纲量：

```python
[50.0, 50.0, pi, 20.0, 20.0, 5.0, 2.0]
```

- 归一化后限制在 `[-20, 20]`；
- GRU 和 projection 位于局部 FP32 precision island；
- 输出后再次应用有效候选 mask，使 padding token 始终为严格零，并阻断无效槽位梯度。

尺度 buffer 使用 `persistent=False`，没有改变参数名称或形状，因此旧 v16.x checkpoint 仍可严格加载。

### 3.3 正确处理 BF16 和 FP16

文件：`cowp/scripts/03_train.py`

修复后的策略：

- **BF16：不启用 GradScaler。** BF16 的指数范围足够大，动态 loss scaling 不是必要机制；
- **FP16：启用 GradScaler。** 检测到非有限梯度时，所有 DDP rank 同步跳过该 optimizer step，并降低 scale；
- 仅在溢出持续发生（连续 8 次，或 scale 已降到 1）时终止；
- BF16/FP32 下若仍出现非有限梯度，则按结构性数据/计算错误 fail loudly；
- 新增运行指标：
  - `runtime/optimizer_steps`
  - `runtime/amp_skipped_steps`
  - `runtime/amp_scaler_enabled`
  - `runtime/amp_scale`

这避免把一次可恢复的 FP16 溢出误判为整个训练失败，同时不掩盖真正的数据损坏。

---

## 4. 论文与代码语义对齐

论文 v16.6 的证书定义要求：

### 4.1 OPR 是保留的自然概率质量

论文定义：

```text
O_i = sum_m p_tilde_im * s_ikm
p_tilde_im = (1-eps_p) * p_im + eps_p / M
```

它不是“冲突根中的条件成功比例”。补丁统一把 OPR 重建为经过 probability floor 后的 retained transported mass。

### 4.2 burden 必须是 same-root 最小安全响应负担

对于每个自然 root：

- 只查看被分配到同一 root 的安全响应；
- 取最小 primitive burden；
- 若没有同根安全响应，使用有限 sentinel `2.0` 表示不可用；
- 不再把无关 root 的最小值相减或混用。

### 4.3 `C_i` 是冲突根上的加权上尾 CVaR

补丁按自然 root 的 floor-smoothed probability mass 计算：

```text
C_i = CVaR_rho({[b*_ikm - beta_i]+}_{m: conflict}; p_tilde_im)
```

不是按 root 数量做等权平均，也不是使用单个全局最小负担。

### 4.4 witness 必须满足论文的联合门控

新的正样本条件严格为：

```text
natural_conflict_mass > delta_c
AND
(tail_burden_excess > gamma OR OPR < alpha)
```

响应不存在不再作为额外、独立的 heuristic gate；没有同根安全响应会通过高 sentinel burden 自然进入 CVaR。

### 4.5 option preservation 不属于 primitive burden

论文附录明确把 option preservation 与 `B_prim` 分开，以避免循环定义。补丁：

- 从旧缓存的 burden component 中移除 option component；
- 使用五个物理/规范性组成训练 primitive burden；
- OPR 单独通过 set-transport / witness 目标监督。

---

## 5. 旧 v9 label cache 的兼容策略

用户当前命令复用了：

```text
SOURCE_NATURAL_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026
```

为了避免重新生成整个 WOMD cache，补丁在训练时通过：

```python
paper_aligned_supervision_batch(batch, loss_weights)
```

使用旧 cache 中已有的 transport primitives 在线重建：

- floor-smoothed root probability；
- natural conflict mass；
- OPR retained mass；
- same-root minimum safe burden；
- conflict-conditioned weighted upper-tail CVaR；
- witness existence；
- candidate false-safe；
- candidate non-coercive feasible；
- 去除 option component 后的 primitive burden。

旧 witness token/interval 仅在旧标签也存在解释正例的位置继续使用；不会把旧的错误 witness existence 强行当成新决策标签。

因此，本补丁可以直接复用现有 v9 augmented cache，不要求先进行全量标签重建。

---

## 6. 修改文件

```text
configs/label_cowp_v16.yaml
configs/model_cowp_v16.yaml
configs/train_cowp_v16.yaml
cowp/data/cache_schema.py
cowp/label/label_engine.py
cowp/label/witness.py
cowp/models/candidate_encoder.py
cowp/models/coordinate.py
cowp/models/cowp_model.py
cowp/models/losses.py
cowp/models/set_transport_head.py
cowp/scripts/03_train.py
label/witness.py
run_cowp_v16_6_dual_gpu.sh
tests/test_v16_6_witness_numeric_fix.py
```

补丁统计：

```text
15 files changed, 970 insertions(+), 103 deletions(-)
```

---

## 7. 新增回归测试覆盖

`tests/test_v16_6_witness_numeric_fix.py` 覆盖：

1. 大全局坐标场景下，无效候选 padding 在 ego-centric 变换后仍为严格零；
2. 无效候选即使含 `1e30` 和 NaN，也不会进入 GRU；
3. BF16 autocast 下候选编码器输出与反向梯度均有限；
4. 无效候选对应输入梯度为零；
5. 有效候选的 NaN 会报告精确索引，而不是被静默替换；
6. 新增尺度 buffer 不进入 state dict，旧 checkpoint 可兼容；
7. GradScaler 仅用于 FP16，不用于 BF16；
8. 旧 v9 标签能重建论文定义的 OPR、冲突质量、same-root CVaR、witness、false-safe 和 NCF；
9. weighted CVaR 按 probability mass 而不是 root count 计算。

---

## 8. 应用方式

### 方式 A：使用完整修复代码包

把完整 ZIP 解压到一个新的工作目录，不要直接覆盖当前唯一代码目录。确认环境和数据路径后运行测试：

```bash
cd /path/to/COWP_fixed
pytest -q
```

### 方式 B：在现有代码树应用补丁

先备份或创建 git commit：

```bash
cd /home/senzeyu2/code/COWP
git status
git apply --check /path/to/COWP_v16_6_witness_nonfinite_and_paper_alignment.patch
git apply /path/to/COWP_v16_6_witness_nonfinite_and_paper_alignment.patch
pytest -q
```

若当前目录不是 git repository，仍可使用 `git apply`；也可退而使用：

```bash
patch -p1 < /path/to/COWP_v16_6_witness_nonfinite_and_paper_alignment.patch
```

---

## 9. 推荐复跑命令

旧输出目录已经写入过 strict provenance，而本补丁改变了训练行为文件与配置哈希。不要直接复用旧失败目录；使用新的 `OUT_ROOT`：

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_5_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
OUT_ROOT=outputs/cowp_v16_6_full_pipeline_v9labels_seed2026_witnessfix_v1 \
AMP_DTYPE=bfloat16 \
BACKGROUND=1 \
RUN_FULL=1 \
bash NEXT_RUN_COMMANDS_V16_6_FULL_CN.sh
```

在 GPU 不支持 BF16 时改为：

```bash
AMP_DTYPE=float16
```

FP16 路径已经具备 DDP 同步的 skip-and-downscale 恢复逻辑。

主要日志：

```text
outputs/cowp_v16_6_full_pipeline_v9labels_seed2026_witnessfix_v1/logs/train_transport_ddp.log
```

建议重点检查：

```text
runtime/optimizer_steps > 0
runtime/amp_scaler_enabled = 0       # BF16 时
runtime/amp_skipped_steps = 0        # BF16 时通常应为 0
loss 及各子损失均为 finite
```

如果新日志出现：

```text
Non-finite feature in a valid candidate trajectory at [batch,candidate,time,feature]=...
```

这表示真实有效候选的 cache 数据已损坏。日志会直接给出 batch 内精确索引，应回查该样本，而不是继续用 `nan_to_num` 掩盖。

---

## 10. 验证状态与边界

已经完成：

- 补丁在未修改原始树上 `git apply --check` 通过；
- Python 语法编译通过；
- 三个运行 shell 脚本语法检查通过；
- 全测试套件 `126 passed`；
- 候选 padding、大坐标、NaN 隔离、BF16 backward、旧 checkpoint state-dict 兼容、论文标签重建均有定向测试。

当前环境不包含：

- 用户实际的 `outputs/...` cache；
- 用户的双 GPU CUDA 运行环境；
- 真实 2044 batch witness epoch 数据。

因此，本报告不能声称已经在用户服务器上完成完整 epoch 或证明最终收敛。补丁已消除从代码与报错路径可确定的结构性故障，并把下一次真实数据异常转化为可定位日志。最终仍需在用户机器上执行上述新 `OUT_ROOT` 复跑，确认实际 cache 和 CUDA/cuDNN 环境下的首批次及后续训练。
