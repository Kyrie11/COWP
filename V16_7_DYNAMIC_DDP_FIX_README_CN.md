# COWP v16.7 动态 DDP 修复说明

## 1. 本次报错的根因

报错不是 DataLoader IPC，也不是 checkpoint/config mismatch。真正失败点是：

```text
RuntimeError: Expected to have finished reduction in the prior iteration ...
this is not compatible with static_graph set to True.
Parameter indices which did not receive grad: 125 126
```

当前 witness 训练同时满足以下条件：

1. `03_train.py` 为 witness/planner 启用了 `find_unused_parameters=True`；
2. 同时错误启用了 `static_graph=True`；
3. COWP 的监督图本来就是逐 batch 动态的：某个 batch 可能没有有效 response、positive witness、explanation token、discriminative candidate 或可用 Waymax outcome；
4. 因此某些 head 在一个 batch 有梯度，在下一个 batch 的梯度为 `None`。

原运行中 DDP 索引 125/126 对应当时 reducer 注册顺序中的：

```text
response_decoder.mode_head.weight
response_decoder.mode_head.bias
```

该 head 通过 response mixture 参与 BCOT/set-transport，但当本 batch 对应监督分支为空时不会进入反向图。这是合法的训练行为，却违反了 `static_graph=True` 的固定反向图假设。

## 2. 为什么不通过“补一个零损失”解决

把未使用输出写成 `output.sum() * 0` 可以强制参数收到零梯度，但会把 `grad=None` 改成 `grad=0`。AdamW 对两者的处理不同：零梯度参数仍可能接受 weight decay 和动量更新，因此会改变原训练逻辑。

本修复保留原有 mask、loss、梯度和 AdamW 语义，只修改 DDP 如何处理动态计算图。

## 3. 代码修复

### 3.1 动态监督阶段禁用 static graph

对以下 stage：

- `response`
- `witness`
- `planner`
- `all`

使用：

```python
find_unused_parameters=True
static_graph=False
```

并保留 `gradient_as_bucket_view=True`（旧 PyTorch 不支持时自动回退）。

### 3.2 永久冻结提前到 DDP/AdamW 之前

原代码注释称永久冻结发生在 DDP/AdamW 前，但 witness/planner 实际到 epoch 循环才冻结。

修复后：

- witness 阶段经过验证的 natural decoder 在 DDP/AdamW 前永久冻结；
- planner 阶段永久冻结 graph、natural decoder、witness decoder；
- 可能在 warm-up 后解冻的 graph/candidate 参数仍在 DDP reducer 和 optimizer 中，不影响原解冻逻辑。

### 3.3 新增 DDP 参数索引清单

训练启动后生成：

```text
<output-dir>/ddp_parameter_manifest_witness.json
<output-dir>/ddp_parameter_manifest_planner.json
```

以后若 PyTorch 再报告参数索引，可直接查到参数名、shape 和 numel。

## 4. 重新运行

代码替换后，原命令保持不变：

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
BACKGROUND=0 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
TRANSPORT_AMP=1 \
PLANNER_AMP=1 \
bash NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```

PyTorch 可能在第一个 batch 输出一次：

```text
find_unused_parameters=True ... did not find any unused parameters
```

这只是性能提示，不是错误。早期 batch 可能刚好使用了全部相关 head，而后续 batch 仍可能跳过可选监督，因此动态 DDP 设置必须保留。

## 5. 链路检查结果

已完成：

- 两进程 Gloo DDP：可选 head 交替有/无梯度，连续训练通过；
- 两进程 DDP：warm-up 参数 `requires_grad` 冻结后再解冻，通过；
- realistic A=6 / M=24 natural forward-loss-backward preflight，通过；
- 全仓库 `pytest -q`：146 passed；
- 全部 shell 文件 `bash -n`：通过；
- v16.7 两个 launcher 引用的 22 个 Python 模块：全部可导入；
- Python compileall：通过；
- natural→transport checkpoint migration 与 planner strict checkpoint 测试：通过。

## 6. 后续 FULL 链路

planner 阶段原来也使用了相同的错误 `static_graph=True` 策略，因此即使 transport 偶然跑完，planner 仍有较高概率出现同类 DDP reduction 报错。本次修复同时覆盖 planner。

`NEXT_RUN_COMMANDS_V16_7_FULL_CN.sh` 的 online-only 依赖逻辑仍保持：只复用同一 `OUT_ROOT` 下通过门禁的 planner checkpoint、机制验证和 BCOT calibration，不重新要求外部 v16.6 natural checkpoint。
