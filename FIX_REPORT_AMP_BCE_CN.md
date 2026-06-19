# AMP + BCE 训练报错修复说明

## 直接报错

本次错误发生在 `natural_loss -> _natural_priority_expectation_loss`：

```text
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
```

原实现先计算：

```python
pred_p = (mix * torch.sigmoid(priority_logits)).sum(dim=-1)
F.binary_cross_entropy(pred_p, tgt)
```

在 `--amp` 下，PyTorch CUDA autocast 禁止概率空间 BCE/BCELoss，因为 sigmoid 后的概率在 fp16/bf16 梯度传播中不稳定。正确做法是使用 logits 空间的 `binary_cross_entropy_with_logits`。

## 根本修复

1. `cowp/models/losses.py`
   - 删除唯一的 `F.binary_cross_entropy(...)` 用法。
   - `_natural_priority_expectation_loss()` 保留论文中的“mode mixture priority expectation”语义：先算期望概率 `pred_p`，再用 `torch.logit(pred_p)` 转回有限 logit，最后调用 `F.binary_cross_entropy_with_logits(...)`。
   - 所有 BCE 类损失统一使用 `binary_cross_entropy_with_logits`。
   - 增加 `_binary_target()`：将标签清洗到有限 `[0,1]`，避免 CUDA BCE kernel 因 NaN、inf、越界 target 报错。
   - 增加 `_safe_float()`、安全版 `masked_mean()`，避免 padding 区域 NaN 通过 `NaN * 0` 污染 loss。
   - 对 source/token 类 label 做 clamp，避免 CUDA scatter / CE 越界。
   - pairwise ADE 在 natural loss 中只计算一次并复用，减少重复计算。

2. `cowp/scripts/03_train.py`
   - AMP 只包裹 model forward；loss 统一在 autocast 外以 fp32 计算。
   - 增加 `--compile` 与 `--fused-adamw` 参数兼容你的训练命令。
   - `torch.compile` 改为 `mode='reduce-overhead'` + `torch._dynamo.config.suppress_errors=True`，遇到不支持的子图自动回退 eager。
   - 修复 compiled model checkpoint 的 `_orig_mod.` 前缀保存/恢复问题。
   - representation/natural 阶段默认跳过 witness/candidate positive oversampling，避免训练前扫描所有 `.npz` 的无关标签。

3. `cowp/models/graph_encoder.py`
   - 修复 `--compile` 下 `if empty.any()` 这类张量驱动 Python 分支可能触发的 Inductor/Triton 编译错误。

4. `cowp/data/dataset.py`
   - Stage A 数据加载只保留模型需要的 `map/conflict_regions` 与 `map/conflict_region_valid`，不再泛读 `map/` 和 `waymax/` 大数组。

## 检查结果

已执行：

```bash
pytest -q
python -m compileall -q cowp tests
```

结果：

```text
24 passed
```

并检查：

```bash
grep -R "F.binary_cross_entropy(" -n cowp tests --exclude-dir='__pycache__'
```

代码中已无概率空间 BCE 调用。

## 建议运行命令

你的原命令可以继续使用：

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage representation \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 8 \
  --amp \
  --compile \
  --fused-adamw \
  --output-dir outputs/checkpoints/representation
```
