# External baselines V5：DTPP / PyTorch tuple-dim reduction 兼容修复

## 现象

DTPP 在第一个训练 batch、进入模型 forward 之前报错：

```text
TypeError: all() received an invalid combination of arguments - got (dim=tuple, )
```

触发点原为：

```python
cand_valid = cand_valid & torch.isfinite(cand).all(dim=(-1, -2))
```

## 根因

当前 `mfrp` 运行环境中的 `torch.Tensor.all` API 只接受单个整数维度，不接受 tuple 维度；因此这属于运行时 PyTorch API 兼容问题，而不是 DTPP loss、梯度或数据集本身首先发生 NaN。

GameFormer 没有在训练中触发，是因为训练入口仅对 DTPP 设置 `require_candidates=True`。GameFormer、PLUTO、PlanT2 的 direct training 不构造 COWP candidate bank。

## 修复

1. 新增 `candidate_geometry_finite()`：先将候选轨迹最后两个维度展平，再执行单维 `all(dim=-1)`，语义仍为“该 candidate 的全部 T x D 元素均 finite”。
2. DTPP adapter 使用该 helper，单个 NaN/Inf candidate 只会被置为 invalid，数值载荷随后 `nan_to_num`，不会污染整个 batch。
3. learned offline eval 复用同一个 helper，因此 GameFormer / DTPP / PLUTO / PlanT2 后续离线测试不会再在相同 API 上失败。
4. 同时移除了 external-baseline 运行链路中的其他多维 tuple reduction：
   - DTPP fallback candidate mask：连续两次 `sum(dim=-1)`；
   - logged-ego candidate matching denominator：连续两次 `sum(dim=-1)`；
   - PlanT2 hazard target：连续两次 `amin(dim=-1)`。
5. NumPy 的 `all(axis=(...))` 保持不变；它与本次 PyTorch API 错误无关。

## 训练与 checkpoint

- `RUN_5_SOTA_BASELINES_COWP.sh` 无需修改用法。
- checkpoint 输出目录和文件名完全不变。
- 本次错误发生在 DTPP 第一个 batch 的 adapter 阶段、optimizer step 之前，因此这一次失败本身不会产生“学坏了的 DTPP 参数”。
- V4 training contract 保持不变；这是运行时兼容修复，不改变训练目标、模型结构、数据语义或 checkpoint 格式。
- 如果并行的 GameFormer 已完整训练并写入 completion marker，下一次 `SKIP_COMPLETED=1` 会自动复用它；否则会重新训练。

## 验证

针对 external baseline / DTPP / baseline integrity / Waymax diagnostics / Waymax rollout / WOMD tensor adapter 的测试：

```text
29 passed
```

另外加入了回归测试，显式验证：

- `[B,K,T,D]` candidate finite mask 输出 `[B,K]`；
- 一个 candidate 含 NaN/Inf 时只 invalid 该 branch；
- external baseline runtime 源码不再使用 `.all(dim=(...))` / `.any(dim=(...))`。

## 原命令

仍可直接运行：

```bash
GPU0=0 GPU1=1 nohup bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all > logs/run.log 2>&1 &
```

建议额外记录一次真实环境版本便于归档：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.__file__)
PY
```

仓库 `requirements.txt` 写的是 `torch>=2.1`，而这次 traceback 显示实际运行环境的 `Tensor.all` 行为与当前 PyTorch 2.x API 不一致；V5 已不依赖该 tuple-dim 行为，因此不要求为了这一个问题升级 torch。
