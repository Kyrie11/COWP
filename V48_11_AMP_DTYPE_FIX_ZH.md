# v48.11 CASTER AMP dtype 修复

## 故障

在 `training.amp=true` 且 `training.amp_dtype=bfloat16` 时，`RecoverySetTournament.forward()` 中：

- `scores` 由 `relative_features.new_zeros(...)` 创建，dtype 跟随 `relative_features`，在实际 v48.11 `candidate_concat_raw` 路径上可能为 `torch.float32`；
- `input_proj`、`MultiheadAttention`、FFN 和 `score` 在线性层 autocast 路径上输出 `torch.bfloat16`；
- `scores[recs] = group_scores` 属于 advanced-index assignment，PyTorch 不允许源和目标 dtype 不一致。

因此抛出：

```text
RuntimeError: Index put requires the source and destination dtypes match,
got Float for the destination and BFloat16 for the source.
```

balanced 与 precision 都调用相同的 v48.11 Stage-T 集合排序器和相同的 bfloat16 AMP 配置，所以两条训练日志都会在首次触发该路径时失败。

## 修复

只在组内分数写回全局向量的 scatter 边界执行：

```python
scores[recs] = group_scores.to(dtype=scores.dtype)
```

该修改：

- 不关闭 AMP；
- 不改变 CASTER 的 attention、居中、排序、nominal pinning、Stage-T/Stage-E 或测试选择逻辑；
- 保持返回张量 dtype 与 `relative_features` 一致；
- dtype 转换仍在 autograd 图中，梯度可以正常回传。

## 验证

已执行：

1. CPU bfloat16 autocast 精确复现原始异常；
2. 修复后混合精度前向与反向通过；
3. nominal 分数仍严格为 0；
4. set-tournament 排列等变性测试通过；
5. v48.11 ordered evidence/simplex 测试通过；
6. 全仓库测试：`148 passed`。

## 替换位置

文件：`src/ocrap/models/ocrap.py`

原代码：

```python
scores[recs] = group_scores
```

替换为压缩版本：

```python
scores[recs] = group_scores.to(dtype=scores.dtype)
```

仓库中的修复版本附带了说明注释和一个 AMP 回归测试。
