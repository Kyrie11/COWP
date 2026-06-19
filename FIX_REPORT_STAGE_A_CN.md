# Stage A representation 训练修复说明

## 本次日志中的直接错误

`train_stage_a.log` 中的错误发生在第一轮第一个 batch 的 forward 阶段，尚未进入 loss：

- `torch._dynamo.exc.BackendCompilerFailed`
- `backend='inductor'`
- Triton 编译报错：`Loop-carried variable _tmp2 has initial type int1 but is re-assigned to int8`
- 堆栈定位到 `cowp/models/graph_encoder.py::_history_mean`

根因是 `--compile` 下 PyTorch Inductor 对布尔归约/张量条件分支的编译 bug。原实现里有 `if empty.any():`，这是数据依赖的 Python 侧张量控制流，在 PyTorch 2.1/2.2 + Triton 上容易触发该错误。

## 已修复

1. `cowp/models/graph_encoder.py`
   - 将 `_history_mean()` 改成无 Python 侧张量分支的写法。
   - 使用 always-on `torch.where()` 处理空历史 agent。
   - 对 history 输入做 `nan_to_num()`，避免 NaN 进入 encoder。

2. `cowp/scripts/03_train.py`
   - `torch.compile` 改为 failure-tolerant：设置 `torch._dynamo.config.suppress_errors=True`，并使用 `mode='reduce-overhead'`。
   - 修复 `torch.compile` 后 checkpoint 保存 `_orig_mod.` 前缀的问题。
   - 增加 `--force-positive-oversampling`。
   - representation/natural 阶段默认跳过 witness/candidate positive oversampling，避免启动时扫描所有 `.npz` 的 witness/candidate 标签。

3. `cowp/data/dataset.py`
   - stage-aware loading 不再用宽泛的 `map/` 前缀，只加载模型真正使用的：
     - `map/conflict_regions`
     - `map/conflict_region_valid`
   - 避免无意加载大 map 辅助数组。

4. 新增测试
   - `tests/test_compile_and_stage_a_fixes.py`
   - 覆盖 `_history_mean()` 空历史行为、Stage A map 精确加载、compiled checkpoint 前缀兼容。

## 建议重新运行

原命令可以继续使用：

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

如果你的 PyTorch 版本仍然对某些子图编译不稳定，代码会自动 fallback 到 eager 子图而不是直接中断训练。

## 本地验证

已执行：

```bash
pytest -q
python -m compileall -q cowp tests
```

结果：`27 passed`，无语法错误。
