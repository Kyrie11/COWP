# COWP Stage-A `cowp/natural/traj` KeyError 修复说明

## 1. 这次报错的直接原因

报错路径是：

```text
COWPModel._agent_history_from_batch(...)
  elif has_womd_state(batch): ...
  elif "state/all" in batch: ...
  else:
      nat = batch["cowp/natural/traj"].float()
KeyError: 'cowp/natural/traj'
```

这说明进入模型时，当前 batch 同时缺少：

- `state/history`
- `state/past/x + state/current/x`
- `state/all`
- `cowp/natural/traj`

训练到 44% 后才出现，说明不是所有数据都错，也不是论文算法目标本身错；更可能是 tensor cache 中混入了少量 partial/旧格式/中断生成的 `.npz`，或者不同 cache 文件使用了不同 state key 命名。

原代码还有一个放大问题：`collate_torch()` 使用所有样本 key 的交集。只要一个样本缺少 `cowp/natural/traj` 或 state 输入，整批 batch 的该 key 就会被删除，最后模型 fallback 到 `batch["cowp/natural/traj"]` 时崩溃。

## 2. 本次修改

### 2.1 统一 state key 命名

文件：`cowp/data/dataset.py`

新增 `_canonicalize_state_aliases()`：

- 将 `womd/state/...` 统一规范化为 `state/...`；
- 避免同一 batch 中有些样本是 `womd/state/history`，有些是 `state/history`，导致交集后状态输入消失；
- 删除重复 alias，避免 CPU/GPU batch 内存重复。

### 2.2 Stage 级别 required-key 检查

文件：`cowp/data/dataset.py`

新增 `_missing_required_for_stage()`：

- representation / natural 阶段必须有 encoder state、critical labels、natural labels；
- response 阶段必须有 candidate 和 response labels；
- witness / planner 阶段必须有 candidate 和 witness / planner labels。

`TorchCOWPDataset` 现在会跳过 partial/invalid cache item，避免单个坏样本污染整批训练。

### 2.3 保留按 stage 限定读取，继续减少 I/O

文件：`cowp/data/dataset.py`

`COWPNpzDataset` 新增 `load(idx, wanted)`，只读取当前 stage 需要的数组。这样不会退回到全量读取 `.npz`，仍然保留上一轮为提速做的按 stage 读取优化。

### 2.4 模型输入路径更鲁棒

文件：`cowp/models/cowp_model.py`

`_agent_history_from_batch()` 现在支持：

- `state/history`
- `womd/state/history`
- `state/past/* + state/current/*`
- `womd/state/past/* + womd/state/current/*`
- `state/all`
- `womd/state/all`

如果真的缺少 state，会给出明确错误信息，而不是静默进入错误 fallback。

### 2.5 trajectory head 改为 agent-centric residual 输出

文件：`cowp/models/cowp_model.py`

论文中的 natural alternatives / safe responses 是 critical agent 的未来轨迹。label 中保存的是绝对坐标 `[x,y,heading,vx,vy,length,width]`。原模型用线性 head 直接预测 WOMD 全局绝对坐标，初始输出接近 0，会导致 Stage-A 初始 loss 很大，也会造成训练不稳定。

本次改为：

- decoder head 预测相对当前 critical agent 状态的 residual；
- 在 `COWPModel` 中加回当前 critical agent anchor；
- loss 仍然对绝对轨迹监督，因此不改变论文算法目标，只改变更合理的坐标参数化。

## 3. 建议训练命令

你原命令可以继续使用：

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
  --prefetch-factor 4 \
  --amp \
  --compile \
  --fused-adamw \
  --output-dir outputs/checkpoints/representation
```

如果本机 PyTorch/Triton 的 Inductor 仍然不稳定，改用：

```bash
--compile --compile-backend aot_eager
```

## 4. 测试结果

当前代码包中运行：

```bash
pytest -q
```

结果：

```text
32 passed
```

新增测试覆盖：

- partial/坏 cache sample 不再导致 Stage-A 中途 KeyError；
- `womd/state/*` 会规范化为 `state/*`；
- natural trajectory 输出会锚定到 critical agent 当前状态附近，而不是全局原点附近。
