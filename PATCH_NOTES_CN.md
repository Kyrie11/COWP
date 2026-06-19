# COWP 优化与修复说明

本版本基于当前上传的论文 tex、COWP.zip 代码与报错文本进行修复。主要解决三类问题：

## 1. representation 阶段 CUDA device-side assert

报错栈显示在 `losses.py::_branch_minade` 的 `branch_gt.any()` 处触发，但这是 CUDA 异步报错的常见表现，真正的问题通常发生在更早的 CUDA kernel。当前 representation 阶段主要执行 `natural_loss`，风险点包括：

- `F.binary_cross_entropy(pred_p, tgt)` 的 target 若存在 NaN、inf、非二值或不在 `[0,1]`，CUDA 会触发 device-side assert。
- `scatter_add_` / `cross_entropy` 相关标签若因为缓存脏值、padding、dtype 转换导致越界，也可能触发 device-side assert。
- 原 `masked_mean` 使用 `value * mask`，如果 padding 区域存在 NaN，`NaN * 0` 仍会传播为 NaN，可能进一步污染 loss。

修复点：

- 对 BCE target 统一做 finite 化与 `[0,1]` clamp。
- 对 source/token 等离散标签做 finite 化、long 转换与合法范围 clamp。
- 对 masked reduction 改为 `torch.where(mask, value, 0)`，避免 masked NaN 传播。
- 对 trajectory、priority、weights 等输入做必要的 `nan_to_num`。
- 对 natural 分支 loss 复用一次 pairwise ADE，避免重复计算。

## 2. 训练速度慢与 GPU 显存占用低

主要瓶颈不在 GPU 显存，而在 CPU/I/O 与重复 loss 计算：

- 原 `TorchCOWPDataset.__getitem__` 先通过 `COWPNpzDataset.__getitem__` 读取 `.npz` 的全部数组，然后再按 stage 过滤。representation/natural 阶段其实不需要 candidate、response、witness、Waymax 等大量字段，但仍会被从磁盘读取，导致训练 I/O-bound，GPU 空等。
- `natural_loss` 中 pairwise ADE 被多次重复计算，包括 set-minADE、mixture NLL、OBS/NEUTRAL/PRIORITY 分支 ADE。

修复点：

- 新增 stage-aware lazy npz loading：只读取当前训练阶段需要的 key。
- representation/natural 阶段不再默认读取 `waymax/` 字段。
- natural loss 内部 pairwise ADE 只计算一次并复用。
- `03_train.py` 新增 `--compile` 与 `--fused-adamw` 可选加速参数。

## 3. 新增鲁棒性测试

新增 `tests/test_training_robustness_optimized.py`：

- 覆盖非法 source、priority NaN/越界时 natural loss 不再崩溃。
- 覆盖 witness token 越界时不再触发 CE index 错误。
- 覆盖 representation stage dataset 不再加载 heavy response array。

本地验证：

```bash
pytest -q
# 24 passed

python -m compileall -q cowp tests
# passed
```

## 推荐训练命令

可先用以下命令验证修复：

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

说明：

- 首次 `--compile` 会有编译开销；如果只是快速 debug，可以先去掉 `--compile`。
- A30 显存较大，修复 I/O 后可尝试把 batch-size 从 128 提到 256 或 384。
- 如果 CPU/磁盘仍成为瓶颈，可进一步将 tensor cache 转为 LMDB/WebDataset/sharded memmap，但这属于数据格式层面的进一步工程优化。
