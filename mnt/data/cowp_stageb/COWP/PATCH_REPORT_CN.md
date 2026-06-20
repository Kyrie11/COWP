# COWP 代码审查与修复说明

本次修改针对三类问题：

1. **`--compile` 报错修复**
   - 报错本质是 PyTorch Inductor/Triton 在编译 `GraphEncoder._history_mean()` 中的布尔/finite 检查相关 kernel 时发生类型不一致：`int1` loop-carried variable 被重赋值为 `int8`。
   - 已将历史状态汇聚函数改为 `torch._dynamo.disable` eager 子图，避免该小预处理函数进入 Inductor，同时保留模型其他部分的 `torch.compile` 能力。
   - 去掉 `_history_mean()` 中的 in-place clamp 和 `nan_to_num` 编译敏感组合，改成显式 `isfinite + where` 清洗。
   - 新增 `--compile-backend {inductor,aot_eager,eager}` 与 `--compile-mode`，如果某些环境下 Inductor 仍报错，可直接使用 `--compile-backend aot_eager` 稳定运行。

2. **训练速度与 GPU 利用率优化**
   - 原 `TorchCOWPDataset` 虽然有 stage-specific key 过滤，但底层 `COWPNpzDataset.__getitem__()` 会先把 `.npz` 中所有数组全部读取进内存，再过滤。对于 WOMD tensor cache，这会把大量当前 stage 不使用的 future/response/witness/map 辅助大数组读进来，是低 GPU 利用率和训练慢的主要原因之一。
   - 已改为在 `np.load` 阶段只 materialize 当前 stage 需要的 key。
   - 原 stage filter 使用宽泛 `state/`、`womd/state/` 前缀，会读入不用的 WOMD future tensors。已改为只读取模型 encoder 真实需要的 past/current state、id、is_sdc、agent_valid、state/history/state/all。
   - `natural_loss` 与 `witness_loss` 去除了多处 Python `if tensor.any()` 的 CUDA 同步点，减少 batch 内隐式 GPU-CPU 同步。
   - CUDA runtime 默认启用 TF32；DataLoader 增加 `--prefetch-factor` 参数。

3. **论文算法一致性修复**
   - `adaptive_beta()` 中原代码错误地用 `scene_states[:, 5]` 当速度，但 d_state 定义是 `[x,y,z,length,width,height,heading,vx,vy,speed,valid]`，第 5 列是 height。已修复为优先使用第 9 列 speed，必要时回退到 vx/vy 范数。
   - `GraphEncoder` 中原本存在 `if False` 导致 agent-conflict evidence 不进入 conflict query 的死代码。已改为用 agent-conflict 聚合初始化 query message；若有 candidate-conflict 分支则继续覆盖为 candidate-conditioned evidence。

本地验证：

```bash
cd /mnt/data/cowp_patched
python -m pytest -q
# 30 passed
```

推荐训练命令：

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

如果你的 PyTorch/Triton 环境继续触发 Inductor 后端错误，使用稳定后端：

```bash
  --compile --compile-backend aot_eager
```

如果主要瓶颈仍是 I/O，进一步建议：

- 确认 `02_build_tensor_cache.py` 生成 cache 时使用默认 `compress=False`，不要对训练 cache 使用压缩 npz。
- 将 `tensor_cache_train` 放在本机 NVMe/SSD，而不是网络盘。
- 用 `--num-workers 8 --prefetch-factor 4` 或 `--num-workers 12 --prefetch-factor 4` 测试吞吐。
- A30 24GB 下 representation stage 可以继续尝试 `--batch-size 192/256`；该模型本身不大，显存 3GB 左右是正常信号，真正瓶颈多半是 NPZ I/O 与 CPU preprocessing。
