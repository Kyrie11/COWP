# COWP Stage-B `pin_memory` 报错与训练命令修复说明

## 1. 报错现象

Stage B response 训练在第 4 个 batch 后，在 DataLoader 的 pin-memory 线程中失败：

```text
RuntimeError: Caught RuntimeError in pin memory thread for device 0
RuntimeError: CUDA error: invalid argument
```

报错发生在下一批数据搬运阶段，而不是 `response_loss()` 内部直接抛出维度错误或 label key 错误。

## 2. 根本原因判断

这不是论文算法设计错误，也不是某个 response label 的语义错误。根本原因是 Stage B 的 dense response supervision 与 DataLoader pinned-memory 组合不稳定：

- Stage B 会加载 `cowp/response/traj`，其典型形状是 `[K, A, R, T, 7]`。
- 当 `batch_size=64`、`num_workers=8`、`prefetch_factor=4` 且 `pin_memory=True/auto` 时，DataLoader 可能同时预取并尝试 page-lock 多个巨大的 batch。
- PyTorch/CUDA 的 pin-memory thread 在这种内存压力下可能报 `CUDA error: invalid argument`。
- pin-memory 只影响 CPU->GPU 数据搬运方式，不影响模型结构、监督信号、loss 定义或预测结果。

## 3. 本次代码修复

### 3.1 训练脚本默认关闭 pin_memory

文件：`cowp/scripts/03_train.py`

- 将 `pin_memory` 默认策略改为关闭。
- 只有用户显式传入 `--pin-memory` 时才尝试开启，并做 sanity check。
- `_to_device(..., non_blocking=...)` 现在只在 pin-memory 实际开启时使用 non-blocking transfer。

这不会改变学习目标，只避免 pinned-memory thread 在 Stage B 中崩溃。

### 3.2 增加 `--prefetch-factor` 参数，并对重标签阶段设置安全默认值

文件：`cowp/scripts/03_train.py`

- 新增 `--prefetch-factor`。
- 对 `response / planner / all` 这类可能包含大 dense label 的阶段，默认使用 `prefetch_factor=1`。
- 用户显式设置大于 1 时会给出警告。

### 3.3 优化 `response_loss()` 内存计算

文件：`cowp/models/losses.py`

- 原代码先对所有 `[B,K,A,R,T,7]` response trajectory 计算 L1，再用 mask 过滤。
- 新代码先选出有效 response slot，再计算 trajectory L1。
- 数学目标不变，仍然只监督 valid response；但减少 padded slot 上的无效计算和中间张量分配。

### 3.4 顺手修复 Stage C 可能遇到的 witness token 脏标签风险

文件：`cowp/models/losses.py`

- `witness_loss()` 中的 token target 在 `.long()` 前增加 `nan_to_num()`。
- 避免未来 Stage C 遇到 NaN/Inf token 时触发 CUDA index/cross-entropy 异常。

### 3.5 增加 `--compile-backend`

文件：`cowp/scripts/03_train.py`

- 支持 `--compile-backend aot_eager`。
- 如果某些机器的 Inductor/Triton 仍不稳定，可以保留 `--compile` 入口但使用更稳的 backend。

## 4. 对其他 train 命令的影响判断

- Stage A representation：不加载 response dense trajectory，之前已成功训练；本次默认关闭 pin-memory 后更稳，不改变训练目标。
- Stage B response：本次报错的核心阶段，已从 pin-memory 和 response loss 两侧修复。
- Stage C witness：不会加载 `cowp/response/traj`，但 witness token target 已做 NaN/Inf 防护。
- Stage D planner：可能加载 `waymax/*` 与 candidate/witness 标签，默认 `prefetch_factor=1`，pin-memory 默认关闭，避免同类 DataLoader 搬运问题。
- Stage all：会加载所有大标签，属于最容易复现内存压力的训练方式；现在默认 `pin_memory=False + prefetch_factor=1`。

## 5. 本地测试

```bash
pytest -q
# 32 passed
```
