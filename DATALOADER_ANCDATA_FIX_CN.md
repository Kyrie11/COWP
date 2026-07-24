# COWP v16.2 DataLoader `ancdata` 稳定性修复说明

## 1. 故障结论

日志中的故障发生在 `eval-before-train` 的 natural 验证阶段：

```text
RuntimeError: received 0 items of ancdata
```

调用栈位于 `torch.utils.data.DataLoader` 主进程反序列化 worker 返回批次时，而不是模型前向、natural loss、反向传播或 DDP collective 内部。

该项目每个样本返回包含大量独立 Tensor 的字典；原启动参数又采用：

- 2 个 DDP rank；
- 每个 rank 8 个 natural DataLoader worker；
- 每个 worker 预取 2 个 batch；
- 训练与验证均启用 persistent workers；
- Linux 默认 `file_descriptor` Tensor 共享策略。

`file_descriptor` 会为共享 CPU storage 传递并缓存文件描述符。多 rank、多 worker、多 Tensor、深预取叠加后，进程接近文件描述符或 IPC 资源边界，接收端可能得到缺失的 ancillary data，最终抛出该错误。

## 2. 修复原则

本次修改只调整数据加载和进程间 Tensor 传输方式，不改变：

- 数据集样本内容与排序规则；
- DDP sampler 逻辑；
- 模型结构与前向计算；
- natural / witness / planner 损失；
- 优化器、学习率、AMP、梯度裁剪；
- checkpoint 评分和早停逻辑。

## 3. 核心修改

### 3.1 新增统一 DataLoader IPC 运行时

新增：

```text
cowp/utils/dataloader_runtime.py
```

默认在 Linux 上使用 `file_system` sharing strategy，避免逐 Tensor 文件描述符传递导致的 `ancdata` 故障。同时记录：

- 当前 sharing strategy；
- 打开文件描述符数量；
- `RLIMIT_NOFILE`；
- `/dev/shm` 可用空间。

可通过以下方式覆盖：

```bash
--sharing-strategy file_descriptor
```

或：

```bash
export COWP_TORCH_SHARING_STRATEGY=file_descriptor
```

### 3.2 训练和验证 DataLoader 解耦

`cowp/scripts/03_train.py` 新增：

```text
--val-num-workers
--val-prefetch-factor
--sharing-strategy
--no-persistent-workers
--persistent-val-workers
```

验证 worker 默认不持久化，以便每次验证结束后释放 worker、队列、共享 storage 和 IPC 资源。

### 3.3 v16.2 启动脚本使用稳健验证配置

`run_cowp_v16_2_dual_gpu.sh` 默认配置：

```text
TORCH_SHARING_STRATEGY=file_system
natural validation: 2 workers, prefetch 1
transport validation: 2 workers, prefetch 1
planner validation: 2 workers, prefetch 1
```

natural 训练仍保留 8 workers / prefetch 2，不改变原训练吞吐策略；只收缩验证阶段资源峰值。

### 3.4 同类入口统一修复

以下直接创建 PyTorch DataLoader 的入口均接入统一 sharing strategy：

- `cowp/scripts/03_train.py`
- `cowp/scripts/20_train_external_baseline.py`
- `cowp/scripts/21_eval_external_baseline.py`
- `cowp/scripts/22_eval_rule_baseline.py`
- `cowp/scripts/35_diagnose_model_anchor.py`
- `cowp/scripts/39_diagnose_learned_natural.py`
- `cowp/waymax_eval/rollout.py`
- `scripts/03_train.py`（旧版直接入口）

## 4. 重新运行

直接执行原命令：

```bash
bash NEXT_RUN_COMMANDS_V16_2_ENGINEERING_SMOKE_CN.sh
```

为避免原失败目录中的 provenance 与新代码签名冲突，smoke 默认输出目录已调整为：

```text
outputs/cowp_v16_2_engineering_smoke_v9labels_seed2026_ancdatafix
```

查看日志：

```bash
tail -f outputs/cowp_v16_2_engineering_smoke_v9labels_seed2026_ancdatafix/logs/smoke_driver.nohup.log
```

启动日志应出现类似内容：

```text
sharing_strategy=file_system
train_workers=8
val_workers=2
val_persistent=False
DataLoader IPC runtime: {...}
```

## 5. 可选保守运行参数

服务器仍存在系统级资源限制时，可临时使用：

```bash
NATURAL_NUM_WORKERS=4 \
NATURAL_VAL_NUM_WORKERS=0 \
NATURAL_PREFETCH_FACTOR=1 \
bash NEXT_RUN_COMMANDS_V16_2_ENGINEERING_SMOKE_CN.sh
```

`val-num-workers=0` 是最保守的同步数据加载模式，只会降低验证读取速度，不改变验证指标。

## 6. 验证结果

已完成：

- 全项目 Python 编译检查；
- 所有 shell 脚本 `bash -n` 语法检查；
- 新增 DataLoader sharing strategy 与验证 worker 生命周期测试；
- 全量测试：`103 passed`。

当前环境没有用户服务器上的 WOMD cache、双 GPU 与对应 CUDA 环境，因此无法在本地完整复现 502 个 validation batch 的真实 smoke 流程；代码级故障路径和所有直接 DataLoader 入口已经覆盖。
