# COWP v16.3 natural 首步 Non-finite gradient 修复说明

## 现象

`eval-before-train` 的 epoch -1 前向与 loss 均为有限值，但 natural epoch 0 的第 1 个训练 batch 在 backward 后触发：

```text
FloatingPointError: Non-finite gradient norm at stage=natural epoch=0 step=1
```

## 根因

CNOB natural decoder 的 residual/control head 为零初始化，首步会产生大量精确零速度状态。原代码在以下两条反向路径中直接计算 `atan2(0, 0)`：

1. `cowp/models/natural_decoder.py`：由总速度和基准速度计算航向；
2. `cowp/models/losses.py`：natural kinematic yaw consistency 计算速度航向。

部分旧版 PyTorch + CUDA 组合中，`atan2(0, 0)` 前向值可以是有限的 0，但 backward 的偏导分母为 0，可能产生 NaN。速度 mask 原先在 `atan2` 之后应用，无法可靠阻止旧 CUDA backward 中的 NaN 传播。因此验证前向正常，而首个训练反向失败。

## 修复

- 在进入 `atan2` 前识别有限的近零速度向量，并替换为常量方向 `(x=1, y=0)`；
- 非有限输入不被隐藏，仍会由现有 model-output safety check 捕获；
- natural decoder 的运动学结构、loss 权重、训练阶段、学习率、优化器、输出目录均未改变；
- 梯度裁剪增加数值稳定 fallback：若仅 FP32 范数累加溢出而每个梯度元素均有限，则使用 FP64 计算同一全局 L2 norm 并执行相同 clipping；若梯度元素本身 NaN/Inf，则输出具体参数路径并终止；
- 恢复脚本保持原 `OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026`。由于代码修复会改变 strict provenance hash，脚本仅在 natural history 尚未生成时备份旧失败运行的 provenance manifest，再在同一目录写入新签名。

## 修改文件

- `cowp/models/natural_decoder.py`
- `cowp/models/losses.py`
- `cowp/scripts/03_train.py`
- `NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh`
- `tests/test_v16_3_numeric_safety.py`

## 验证

```text
110 passed
```

新增回归覆盖：

- exact zero-velocity CNOB backward 梯度有限；
- 超大但有限的 FP32 梯度不会因 norm reduction overflow 被误报；
- 恢复脚本保持原输出目录并只轮换失败运行的旧 provenance。

## 运行

仍使用原命令：

```bash
BACKGROUND=1 bash NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh
```
