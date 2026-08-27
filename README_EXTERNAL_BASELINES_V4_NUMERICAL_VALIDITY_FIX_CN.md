# 外部 Baseline V4 数值稳定性与有效性修复（2026-08-27）

本补丁不改变 `RUN_5_SOTA_BASELINES_COWP.sh` 的调用方式，也不改变 `outputs/external_sota5_v16_8_33/<baseline>/external_<baseline>_best.pt` 等 checkpoint 路径。

## 根因与修复

1. **GameFormer 地图 mask 错误**：旧实现以局部坐标 `x==0` 判断 padding，并用“段内任一点为 padding 则整段 padding”的归约；ego 坐标系中真实车道经常穿过 `x=0`，会误屏蔽有效地图段。V4 从 WOMD `roadgraph_samples/valid` 显式传播 point-valid mask，并在 max pooling 前屏蔽无效点。
2. **DTPP/PLUTO/PlanT2 同类问题**：DTPP V3 已从 `x==0` 改为 all-zero point，但仍无法区分“真实原点”和 padding；PLUTO/PlanT2 仍以 `xy!=0` 推断地图有效性。V4 四个 learned baseline 统一使用源有效性。
3. **混合精度策略不一致**：旧启动脚本默认给 GameFormer/PLUTO/PlanT2 强制 BF16，仅 DTPP 默认 FP32。V4 四者默认全部 FP32；如确需做 AMP ablation，可设置 `GAMEFORMER_AMP=1`、`DTPP_AMP=1`、`PLUTO_AMP=1` 或 `PLANT2_AMP=1`，dtype 由 `EXTERNAL_AMP_DTYPE` 控制。
4. **错误被 epoch 末 2% 阈值遮蔽**：旧训练循环允许先跳过非有限 loss/gradient，最后才以 `max_skip_fraction` 报错。V4 默认 `max_numerical_skip_fraction=0`，第一次数值异常立即失败并保留 baseline/epoch/batch/parameter 或 gradient 诊断；数据/监督跳过仍保留总比例保护。
5. **单个坏 proposal 污染整 batch**：V4 将 NaN/Inf candidate 逐 proposal 判无效并清零其数值载荷；future 的非有限时间点只撤销对应 supervision validity。
6. **闭环/离线测试防护**：learned/rule baseline 的候选选择都加入 trajectory finiteness；checkpoint 加载拒绝非有限权重；Waymax action 不再用 `nan_to_num` 静默隐藏坏动作。
7. **旧 checkpoint 自动失效**：completion marker 新增 `contract_version=v4_explicit_validity_fp32_20260827`。默认 `SKIP_COMPLETED=1` 只复用满足新 contract 和当前 AMP/DTPP-cost 配置的模型；旧 V2/V3 会在原路径上重新训练并覆盖。

## 原命令保持不变

```bash
GPU0=0 GPU1=1 nohup bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all > logs/run.log 2>&1 &
```

```bash
PARALLEL2=1 \
GPU0=0 GPU1=1 \
WOMD_VALIDATION_TFEXAMPLE_DIR=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation \
bash RUN_5_SOTA_BASELINES_COWP.sh waymax all
```

建议先完整重跑 `train_parallel2 all`，再跑 Waymax；旧完成标记不会被误复用。
