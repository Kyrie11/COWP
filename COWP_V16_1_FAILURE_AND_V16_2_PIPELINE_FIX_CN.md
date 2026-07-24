# COWP v16.1 失败诊断与 v16.2 全流程工程修复

## 1. 本轮为什么没有实验结果

两个后台日志在同一个位置终止：启动阶段的 `pytest -q`。错误不是训练数据、DDP、
模型 anchor 或 Waymax，而是服务器 PyTorch 版本不支持：

```python
mode_mask.any(dim=(0, 1))
```

v16.1 已经修复了原先丢失 mode 维度的问题，但新的写法依赖较新的 PyTorch API。
服务器运行时只接受单个 `dim`，因此 5 个测试失败，脚本在 `set -e` 下立即退出。
训练尚未启动，没有 checkpoint、natural gate、planner、selector 或闭环指标。

修复为完全等价且兼容旧版 PyTorch 的写法：

```python
mode_mask.any(dim=0).any(dim=0)
```

## 2. 额外发现的全流程错误

`NEXT_RUN_COMMANDS_V16_1_FULL_CN.sh` 虽然名为 FULL，但没有设置 `RUN_FULL=1`。
即使前序阶段全部成功，默认也只会运行 probe，不会生成正式 full Waymax 结果。
v16.2 的 FULL 脚本已默认设置：

```bash
RUN_PROBE=1
RUN_FULL=1
REQUIRE_WAYMAX_PREFLIGHT=1
```

## 3. v16.2 的工程保证

- GPU 训练前生成 `eval/pipeline_preflight.json`，检查环境、配置、关键模块、CUDA、
  torchrun、Waymax 依赖以及真实 A=6/M=24 natural forward/loss/backward。
- 全部测试通过后才开始 cache gate、anchor、训练和评测。
- 并行任务使用 `wait_all`，不会因第一个子任务失败而遗留其他后台进程。
- full COWP 两个 shard 可单独续跑，已有合法 JSON 不重复计算。
- 流程结束前必须生成 `eval/pipeline_completion_report.json`。
- probe/full 报告必须同时包含 CR、offroad 与 EP/progress，缺任一项都不能标记完成。
- strict 模式不绕过算法质量 gate；另提供小规模 engineering smoke，只用于确认代码路径。

## 4. 建议运行顺序

### 4.1 先跑工程全链路 smoke

```bash
bash NEXT_RUN_COMMANDS_V16_2_ENGINEERING_SMOKE_CN.sh
```

该流程仅跑 1 epoch 和少量 Waymax 场景，允许质量 gate 失败后继续，以发现 transport、
planner、selector、Waymax 接口错误。输出不可用于论文。

### 4.2 再跑严格完整实验

```bash
bash NEXT_RUN_COMMANDS_V16_2_FULL_CN.sh
```

严格流程保持 natural、mechanism 等所有门禁。若工程预检和测试通过，但某个质量 gate
失败，应将其判定为模型/算法结果，而不是工程错误。

### 4.3 查看状态

```bash
bash CHECK_RUN_STATUS_V16_2.sh

tail -f outputs/cowp_v16_2_pipeline_v9labels_seed2026/logs/full_driver.nohup.log
```

## 5. 结果文件

严格 full 完成后至少应存在：

- `eval/pipeline_preflight.json`
- `checkpoints/natural/cowp_natural_best.pt`
- `eval/learned_offline/natural_basis_gate.json`
- `eval/learned_offline/natural_effectiveness_gate.json`
- `checkpoints/transport/cowp_witness_best.pt`
- `checkpoints/planner/cowp_planner_best.pt`
- `eval/learned_offline/mechanism_verification.json`
- `eval/probe/delta_conventional_vs_root_transport.json`
- `eval/waymax/delta_conventional_vs_cowp.json`
- `eval/waymax/delta_planner_vs_cowp.json`
- `eval/pipeline_completion_report.json`

其中闭环对比必须包含 CR、OffroadRate/Offroad 和 EP。

## 6. 本轮没有修改算法

v16.2 没有改变 decoder、loss、OBS capacity、planner、selector 或标签定义。下一轮完整
结果仍可用于验证 v16 算法，不会混入新的算法变量。
