# 保留的旧版 V4 实验说明（已适配为 0.9.16 的测试副本）

**请优先使用上级目录 `../README_CN.md` 和 `../scripts/`。本文件属于原 COWP 的历史 V4 操作记录，部分旧的输出根目录、命令名与数据接口可能不适用于新 RIDE 验证流程。**

# V4 代码执行说明（先读）

## 0. 服务器安装与环境隔离

请**不要在运行中的 CARLA/world 上使用其他 tick 客户端**。本代码控制同步帧、每条实验重启 CARLA 地图、销毁自己的 actors 并恢复异步模式。按 CARLA 官方同步模式要求，不得并行启动多个 Traffic Manager 同步服务。

Waymax 测试使用你的 WOMD TFExample 原始 1.3.1 记录，仍依赖服务器已安装的 TensorFlow/JAX/Waymax。

CARLA 0.9.16 Python 解释器需要能导入 `carla` 和 PythonAPI/agents。假设 `$CARLA_ROOT` 指向 CARLA 0.9.16 根目录：

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
```

在另一个终端（Python 环境必须与 0.9.16 包兼容）：

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH"
# CARLA 的 PythonAPI/carla/dist 下若是 .egg，需加入与你的 Python 版本匹配的 egg 路径；
# 或按照 CARLA 0.9.16 官方说明 pip install 对应 .whl。
python -c 'import carla; from agents.navigation.behavior_agent import BehaviorAgent; print("ok")'
```

## 1. 先修 Waymax（避免重复浪费 22 场景）

```bash
export RAW_WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export EICDRR_FULL_ROOT=/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh tests
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh waymax_inspect
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh waymax_smoke
```

检查 `$EICDRR_FULL_ROOT/h2_gate_v4/waymax_smoke_1.json`，要求 complete=1、errors=0、两种 ego 目标速度分支成功运行。只有 smoke 通过才运行：

```bash
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh waymax_priority22
```

如仍报告 reset 位置 mismatch，请上传错误和 raw scenario_id / Waymax init_steps/timestep 诊断。**不要用提升位置阈值或改 NPZ 坐标来“修复”。**

## 2. CARLA 快速验证

```bash
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh carla_preflight
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh carla_smoke
```

`carla_smoke.jsonl` 必须有 `type=manifest` 和至少 6 条 `type=trial`，不能全部 `simulation_exception`。需要存在有效的 start state、lane validity、实际 ego speed / displacement，`carla_smoke_analysis.json` 中有效 contrasts 不为零。没有交互碰撞是允许的，不要偷偷增大速度来强迫正例。

如果 Town04 无法找到直行长车道，可以试：

```bash
CARLA_TOWN=Town06 bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh carla_smoke
```

## 3. 开发集机制实验（成功后才能继续）

```bash
CARLA_ANCHORS=3 CARLA_REPEATS=3 bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh carla_mechanism_dev
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh carla_joint_h1_pilot
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh carla_h3_backup_pilot
```

这三个会产生 JSONL 原始轨迹与汇总 JSON。请完整上传 JSONL 和汇总，包括失败试验。预先定义 good/bad 的反事实对和任务进展阈值；CARLA joint witness/H3 子实验属于开发期筛查。

### 什么算有效结果

- `strict_third_party_collision_preventable`: 明确两个非 ego 车辆在 bad 分支相撞、good 分支不相撞，ego 两分支无碰撞，地图有效且进展匹配。
- `cross_family_same_action_strict_cases`: 两套 CARLA 反应策略在相同 anchor/seed/bad/good 且初始状态匹配时都有严格反例。CARLA 模拟的证据，不是真实驾驶者因果真值。
- `bounded_joint_safe_witnesses`: 对相同 ego bad action 找到有限候选库中的实际物理合法、安全、进展匹配响应。为正向存在性证据；零不是不可行证明。
- `recovery_prefix_parity_pass`: 恢复和未恢复分支在干预切换前的物理状态相同。后缀安全只说明此 backup 候选可行，不能证明所有策略不再可恢复。

## 4. 不包含的能力（避免夸大）

CARLA 本模块不自动把 WOMD roadgraph 转成 CARLA/OpenDRIVE，不会假称同一 WOMD 场景在两个平台完全复现。也没有真的运行 GameFormer、RA-CBF、人的反事实实验。后续 H4 仍需同一候选 action 接口、可训练权重/官方实现和完整 baseline 结果。
