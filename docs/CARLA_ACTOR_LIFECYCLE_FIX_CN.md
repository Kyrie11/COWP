# RIDE — CARLA actor 生命周期、伪初始碰撞及配对实验审计 (2026-10-27)

## 上传结果的直接证据

- `convoy_actors_smoke01/carla/convoy.jsonl`：1 个 BehaviorAgent trial，`trace=[]`、三车 `initial_states.xyz=[0,0,0]`、`initial_overlap=True`、`reason=overlap_after_warmup`，无 `metrics.valid=True`。虽然有 `run_end.completed=true`，它不是完成了有效物理仿真的证明。
- 冻结 Town04 的实际几何：ego (-511.623,219.836), A (-511.026,199.767), B (-510.758,182.064)。与读出的全零位置不一致；初始重叠应判为未初始化读数污染，不能作为 RQ1/RQ2 的发现。
- `convoy_tm_smoke01/carla/convoy.jsonl`：只有 manifest + `run_error`，没有 trial；`ACTOR_RESET_WORLD_NOT_EMPTY: 3` 阻止了正式 TM 仿真。
- BehaviorAgent 运行日志有六条 “attempting to destroy an actor that is already dead” 警告，其中三条对应 vehicle 147–149。与下一个 TM 运行发现 3 辆车数量相符，但不能仅靠数字断言其身份。
- `convoy_audit.json`：有效配对对比为 0，缺失预注册动作配对为 4；`convoy_joint_backup_audit.json` 也是 0 个有效配对。

## 故障机制

原版 `run_trial()` 在同步模式 `world.apply_settings(synchronous_mode=True)` 后立即 `try_spawn_actor` 三车和传感器、设置目标速度，并在 `warmup_ticks=0` 时读取 `get_location/get_transform`。缺乏同步 tick；初始状态可能仍是未初始化值。随后检测三车零坐标相互重叠，函数提前退出，没有跑任何正式时步。销毁时调用多个 `actor.destroy()`，未严格验证服务端 Actor 注册表清理结果，因此下一轮可能遇到既有车辆。本问题与单卡显存不足、TM 端口冲突无直接证据关联。

## 新修复的关键点

1. 每条 trial 在生成车辆和传感器后，无条件先执行 `world.tick()`，再逐车与已冻结的锚点位置比较；误差超过 3m 时，立即以 `CARLA_SPAWN_NOT_INITIALIZED` 终止（不能返回合法观测值）。
2. 生成 RIDE 车辆时填写 blueprint `role_name=ride.<随机试验UID>.<role>`；trial JSON 记录 trial_uid 和 actor_ids。旧代码生成的无标签车辆不会自动认领。
3. 清理通过 `client.apply_batch_sync([carla.command.DestroyActor(id), ...], False)`，同步 world 下推进一次 tick，再读取服务端 Actor 列表确认 ID 确实消失；失败时抛出 `RIDE_TRIAL_NOT_CLEAN`，避免污染后续 trial。
4. 新 CLI `python -m ride.carla.actor_lifecycle` 默认只读，显示 Actor ID、role_name、位置，允许明确授权后只按标签清理、或精确指定旧 Actor ID。**不会自动清掉所有车辆，也不会调用 `reload_world()`**。
5. `03_run_carla_case.sh` 运行分析后再启动 `smoke_validate`，只有至少 1 条 trace 非空且 `metrics.valid=True` 才 exit 0（可用 `RIDE_SMOKE_REQUIRE_VALID=0` 仅调试无效场景）。这个检查不证明 RQ1/RQ2 或行为模型科学有效。

## 服务器操作

先解压到单独的新目录，例如 `~/code/RIDE_ACTOR_FIX_20261027`。在每一个控制终端设置：

```bash
cd ~/code/RIDE_ACTOR_FIX_20261027
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2002 CARLA_TM_PORT=8102 CARLA_TOWN=Town04
export RIDE_WORLD_RESET=actors
source scripts/00_env.sh

# 1) 无副作用检查卡在服务器中的车辆
python -m ride.carla.actor_lifecycle
```

如果你**确认这是独占的 CARLA 2002 实例**，而车辆确实是本次实验遗留，可根据检查输出中的实际 ID 精确删除（下面 147 等是上一次日志中的示例，当前须重新核对）：

```bash
python -m ride.carla.actor_lifecycle \
  --cleanup-ids 147 148 149 \
  --confirm DELETE_THE_LISTED_IDS

python -m ride.carla.actor_lifecycle  # 验证 vehicle 数量已变为 0
```

若已有车辆不是你控制的，**不要删除**；用独立空闲端口启动新的 CARLA。对新版本留下的标有 `role_name=ride.*` 的车，确认专用实例没有其他 RIDE 任务后可以：

```bash
python -m ride.carla.actor_lifecycle \
  --cleanup-owned --confirm DELETE_THE_LISTED_IDS
```

清理不能消除 Unreal 内部隐藏历史；若本次服务端存在其他未知 Actor、TM 控制器或同步 tick 客户端，不要把这次 actor-reset 当成已验证的严格因果复现实验。

### 最小物理试验 A / B

```bash
export RIDE_ANCHORS=1 RIDE_REPEATS=1 RIDE_SEED=17
export RIDE_PROFILES='early_slow'
export RIDE_FAMILIES='behavior'
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_actorfix_behavior01
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy

# 确保上面的 RIDE_ACTOR_CLEANUP 确认无遗留后
export RIDE_FAMILIES='tm'
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_actorfix_tm01
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
```

成功的**必要而不充分**日志特征：
- `RIDE_SPAWN_PRIME_TICK_DONE`，三辆车位置与 -511 附近的冻结锚点一致，而不是 (0,0,0)。
- `RIDE_ACTOR_CLEANUP ... "remaining_ids": []`。
- `RIDE_SMOKE_PASS`、valid_trajectories≥1、trace 长度约 160（8 秒 / 0.05 秒）。
- 若已成功但被 `offroad_or_lane_departure` 等原因拒绝，那仍不算科研有效 trial；需检查地图与轨迹，而不是强行标记 valid。
- 原实现把 `is_at_traffic_light() + 红灯 + 车速>0.5` 直接判为违规，但这只是位于交通灯触发范围内时仍在移动，并非经过停止线的证据。本补丁将其保留为 `red_light_trigger_motion_proxy_observed`，不再据此否决物理轨迹，明确标注 `traffic_light_legality_verified=false`；正式法律合规结论需要额外停止线跨越检测。

### 完整配对机制筛查

只有两组 smoke 都正常后，用**新输出根目录**保留同一组 Town、seed、anchor：

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_actorfix_pair01
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='early_slow late_slow mild_early mild_late'
export RIDE_REPEATS=2
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

当前 `RIDE_SMOKE_PASS` 只是确保至少有一条有效轨迹；若要检验 RQ1/RQ2 必须再确认所有预注册动作对足够完整、起始状态匹配、实际 Ego 轨迹执行且任务进展匹配、风险变化由 ego 干预而不是场景本身引起。**当前两个上传结果均不足以支持 RQ1/RQ2 成立。**

## 已知实验设计局限

- Actor-reset 只能重建 RIDE 车辆、重设信号灯和种子，不能保证 Unreal/Traffic Manager 内部全部状态与此前实验完全一致。重复 seed、不同顺序、不同起始快照可作为额外鲁棒性验证。
- 两组单臂 smoke 不能计算配对效应；不能使用 `run_end.completed` 替代 `metrics.valid`。
- TTC 使用 follower 同路同车道几何近似；acceleration 来自速度差分；急刹、合法性、传感器碰撞需要按照现有有效性筛选谨慎解释。
- BehaviorAgent 和 TM 都来自 CARLA，不能替代真实异质司机反应校准，也不是 RACP/HYPE 等强基线。完整 RIDE 论文仍须经过原生 WOMD、独立反应环境和 RQ3 基线实验。

本修复通过的是本地模拟单元测试，不是服务器上的 CARLA 物理验证。碰到 CARLA Unreal SIGSEGV 时必须检查服务端日志，不应把 Python 的结果解释为硬件缺陷的充分证据。
