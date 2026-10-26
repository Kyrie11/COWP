# RIDE / CARLA 0.9.16：`NO_ELIGIBLE_MAP_ANCHOR` 诊断与修复

## 触发点（已经证实的代码事实）

`scripts/03_precommit_scenes.sh convoy` 调用 `python -m ride.carla.run --archetype convoy --town "$CARLA_TOWN" --manifest-only`。报错发生于生成物理试验之前，`build_anchors` 返回空列表。此时并未执行 convoy 的干预配对，不能把它算作一个 RQ1/RQ2 的阴性样本。

旧版本 `build_anchors` 对 convoy **只扫描 `get_spawn_points()`**。旧 `convoy_candidate` 要求道路锚点向后 18 米、向前 20 米及 100 米有单独的 `previous/next` 结果，要求 B、A、ego、end **全部**属于相同的 `road_id` / `lane_id`，且首尾方向差小于 12°；这比模型研究所需的“连续、可驾驶、平滑、非路口道路”更苛刻。CARLA 0.9.16 自身的 `.next(d)` 允许在路口有多个返回值，`next(100)` 也不能验证路径中间是否交叉路口。

无法在此沙箱运行用户的 CARLA 服务端，**因此不能断言 Town04 的哪个具体 road_id 是有效的，也不能保证修复后 Town04 必定有锚点**。具体是哪一个旧约束导致 Town04 无解，需要用户服务器上新加入的逐类拒绝统计。

## 修复后的锚点约束（v2，不混用旧协议）

- 先尝试推荐出生点；不足时从 `Map.generate_waypoints(10.)` 生成所有车道采样，确定性排序和 shuffle，避免漏掉没有预设出生点的合法路段。
- B → A 为 18 米，A → ego 为 20 米，A → end 为 100 米，**不改变实验角色或纵向目标**。分别向后每 3 米、向前每 5 米检查下一路点是否唯一、可驾驶、非 junction、车道连续，且无局部急转。
- 初始跟车对和 ego 的近段走廊要求同一个 OpenDRIVE `road_id`；只允许更远处**非路口且车道连续**的自然道路 ID 变化。总航向变化上限 25°；这比 12° 宽，但依然禁止急转、U-turn、路口穿越。
- 选用的 waypoint 的三维位置、`road_id`、`lane_id`、朝向继续冻结并在物理运行时复核。新增 `anchor_protocol_version`，不允许旧清单进入新版物理实验。
- `--inspect-anchors` 和 `scripts/01b_probe_anchors.sh` 只调用地图筛选，不写冻结清单或 outcome JSONL；在 stdout 打印 `CARLA_MAP_ANCHOR_DIAGNOSTICS`，便于决定地图/模板是否合理。

## 必须在服务器执行的命令

在 CARLA 0.9.16 运行中、没有其他实验共享该服务端时，在 RIDE 根目录：

```bash
export CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m ride.doctor --server
bash scripts/01b_probe_anchors.sh convoy Town04
```

若诊断 `eligible_anchors=0`，可以**单独**继续：

```bash
bash scripts/01b_probe_anchors.sh convoy Town05
bash scripts/01b_probe_anchors.sh convoy Town03
```

只有某个地图 `eligible_anchors > 0` 才把它用于冻结：

```bash
export CARLA_TOWN=Town04  # 换成探测确实有候选的 Town
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_v2_town04
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

每次换地图或模板版本，使用独立 `RIDE_OUTPUT_ROOT`。冻结阶段不会覆盖已经存在的 JSONL；不要删除历史证据或将不同地图的锚点合并。`--inspect-anchors` 虽然不输出结果文件，但**会 `load_world` 切换 CARLA 地图**，不能与正在运行的试验并行。

## 如何解读统计

- `spawn_points_scanned`：预设出生点检查数；零合法不代表整个地图无解。
- `map_waypoints_scanned`：从全图 10m 采样搜索的候选数；如果没有执行，通常表示在出生点已找够。
- `rear_lane_end_or_branch`、`forward_lane_end_or_branch`：向后/向前路径终止或分叉。
- `*_junction_or_non_driving`：途经路口或离开可驾驶车道。
- `*_lane_identity_changed`：路线中途车道 ID 变化。
- `corridor_curvature_over_25deg`：需要更大弯度，不能直接当成道路不安全，更不能为了获得正例而无限放宽上限。
- `initial_corridor_road_id_changed`：三辆车附近跨 OpenDRIVE 道路分界，不适合当前固定车辆对的 TTC 计算。

## 验证与边界

当前环境完成了 31 项回归/单元测试、脚本语法及模块编译；新增测试覆盖：预设出生点之外的连续车道、跨未来 road_id 的正常路段、被藏在 100 米内的路口，以及只读 `--inspect-anchors`。以上均为 **模拟 waypoint 客户端的代码测试**，不等同于实机 CARLA 0.9.16 地图几何或物理执行成功。需要用服务器的诊断日志决定是否存在实际地图锚点。
