# RIDE v11 → v12：科学问题、反证线索与可复现实验

**日期：2026-10-28。来源：本轮三个上传包（v11 工程、v11_chain_m0_17、v11_chain_grid_town04_seed17）。** 这份报告是对真实 JSONL/审计文件的回放分析。**没有**在用户的 CARLA 服务器上运行 v12，也没有验证真实交通中的因果效应。

## 1. 先回答为什么第五步拒绝

`v11_chain_m0_17/carla/chain_m0_gate.json` 的状态是 `ENGINEERING_EXCEPTION_BLOCKER`，不是 `M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF`。原始 M0 有 20 个 trial、12 个有效、5 个 `AttributeError: 'NoneType' object has no attribute 'is_junction'`、3 个 TM `offroad_or_lane_departure`（无效原因计数有可能在一条 trial 上重叠）。有效的同模型比较 4 组；跨 TM/Behavior 的任务匹配比较 1 组，来自一个 anchor。`scripts/33_v10_chain_confirm.sh` 的第一项门禁要求 dev M0 无异常，故拒绝有依据，不能通过删掉失败记录或强制跳过。

异常发生在 `ride/carla/run.py` 调用 `BehaviorAgent.run_step`，对照 traceback 源于 `behavior_agent.py` 对 `_incoming_waypoint.is_junction` 的访问，此时 `_incoming_waypoint=None`。在 v11 路由前置检查通过后队列仍可能于运行中耗尽；这与 `LocalPlanner.get_incoming_waypoint_and_direction()` 在队列为空时返回 `None` 一致。v12 新增**明确记录的沿非路口唯一车道前视路线**，并在冻结 anchor 和仿真开始时检查 75m/约 78m 的路线覆盖。它使用 BehaviorAgent 自身的 `set_global_plan` 导航接口，不修改其跟驰、刹车和决策代码；如果道路不能生成完整路线，继续拒绝该 anchor。它是一个新的实验协议，**不能用 v12 的结果追认旧 v11 实验**。

## 2. 全量四格数据：有一个干净实验，也有真实负面结果

| B-C / A-B (m) | trial | valid | simulator exception | task-matched cross-family action pairs | strict risk rank flip | B→C代理（v11 冻结阈值） |
|---|---:|---:|---:|---:|---:|---:|
| 20 / 35 | 40 | 40 | 0 | 4 | 0 | 0 |
| 20 / 45 | 40 | 32 | 5 | 4 | 0 | 0 |
| 25 / 35 | 40 | 32 | 5 | 4 | 0 | 0 |
| 25 / 45 | 40 | 32 | 5 | 2 | 0 | 0 |
| **总计** | **160** | **136** | **15** | **14** | **0** | **0** |

请注意：这些是相同地图 Town04、seed17，多个 cell 重用同一个道路区域，14 个动作对**不是**14 个 IID 路段。原始 M0 的 25/35 配置与 Grid 也有重合，禁止合并计数后作显著性推断。四格中有 74 组有效的相对于 `cruise` 的负对照；其中没有新的 C-specific 第三方主要危险。更重要的是，**最干净的 20/35 cell 已完成 40/40 有效实验，仍没有主端点翻转**；因此「完全因为工程故障没有正例」已经不准确。

## 3. 保留原研究题目的边界，重新定义递进机制

RIDE 工作标题 **Response-Model-Induced Decision Errors in Interactive Driving** 可以继续作为待证伪的 H-RIDE；不要把现象当成已经建立的事实。

- **M1：ego→B 直接响应。** 相同初态、执行前缀匹配，ego 干预后 B 相对于 `cruise` 是否出现额外减速或舒适负担。需要做动作时序检查和相同 B 身份检查。
- **M2：B→C 次生传播。** 观察 B 先变化、C 后变化的时序只是必要线索；要证明 B 的介导作用，后续需有 `do(B response)` 或冻结 B 轨迹的 mediator intervention，而不能仅凭两个制动发生时间的先后给出因果结论。
- **RQ2：可避免的第三方外部性。** 同一个场景在**两条均合法且任务进展匹配**的 ego 动作下，对固定 B-C 的 TTC/碰撞和 C 额外负担产生可重复的差异；`cruise` 只能作为机制负对照，不能直接替代进展匹配动作。
- **RQ1：决策相关的模型失配。** 同一冻结候选对，在不同响应证据下发生非平局风险排序翻转；之后需要 ex-ante 预测器评分和实际被选择动作记录才能说某个 planner 选错。
- **RQ3：强基线是否仍有剩余缺口。** hard courtesy、risk-aware response planning、GameFormer-Planner 等在同一候选预算、同一可见信息及独立响应环境中调优评测。当前没有这样的结果，不能宣称强算法系统性失败。

## 4. 观察到但不可事后追认的探索性线索

四格共 74 组有效 `cruise` 对照：TM 38，BehaviorAgent 36。保持至少 0.2 秒和最大 3 秒 B→C 滞后的时序要求，逐阈值**事后敏感性分析**：

| 额外速度损失阈值 B/C (m/s) | TM B→C | Behavior B→C |
|---|---:|---:|
| 0.5 / 0.3（v11 预注册）| 0/38 | 0/36 |
| 0.4 / 0.3 | 0/38 | 0/36 |
| 0.3 / 0.3 | 6/38 | 0/36 |
| 0.2 / 0.2 | 10/38 | 0/36 |
| 0.1 / 0.1 | 12/38 | 0/36 |

原来 0.5 的 B 阈值在本批次观测中几乎卡在最大响应（约 0.497 m/s）上，令所有较弱变化成为零；但较低阈值的正例不能算 v11 的正式假设通过。另一条探索性指标是 C 车舒适制动超限积分（`0.05 × Σ max(0, -a_C-2.0)`，干预后、与 cruise 做差）：9/74 组的增量超过 0.1 m/s，均来自 **20/35 cell 的 TM 分支**。该 cell 的 TM 13 组对照中增量均值约 +0.208 m/s，而 BehaviorAgent 13 组约 +0.001 m/s。它可能提示模型敏感的第三方代价，但阈值、cell、控制器和结果是在开发集观察后提炼的，必须新冻结、跨走廊/seed、独立响应环境复核。

不能将这类连续负担直接转成“碰撞风险”。在 7.5m/s 附近、20–25m B-C 间距、约 +0.5m/s 后车速度差下，车辆净间距通常十余米，初始 TTC 会远大于 1.5s；若 B 不强制动，就没有理由期待频繁出现 1.5s TTC 或 B-C 碰撞。因此 v11 的**极端风险端点稀疏**不完全是实现错误，也与场景物理参数有关。

## 5. v12 改动范围

- `ride/carla/route_protocol.py`：唯一连续非路口路线搜索、明确路由覆盖、CARLA BehaviorAgent 固定 `LANEFOLLOW` route 导入，不改其原生 car-following 决策；失败不自动切换策略。
- `ride/carla_scenarios.py`：链式场景冻结前检查 A/B/C 各自至少 75m 单一连续车道；新的 `ride_lane_corridor_v12_20261028`。
- `ride/carla/run.py`：显式 `--behavior-route-protocol frozen_corridor` 和 `--cutin-paired-amplitude-mps`；冻结/执行两阶段严格锁参数与路由模式。
- `ride/carla/protocol.py`：新增 `speed_early/speed_late`，固定 2.5 秒并线时机、在 1.5–3.5 和 3.5–5.5 秒施加等幅反向补偿速度扰动，**目标**速度积分相同，但实际行驶进展仍需实测匹配。建议 7.5±0.5 m/s；不得超过当前路段限速。
- `ride/v12_sensitivity.py`：只读 post-hoc 敏感性，记录 TM/BA、B/C 时间先后、C-specific hazard、C/B 舒适制动负担、无效数据为 UNKNOWN。
- `ride/v12_speed_pair_audit.py`：新速度对在双响应模型下的实际任务匹配、风险排序；只报告事后 oracle 风险排序，不冒充 ex-ante planner choice。
- `scripts/37_v12_chain_m0.sh`：用全新的输出目录冻结并执行 7 个动作、两种响应模型；保留异常。
- `scripts/38_v12_reaudit_grid.sh`：只读重新审计已有 v11 四格日志。
- `scripts/39_v12_chain_confirm.sh`：新版本冻结、路由模式、速度扰动及两个独立 dev anchor 的确认前置门禁；拒绝沿用 v11 失败的 M0。

本地 `pytest`、`compileall`、`bash -n` 验证通过；本地不含 CARLA 0.9.16 服务端，**没有声称 v12 新路线在真实 Unreal 中成功执行**。

## 6. 下一步完整指令

### 6.1 安装和离线测试

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip -q RIDE_V12_MECHANISM_AUDIT_20261028.zip -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V12_MECHANISM_AUDIT_20261028
conda activate mfrp
python -m pip install -e '.[test]'
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export DATASET_ROOT=/data0/senzeyu2/dataset
source scripts/00_env.sh
python -m pytest -q
python -m compileall -q ride
bash -n scripts/*.sh
```

### 6.2 地图预检（只操作自己的独占 CARLA 服务器）

```bash
export CARLA_TOWN=Town04
RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04
bash scripts/29_v9_carla_preflight.sh Town04
```

### 6.3 新协议 M0（推荐优先：20m/35m）

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v12_dev_bc20_rear35_seed17
export CARLA_TOWN=Town04
export RIDE_SEED=17 RIDE_ANCHORS=4 RIDE_REPEATS=1
export RIDE_WORLD_RESET=actors RIDE_WARMUP_TICKS=20
export RIDE_CRUISE_MPS=7.5
export RIDE_CHAIN_BC_GAP_M=20
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_CHAIN_C_SPEED_DELTA_MPS=0.5
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS=0.5
bash scripts/37_v12_chain_m0.sh
cat "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json"
cat "$RIDE_OUTPUT_ROOT/carla/v12_speed_pair_audit.json"
```

这批实验预期 4 anchor × 2 family × 7 candidate = 56 trial（若无法冻结足够 anchor，明确失败，不改变 denominator）。首要检查 `behavior_route_configuration` 的 `route_length_m`、是否消除了 `NoneType.is_junction`、ego lane dwell、真实进展匹配及是否新获得 M1 弱响应；此步不应预设一定发生危险。

如果 `BEHAVIOR_ROUTE_END_OR_BRANCH` 或 `BEHAVIOR_ROUTE_DISCONTINUITY` 使 anchor 不足，不要绕过路线检查，先在 Town04 其他区域或空闲 Town05 独立服务端中搜索合格道路。不同道路必须以新输出目录冻结。若保持同源数据环境，请不要混入 v11 结果。

### 6.4 继续做全量正式确认（只在新 v12 开发 M0 干净后）

```bash
export RIDE_DEV_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/v12_dev_bc20_rear35_seed17/frozen/chain_cut_in_anchors.jsonl
export RIDE_CONFIRM_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v12_confirm_bc20_rear35
export RIDE_CONFIRM_SEEDS='23 41 73'
export RIDE_ANCHORS=6
export CARLA_TOWN=Town04
# 同样保留 dev 的 7.5、20/35、C+0.5、3.5/20、paired amp0.5、warmup20
bash scripts/39_v12_chain_confirm.sh
```

脚本会先拒绝版本、路线模式和参数漂移，检查开发集无仿真异常且至少两个独立 anchor 有有效跨模型对，然后在同地图排除距开发 anchor 90m 内的候选位置。若没有足够 holdout 道路，请新开 Town05 服务器、显式切图并创建新确认目录；不得放松场景距离以“获得显著性”。

### 6.5 只读重算旧 v11 四格（无需 CARLA，建议归档）

```bash
export RIDE_V11_GRID_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v11_chain_grid_town04_seed17
export RIDE_V12_AUDIT_OUTPUT=/data0/senzeyu2/dataset/RIDE/evidence/v11_grid_v12_readonly_audit.json
bash scripts/38_v12_reaudit_grid.sh
```

## 7. 提前定义停止/转向条件

- 新 v12 在多个独立、有效 corridor 中仍没有 B 和 C 的负对照差异：**降低或放弃“典型合法并线会产生严重次生风险”的主张**；当前场景不能支持普遍现象。
- 只有 TM 出现弱传递、BehaviorAgent/独立 learned responder 不复现：可以研究仿真器/控制器反应差异，但不能声称人类交通存在稳定机制。
- M1/M2 弱负担可以复现，但任务匹配的选错没有出现，或强 hard-courtesy 已解决：将标题范围从“系统性决策错误”收窄为第三方负担估计/响应证据可靠性，而不是制造一个不存在的 RQ3。
- M1/M2、合法任务匹配 RQ2、严格跨响应排序 RQ1 均在独立场景和 learned environment 复现，且强规划器在事前决策中仍有残余错误：才设计新算法，如响应证据可信区间、第三方风险负担约束、按决策价值分配反事实仿真预算。

关键统计单位是**独立道路场景/走廊**而不是 56 个 trial；正式研究至少需要多个 corridor、seed、地图和不同架构的响应模型。未观测正例应公开原分母及不确定性。
