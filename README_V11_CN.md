# RIDE v11：机制证据、实验保真和 CARLA 四车连锁研究（2026-10-28）

## 0. 适用范围

基于用户上传 `RIDE_V10_MECHANISM_GATE_20261028(1).zip` 修改，保留原有脚本及 v10 工作流。
新增的是 **假设验证工具**，而不是经过现实交通验证的新规划算法。

目前 RQ1 = 未证实；RQ2 = 未证实；RQ3 = 尚无可用强规划器的事前动作选择。不能宣称“强规划器系统性失败”。

**必须保持四层证据独立**：M0 正确场景/有效执行；M1 ego 干预和 B 反应；M2 B–C 次生风险、任务匹配可避免性；M3 两种响应证据导致 planner 事前选择错误。每级不通过就不引用下一级的结果。

## 1. 本次上传 v10 的审计结果（只来自上传文件）

| Panel | trial | `valid=true` | 同 family 合格动作对 | 跨 family 合格对 | 严格风险排序翻转 | 有效 B–C 碰撞 |
|---|---:|---:|---:|---:|---:|---:|
| threecar seed17 | 20 | 20 | 8 | 3 | 0 | 不适用（观察 A–B） |
| chain-dev seed17 | 20 | 13 | 4 | 2 | 0 | 0 |
| chain-confirm seed23 | 60 | 38 | 11 | 4 | 0 | 0 |

**仅有 seed23 确认包；seed41、seed73 没有上传**。四车开发集 `ENGINEERING_EXCEPTION_BLOCKER`，不该作为已通过的确认前提。确认包中的 60 次试验拥有 `run_end`，但有 22 次无效：冻结 `follower_c` 车道/road_id 恢复失败 10 次，BehaviorAgent `NoneType.is_junction` 路由异常 4 次，以及两类控制器的越界 8 次。开发集中则有 BehaviorAgent 路由异常4次、TM 越界3次。

汇总 100 次 trial，71 valid、29 invalid、18 例 simulation_exception（包括 10 个冻结恢复错误与 8 个 BehaviorAgent 运行异常）。

三组 Panel 不是一个独立统计实验，9 组跨模型合格比较**不可当成 9 个独立道路案例**。全部 10 个 anchor 坐标聚成 7 个 90m 物理簇；拥有合格跨模型比较的 anchor 来自 5 个簇。三车与四车发生部分地理重叠。

在本轮的四车负对照中，21 个 action vs cruise 比较满足初态/前缀门禁，预注册的探索性 B 先于 C 额外减速的时序代理计数为 0。大部分 B–C 最小净间距约19~22m，TTC 多为几十秒或 `9999`（无闭合）的哨兵值。四车拓扑正确**不等于**风险传播机制已出现。当前二元碰撞、hard brake、short TTC 主端点都没有通过有效行动对识别出新增风险。

## 2. v11 源码变更

- `ride/carla_scenarios.py`：将冻结坐标精度从 3 位小数提升到 7 位（精度问题是合理嫌疑，不能单凭日志确认唯一根因）；构建阶段对每个角色用与 replay 相同的 CARLA `get_waypoint` 执行 round-trip 检查；边界不稳定的 anchor 提前剔除并报告，不让单个错误 anchor 重复污染十次 trial。场景协议更名为 `ride_lane_corridor_v11_20261028`，旧 frozen 不可混用。
- `ride/carla_scenarios.py`：**单个 Panel 内**的独立位置最小间距由65m提高到至少90m；新的检验只能提高几何隔离，并不能保证统计独立性。
- `scripts/33_v10_chain_confirm.sh`：开发 `chain_m0_gate.json` 必须以完整、无工程异常、有效跨模型 pair 的状态通过；只有“有两组 pair”不能绕过 `ENGINEERING_EXCEPTION_BLOCKER`。
- `ride/carla/run.py`：在 BehaviorAgent 开始推理前检查 route incoming waypoint，遇到不完整的路由明确保留失败原因，不做无依据的控制器回退。避免将局部 `NoneType.is_junction` 报错悄悄当成安全反应。
- `ride/carla/roles.py` + `run.py`：提供两个**探索性**混合响应组合：`mixed_b_tm_c_behavior` 与 `mixed_b_behavior_c_tm`。A 与 B 采用其中一种模型，C 用另一种；适用于 `chain_cut_in`。这是 CARLA 环境内的异质控制器压力测试，不能据此声称两组独立人类反事实证据。
- `ride/v11_chain_evidence.py`：只读读取任意多个 v10/v11 JSONL、统计严格 M0/RQ1 分母与响应时间链，负对照必须初态相符、时间轴对齐、干预前 ego/B/C 位置和速度足够接近。不将额外速度损失自动视为碰撞，也不将 B先于C 的时序证据称为因果中介证明。输出所有 invalid/exception 与 90m 位置聚类。
- `scripts/35_v11_chain_discovery.sh`：预注册 2×2 机制诊断网格（B–C 20/25m、rear gap 35/45m），单元独立冻结、独立输出，不选择有利单元后删掉其余。

## 3. 服务器安装与回归

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip -q RIDE_V11_CHAIN_CAUSAL_AUDIT_20261028.zip -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V11_CHAIN_CAUSAL_AUDIT_20261028
conda activate mfrp
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pip install -e '.[test]'
python -m pytest tests -q
python -m compileall -q ride
bash -n scripts/*.sh
```

CARLA headless 请使用原有 `scripts/02_start_carla_headless.sh` 在独立终端启动；切图只能在**没有其他运行任务**时，由操作者在独立 CARLA 进程中执行：

```bash
export CARLA_TOWN=Town04
RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04
bash scripts/29_v9_carla_preflight.sh Town04
```

## 4. 首次 v11 四车 M0：不要直接大批量 seed

```bash
export CARLA_TOWN=Town04
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v11_chain_m0_17
export RIDE_SEED=17 RIDE_ANCHORS=2 RIDE_REPEATS=1
export RIDE_WORLD_RESET=actors RIDE_WARMUP_TICKS=20
export RIDE_CRUISE_MPS=7.5
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_CHAIN_BC_GAP_M=25
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_CHAIN_C_SPEED_DELTA_MPS=0.5
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
bash scripts/31_v10_chain_m0.sh
python -m ride.v11_chain_evidence \
  --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/v11_evidence.json"
```

上面的 7.5+0.5=8.0m/s 低于此前 Town04 30km/h 的速度限制。若其他路段限速不同，仍以 `configure_tm_speed` 的显式检查为准。

**第一批只检查**：`chain_m0_gate.json` 是否排除了 ENGINEERING_EXCEPTION_BLOCKER；是否出现新的 frozen roundtrip 拒绝计数；真实完成并线但 ego 无碰撞的有效 pair 是否足够；B/C 速度变化是否可归因于干预后的时间窗口。

## 5. 探索性四格机制网格：未达到 M0 之前不要运行

```bash
export RIDE_V11_GRID_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v11_chain_grid_town04_seed17
export CARLA_TOWN=Town04
export RIDE_SEED=17 RIDE_ANCHORS=4
export RIDE_CRUISE_MPS=7.5 RIDE_CHAIN_C_SPEED_DELTA_MPS=0.5
export RIDE_V11_BC_GRID='20 25'
export RIDE_V11_REAR_GRID='35 45'
bash scripts/35_v11_chain_discovery.sh
```

这只是候选机制发现与支持性边界测量；如果所有 4 个格子都无 C 额外减速，不应该通过再缩小到不合理车距制造碰撞。应当记录失败并考虑有限证据路线，而非继续调整以获得偶发极端案例。

## 6. 异质响应（必须先验证常规 M0，探索性，不纳入 RQ1 主表）

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v11_chain_mixed_seed23
export CARLA_TOWN=Town04 RIDE_SEED=23 RIDE_ANCHORS=2 RIDE_REPEATS=1
export RIDE_CRUISE_MPS=7.5 RIDE_CHAIN_C_SPEED_DELTA_MPS=0.5
export RIDE_CHAIN_BC_GAP_M=25 RIDE_CUTIN_REAR_GAP_M=35
bash scripts/36_v11_mixed_response_smoke.sh
```

这个执行四种 response 组合，但统一 RQ1 主比较仍是单独的 TM、BehaviorAgent，两种 mixed 只作为**跨控制器异质性敏感性分析**。

## 7. 何时进入真正确认 / RQ3

- M0：每条 corridor 成功初始化、每个角色真实可恢复、无 `simulation_exception`、至少一组 tm/behavior 的有效且任务匹配动作对；失败数据都保留。
- M1：在 matched cruise 下，ego 机动使 B 的轨迹/速度**发生显著差异**，开始时刻不早于 ego 动作，且前缀没有未解释分叉。
- M2：B 响应后，C 额外减速并出现 B–C 风险变化；对另一条**任务匹配** ego 动作风险显著更低。连续负担不应事后覆盖预注册的主碰撞/TTC/急刹端点。
- M3：两个响应模型对任务匹配候选的严格主端点排序确实相反；且真正规划器在反事实执行之前**冻结**了评分/选择。在独立评测环境中证明选择错误可避免。

目前 M0 仅三车通过，四车仍不稳定；M1~M3 不能宣称已通过。即使 CARLA 两模型都出现 M2，也必须用 WOMD/nuPlan 校准、ReactSim SMART/MTR 与不同反应架构独立验证。

确认集仅在**开发 M0 无异常**且研究协议和参数冻结后使用。旧脚本 `scripts/33_v10_chain_confirm.sh` 仍保留路径兼容，但已增加严格 M0 status 校验。一个 seed 下多个动作不代表多个独立道路场景。确保 Town04 holdout 使用90m排斥；跨 Town 则需单独启动预检查地图。

## 8. 论文立意的停止/转向条件

- 如果有效、多个物理 corridor、多种响应模型下始终没有 B→C 风险传播，应缩小/放弃“systematic third-party failure”命题，不要追逐碰撞。
- 如果 B/C 连续差异存在但碰撞主端点没有决策差异，可转向 **interaction burden as decision-relevant third-party externality**，但需要重新与 Courteous/RACP/hard social constraints 比较并做全新预注册 holdout。
- 如果主要矛盾是有限 response-bank 未找到 witness 却被当成不可恢复，考虑 **Decision-Critical Behavioral Recourse Evidence**：保留 `VERIFIED_FEASIBLE / CERTIFIED_INFEASIBLE / UNKNOWN`，但必须证明有新增决策价值且不靠停车过于保守。
- 只有 RQ1/RQ2、独立反应复核、strong fair baselines 均成立才开始设计新的核心算法及完整 RQ3 对照。
