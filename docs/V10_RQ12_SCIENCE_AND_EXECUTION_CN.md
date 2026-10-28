# RIDE v10 科学审计、四车连锁场景及服务器复现实验（2026-10-28）

> 数据基准：本轮用户上传的 `RIDE_V9_CARLA_STABILITY_20261028_MAPFIX.zip` 和 `rq12_v8_m0_smoke(1).zip`。原有 v9 的安全地图切换 `scripts/02c_prepare_carla_town.sh` 被完整保留。本文所说的 v10「通过测试」均指静态/Mock 单元测试，**不代表在用户 CARLA 0.9.16 服务器上运行过 v10 四车场景**。

## 1. 科学状态——不要混淆工程、模型敏感性与决策失效

RQ1：固定同场景、同初态、同一对可行且任务进展匹配的 ego 行动，比较 TM 与 BehaviorAgent 估计的第三方安全风险**严格顺序是否反转**。这个指标是基于仿真后 outcome 的反事实诊断，并不是上线规划器提前作出的选择。RQ2：在相同环境、相同响应模型和匹配的 ego progress 下，是否存在可归因且可通过替代合法行动避免的第三方碰撞/异常制动/短 TTC，并要求独立环境复核。RQ3：上述现象成立后，现有强规划方法是否仍错误接受高风险行动。

### 最新 M0 直接取证

| 项目 | 上传证据结果 |
|---|---:|
| `run_end` 完整、源 JSONL trial | 20/20 |
| TM / BehaviorAgent 试验 | 10/10，10/10 |
| `valid=true` | 20/20 |
| 真实完成并线机制 | 16/20（各响应模型 8/10，另外的 cruise 无并线） |
| Ego 碰撞 / 原先 A–B 物理碰撞 | 0 / 0 |
| 有效跨响应模型、任务匹配早晚行动比较 | **3** |
| 支撑这三组比较的独立物理 anchor / seed | **2 / 1** |
| 预先冻结的三元安全键严格排序翻转 | **0** |
| 已有官方强 planner 的 ex-ante 选择 | **0** |

证据归档中没有此次新 Town05 confirm 的有效 JSONL，因此**不能推断 Town05 的任何 RQ1/RQ2 成立情况**。Town04 M0 通过仅意味着实验执行与至少部分任务匹配门禁正常，不是「现实道路上发生这种系统性错误」的结论。

### 现象线索：连续制动负担 ≠ RQ1 完成

旧 M0 下观测到的最大后车速度：TM `8.3949 m/s`，BehaviorAgent `12.1285 m/s`；BehaviorAgent 有 `5/10` 次试验超过 nominal 8m/s 至少 1.5m/s，TM 为 `0/10`。这进一步说明**两个家族的速度条件原本没有对齐**；v10 的名义上限约束属于新控制协议，在 CARLA 中未实测其有效性，不能用历史 v9 数据填充新结果。

在第二个物理 anchor 的 BehaviorAgent 下，rear B 的 `cruise` 负对照约 4.25 秒起持续急刹，基线制动超额积分约 0.633 m/s；`early_slow` 约 1.034 m/s，`late_slow` 约 0.721 m/s。TM 的相应负担接近 0。注意 `early_slow/late_slow` 在 cut-in 实际**只改变并线触发时刻，保持 ego 8m/s 名义纵向速度**；名称 `slow` 是遗留标签，不能将此观察写作「早/晚制动对比」。后车受背景跟车影响，连续负担应与 matched `cruise` 在**同模型、同初态、相同绝对时间**比较，若行动前已分叉则标记为不能归因。严禁事后改变原主端点以宣称 RQ1 通过。

## 2. 本版工程变更与边界

1. **新增四车 `chain_cut_in`**：ego 从相邻车道并入前车 A 与直接受影响后车 B 之间，B 后有第三方跟车 C；主要碰撞、后车短 TTC 与制动风险固定记录 B–C，不再把被 ego 插入隔开的 A–B 错当作最敏感的连锁追尾风险对。C 通过 CARLA 地图真实向后 waypoint 搜索构建，检测初始重叠和道路边界；找不到就拒绝场景，**不生成虚构坐标**。
2. **名义速度公平性**：TM 使用原有严格限速校验、名义速度转换；BehaviorAgent 设置 `_behavior.max_speed <= 3.6*nominal_mps`、`speed_lim_dist=0`，只限制最高速度，不改其原始跟驰/障碍物行为算法；将实际参数完整写入 trial。新协议名 `nominal_cap`；对照 `native` 可另行冻结独立实验。注意这是条件标准化，不意味着两控制器获得完全相同的速度分布。
3. **初态/反事实有效性**：四车 `ego/A/B/C` 均记录位置、速度、yaw、世界速度向量；跨模型配对校验 A/B/C 初态、ego 执行轨迹、任务进展；负对照的制动增量还要求**ego 首次横向指令前**行动/控制轨迹接近。若操控或车速历史在 `do(a)` 前已不同，记录 `pre_action_trajectory_mismatch`，不计为额外归因证据。
4. **成对风险的纵向几何**：B/C TTC 仅当物理车道中心线相符、相对朝向一致、位于所冻结道路走廊时计算，允许合法 OpenDRIVE road ID 分段变化；缺失 TTC 不得假设为安全。
5. **主辅端点严格分离**：原 collision / sustained-hard-braking / short-TTC 三元主端点不变；连续负担、第一次制动时间及相对于 cruise 的负担增量作为探索性诊断，重新设计的四车 B–C 主风险属于**新的 v10 实验协议**，不得和 v9 A–B 直接合表比较。
6. **保持用户修复的地图流程**：`RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04` 仍可用，但**仅在自己控制、空闲的 CARLA 服务端做一次显式地图切换**；仿真/确认批次不触发自动 `load_world`。若切 Town05 仍发生 Signal 11，首先保留 Unreal 日志并在独立进程验证，不是 RIDE 科学负例。
7. **版本冻结**：`ride_lane_corridor_v10_20261028` 包括 C 的初始间距/速度差及 BehaviorAgent 名义速度协议；旧 v9 场景清单与 v10 执行不可混合。`RIDE_SMOKE_REQUIRE_COMPLETE=1`，无效 trial 必须保存并计入分母。

目前**没有**实现：从 WOMD/nuPlan 复刻真人反事实；所有合法交通灯规则检验；正式预推理 planner action scoring；学习式响应模型的交叉验证；完全独立于 CARLA 的第二物理环境；已验证的四车实验成果。必须如实表述。

## 3. 安装和本地代码检查

把 v10 压缩包上传到服务器后：

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip -q RIDE_V10_MECHANISM_GATE_20261028.zip -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V10_MECHANISM_GATE_20261028
conda activate mfrp
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m ride.doctor --server
```

如果服务端未运行，先从独立 tmux 终端 `source scripts/00_env.sh && export RIDE_CARLA_QUALITY=Epic RIDE_CARLA_START_MAP=Town04 && bash scripts/02_start_carla_headless.sh`；运行时 CARLA 不应已被其他实验占用。若已运行且地图不是 Town04，先在自己控制的空闲服务端：

```bash
export CARLA_TOWN=Town04
RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04
bash scripts/29_v9_carla_preflight.sh Town04
```

安全说明：Town05 有已知 Unreal `Signal 11` 风险，不在正在跑试验的世界中切地图。`-RenderOffScreen` 是无头渲染，不代表不初始化 GPU；不要未经确认将引擎崩溃归因于 Python 或 DB。

## 4. 可选：先对旧 M0 数据作**只读重审计**

```bash
# 把旧试验原始 JSONL 的真实路径填进来；分析器不会改变文件
OLD=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_m0_smoke/carla/cut_in.jsonl
bash scripts/32_v10_reaudit.sh "$OLD" \
  /data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_m0_smoke/carla/v10_legacy_audit.json
```

结果仍应是 20 valid、3 valid cross-family、0 strict reversal（如果你指向的确实是本轮上传的文件）。`v10_legacy_audit` 只分析老实验，不能更改旧 freeze 的协议；旧数据不应进入 v10 主试验汇总。

## 5. 先独立测试三车 v10 控制组 M0

```bash
export CARLA_TOWN=Town04
export RIDE_SEED=17
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export RIDE_WARMUP_TICKS=20
export RIDE_WORLD_RESET=actors
export RIDE_CRUISE_MPS=8
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v10_threecar_m0
bash scripts/23_v8_smoke.sh
python -m ride.v10_mechanism_report \
  --input "$RIDE_OUTPUT_ROOT/carla/cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/threecar_v10_mechanism_report.json"
```

这一批是**新的速度协议**，不能与 v9 BehaviorAgent 原生行为的结果当作重复验证。重点检查报告中两族的前车速度分布、有效双动作对、`cruise` 的背景制动与是否有不可比初态。只要仍有严重配置异常，停止在更大 N 上运行。

## 6. 新的四车连锁场景 M0：必须先过物理门禁

```bash
export RIDE_CHAIN_BC_GAP_M=25
export RIDE_CHAIN_C_SPEED_DELTA_MPS=0
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v10_chain_dev_m0
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export CARLA_TOWN=Town04
bash scripts/31_v10_chain_m0.sh
cat "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json"
cat "$RIDE_OUTPUT_ROOT/carla/chain_v10_mechanism_report.json"
```

必要条件：四车均正确生成且不初始重叠；B/C 在真实车道同向且制动/碰撞/TTC 有适用时步；ego 安全进目标车道；TM 和 BA 在两个匹配动作之间能比较；负对照不能在干预前已经出现无法解释的轨迹漂移。`M0` 即便通过也不意味着存在可避免第三方风险。

若 v10 四车低风险安全运行而未产生 B–C 风险，先保留负例；可以独立新建探索性协议，**事前**离散选定 B–C 间距（25/30m）与 C 速度差（0/0.2m/s），每个 cell 冻结后独立运行，不通过事后调参数只留碰撞正例。在限速为 30km/h 的路段，任何新速度目标都必须在道路限速内，否则程序应中止。

## 7. 发现集扩展与真正确认

只有 v10 chain M0 完成且有可比动作对，才能将开发集 `RIDE_ANCHORS` 扩至 6、仍优先一次 seed，作为独立**发现集**。注意不能用本次仅有 2 anchors 的开发 M0 直接支持人口层面统计。请使用全新 `RIDE_OUTPUT_ROOT`、事先确定的参数，一条地理 corridor 在多个 seed 下只能算一个物理 cluster。

当已经有充分开发样本并冻结超参数后，才在至少 3 seeds、多个与开发集分离的物理 corridor 做确认：

```bash
export RIDE_DEV_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v10_chain_dev_m0/frozen/chain_cut_in_anchors.jsonl
export RIDE_CONFIRM_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v10_chain_confirm
export RIDE_CONFIRM_SEEDS='23 41 73'
export RIDE_ANCHORS=6
# 仅在你的 CARLA 服务器确认已经加载的地图上运行，保证不触发自动切换
export CARLA_TOWN=Town04
bash scripts/33_v10_chain_confirm.sh
```

如果为了跨地图而使用 Town05，应在空闲独立进程中做地图准备，确认 `scripts/29_v9_carla_preflight.sh Town05` 正常后才运行。`33` 会要求开发 chain M0 report 有有效跨 family pair，核对控制参数、冻结协议和物理 corridor 空间排除；缺场景就失败，绝不把缺失场景计为 0 风险。跨 Town05 的空间坐标不同，报告仍应把 Town 视为测试域变量，避免只有 Town04 的结论过度外推。

## 8. 严格的 RQ1/RQ2/RQ3 判定逻辑

- **RQ1 现象探索**：同一 initial state、两条真实执行且进度匹配的 ego actions、同一前置响应族信息；两家模型完整执行后的预注册安全键呈严格相反顺序。两模型都是 CARLA family，仍不代表人类真值。未来须校准独立 learned response environment，并报告是否为 task-relevant ranking flip。
- **RQ2 现象探索**：至少在一个响应模型内，用同 scene/corridor、相同 B/C 身份和等进展行动构成可避免 third-party B–C 风险差。强制记录 ego 自身碰撞、道路/规则、处理前负担、模型/种子、有效分母。**不能把 cruise 当成 task-matched RQ2 替代行动**，它仅是机制负对照。进一步通过独立响应模型、非 CARLA 数据源和不失去进展的可替代动作复核。
- **RQ3**：只有前两个环节形成冻结、跨场景支持的现象，再接 hard courtesy、risk-aware contingency 和官方 GameFormer-Planner，在相同候选池、信息预算和真实 rollout 之前写出选择与模型 SHA。当前 v10 只创建物理评测层，没有正式 planner baseline。

对连续负担的发现，应**预先注册为下一批 primary 或 co-primary 研究方案**之后再新冻结执行，而不是修改现有 v9 或本次 v10 的原始 endpoint。保持碰撞 / 短 TTC / 制动的多指标原始结论和失效类型分层。统计以唯一 corridor 为独立单位 bootstrap/置换，不把 2 anchors x 2 families x N profiles 当成独立 N。报告没有发现的区间及样本选择流程。

## 9. 论文故事的可证伪转向

路线 A（保留 RIDE）：若真正在四车 B–C 对中发现可避免风险且响应建模能改变 task-matched ego 动作选择，针对「响应假设所诱发的具体第三方风险选择错误」形成不足以被 hard courtesy / interaction prediction 直接处理的 gap。需要正面与 Bosch SMART-reactive planner ranking shift、ReactSim-Bench agent-agent collision stress 对比；不能把「reactive simulator 改变规划分数」冒充原创点。

路线 B（可能转向）：若越来越多有效跨模型配对仍无严格反转、唯一差别是连续负担/有限响应库不确定性，转为 **Decision-Critical Behavioral Recourse Evidence**：在明确交通负担语义、有限验证预算下，规划器怎样识别 `VERIFIED_SAFE`、`CERTIFIED_INFEASIBLE_IN_SPECIFIED_DOMAIN`、`UNKNOWN`，并检验错误接受/错误拒绝是否比现有 decision-aware prediction 更严重。仅 finite-bank no-witness 不推出 UNRECOVERABLE；仅改标题不能构成新颖性。

**此时既不能声称 RQ1/RQ2 通过，也不能声称研究中的真实人类失效已由 CARLA 证明。** 决策目标是建立可核验的研究发现，不是人为制造正例。
