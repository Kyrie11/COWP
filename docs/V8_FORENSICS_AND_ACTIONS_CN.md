# RIDE v7 真正发现了什么：论文主线与科研行动审计

**审计输入**：`自动驾驶论文定位分析(3).md` 最新 RIDE 故事；RIDE(3).zip v7；`rq12_v7_dev_protocol` 和 confirm seeds 17/23/41 原始 `cut_in.jsonl`；`diagnostics.zip/val_db_coverage.json`；上传的 ReactSim-Bench 和 GameFormer-Planner 源码。没有 CARLA/nuPlan 服务器，也没有实测用户的新模型运行，下面是对已有证据的审计和离线代码修复。

## 故事与三层科学问题

保留工作标题 **RIDE: Response-Model-Induced Decision Errors in Interactive Driving**。唯一可证伪上层问题：在无共享响应协议的异质交通中，规划器是否因为不可靠响应证据，在近似同任务进展的合法 ego 动作对中选择可避免的第三方高风险动作？

- **RQ1（必要、不充分）**：同初态/同样两个有效 ego 候选、同一第三方车辆对，在两种响应模型下是否出现 **严格**风险排序翻转？必须区分模型预测误差、后验排序改变、事先锁定的 planner 选择错误。
- **RQ2（必要、不充分）**：相对于任务匹配 alternative，在独立 response environment 中是否有可归因、可避免的他车急刹/持续制动负担、短 TTC、第三方碰撞？`cruise` 不是任务匹配 alternative，交通参与者自身背景制动不能算作干预风险。
- **RQ3（待启动）**：固定模型可见信息、候选集合、训练预算与反应环境后，充分调优的 hard courtesy、risk-aware selector、完整 GameFormer-Planner 是否仍在 RQ1/RQ2 证实的机制上选错？Bosch/ReactSim 是评测反应环境，不是两个 learned ego planners。

数学对象可表示为：同场景、同初态 s、同候选 a,b 和反应环境 P，固定第三方风险/负担 `L_P(s,a)`, `L_P(s,b)`。RQ1 不仅要求 `L_P(s,a) != L_Q(s,a)`，而要 `sign(L_P(s,a)-L_P(s,b)) * sign(L_Q(s,a)-L_Q(s,b)) == -1`。RQ2 需要 `L_P(s,a)-L_P(s,b) > delta`, 相同任务进展且合法/碰撞排除。真实人类对不同 ego 未来的因果效果 **不能** 仅凭日志 WOMD 或两个 CARLA 控制器识别。

## 四组原始实验结果

| panel | raw trial | valid | `mechanism_executed` | within-family task-matched | cross-family task-matched | strict reversal | A–B pair collisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 100 | 87 | 80 | 26 | 9 | 0 | 0 |
| confirm seed17 | 80 | 38 | 64 | 5 | 0 | 0 | 9 |
| confirm seed23 | 80 | 34 | 64 | 6 | 0 | 0 | 10 |
| confirm seed41 | 80 | 40 | 60 | 14 | 0 | 0 | 7 |

上述 26 次第三方碰撞均来自 **无效 trial**，只在 TM；其中有些在 `cruise` 负对照就存在。它们既不能证明 RQ2，也不是有意义的“传统方法失误”标签。开发集的九个跨模型 pair 也不能当成 9 个独立道路 cluster，零反转只是局部负面信号，不足以证明命题不存在。

### 导致确认失效的三个具体混杂因素

1. **速度目标不公平**。在 v7 `ride/carla/run.py`，TM 用 `tm.vehicle_percentage_speed_difference(a,0.)`，本意 9m/s，实际是按地图限速行驶。三个确认 seed 的 TM 多数后车 trial 达到超过预设名义速度两倍；BehaviorAgent 并无同样速度轨迹。确认碰撞和越界首先是模型控制协议不一致的证据，不能解释为细粒度响应证据失准。v8 通过限速 km/h → 百分比转换，并保留额外速度漂移诊断。
2. **控制协议从 dev 漂移**。Dev `cutin_late_switch_s=3.5`、`lateral_lookahead_m=20`；三个确认集为 4.5、15。改变的是实验动作本身，不能叫同一 protocol 的 confirmation。v8 `protocol_lock.py` 必须完全一致。
3. **corridor 重复**。Seed17 的八个确认 anchor 到 dev 最近距离全部为 0m；seed23 多数到 dev 在 90m 内；seed41 也有重复或邻近。物理 corridor 应为统计聚类单位，seed 是随机重复不是独立场景。v8 holdout 有 >=90m 空间排斥；不同 Town 更稳妥。若 Town05 找不到有效 corridor，要保留失败并另报 town，而不是后验降低门槛。

### 进一步有效性边界

- `mechanism_executed` 不隐含 `valid=true`。必须同 trial 满足无 ego 碰撞、初态/速度/进展匹配、合法道路与真实入道。
- 原 v7 dev BehaviorAgent 有 9 次 `AttributeError`（路线局部 `NoneType.is_junction`），confirm 某些 seed 有 RPC 失败。异常是删失，不能被算作安全负例。
- CARLA `traffic_light_legality_verified=false`，不能使用现有 intersection 的红灯触发运动代理判断法规合法性；需要停线、灯相、地图 conflict polygon 和路线弧长。
- 目前是 **后验评测排序**。即使修复后出现风险翻转，尚不能声称 GameFormer/RACP/强 planner 在推理时选错，直到 rollout 前锁定 choice。
- Continuous braking (`burden`) 的非零差异值得保留为预注册 v8 **secondary endpoint**；旧论文主端点碰撞/持续急刹/短 TTC 保留，不可事后替换。

## nuPlan 不是一个“重下即可”的单问题

你现有 8732 个本地 DB 的四城地图链接可见，但 ReactSim 官方 val 2636 个场景在 1984 个日志中，按文件名查 1036 个不同日志不存在，影响 1501 个场景（其余 1135 个仅表示数据库文件存在）。缺失主要集中于 2021 年 5/6/7 月，分别 58/557/421 条日志。这个时间分布强烈提示本地档案或版本覆盖不同，而不是单个损坏。`RIDE_NUPLAN_MISSING_LOGS_20261028.txt` 是最小核查清单，并非已核实的官方可下载对象清单。

前次 `2cdb26865dfa5261` `FILTERED_SCENE_BOUNDARY` 则说明 **某个 DB 和 token 均存在**，但查询器过滤。必须核对 ReactSim 作者 ID 文件的 hash/commit、nuPlan devkit 版本、scene 边界筛选、原始 DB 时段。重新下载相同字节通常不解决它；但若当前数据库是版本或截断不一致，正确来源的 DB 也可能改变场景边界，需要核对实际归档完整性，而不是无条件断言永不重下。不能绕过完整性查询还称官方 benchmark。

分步处理：`22_reactsim_db_coverage.sh` → `27_v8_nuplan_recovery.sh` → 查其他原始数据根目录/官方 AWS 归档清单与大小 → 仅增量补齐确认属于同版本的缺失归档 → `17_reactsim_preprocess_checked.sh inspect-val` → 分类 raw-token/boundary/SQL join/校验错误 → 只有全部满足官方条件才做 2636 完整验证。研究小子集由 `28_reactsim_research_subset.sh` 在真实 nuPlan 查询下产生、保存原官方列表 SHA256、明确不官方且有选择偏差，可做技术 smoke，不可当正式泛化实验。

## 下一批实验的预先注册 endpoint/stop rules

### M0：控制协议有效性（必须先满足）

每个响应 family 的 `cruise` 负对照无 ego 或 A–B 异常碰撞；比较同一初态三车位置误差 <=0.5m、初速度误差 <=0.5m/s；目标动作入道且驻留；每组两个 action 均无 ego 碰撞和离道；ego 终点 progress 误差 <=3m；TM 与 Behavior 的后车 **cruise** 控制须接近同一物理速度标定（v8 报告持续 >=0.5s 超 1.5× 名义速度的异常）。记录 100% aborted/invalid，不选择性删除。

### RQ1：区别真实决策临界与纯模型误差

每个 corridor 预先固定两个 action 对：`early_slow/late_slow`, `mild_early/mild_late`；相同候选执行轨迹应足够接近（源码 `same_ego_execution` 给出位置/速度阈值）；至少两类模型且排序严格相反、不计平局；报告模型可靠性与差异，按 corridor 聚类重采样。先在 CARLA 找机制，但正式至少需要一个模型结构不同的 nuPlan reactive evaluator 或按 WOMD 观察有效性校准。最终必须有 ex-ante planner 把候选选错的记录，不能只计算后验 risk key。

### RQ2：任务匹配、干预依赖和风险可避免性

A/B 需要同 road legality、无 ego collision、差 <=3m、固定 follower_a/follower_b 和同一状态、双方完整 trace。主指标保留第三方碰撞、持续急刹和短 TTC；次级是第三方 B `burden(action)-burden(cruise)` 的累计量、强制退让开始相对横移指令的时差、以及负担/进展 Pareto 前沿。`cruise` 是背景负对照，不是同任务替代；若在 ego 发出横移命令以前就发生负担，要作为背景或 anticipation 并显式解释。

### RQ3（可以预部署，不可报告成立）

先在已存在的 nuPlan DB 上分别复现 Bosch SMART 与原生 GameFormer-Planner，记录 Git/权重 SHA/环境锁；但它们当前**不能作为两个 Planner baseline 主表**：前者是响应环境，后者是完整 learned planner。GameFormer-Planner `run_nuplan_test.py` 会无头环境中启动交互 NuBoard，v8 提供只抑制 GUI 的 runner 副本（默认无修改作者源码）；源文件无作者预训练权重，必须取得精确权重或重新训练后注明非官方训练设置。Bosch README 要求在隔离 nuPlan-devkit 拷贝中替换 `nuplan/planning`，不能覆盖已有工程。评估矩阵：planner=(IDM/hard courtesy/selector/GameFormer-Planner) x environment=(nuPlan IDM/Bosch SMART/ReactSim SMART/MTR)，先保证 `scenario_token`、初态、候选动作与控制接口可对齐。SMART 在 Bosch/ReactSim 之间共享模型家族，不是两个独立行为架构；MTR 增加结构多样性。

### 研究路线转向（不能因为故事需要而强行筛选正例）

- 多独立 corridor、同控制协议的完整配对和多个 learned response environment 下，**没有任何严格排序翻转**：降低模型诱导系统性选错这条主张，转向可决策的响应证据/行为可恢复性检验，停止扩大搜索直到偶然撞上“正例”。
- 排序不翻转但连续负担有可重复正的干预增量：可研究 **progress-conditioned third-party externality sensitivity**，但必须先击败 hard courtesy/FeAR/社会风险约束，不能称发现他车代价的新概念。
- 有显著有限响应库 `UNKNOWN→verified feasible` 正向见证，但独立响应风险现象弱：考虑 **Decision-Critical Behavioral Recourse Evidence**（三值证据 verified feasible / certified infeasible / UNKNOWN，在计算预算下减少错选），明确对比 PDT、RACP、HYPE、decision-aware uncertainty；没有可验证的决策差异则不足以成为新贡献。
- 仅 CARLA TM/BehaviorAgent 发生变化，SMART/MTR 复核失败：结论限制为仿真器-控制器偏差，而不是“真实混合交通强规划器系统性失败”。

## 使用数据的定位

- WOMD v1.3.1 validation/interactive 已可用于开发和 observational calibration；不因目前 RQ1/RQ2 门禁问题新增下载 training/test。WOMD logged ego future 不等于 do(ego action)。
- CARLA 控制场景只用于高可解释的 M0/干预机制筛查；S1 cut-in、S2 convoy 是当前可操作物理机制；S3 signalized intersection 必须有停线/路权验证；S4 support-gap 是 S1 型道路上的有限证据协议，非第四条独立道路拓扑。
- ReactSim-Bench 官方 2636 是非日志 AV 干预的 stress benchmark，对 RQ1/RQ2 独立响应比 self-defined WOMD 最有价值；不能从 stress 集合估计自然道路发生率。
- Bosch SMART-reactive 可提供第二个测试平台；不要将相同 SMART 学习器两次实现误报完全独立模型。

**结论**：当前 RQ1/RQ2 未通过，确认集由于速度和控制协议错误无法判定趋向，开发集 `9 对 / 0 翻转` 是局部负面信号；下一阶段是 **修复速度—冻结同协议—新 corridor—独立 learned environment—正式 baseline choice**。先把 M0 与样本分母修好，再决定 RIDE 算法是否值得设计。
