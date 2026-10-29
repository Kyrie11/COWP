# RIDE v14：第三方反事实敏感性、次生风险及规划选择的科学审计与实验手册

版本：2026-10-29。**先读科学结论，再执行代码。** 本版本在用户上传的 v13 完整源码上继续开发；此前的 v7–v13 文件保留为历史实验。禁止把不同协议/不同道路的结果无条件合并。

## 一、最终研究定位

**暂定工作名：RIDE — Response-Model-Induced Decision Errors in Interactive Driving。**

初始故事的核心不是 `predictor ADE`、也不是单纯 courtesy 或有限库 `UNKNOWN`：在异质、无共同响应协议的混合交通中，规划器会基于不完整/失准的他车响应假设选择 ego 动作；一个对 ego 自身安全的候选，可能对后方固定车辆链条造成可避免的负担和安全风险，尤其当响应模型将两个候选的第三方反事实后果压缩成假平局或错误排序时。

建议顶层问题：*When and how does underestimated counterfactual response sensitivity of downstream third parties induce avoidable errors in interactive AV planning? Can it amplify into secondary safety hazards under independently controlled heterogeneous traffic?*

形式化：固定初态 s、任务进展相近的两个合法 ego 动作 a,b、固定第三方 C 和响应模型 M、独立执行环境 E。

- `D_M = L_C^M(a) - L_C^M(b)`，`D_E = L_C^E(a) - L_C^E(b)`；反事实差分失配 `epsilon_CF = D_M - D_E`。
- 第一种失败：`abs(D_M) <= tau`，但 `abs(D_E) >= delta` 且选错动作；第二种：`sign(D_M)` 与 `sign(D_E)` 相反且选错；第三种：模型把超过第三方风险/制动预算的动作误接收，低风险替代仍可完成近似任务进展。
- 这些只是**必要中间机制**。只有事前冻结的选择加上独立执行中显著更差的结果，才叫诊断性选择错误；CARLA 行为模型不能冒充人类真实反事实。
- 负担量 `L_C^comfort(a) = ∫_{t≥1s} max(0,-accel_C(t)-2m/s²)dt`，单位 m/s。它是舒适负担代理，**不是事故发生概率**。
- 次生安全结果单独报告：B–C 的实际角色确认碰撞；B–C 最小 TTC 与其适用帧数；C 严重制动连续持续时间；按限定独立制动响应计算的安全裕度（仅条件性理论屏查）。

**三个新的研究层次：**

1. `M0 / M1`：初始状态一致、所有动作可物理执行、ego 不碰撞、真正并线，ego 改变 B 的行动且 C 的响应有合法可归因证据。B 先变、C 后变只是中间代理，不证明 B 的介导效应。
2. `RQ1 + RQ2`：反事实敏感性失配 -> 事前选择错误 -> 任务匹配的、可避免的 C 负担，随后考察是否进一步放大为严重制动、短 TTC 或 B–C 次生物理碰撞。三种危险严重度分层，不得互相替代。
3. `RQ3`：强 courtesy B-only/B+C、多响应风险规划、真实 HYPE、GameFormer-Planner、RACP 在公平可见信息、候选池、预算下是否还有剩余错误。透明 RIDE source selector 不是官方强基线。

## 二、直接读取三份 v13 包得到的结果

| 指标 | Behavior→TM Seed17 | Behavior→TM Seed107 | Behavior→Behavior Seed89 |
|---|---:|---:|---:|
| 原审计 `valid_comparisons` 行数（两种选择规则） | 12 | 22 | 8 |
| 不重复的**物理动作对** | 6 | 11 | 4 |
| 来源模型诊断性选择错误，`SOURCE_THIRD_PARTY` | 1 | 2 | 0 |
| 出错物理 anchor 数 | 1 | 2 | 0 |
| 观测 B–C 物理碰撞 | 0 | 0 | 0 |

**重要去重：** 原报告 Seed107 `model_blind_distinctions=4` 与 `strict_wrong_model_orders=2` 是按两种透明选择策略重复记录；不能称为 4 个独立平局失配和 2 个独立严格错误排序。唯一动作对口径分别是 **2** 个模型遗漏差分与 **1** 个严格连续负担方向错误。Seed17 的两条模型差异记录对应 1 个动作对。原报告总行数不是样本量。

具体的模型相关选择案例：

1. `Seed17/Anchor3 speed_early vs speed_late`：source C 负担分别约 **0.1961 / 0.1945**，target 约 **0.5026 / 0.9627 m/s**。来源近似平局，选择了 `speed_late`，真实目标环境中的替代动作可减少约 **0.4601 m/s**。有任务匹配的模拟实证，无第三方危险 TTC 或碰撞。
2. `Seed107/Anchor4 speed_early vs speed_late`：source 约 **0.200 / 0.092**（偏好晚动作），target 约 **1.901 / 4.243**（早动作显著较优），失配更强。但 C 在 4.50–4.55s 记录过 **-20.56、-22.76 m/s²** 极端纵向加速度；应排除控制器/车辆物理瞬态后才能作为主结果。对每帧加速度截断到 ±6m/s² 后，候选成本差仍约 0.776 m/s；中值滤波后约 0.702m/s，均属**事后鲁棒性诊断**。
3. `Seed107/Anchor5 early_slow vs late_slow`：target C 负担约 **1.900 / 2.104 m/s**。按 4m/s²、持续 0.2s 的二元硬刹指标，前者持续约 0.15s，后者约 0.20s。改为 3.5m/s² 两者都触发；4.5m/s² 两者都不触发。该风险代理临界，B–C 最小 TTC 仍较大，不是严重安全失败的坚实证据。

Placebo Seed89 **不是同一道路 / 同一 seed 的严格负对照**。当前已观察到的是 CARLA TM/BehaviorAgent 间第三方反事实负担估计可能失配，发生诊断性选择错误；**没有**证明真实人类行为的风险，更没有证明次生 B–C 碰撞或 GameFormer/RACP 失败。

## 三、升级安全问题之前必须明确的负面结果

对 Seed107 的 42 条目标 TM 动作轨迹，用四车 B–C 间距、当前 B/C 速度以及近似常减速度做**条件性停止裕度**筛查：

- 当 C 响应延迟 0.6s、C 最大减速度 3.5m/s²、B 假设减速度 5m/s² 时，42 条动作均无模型屏查定义的负停止裕度；实际碰撞数始终为零。
- 当仅把 C 假设响应延迟增至 0.9s，约 24/42 条动作出现**假设的**负停止裕度；这一结果**不是**实际 CARLA 碰撞，也不能证明哪条任务匹配候选一定优于另一条，因为许多负裕度同时出现在两个候选中。
- 一般而言，增加后车数量、延迟或密度**不保证**风险单调增加：安全间隙、协同减速及不同控制器的制动策略可能抑制传播。复杂交通增加了可能的交互路径，但必须用受控实验检验具体的条件分布。

因此，“反事实敏感性失配导致次生安全失败”是待验证的**有条件命题**，不能把延迟设大到碰撞就宣称真实交通存在该缺陷。

## 四、v14 具体新增模块与界限

- `ride.v14_counterfactual_audit`：只读重新计算锁定选择/复现哈希，按唯一动作对计数，识别 >10m/s² 极端加速度，报告截断 ±6m/s²、±5m/s²、中值滤波、4m/s² 硬刹阈值邻域的事后敏感性。不能替代预注册结果。
- `ride.v14_frozen_replay` + `scripts/42_v14_replay_frozen_two_stage.sh`：使用**完全相同物理道路几何**的 v13 冻结 anchor，在新输出目录做同 seed 同道路、不同响应模型/同模型 placebo 的前瞻性选动作。改 seed 仅改变随机设置，不产生独立物理道路样本。
- `ride.v14_safety_envelope`：车距/制动常减速度假设下的条件性停止裕度，不是形式化 RSS 安全保证，也不是 CARLA 物理碰撞。
- `ride.carla.lagged_controller`：仅延迟 C 的**制动命令**（不延迟转向或油门），可设置制动命令上限。此为**人为控制器能力消融**，不是经过人类标定的反应时间模型。
- `scripts/43_v14_safety_stress_carla.sh`：CARLA 同一批冻结场景并行跑三组：TM 全体、B 为 TM/C 为正常 BehaviorAgent、B 为 TM/C 为延迟制动 BehaviorAgent。比较 C 延迟效果时**必须用后两者**（相同 C 控制器）比较；TM 全体仅作模型家族背景参照。
- `ride.v14_safety_stress_audit`：保留无效动作/异常；真实碰撞只承认带精确 B、C 角色的碰撞传感器事件。判定“动作可避免的次生碰撞候选”还要求同响应模型、同初态、ego 进展匹配、`cruise` 负对照不碰撞。另检查两个 C 控制实验中 ego 与 B 轨迹是否也保持一致。
- `scripts/45_v14_locked_source_to_safety_stress.sh`：新增的**事前完整响应证据选择 → 锁定 SHA → 延迟 C 目标执行**协议。来源 B=TM/C=正常 Behavior，目标 B=TM/C=同一 Behavior 但仅延迟/限幅 C 制动。它直接检查受控 C 反应差异是否产生规划选择损失。Source selector 仍是透明模拟器评分器，不是 GameFormer 或 RACP。

上述 CARLA 新脚本在交付环境中**没有运行 CARLA 服务器实证**。本地验证覆盖 Python 编译、shell 语法和离线单元测试。若闭环仍出现 waypoint、reset、异常加速度问题，必须保留 `UNKNOWN`，不能用异常轨迹作正例。

## 五、服务器运行：环境与原始证据复算

请解压完整 v14 源码到：

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip -q RIDE_V14_COUNTERFACTUAL_SAFETY_20261029.zip -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V14_COUNTERFACTUAL_SAFETY_20261029
conda activate mfrp
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m compileall -q ride
bash -n scripts/*.sh
python -m ride.doctor --server
```

CARLA 0.9.16 须无头服务已经运行。验证当前世界地图，只有你独占 server 时才能切地图：

```bash
export CARLA_TOWN=Town04
RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04
bash scripts/29_v9_carla_preflight.sh Town04
```

如要从解压后的 v13 三份证据重新计算 v14，请设置**实际的解压目录**；例如：

```bash
export RIDE_V14_AUDIT_ROOTS='/path/to/v13_dev_behavior_to_tm_seed17 /path/to/v13_new_confirm_seed_107 /path/to/v13_placebo_behavior_seed89'
export RIDE_V14_AUDIT_OUTPUT=/data0/senzeyu2/dataset/RIDE/evidence/v14_v13_forensic.json
bash scripts/44_v14_readonly_audit.sh
```

### A. 最优先：同一冻结 anchor 的严格重复与同模型 placebo

复用已经出现失配的 **Seed17/Anchor3** 和 **Seed107/Anchor4/5**，进行以下配对实验。所有重复使用相同道路几何与同 seed 开始；随后在单独目录尝试其他 seed（随机性检验）。不要把 anchor3/4/5 当作新的 holdout。

```bash
# 同场景、同 seed 的 BehaviorAgent -> BehaviorAgent
export RIDE_V14_ORIGINAL_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/v13_dev_behavior_to_tm_seed17/freeze/frozen/chain_cut_in_anchors.jsonl
export RIDE_V14_ANCHORS='3'
export RIDE_V14_SEED=17
export RIDE_V14_SOURCE_FAMILY=behavior
export RIDE_V14_TARGET_FAMILY=behavior
export RIDE_V14_PLACEBO=1
export RIDE_V14_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_anchor3_behavior_placebo_repeat1
bash scripts/42_v14_replay_frozen_two_stage.sh

# 同一几何/seed，跨模型再做一次 BehaviorAgent -> TM
export RIDE_V14_TARGET_FAMILY=tm
unset RIDE_V14_PLACEBO
export RIDE_V14_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_anchor3_behavior_tm_repeat1
bash scripts/42_v14_replay_frozen_two_stage.sh

# 不同几何/seed 的 Seed107 两个 anchor，重复相同测试
export RIDE_V14_ORIGINAL_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/v13_new_confirm_seed_107/freeze/frozen/chain_cut_in_anchors.jsonl
export RIDE_V14_ANCHORS='4 5'
export RIDE_V14_SEED=107
export RIDE_V14_TARGET_FAMILY=behavior
export RIDE_V14_PLACEBO=1
export RIDE_V14_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_anchor45_behavior_placebo_repeat1
bash scripts/42_v14_replay_frozen_two_stage.sh
```

下一步对同一 anchor 重复 5 次**每次创建新 `RIDE_V14_ROOT`**，且分别做 `behavior→tm`、`behavior→behavior`、`tm→tm`（后者需 Source=Target=tm + placebo=1）；比较 source-target 后果差分的随机性、极端加速度和有效样本删失。禁止复用既有输出目录，不得把不同协议混做单一统计试验。

### B. 先做响应控制器隔离的物理压力测试

**必须先做零延迟/无制动限幅控制组**，然后预注册延迟与制动上限的逐因素实验，而不是只汇报能够撞车的设置。每个 cell 建新目录且固定其他参数。下面为四格开发筛查，数值是**人为实验条件，不是人类实测反应时间**。

```bash
export CARLA_TOWN=Town04
export RIDE_SEED=171
export RIDE_ANCHORS=2
export RIDE_WORLD_RESET=actors
export RIDE_EXCLUDE_ANCHORS_FROM='/data0/senzeyu2/dataset/RIDE/evidence/v13_dev_behavior_to_tm_seed17/freeze/frozen/chain_cut_in_anchors.jsonl /data0/senzeyu2/dataset/RIDE/evidence/v13_new_confirm_seed_107/freeze/frozen/chain_cut_in_anchors.jsonl'
export RIDE_EXCLUDE_RADIUS_M=90

# 基础 C 行为：无延迟且不限制刹车
export RIDE_CHAIN_C_CONTROL_DELAY_S=0
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=1.0
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_base_seed171
bash scripts/43_v14_safety_stress_carla.sh

# 单独测试 C 制动命令的 0.3 秒延迟
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=1.0
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_delay03_seed171
bash scripts/43_v14_safety_stress_carla.sh

# 单独测试 C 制动命令限幅
export RIDE_CHAIN_C_CONTROL_DELAY_S=0
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_brake085_seed171
bash scripts/43_v14_safety_stress_carla.sh

# 交互项：延迟 + 限幅
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_delay03_brake085_seed171
bash scripts/43_v14_safety_stress_carla.sh
```

重点查每个 cell 的 `carla/v14_physical_stress_audit.json`：是否有 `AVOIDABLE_SIMULATED_BC_COLLISION_CANDIDATE`、是否有 `BACKGROUND_BC_COLLISION_NOT_EGO_ATTRIBUTABLE`、`same_controller_C_vs_lagged_C_parity` 的 `initial_match/ego_execution_match/B_execution_match`。再查 `carla/v14_hypothetical_stop_margin.json`，其中负裕度仅用于选下一轮压力条件，不是撞车数据。

### C. 如果需要直接检验“失配导致事前错误选择 → 次生安全”

这个协议比直接跑碰撞压力网格更贴近 RIDE。来源模型用 `A/B=TM，C=原始 BehaviorAgent` 完成 lookahead，**先锁定动作**，之后 `A/B=TM、C=延迟/限幅制动的 BehaviorAgent` 执行完整候选。来源和目标保证相同冻结几何、控制器除了 C 的制动命令外保持不变。例子：

```bash
export CARLA_TOWN=Town04
export RIDE_SEED=173
export RIDE_ANCHORS=3
export RIDE_WORLD_RESET=actors
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
export RIDE_V14_LOCKED_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v14_locked_c_stress_seed173
bash scripts/45_v14_locked_source_to_safety_stress.sh
```

查看：`v13_target_decision_audit.json`（事前冻结选择对后果的错误判别）、`v14_readonly_physics_audit.json`（加速度和成本稳健性）、`v14_physical_safety_audit.json`（实测 B–C sensor 和负对照）、`v14_hypothetical_margin.json`（仅假设屏查）。

即使出现物理 B–C 碰撞，也必须同时满足：

- 初始状态和路权/道路合法性完全一致，ego 两条动作实际不同且任务进展在预设 3m 之内，ego 自身没有碰撞。
- `cruise` 未出现同一 B–C 碰撞；低风险替代在**相同目标环境**中没有碰撞，不允许仅比较不同 family 的两条动作。
- 真实碰撞传感器的车辆角色精确是 B–C，排除初始化、地图和控制器切换异常。
- 来源模型按锁定规则原本确实选择高风险动作；C 制动延迟的对照相对正常 C 控制器是唯一主要处理差异，B/ego 轨迹的一致性检验通过。
- 若要进一步称为 `ego -> B -> C` **因果介导**，需要追加动力学合理的 B 响应受控重放 / 克隆干预：固定 ego 和 C 初态，只改变 B 的自然可行制动轨迹，对照 C 反应。观察到 B 先减速、C 后减速不够。

### D. 后续复杂车流实验：按机制顺序扩展，而不是先增加车辆到 10 辆

仅在四车中证明可避免的实测危险端点后，冻结一个 5/6 车排队版本，保持相同相邻车距/名义速度、驾驶员反应分布和 ego 动作池，单独改变车辆数；比较固定最远后车风险、任何车对风险、每车平均风险（分别报告）、事件传播深度与延迟。车辆数越多，`any collision` 的基数自然上升，应控制基数和暴露量；**不能把这种基数效应当成反事实敏感性随密度增强**。

## 六、论文转向与进入算法设计的停止条件

| 科学结果 | 允许发表的主张 | 下一步 |
|---|---|---|
| 稳定虚假平局/连续负担选择损失，无危险端点 | 反事实响应敏感性失配导致可避免负担，但未证明安全事故 | 需跨道路、独立模型验证；将 severe collision 移出主贡献 |
| 仅 TM 发生极端加速度或边界阈值事件 | 控制器/仿真器敏感性，尚不是坚实安全机制 | 修复车辆物理与重置，改变指标并在新开发协议预注册 |
| 合法任务匹配 B–C 近失或物理碰撞，负对照安全且同模型 repeat 稳定 | CARLA 条件下存在可避免次生安全失败 | 增加 B mediation 对照、SMART/MTR 独立响应和强 planner 选择 |
| 只在通过人为延迟/刹车限幅设置时撞车，无观测合理性支持 | 存在受控物理机制，不是现实道路系统性失效 | 用 WOMD/nuPlan 跟驰数据校准参数；不能直接将应力测试当主证明 |
| 多模型多道路有效样本中的敏感性差分和安全后果都消失 | 当前 RIDE safety-first 论断缺乏支持 | 转为决策相关第三方响应误差/负担，或终止故事而非调参制造事故 |
| 强 courtesy B+C 或风险多响应选择器已经完全避免错误 | 尚无“强算法仍遗漏”的 research gap | 收窄到公平基线确实没处理的证据验证预算问题，或放弃新方法 |

初步建议使用新冻结确认集 ≥20 个独立物理道路簇，至少 3 个 seed 与至少一个独立 learned-response 环境来估计不确定性；这不是“满足即证明 CCF-A”的神奇数值。以道路簇为 bootstrap 单元，保留所有无效/UNKNOWN，展示 ego progress–risk Pareto 曲线。不把一条走廊的多个 seed 当独立交通故事。严重碰撞为稀有事件，样本设计还需按发生率和置信区间作专门功效分析。

ReactSim-Bench 官方 2,636 个干预场景、SMART/MTR 是对真实数据路网的独立 learned-response 评估工具，不是反事实人类真值；结合既有 nuPlan DB/map 先复现作者官方推理再适配两动作同车辆身份协议。没有必要为 v14 先下载新的 WOMD 大规模训练分片。真正强规划基线接入前，先实现充分调参的透明 hard courtesy（仅 B、B+C）、risk-aware multi-response 两种选择器，然后在统一输入/预算下对官方 GameFormer-Planner、RACP/HYPE 做事前选择比较。**没有官方复现结果不能声称它们系统失败。**

## 七、科学诚信与运行安全

- 这里的延迟制动控制器仅用于隔离响应能力这一自变量。不要模糊成模拟真实驾驶员 0.3 秒反应延迟；它没有调整感知、意图或跟车安全边界，且制动命令比例不等同减速度。
- 只在独占的 CARLA 服务端运行、切地图；不覆盖旧 manifest，不在中途改变阈值/候选集合。
- `minimum TTC` 必须附同车道适用帧数；极端加速度必须连同发生帧、碰撞传感器、reset 状态公开；要报告原始和物理过滤后的两个结果。
- 停止裕度负数只是简化纵向模型的条件性脆弱性；真实安全失败只承认可复现、有有效对照的物理事件或明确定义的近失指标。
- v14 不修改历史 v13 审计数据本身，新协议产物全部另建目录。若 CARLA 服务器环境失败，本地测试通过不能替代真实仿真验证。
