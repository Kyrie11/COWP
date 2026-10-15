# COWP EICDRR V4：科学判决、修复说明、预注册实验协议

## 1. 输入证据（仅来自用户上传的 h2_gate_v3(1).zip）

- `waymax_idm_route_policy_8s.json`: attempted=22，complete=0，errors=22，其中 18 次“Waymax SDC position disagrees with native cohort”，4 次“waymax.env has no MultiAgentEnvironment”。这不是没有失败场景，而是评测完全未运行起来。
- `observational_fit_focal.json`: 320 次尝试、308 个有效。近距离运动分层 n=272：Cautious/CV 2 秒 ADE 差 +0.2267 m（配对 bootstrap 95% CI [0.1284,0.3331]）；Heterogeneous/CV 差 +0.1922 m（[0.1031,0.2840]）。这些 IDC/IDM 模型未通过与简单 CV 的观测拟合比较，不能当作人类干预反事实真值。
- `fixed_prefix_4_to_8.json`: 22/22 有效、前缀控制相同。至少一个配置严格正例在 4s 与 8s 均出现的场景 3 个；按配置相同动作与同车辆 pair 的交集：Cautious 0、Assertive 4、Heterogeneous 1。依然是同一简化物理及响应家族的结果，不能外推到现实。
- `validation_interactive_inventory.json`: 150/150 个**文件名和文件大小**检查通过，共约 78.0 GB；不代表已经解码或执行 interactive 反事实试验。
- 本轮压缩包未包含 `ttc_rule_dev.json`、CARLA 执行结果、真实 GameFormer/RA-CBF 配对结果或 H1 joint witness；“没运行”和“运行得到负结果”必须分开。

## 2. 当前结论

H2 现实意义上未确认、也不能被本轮否定。三问：
1. 原生地图/有效控制下的 WOMD 正例：**未测**（Waymax 0/22 完成）。
2. 同一原始他车 ID 对，在两个独立驾驶策略中同干预复现：**未测**。TTC/IDM 同类物理不够；CARLA TM/BehaviorAgent 可作为真实 CARLA 物理的两套策略，但仍非人类反事实，亦非直接 WOMD 回放。
3. GameFormer/RA-CBF 等强规划基线是否选坏动作且存在任务匹配的更优替代：**未测**，需固定共同动作接口与目标，不能用 oracle 事后排序冒充 baseline。

三个性质分别为：Joint feasibility 尚无 SAT witness / bounded UNSAT 证明；Decentralized realizability 尚无合格跨策略响应概率评估；Ego-conditioned preventability 仅有简化模型候选。H1、H2、H3、H4 均未被严格关闭。

## 3. 本次实施的修复

1. `research_probe/waymax_paired_idm.py`：原逻辑在 env.reset 前比较 simulator current state 与 WOMD current 坐标，可能混淆 warmup 时间（官方 EnvironmentConfig.init_steps=11）；改为先校验原始 TFExample 当前 SDC 与 NPZ，然后在 env.reset 后比较模拟 SDC。禁止跳过、平移或静默修正不匹配。适配 `env.BaseEnvironment`（保留旧版 MultiAgentEnvironment 分支）。**未经用户服务器上的 Waymax 实测**。
2. `research_probe/carla_h2/run_carla.py`：CARLA 0.9.15 同一地图、车道、种子和三车物理启动点的可重复场景；ego 由 PID 实际控制并报告速度跟踪误差；非 ego 分别使用 Traffic Manager 和 BehaviorAgent。每个分支重新加载世界并重置随机种子。基线只有行为策略，不是 GameFormer/RA-CBF。
3. `research_probe/carla_h2/protocol.py`：早减速后补偿与晚减速后补偿的目标速度积分相等，但**实际执行后还需要严格进展匹配**（默认 ±3m）；单独记录第三方碰撞、负担减速度、短 TTC，不把风险代理当真实碰撞。
4. `research_probe/carla_h2/analyze_carla.py`：要求同起点、同 seed、同动作/车辆对、真实 ego 进展配对；跨策略比较再次要求初始物理状态匹配。有限 joint candidate 搜索仅能构造正向 SAT witness，空搜索结果只应给 UNKNOWN。
5. H3 pilot：同一 ego 固定前缀到 3.5 秒后，分别执行“停车/恢复巡航”备份；验证所有角色在前缀终点的位置和速度匹配；通过的备份是“此有限支持中存在恢复 witness”，失败不能证明真正的不可恢复。

**测试**：在当前分析容器中可运行的 84 项 pytest 已通过；因无 CARLA 服务端和原始 WOMD，此交付没有伪造 CARLA 或新 Waymax 成功结果。

## 4. CARLA 与 WOMD 的作用边界

- CARLA 的 `Town04` 直道路三车跟驰/制动传播是**明确设计的压力场景**，不是迁移 WOMD 真实道路几何，也不能用人工构造碰撞宣称 WOMD 中这种事件频繁。
- CARLA TM 与 BehaviorAgent 代码结构不同，有独立策略，但共享 CARLA 的地图、动力学、检测与交通约束。故“同结构独立响应模型”严格来说是**两套不同软件策略，共享同一物理模拟平台**。多随机种子、不同天气与地图仍需进一步评估。
- WOMD validation_interactive 的 150 个分片文件存在，建议优先对它进行场景角色审计：判断目标交互 pair 是否实际包含 SDC，并与普通 validation 按 scenario_id 去重。把此富集集合单独报告，不能与自然频率混算。
- CARLA 里的 joint_witness 是由有限 2 车目标速度组合构成；通过后得到同 ego 干预下存在一个物理安全响应的**正向证据**。不能从未搜索到解推出真实驾驶者无解；不允许用学习 proposal 的未发现直接作为 UNSAT。

## 5. 假设优先级与科学止损条件

**Gate 0：评测可信性**
- Waymax 先跑 1 条。需成功完成、原始 SDC ID/当前位置匹配、地图指标无异常；如果失败先只修复接口。
- CARLA 先跑一个 Town04 anchor、一组 seed、三个预注册的 ego profile、两套 responder；任何错误、缺失道路状态、初始状态不一致必须报告 failure，不可算阴性场景。

**Gate 1：H2 模拟可避免风险（近期核心）**
- 在 development 场景集固定 action profiles，按相同 seed 配对 CARLA TM 与 BehaviorAgent。
- 首要严格终点：同一第三方车辆 pair 的物理碰撞在 bad action 存在、good action 不存在；两个 ego 分支均无碰撞且合法，实际路线任务位移差 <=3 m。次级终点：硬制动、短 TTC、其他不稳定响应证据。不得拿次级终点替换严格主终点。
- 需要保存每条轨迹、实际控制、碰撞传感器、角色 ID、地图和异常；报告全部 denominator 和配对区间，而不只展示挑出的案例。
- 仅 CARLA 正例 -> **CARLA 物理压力案例**；原生 WOMD Waymax 正例和另一结构模型成立 -> 可研究 sim-to-WOMD 的外部一致性。**仍非真人因果证明**。

**Gate 2：H1 正向联合可行**
- 固定 bad ego action 后，搜索 responder target speeds，并完整执行物理仿真。
- 若找到无碰撞、合法、无异常急刹且角色进展限制满足的轨迹 => model-bounded SAT witness。没找到 => UNKNOWN。
- 更强的 decentralized gap 应要求**相同 ego action**下 joint witness 安全而独立策略有失败，并对空间/任务/时间做公平限制。

**Gate 3：H3 可恢复性**
- 同一前缀执行后的所有 agent 位置/速度一致；搜索停车/巡航/更多 backup 控制的后缀并验证物理约束。
- 仅有 backup support collapse 不构成递归不可行证明；需要更宽支持、鲁棒可达集下界或边界证明才能声称不存在恢复策略。

**Gate 4：H4 规划器方法**
- 当有稳健 H2 模拟正例后，再在共同 action proposal、完全同一反应策略、相同算力条件下执行 COWP、social-cost/hard courtesy、GameFormer adapted planner、RA-CBF adapted controller 与物理安全基线。
- GameFormer/RA-CBF 的移植结果必须标明是自行适配，未训练 checkpoint 时不得冠以作者官方结果。

**研究止损**：如果 Gate 0/1 完善并在交互富集场景和独立策略上仍找不到稳定、任务匹配的外部化风险，停止宣称 ego-induced third-party collision 的普遍现实失败。选择更可验证的*action-conditioned response uncertainty causing unqualified planning decisions*，直接评估决策排序翻转、预测到实际响应的不一致、证据缺乏下的过度置信与选择性拒绝。

## 6. 论文定位：暂时保留上层问题，弱化 Same-root

建议目前以：

> **When is an interactive planner justified in relying on another agent's safe response under model disagreement and incomplete recourse support?**

作为科学问题的可证伪版本。

这是“证据资格”的问题，不是已证明新的真实驾驶失败。核心卖点若要到高水平，需给出：真实系统/物理模拟发生的未认证依赖风险；明确的证据分级（existence / bounded exclusion / unknown）；高价值的可计算证书或可靠性估计，以及相同任务下的闭环改进。局部 normal-root/same-root 可作为条件约束和消融变量，非现象首创。

如果后续 H2 真正闭合，升级为 *intervention-conditioned decentralized response realizability with evidence-calibrated recourse*；如果没有，主线转向 *decision-relevant reliability of finite reaction certificates under model uncertainty*。后者依然需要系统失败正例，不能仅凭 CV 比 IDM 更好就宣称论文贡献。

## 7. 数据伦理与科学表述

自然 WOMD 记录中的实际行为不是相同情境的随机化 ego 干预实验，不支持直接估计真实司机的 do-effect；CARLA 是软件反应代理，不是自然人；有限 candidate 集没有反应不代表所有合法响应都不存在；scene-selected development 反例比例不能报告成现实事故频率。明确保存 seeds、地图、车辆 IDs、异常与失败试验总数，并冻结独立 holdout 前所有参数。
