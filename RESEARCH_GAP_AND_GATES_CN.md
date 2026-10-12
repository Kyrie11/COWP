# 科学研究问题审查与验证 Gate（2026-10-11）

## 1. 目前**没有**足以宣称新科学问题已成立的证据

本项目当前定位为：**在无共享未来计划的混合交通中，ego 干预是否会造成原本可接受的他车响应相互不兼容、独立执行时失配、或造成下一步安全退出丧失；这种失效能否被等任务成本的替代 ego 动作避免，并且能否超越已有的非集中式 contingency MPC 和责任安全约束解释？**

不是“首次研究无通信多车避险”，也不是“首次研究递归可行性”，不以有限响应库搜索失败宣称真实无解。论文第一步应检验该问题是否存在**方法无关、反应模型不敏感、足够频繁且具有规划决策价值**的实例。

## 2. 最高优先级先行工作与差异化风险

| 文献及链接 | 已有技术内容 | 本项目必须具体补证的内容 | 正面 baseline 角色 |
|---|---|---|---|
| Studt & Schildbach（2026.05；另有 2026.08 有限感知扩展 arXiv:2608.30874）, [Decentralized Contingency MPC based on Safe Sets (2026)](https://arxiv.org/abs/2605.10738) | 无共享规划轨迹、只看状态、多个非线性独立智能体、backup、递归可行性证明 | 异质人类/AV 非共享协议、责任和 burden、ego 动作条件下的实测风险；不能声称其不考虑未来安全 | **第一强基线，必须复现或与作者代码对齐** |
| Vielmetti et al., [R3R (ACC 2026)](https://arxiv.org/abs/2510.06436) | 异步去中心多智能体、通信约束、安全门控、无限时域保证 | 道路混合交通、私有意图与非同质协议下的 coverage/safety/efficiency | **第二强基线** |
| Cosner et al., [RA-CBF (ICRA 2023)](https://arxiv.org/abs/2303.03504) | 在真实驾驶数据上学习场景责任分配，不显式协调也能合成安全控制 | 责任分配是否足以保证实测联合响应/未来备份，不能先验假定不足 | 强社会-安全基线 |
| Sun et al., [Courteous Autonomous Cars (2018)](https://arxiv.org/abs/1808.02633) | 反事实 courtesy 增量与影响其他司机的行为成本 | 判断正常礼貌代价和联合安全失效之间的区别 | Soft/Hard Courtesy |
| George et al., [FeAR (2023)](https://arxiv.org/abs/2305.15003) | 行动空间压缩与因果责任 | 动作条件下的联合可执行性与时间演化增量是否独立 | FeAR-style baseline |
| Luo et al., [JFP (CoRL 2022 / PMLR 2023)](https://proceedings.mlr.press/v205/luo23a.html) | 解码阶段保证联合预测轨迹一致性，WOMD 交互评测 | 联合预测一致性 vs 动作条件安全备份和独立执行差异 | 强 joint predictor |
| Huang et al., [GameFormer (ICCV 2023)](https://arxiv.org/abs/2303.05760) | 交互式博弈与规划；不是简单的单车预测 | 与动作影响下的 recourse/备份对照 | 强交互规划 |
| Wabersich & Zeilinger, [Predictive Safety Filter (Automatica 2021)](https://doi.org/10.1016/j.automatica.2021.109597) | MPC safety filtering 及安全备份 | 必须区别于普通 ego safety-backup | Physical safety baseline |
| Liu et al., [Secondary risks from defensive braking (2025)](https://doi.org/10.1016/j.aap.2025.108176) | 真实问题动机：连续防御制动诱发后方次生风险 | 不可声称“首次发现 defensive braking secondary risk” | 场景出处/行为模型校准参考 |
| Gulino et al., [Waymax (NeurIPS Datasets 2023)](https://github.com/waymo-research/waymax) | logged replay, IDM, 多主体 rollout API | log replay 不产生真实反事实反应 | 评测平台/接口 |

**必须直接核查 Contingency MPC 的前提和保证范围，不应用“现有工作只优化 ego”概括上述文献。** 2026 先行工作的证明通常依赖同质的规则协议/感知约束；但是否因此存在实际 gap，必须通过异质反应和匹配任务表现的 baseline 实验证实。

## 3. 四个不同的真值层次

1. **Individual existence**：每名他车对固定 ego 轨迹，是否有独立的低负担可行响应。
2. **Joint existence**：这些响应是否可在所有他车之间同时满足碰撞/动力学/规则约束。Joint existence 是 offline oracle 证据，不是他车将协商执行的承诺。
3. **Decentralized realization**：他车各自依赖观测和私有意图、不交换未来计划时，实际是否能做出联合安全反应；需要独立校准的 reactive agents。
4. **Successor recoverability**：基于真实 controller emitted action 的后继状态，是否仍有符合约束的连续备份。当前脚本只实现候选第一段的简化动力学 proxy，未接 Waymax action projection。

数学定义见 `interactive_planning_research_revision.tex` 第 3 节。`Joint CSP feasible` 不推出 `Decentralized outcome safe`。此外，有限库非空可确证库中存在可行解；有限库为空**不能证明连续空间无解**。

## 4. 当前新脚本能够做什么，不能做什么

`research_probe.probe` 独立于 COWP witness/RCOT/BCOT 伪标签，使用从 tensor-cache 读出的 **current** agent states 和冻结的 ego candidates，给每名邻车生成固定航向下恒加速度轨迹，测试：

- `pairwise_joint_discordance`：每名 agent 都有单独的 ego-safe primitive，但离散联合 CSP 无解；
- `joint_but_local_conflict`：CSP 存在联合解，但每辆车不通信、独自选 minimum-|acceleration| 时互相冲突；
- `temporal_collapse_proxy`：CSP 可行、各车独立选择也相容，但按选定响应执行 0.8s 后，采样的 ego-backup / responder 候选失去可行性；
- `avoidable_collapse_proxy`：同时存在行驶距离差不超过 3m 的另一条候选，在相同 proxy 下可继续恢复；
- 三层 selector：`pairwise_utility`、`joint_existence_utility`、`discrete_successor_backup`，共享同样候选银行。

它采用简单圆盘碰撞和恒加速度控制，既不是 Waymax，也没有考查地图/优先权/驾驶意图多样性，因此它**只能作为筛查探针**，不是数据集中真实发生机制的证据。输出的 `positive_scenes` 不得写成真实事故频率。Toy synthetic selftest 更不能作为场景创新证据。

## 5. 预注册实验决策逻辑

- **Gate 0, integrity**：基础算法检查成功，scenario 来源/去重/划分/使用当前状态逻辑成立；cache 缺失、批量不能读取或 schema 错误 → **BLOCK**。
- **Gate 1, development discovery**：80 个未看过结果的 cache 文件，固定 3-neighbor、10-candidate bank，主响应加速度集合 `[-4,-2,0,1.5]`，后继 proxy 固定 0.8s；全部 4 类诊断逐场景保存。**不以正例数直接宣称 GO**。
- **Gate 2, proposal-family sensitivity**：同一批 ID，改为 `[-5,-3,0,2]`，用 `research_probe.analyze` 比较**共同场景**中持续出现的正例。如果只在弱 primitive bank 下出现，首先怀疑 proposal incompleteness。
- **Gate 3, geometric and motion validator**：对共同正例在服务器上补充 **Waymax 原始 action projection**、车体几何、路图、横向可行 primitive、更多非关键 agent，外加自主/预测控制器实际 emit 的首拍动作。计算真 joint CSP 与时序继续能力。只有跨更强响应模型维持的案例才能做机制分析。
- **Gate 4, independent behavior**：使用已校准 IDM / 其他 reactive agents、多种驾驶风格、同一固定 scene seeds，测真实反应和安全/行驶效率。不允许依赖当前 `logged_replay` 来宣称因果反应。
- **Gate 5, novelty test**：在相同场景、可获得观测、计算预算、车辆动力学和 matched-progress/collision-rate 下，**忠实实现** Studt & Schildbach（2026.05；另有 2026.08 有限感知扩展 arXiv:2608.30874） 的 decentralized contingency MPC 或说明前提差异并提供等价对照。若它已经解决全部失败，收窄题目到其覆盖范围外经实验确认的异质驾驶行为/动态责任需求；否则不可宣称新 gap。
- **Gate 6, publication**：最终算法冻结后换全新 selection-blind WOMD scenes，≥3 个独立 seed，paired confidence intervals 和人类标注样本，不能复用历史 exact200 开发集作最终结论。

**如果 Gate 2 的个体/联合差异几乎全是 proposal-bank artefact，或 Gate 5 的强 baseline 已把问题解决，不应继续写“IRC 新场景”故事。**

## 6. 何时需要多 agent 同时运行自有 planner？

**主任务不要求所有 agent 都运行新算法。** 主要场景应该是一个 ego 规划 + 相互独立的人类/机器驾驶响应，互相不能指定下一步动作；offline joint solver 仅用于研究时的存在性上界。将所有车辆都运行相同 planner 会加入一个隐含公共协议，反而削弱异质驾驶的主要挑战。等主问题成立之后，可以额外加入各车都运行同一 planner 的 multi-ego 无通信泛化实验，但它不应替代主 benchmark。
