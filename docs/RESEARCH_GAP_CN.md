# RIDE: 研究问题、14 篇参考论文的创新边界与可证伪判据

## 科学标题与范围

**暂定标题**: RIDE: Response-Model-Induced Decision Errors in Interactive Driving.

这一标题是预注册研究问题，不是已验证算法贡献。严谨且可验证的主张是：存在 ego 候选决策被有限/失准/未校准的响应模型改变，在任务匹配的备选动作集合中导致更高、可独立评估的第三方风险或行为负担；这种现象不能由常规安全、强 courtesy、RACP、ego-conditioned risk-aware planning 解释掉。

拒绝以下原创性主张：
1. 首次考虑他车被迫制动（Courteous 已研究）。
2. 首次保护其他车的行动空间（FeAR 已研究）。
3. 首次考虑多模态他车响应（RACP、GameFormer、JFP）。
4. 首次使用 ego-conditioned prediction 选动作（HYPE、GameFormer）。
5. 首次发现现实化 reactive agents 改变 planner 表现（When Planners Meet Reality、nuPlan-R）。
6. 首次评估 off-log ego 行为引发他车碰撞（ReactSim-Bench）。
7. 首次处理交互导致的分布偏移（Mirzaeedodangeh et al.，L4DC 2026）。
8. 首次研究去中心化递归安全（Studt 2026、R3R）。

## 准确的理论对象

给定同一场景 s、同样的有限候选集合 A(s)，以及预测响应模型 \hat P：

    a_hat = argmin_{a in A}  J_pred(s,a ; P_hat)

由 *独立评测响应环境* P_ref（可能有多个不同结构模型）估计真实执行任务代价：

    Regret_{P_ref}(s,P_hat)
       = J_ref(s, a_hat) - min_{a' in A_match(s,a_hat)} J_ref(s,a')

`A_match` 包含合法、动力学可执行、路线进展接近且安全/舒适性可比较的动作。对不可用/未认证的响应证据，结果必须区分 `positive witness`、`bounded negative` 和 `unknown`；采样为空不等于真实不可行。

J 的主评价终点须事先冻结：第三方 pair collision，ego safety，规则合规；急刹、TTC、交通效率作为独立的次级代理。不可用 teacher 的自己的 burden score 当独立真值。

## 三个 RQ 与停止条件

### RQ1：反应模型失准是否转化为 action ranking shift？

设计：相同场景、历史/候选、随机种子、执行动作；独立的 responder/model 产生不同风险估计。记录完整的候选排序，而非仅 ADE 或单动作碰撞。

未找到 action ranking change -> RQ1 暂不支持；发现模型分歧只属于 simulator sensitivity，尚不是因果强实验。

### RQ2：是否存在任务匹配、可避免的第三方 *decision regret*？

设计：同一场景的动作 a 和 a' 进展 |delta| <=3 m，ego 无碰撞，动力学/道路规则满足；第三方精确 pair 的碰撞或已预注册负担在 a 和 a' 中发生差异。优先要求两种**结构独立且在真实观测下校准**的反应模型复现，而不是同一 IDM 三组参数。

不存在严格案例 -> 不应声称发现了现实系统性可避免事故；可视情况研究 evidence-qualified response support，但另需独立技术 gap。

### RQ3：强基线是否仍系统性作出错误候选选择？

先锁定 Action IDs 和进展，再运行完整基线：hard-constrained courtesy（充分调参）、RACP、GameFormer/HYPE、有限反事实响应证据方法、可行的责任安全过滤器。保证同等输入、模型权重、算力及 reactive evaluation，不用对某个算法在其原始理论保证前提外的试验冒充官方保证被推翻。

如果 hard courtesy 或 RACP 在 RQ2 候选上已完全处理，就必须转向更具体的 residual gap，不再强行推广 RIDE。

## 14 篇原文覆盖矩阵（由用户上传 PDF 提炼）

| 论文 | 自身已覆盖 | 仍需要通过实验确认的新差异 |
|---|---|---|
| Sun et al., Courteous Autonomous Cars | induced human cost, courtesy loss | response-model errors are decision-relevant under matched alternatives? |
| George et al., FeAR | ex-post action-space reduction and causal responsibility | continuous priority-aware, planning-time evidence, not mere FeAR extension |
| Mustafa et al., RACP | Bayesian multiple policies, risk-aware contingent actions, ego-induced reactivity | misspecification-induced selection error outside belief model? |
| Vielmetti et al., R3R | asynchronous backup with limited communication | non-cooperative human traffic outside shared backup assumptions |
| Yu et al., HYPE | proposal-conditioned occupancy, MCTS, nuPlan reactive eval | systematic decision-level regret under model disagreements? |
| Hagedorn et al., When Planners Meet Reality | SMART reactive benchmark, 14 planners performance shift | causal mechanism at frozen action selection and third-party pair risk |
| Peng et al., nuPlan-R | learned diffusion-based reaction simulation, reactive metrics | experiment platform, not third-party reliance planning method |
| Studt & Schildbach | state-only contingency MPC with consensual rule set | heterogeneity outside common rule protocol; preserve proper assumptions |
| Zhang et al., ReactSim-Bench | 2636 off-log AV deviations, agent-agent metrics | planner choice/regret of known reaction evidence; not new benchmark reactivity |
| Huang et al., GameFormer | game-theoretic interaction and conditional planning | decision regret not already handled by multi-agent prediction itself |
| Cosner et al., RA-CBF | responsibility allocations and CBF safety controller | conditional behavioral reliance beyond formal safety set |
| Luo et al., JFP | consistent joint future distribution | joint consistency vs decentralized realization, action-choice relevance |
| Mirzaeedodangeh et al., L4DC 2026 | robust conformal safety under policy-induced distribution shift | third-party task-matched decision burden and finite recourse support |
| Lyu et al., Risk-based Socially-Compliant Planning | risk tracing and hard social compliance | independence of burden-relative safety evidence from teacher |

需补查、尚未用户上传的近邻文献：2026 Bridging Predictive Uncertainty and Safe Action (arXiv:2606.03296), 2026 autonomous driving surveys, CausalPlanner (2026)。不要将本表当作穷尽式专利/新颖性证明。

## 四类情境：科学证据等级

1. `cut_in`：并线与后车被迫刹车；类似 Courteous，**不能作为原创现象**，用于证实第三方决策归因与 hard courtesy 区分。
2. `convoy`：ego 干预引发下游车辆连锁制动/追尾；必须区分旧事故、独立交通动作和 ego 诱发。
3. `intersection`：理论 joint safe response 可以构造，但独立 Traffic Manager / BehaviorAgent 不一定采用；正向 SAT 见证不等于真实驾驶员协同协议。
4. `support_gap`：窄响应候选库查无有效 recourse、扩张后出现响应 witness；仅可证明**有限 proposal 未覆盖**，无解不能被解释为全空间无法恢复。

S1-S4 都是研究中的 *candidate stress archetypes*，不是从原始 WOMD 采样并标注的真实交通事件。CARLA 物理仿真依然有 behavior model 假设。只有 RQ1-RQ3 得到独立、重复支持，才允许研究 RIDE 算法。

## 统计与避免自证

- 冻结 development/validation/test，以 scene ID+原始 segment 去重；历史 Compact-5k 仅作开发。
- 审计全部尝试、报错、无匹配动作、不合法场景，不只展示精挑案例。
- 主指标用 per-scene paired effect、collision regret、precision of unsafe selection、coverage vs regret、ego progress 和常规 safety。
- 对稀有事故报告 scenario cluster bootstrap CI；无真实场景则不得从合成 demo 的 count 推总体频率。
- 需要单独测 ground-truth response calibration / support，不能让 同源 teacher 生成 root 和 oracle 同时担任 label 与评测模型。

## 2026-09 至 2026-10 的必要增补文献：缩小“决策后悔值”原创性

以下是对用户 14 篇上传论文进行比较后，再通过外部检索发现的**新增近邻**，不是用户 ZIP 的组成部分：

- [From Proxy Learning to Driving Decisions (PDT), arXiv:2609.02688](https://arxiv.org/abs/2609.02688) 已经从 proxy loss、candidate ordering、support-versus-selection regret 讨论预测知识是否能转为真实规划效用，使用 NAVSIM 评测。不能把一般性的 proxy-to-decision transfer/regret 命名为原创。
- [Evaluation Is All You Need for Multi-Modal Autonomous Driving, arXiv:2609.30818](https://arxiv.org/abs/2609.30818) 已指出最优候选与选择器之间的 gap，并提出风险感知轨迹评分机制。不能仅以“评价器没选到最好的候选”为原创。
- [INTERACT, arXiv:2609.31137](https://arxiv.org/abs/2609.31137) 已将 ego intent-conditioned 反应预测与连续局部优化耦合，强调对候选反应结构的泛化。不能将 ego-conditioned prediction 作为新主张。
- [SIGMA, arXiv:2610.09520](https://arxiv.org/abs/2610.09520) 讨论 uncertainty 对真实 MPC 规划选择/收益的影响。不能将 decision-aware uncertainty 作为首次提出。
- [Bridging Predictive Uncertainty and Safe Action, arXiv:2606.03296](https://arxiv.org/abs/2606.03296) 需要进一步核对方法细节，作为未完成的 novelty review 项目。

**仍有待实证检验的交叉科学对象**：*task-matched ego-intervention-conditioned third-party externality*（明确的非 ego–非 ego vehicle pair、交通责任/自然行为、反应候选的证据有效性），以及此类外部风险的模型失准是否真正反转了 planner 对 ego 候选的排序。在至少两类独立、真实观测校准的反应评测上，固定动作及场景、冻结强基线的选动作结果，分别比较（a）物理风险，（b）负担指标，（c）匹配进展，（d）root 证据；只用合成 CARLA TM/BehaviorAgent 不能证明现实交通因果失败。

因此阶段标题 **RIDE: Response-Model-Induced Decision Errors in Interactive Driving** 仅为“研究问题探针工程”名称，不等于论文已经具备原创性。证实 RQ1/RQ2/RQ3 且展示有别于 PDT、iDriveVLA、RACP 和 INTERACT 的必要机制后，才锁定投稿标题。

## 工程与证据对应关系

| 科学任务 | 执行脚本 | 证据及限制 |
|---|---|---|
| 环境、CARLA 版本核对 | `scripts/01_doctor.sh` | Python 路径与版本，不是物理验证 |
| 固定四类 anchor（先看过去/地图，不看 outcome） | `scripts/03_precommit_scenes.sh` | 冻结地图和几何选择，非现实场景流行率 |
| CARLA 同初始状态反事实配对 | `scripts/03_run_carla_case.sh` | 真实 CARLA 物理，不是人类 do-effect |
| 可控 responder 有限候选库 | `scripts/03_support_ablation.sh` | 正向存在性见证与 UNKNOWN |
| 外部反应导致风险排序翻转 | `ride.ranking` / `04_analyze_carla.sh` | 后验评价器，不是实际 planner 在线选动作 |
| 固定第三方 pair 的风险归因 | `ride.evidence` | 带进展差/初始不符删失与原始计数 |
| 冻结 ex-ante 预测的 proxy 基线选动作 | `ride.selectors` / `scripts/06a_lock_proxy_selectors.sh` | 参考评分器，不是原作者 RACP/HYPE |
| 强 baseline 外部选动作后验审计 | `ride.decision` / `scripts/06_compare_locked_planners.sh` | 必须提供真实外部 checkpoint 推理选择和 hash |
| 原生 WOMD 盲抽样 | `scripts/09_native_womd_cohort.sh` | 原生记录、非用户原模型标签，仍可能有行为数据选择偏差 |
| WOMD 原始记录反应预测拟合 | `scripts/11_observational_response_calibration.sh` | 观测拟合，不识别真正的他车干预响应 |
| ReactSim 公开基准数据 | `scripts/07_download_reactsim_data.sh` | 只下载作者的 ego 反事实动作；还需要 nuPlan + benchmark 模型 |

## 必须停止研究的明确条件

- **Gate 0 不通过**：无法保证 CARLA 0.9.16、地图有效、初始状态配对、同 profile 进展匹配；应只修工程。
- **RQ1 不通过**：独立反应模型改变预测值，却没有稳定改变候选风险排序。不要继续宣称 decision-critical miscalibration。
- **RQ2 不通过**：不存在初始合法、任务匹配、ego 非碰撞的第三方正例；此时优先研究有限支持/UNKNOWN，不把 S1~S4 当作失败发现。
- **RQ3 不通过**：hard courtesy、RACP/INTERACT/HYPE 等在公平比较下已完全处理，或者所谓差异只来自候选多、停得更久；应更改论文故事。
- **只在同 CARLA TM/BehaviorAgent 出现现象**：最多是 simulator-specific pilot，不代表人类混合交通系统性失败。
