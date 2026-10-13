# COWP / EIC-DRR：2026-10-13 科学判定、代码审计与下一轮实验计划

> **结论（NO SCIENTIFIC GO；保留研究方向，不确认新失效机制）**：当前原生 WOMD 数据只是真实交通情境的初态来源，并不能提供 ego 干预下他车的真实反事实结果。三个反应参数组同属一个简化纵向 IDM-inspired 模型，**并非三种独立响应模型**。80 个检查场景（76 eligible）中，单参数组只有 1 个场景的固定非 ego 车辆对碰撞随 ego 动作变化；没有双侧 ±3 m 进展匹配的安全替代动作，更没有跨参数组稳定案例。没有实际完成 GameFormer / RA-CBF / JFP 的配对闭环比较。该负结果要求先扩大可检验动作和反应模型，而非重新命名现象。

## 1. 论文 Idea、现有算法及新问题的准确位置

原论文 COWP 的技术主干是：历史 WOMD/Waymax 场景，ego proposal bank → observational / ego-neutral / priority-preserving natural-root 构造 → RCOT/BCOT 低负担响应搜索 → protected-priority 判据 → hard-first 选择，失败时显式 uncertified fallback。模型用同一自然意图 root 对照他车为了避让某个 ego 候选必须承担的减速、时间损失或优先权退让。RCRSO 的多样响应 proposal 改善 support，但不会自动提供反事实完备性证明。Compact-5k 包含 train=5,000、validation=1,000、heldout-test=1,200；历史筛选缓存、机制标签、有限 proposal search 与开发流程可能引入数据依赖。真实执行闭环与模型内部的 joint CSP 不能混为一谈。

第四次 idea 升级建议把焦点收敛到 **ego-intervention-conditioned decentralized response realizability (EIC-DRR)**，我赞同把它设为“下一轮可证伪的科学假设”，而不赞同今天就确立为实证贡献。这一判断与过去的 75 场景几何审计一致：圆盘模型的 14/75 H1 正例，在朝向矩形下剩 2/75；两例均存在初始重叠；H2 的两个剩余例子对 10 条 ego 候选均不变；H3 检出 0。既有数据不能支持“新型真实失败已经确认”。

## 2. 三份新 JSON 可确立的观察事实

输入：`cohort_manifest.json`、`reactive_screen.json`、`baseline_readiness.json`。

- `cohort_manifest.json`：WOMD validation TFExample 前 4/150 分片；1,208 条 inspected、1,189 条 usable；hash-reservoir 联集 231 unique scene，混合 blind 与 interaction strata（两 stratum 可重合）。场景选择只依赖当前/过去，但只扫描少数分片，没有代表全 WOMD 的统计采样设计；原 manifest `excluded_scenario_id_count=0`。
- `reactive_screen.json`：80 条来自 scenario ID 排序 manifest 的前缀；4 个 initial-overlap 剔除；76 条 eligible。不能将 76/80 或 0/76 作为总体发生率及安全性估计。
- 不同设置下出现 non-ego–non-ego 碰撞的场景数：cautious **1**、assertive **10**、heterogeneous **4**。
- `any collision` 随 ego action 变化：13/8/12，但包含大量 *ego–other* 碰撞，并不是 H2 的“响应车辆间碰撞”的新证据。**固定非 ego collision-pair** 随动作变化的场景为 **0/0/1**。
- 唯一 ID 为 `4689a7fb2d20970b`，在 heterogeneous 中，危险动作 a=-3.0 m/s² 的 4s ego 位移 34.99 m，三个安全替代动作 a'=-1.5/0/+1.5 的位移分别 46.99/58.99/70.99 m（相差 12/24/36 m）。它们满足“替代动作**不更慢**”的单侧条件，**不满足**“进展相近：位移差绝对值≤3m”的双侧条件。注意即便 ±3m 能通过，Euclidean displacement 也不等于正式 route progress。
- 因此 **同模型三设置、双侧 ±3m、固定非 ego pair、避开所有碰撞、相同 action pair** 的稳定场景数：**0**。
- 原来的 `all_regimes_fit_observations` 用短期未来 ADE，不是针对 `do(ego action)` 的行为校准；cautious/assertive/heterogeneous 只改参数，共享结构，不能用于“三个独立模型”论断。
- 本次已有全体 native ID 与旧 Compact-5k validation **7**、heldout **5** 的 exact-ID overlap；前80条中与旧 validation 重合 **2**。已提取原 Compact-5k train/val/heldout 的 **7,200 唯一场景 ID** 到 `reference_manifests/compact_all_7200_ids.txt`，下一轮选择要 `--exclude-ids`。

`outputs/native_paired_audit.json` 可逐场景核对计数，不包含“仿真跑通”的伪造数据。

### 2.1 一个容易忽略的归因缺口

当 ego 动作改变时，别的两辆车碰撞不一定是 ego 引发：可能两者在参考分支里就会碰，或同一自然反应模型本身在高密度场景不稳定。必须同时保存每个固定非 ego pair 的：实际干预风险、ego 被响应者忽略的 ghost reference 风险、匹配的安全替代动作风险及实际首次碰撞步骤。ghost reference 是一个**模型结构敏感性对照**，不是将 ego 物理移除的真实因果反事实。新脚本 `paired_intervention.py` 实现这一限定的对照，必须在服务器原生 NPZ 上运行后才能判定其输出。

## 3. H1–H4 当前判决与需要的实验

| 假设 | 可证伪的精确定义 | 当前证据 | 下一步必须做什么 |
|---|---|---|---|
| H1：个体安全响应存在而联合相容集合为空 | 对确定 ego a、同一 root，所有参与者分别存在低负担 verified response，但不存在联合互不冲突组合 | 历史圆盘例基本为几何假阳性，native 没有枚举 root-conditioned joint response sets | 排除初始碰撞；带朝向车体、连续时间碰撞和路权/路面约束；加密 response proposals；只报告“本搜索库不可行”，不可当真实数学无解 |
| H2：联合相容存在、独立选择响应失败且 ego 可以改变该风险 | 预定义相同 exogenous seeds 下，固定非 ego pair 的 `P_dec(risk|do(a)) - P_dec(risk|do(a')) > δ`，同时 `|route_progress(a)-route_progress(a')|≤κ`，并有联合安全 witness | native 只有一参数设置 1 场景显示动作敏感，但没有相似进展替代，且并未证明联合存在或独立策略校准 | 最优先；真实 ego projected control、强可行 action generation、Waymax reactive + 与 IDM 不同结构的学习/规则反应器；按车辆对归因和同场景/同种子配对 |
| H3：执行后 successor viability 丧失且可避免 | actual first action 后状态进入经明确定义的 backup-viability 不合格域，而 matched alternative 不进入 | 历史 0；本轮未运行 | 暂不作为主贡献；若 H2 成立，再用 reachable backup / contingency MPC 验证，否则停做 |
| H4：现有强基线不能有效解决 H2 | 同观测、动作空间、响应器、计算预算下，EIC-DRR 算法比强闭环 planner 在归因风险和进展-安全 Pareto 上有优势 | 未运行；`baseline_readiness` 明确未就绪 | H2 gate 通过后才比较；先原生 Waymax IDM + matched MPC，再 GameFormer 适配、RA-CBF，JFP 只用于预测诊断；必要时 Studt / R3R，保留原假设 |

实验的首要目标不是凑出 4 条漂亮百分比，而是**把一个从真实日志初态出发的、可归因、可替代、可跨模型复现的例子**建立为因果模拟证据。这个结论仅相对于指定模拟器，不能自动外推至真实人类反应。

## 4. 研究问题怎样才能有足够强的卖点？

**主命题（有条件推荐）：** *Planning under Intervention-Conditioned Decentralized Response Uncertainty*，更精确的物理语义是“ego 的可执行干预如何在自然驾驶意图与约束下，改变其他驾驶者独立选出的低负担反应的**实现风险**，并能否以任务可比的替代动作减少这种风险”。

定义单个模型 P 与同初态 s、同噪声分支的配对响应风险差：

`D_P(s;a,a') = E[L_non-ego | s,do(a)] - E[L_non-ego | s,do(a')]`

其中 L 预注册为固定非 ego pair 的 OBB collision、急制动或针对优先权的 burden；碰撞与负担分别汇报。事先固定双侧匹配标准 `abs(route_progress(a)-route_progress(a'))<=κ`，及舒适性、路线可行性、限速/红灯等条件；单侧效率不下降只算 dominance 对照，不能声称 matched progress。对多个真正不同的模型族 M，可用最坏模型风险差及 scenario-level 配对 bootstrap 的单侧置信下界，不能拿三个 IDM 参数调节当独立复现。

比“joint consistency”更具体，也更难被已有工作完全覆盖，但该科学对象**不是自动出现的原创理论贡献**：courtesy cost、FeAR 的行为空间缩减、RA-CBF 的责任分配、Studt 的无通信递归可行性、GameFormer/JFP 的 joint prediction 都与此邻近；**存在联合响应 ≠ 独立驾驶者必然选中该响应**本身是集合论和概率逻辑的直接事实，不能当论文卖点。真正可能区分的是：**在真实情境中，特定 ego 干预的可检出/可避免风险与行为模型不确定性条件下的可审计 witness + 保守但不僵死的 plan selection**。

**算法方向（有成功案例才实施）：** 把原 COWP root-conditioned burden verification 用作 response support evaluator，增加一层干预条件下的 realization-risk lower confidence bound。对候选 ego action，输出三值标签：

1. **Verified admissible**：在预定义执行/动力学/几何和误差界下，构造出足够强的安全 witness；
2. **Verified inadmissible**：只有在有严格完备搜索/上界证明时才允许；
3. **Undetermined**：有限 proposal bank 未找到安全响应、反应器分歧或者统计不确定性过大，一律不得宣称“被迫退让/无解”。

如果后续验证失败，请将算法贡献保留为 **priority-aware same-root burden-conditioned planning + evidence-qualified recourse support**，而不是维持 EIC-DRR 的强失效叙事。这条回退比把双重 option-set / temporal collapse 过早包装成统一理论更稳。

## 5. 后续实验：四道按顺序通过的 gate

**G0 数据真实性与隔离。** 用原生 validation 所有 150 个分片（或显式注册的多分片代表性抽样），在 selection 时只看当前/过去，排除全部 Compact-5k 7,200 ID，冻结 salt、source shard/record index、重复记录去重与 ID 哈希。候选评估场景需排除初始所有动力学有效实体间 OBB 重叠，但不能仅筛选阳性标签。报告多分片/多场景错误率、sampling flow、strata 权重与场景级置信区间；不能把 interaction-stratum 比例解释为自然频率。

**G1 同场景动作归因。** 从当前原生 NPZ 运行 `reactive_stress`（参考）与 `paired_intervention`（ghost ego 对照）。对固定 other-other pair、首碰撞时间、他车最小减速度、ego 自身碰撞、非 ego 累计风险分别记录，保持每对动作同一状态、固定反应参数/种子。action grid 至少有已声明的纵向安全候选；高保真之前必须允许改变合流时机和纵向空间余量，而非仅固定朝向 4 个加速度。

**G2 Waymax reactive 闭环与独立模型族。** 先尝试 `waymax_paired_idm.py` pilot，确保配对重现和所选 ID 对得上；它仍是 IDM 家族且沿日志未来空间 route 行驶，ego 干预是期望速度不是完整控制器 emitted action；非车辆是 expert replay，因此最多算 Waymax 结构化 reactive 基线，不可当“验证独立模型”。之后接入与 IDM **不同结构** 的 map/rule-constrained reactive model 和训练于不重叠数据的 learned/reactive model，并对原始 WOMD 未来做 holdout predictive + calibration diagnostics（但仍不等于 counterfactual ground truth）。计算固定 scene cluster 的风险差 CI，按 action pair 同种子配对，分别展示 collision、newly induced hard braking 和 burden。

**G3 新算法和 H4 比较。** 只有 G1/G2 出现稳定、任务相近的失败/可避免场景才实现新 EIC-DRR 层，并在同一评测协议对比 baseline。若 G2 再次 0，停止 EIC-DRR 论文主题升级，以已有 COWP 问题聚焦更可信的 burden feasibility 与 proposal coverage 问题。

说明：本包只实现 G0 的 exclusion/索引改进、G1 的审计/简化 ghost 分支、G2 的 Waymax 官方 IDM API 适配**实验性原型**。尚未实现独立 learned responder、真正完整的 EIC-DRR planner、完整 H1/H3 证明或成熟强 baseline；任何运行命令都不能绕过这些科学缺项。

## 6. 强基线排序：是否现在做 GameFormer 和 RA-CBF 配对闭环？

**现在不宜把 GameFormer+RA-CBF 同时作为最先投入的复现任务。** 首先应确保 G1/G2 同平台风险事件稳定存在。优先做：

1. **Waymax 官方 reactive IDM + 你的原 COWP + 一个强 safety/contingency MPC（可含 common-rule decentralized backup）**，统一 controller emitted action、物理障碍几何、事故判据、地图合法性和进展/舒适约束。Waymax 自身沿日志 route 的 IDM 不等于另外一个独立的真实人类模型。
2. **GameFormer**：上传源码存在，运行的是 WOMD open-loop evaluation 路线；当前无匹配 checkpoint、Scenario proto+地图输入未就绪，不能把导出的 open-loop 预测轨迹直接计作 decentralized closed-loop。可先训练/取得 checkpoint 跑同场景 open-loop，再将其 ego trajectory policy 以 receding horizon 方式投入同一 Waymax reactive rollout。务必报告非官方接口封装和是否使用 GT future。
3. **RA-CBF**：源码有 `tbsim/safety_funcs/cbfs.py`，无作者完整 closed-loop 环境/分发的训练 checkpoint；原实现与 Trajdata/nuScenes 等协议相关。应先跑通原仓库可复现示例，再移植到 Waymax，对能力、责任分配假设及安全集调整公开标注为**本地适配**，不能宣称复现了论文官方完整成绩。适用于验证“责任分配安全滤波能否替代本方法”的 H4。
4. **JFP**：你此轮实际挂载的七个文件中没有可打开的 JFP PDF/源码；旧分析已将其作为 joint/conditional prediction 论文审视。应做 *joint prediction consistency* 诊断，而不把它称为闭环安全控制器。若要对比强 **planner**，优先找可运行的 DTPP/交互 MPC 或提供已训练 checkpoint 和 closed-loop policy 的 GameFormer。
5. **Studt 的 decentralized contingency MPC** 与 **R3R 官方 Julia**：前者直接挑战“无通信下保留 future backup”的创新点，不能回避；后者含通信/已承诺未来轨迹，不能默默去掉通信再声称作者方法失败。务必维护各自原始假设。

## 7. 要报告的评估表格与统计设计

每一方案在 **相同场景、相同初始状态、相同地图、相同其他 agent seed** 下运行：ego-control collision、non-ego/non-ego collision（相同固定 pair）、new braking events、near misses、responsibility-weighted burden、路线进展、红灯/越线/逆行、最小 backup viability、算力/延迟、成功执行率和 rollout errors。每个指标把 `native_observed`、`simplified_reactive_model`、`Waymax_reactive_IDM`、`independent_rule_responder`、`independent_learned_responder` 分开列。正式风险比率提供以 **scene ID 为重采样单位** 的 bootstrap CI；禁止将同一 scene 下 4 组 ego action ×3 response 参数当 12 个独立观察。不同随机种子用 common random numbers 配对；合理报告 conditional risk difference 和 effect heterogeneity，而不是正例比例唯一判断。

若真的拿到 1–5 个压力案例，它们只能支撑 **case-based mechanism witness**，不能支撑全分布 prevalence claim。要升级到 CCF-A 规模还必须在未被方法开发触碰的原生 holdout 上稳定重复，报告跨场景统计和强 baseline 实际差异。阈值（如初始 3 m 仅为 pilot）应在 final holdout 前冻结、用物理/任务语义解释，不能以发现正例为目的反向拟合阈值。

## 8. 交付内容说明与复现限制

- `research_probe/paired_evidence.py`：对**已经上传的 JSON** 做保守、固定 non-ego pair + progress matched 审计；本地执行过，不需要原始 NPZ。
- `research_probe/paired_intervention.py`：ghost-ego reference 对照和双侧 progress-matched 报告；已跑基本单元测试，**没有**用户服务器的 NPZ 可执行真实 cohort。
- `research_probe/reactive_stress.py`：增加响应忽略 ego leader 的参照开关和 hash-order 采样，保留原功能。
- `research_probe/native_womd.py`：在新 cohort manifest 中保存 source record index，配合全面 exact-ID 排除表。
- `research_probe/waymax_paired_idm.py`：基于官方 Waymax `IDMRoutePolicy` + `MultiAgentEnvironment` API 的**试验型** TFExample 配对 rollout，未在本沙箱安装 Waymax/JAX/TensorFlow 和真实 TFRecords 的条件下运行，不保证版本零修改即工作；它只帮你拿到真实执行/集成错误回传，不能作为新实验成功的证据。
- `outputs/native_paired_audit.json`：只包含原 JSON 统计结果，不包含虚构闭环结果。
- `interactive_planning_revised_eicdrr.tex` + `.bib` + `.pdf`：在保留 COWP 方法段落的同时，将主叙事改成可证伪的问题定义，明确负证据、数据泄漏及基线尚不可用。
- `NEXT_EICDRR_COMMANDS_CN.sh`：首批服务器命令，严格不覆盖已有实验输出。进入服务器前建议以 git/sha256 冻结代码和配置。

下一轮请上传：新 cohort `cohort_manifest.json`、`reactive_screen_hash.json`、`paired_ghost.json`、`waymax_pilot.json` 以及 `baseline_readiness_latest.json`；必须保留每项 `errors` 原样。没有原生 cohort `.npz` 时，无法在当前环境独立生成 Waymax 真闭环结果。
