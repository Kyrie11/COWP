# COWP / EICDRR 科学问题、代码审计与证据门控报告

审计对象：用户上传的 `interactive_planning_revised_eicdrr.tex`、`idea升级.md`、`大模型建议.md`、`COWP.zip`、`formal_v16_8_24_compact_full_5k.zip`、`reactive_screen.json`、`cohort_manifest.json`、`baseline_readiness.json`。日期 2026-10-13。

## 0. 执行摘要

**目前不能声称「在真实 WOMD 交通情境中，ego 行为导致去中心化响应可实现性失败，且不同独立反应模型一致证实此现象」**。实际完成的是：在从 4/150 个 validation TFExample 分片构建的历史原生 cohort 前 80 条中，对 76 个可用场景使用一套带三组参数的简化固定朝向 IDM 类模型的屏幕筛选。未运行真实反事实人类驾驶，未报告 Waymax reactive 运行结果，未对 GameFormer、RA-CBF、JFP 执行共同场景下的配对闭环评测。

更重要的是，在当前上传源码的 cohort writer 中发现 **SDC track 的排列与保存的 ego_index=0 不一致的可能**。原始 WOMD 的 SDC 不保证为第零条；遗留 writer 对所有 raw tracks 不重排，却强行写 ego_index=0。上传 JSON 未含用于原位复核的 NPZ；所以不宣称已有 80 条都受污染，但必须重新生成并验证。旧 manifest 同时没有 Waymax adapter 要求的 `source_record_index`。修复版对保存的场景仅保留真正 SDC（索引 0）和所选邻车，保存 `source_ego_index`、`selected_source_indices` 和 `source_record_index`，增加真实原记录一致性验证。

## 1. 当前确切支持的科学事实

- Cohort：扫描 4/150 个 validation TFExample 分片、1208 条记录、1189 条初步 usable；231 个 unique scenario ID（并非整库随机代表样本）；留出的是场景历史而非有反事实标签的完整数据。`excluded_scenario_id_count=0`，故尚未做对历史开发样本的强制排除。
- Screen：取排序后前 80 条，76 eligible、4 initial oriented overlap excluded、errors=0；只用 4 个恒定 ego 加速度：-3、-1.5、0、+1.5 m/s²。
- 参数组 `cautious/assertive/heterogeneous` 场景级别 any collision 分别 14/18/15；non-ego/non-ego collision 分别 1/10/4；后者随 ego 动作发生变化的场景分别 0/0/1；满足 ±3m 双边 progress-matched 替代的分别 0/0/0；跨参数组三者共同支持的场景为 0。
- 唯一异动例：`4689a7fb2d20970b`，heterogeneous 下 ego -3.0 时非 ego pair `(5,9)` 碰撞，另三动作不碰撞；最近无碰撞替代 -1.5 相比 -3.0 多行驶约 12m，且另外两组不复现。仅构成 simulator hypothesis，不能说现实失败成立。
- 当前 source code 中的三组策略仍然共享一个 IDM 纵向动力学结构。原始 WOMD future 1s ADE 即使小，也只说明模拟轨迹与一个未干预的原始 future 较近，不构成未观测 ego 干预的行为校准。
- Compact-5k：不是简单全 WOMD 无偏抽样：5000 train + 1000 internal val + 1200 heldout-test，历史提升缓存选取，heldout test 亦取自 WOMD 官方 validation 而非官方隐藏 test。1200 场景中历史标签模型支持检查可过，但 `priority_false_safe` 14281/28327 候选是方法和响应库定义的伪标签，而非独立 reactive 事故证据。
- 历史 H1：circle 14/75 变 oriented rectangle 2/75，剩余2个具有初始车辆重叠；所以 0 个可归因到无初始冲突状态的新 H1 强证据。历史 H2 2/75 候选不变；历史 H3 0/75；H4 未测。

### 不能混淆的证据层次

1. 真实 WOMD 的记录：支持场景初始条件与观察到的真实共现行为；**不能为不同 ego 动作提供潜在结果**。
2. 简化模型的 paired simulations：支持模型内干预效应（若 identity、pair、ghost、progress 全通过）；不能证明实际驾驶者的因果响应。
3. Waymax reactive 场景：支撑在特定明示响应策略下的闭环行为，而非真实人的干预真值。
4. 不同模型族+任务匹配+冻结 holdout+校准：才是强论文证据。不知道的标签必须列为 `undetermined`。

## 2. 建议论文定位及条件性卖点

科学问题不要回到过宽的 **generic joint-consistency**，也不要在没有数据时提升到已经发现了 universal failure 或 intervention-induced recovery collapse。当前最有区分度的可检验定位为：

> **Intervention-conditioned behavioral realizability of response-compatible plans**：给定混合交通中仅能独立决策的周车，某个 ego 实际执行控制是否改变他车对之间的实现风险，且能否在保持合法性、舒适性和路线进展接近的前提下通过另一 ego 动作降低这种风险？

直觉：联合安全解“存在”和独立驾驶者“会执行到它”是不同的判断。论文贡献不应只是写出不等式 `J != empty` 不蕴含 `q_dec` 高；Studt 的无通信、state-only、consensual-rule 递归可行性和 RA-CBF 的责任约束都已明确占位。**可能的新贡献**须在真实交互条件下定义并解决 `J-exists / behavioral-selectability` 的 gap，以及在响应模型不确定和有限候选搜索下的证据等级。

更有潜力的差异化研究点（按风险由低至高）：

- **Ex-ante, intent/priority-conditioned behavioral recourse dependence**：现有 COWP 的 same-root 与 burden 机制可提供不同于 soft courtesy 的结构化存在性见证，但必须区分有限库下的 no-witness 与真实无解。
- **Action-paired, progress-matched responder impact**：预先冻结 ego 干预，识别同一他车对的差异，明示物理碰撞、响应负担、背景风险和可避免性。仅改变本车 collision 不能当做他车失效。
- **Uncertainty-aware three-valued feasibility**：证据充分时支持、精度充分排除研究门槛时否定，其余未决定；不是拿神经网络分数阈值当“安全证书”。理论/算法亮点需要给出实际可证明的上下界和置信度控制；当前尚未实现。
- **Joint response existence versus decentralized execution**：如果不同性质的反应模型均展示正例，才能成为最有分量的主贡献；若失败则主动收敛回 COWP 的自然 root、负担与有限支持覆盖不确定性。

### 提议精确定义

记固定状态 `s`、ego 已执行动作 `a`、独立响应策略模型 `P`，`Y_ij(s,a,omega)` 为非 ego pair `(i,j)` 的风险。定义

`Delta_ij^P(s;a,a') = E[Y_ij(s,a,omega)-Y_ij(s,a',omega)]`，同一场景、同类周车策略、共享环境随机数和真实执行动作。要求 `(i,j)` 在初始时刻无碰撞、固定 pair、无无关 ego 自撞解释，`|route_progress(a)-route_progress(a')|<=3m`（此值仅开发期预设，正式实验独立冻结），再检查物理可行、道路合法、jerk/加速度、责任与负担。

“joint set nonempty”只能证明存在某个可行响应组合，不能直接推出可实现概率高。有限 response bank 中搜索不到不能推出真正 empty。约束求解器只对明示有限问题保证正确性。

## 3. 为什么必须先改代码，不是立刻增加大型模型

现有 constant controls (-3,-1.5,0,+1.5) 导致四秒 ego displacement 强烈不同。唯一变化案例差距从 +12m 起，根本无法证明相同驾驶任务下“可避免”；加密恒定加速度网格可以做敏感性分析，但不能自动产生更公平的对照。

本次新增 `research_probe/matched_response_test.py`：对每个基础加速度 `b∈{-1,0,1}`，定义平滑正/负 timing pulse `a_k=b ± A*cos(2*pi*(k+.5)/N)`，`A∈{1,2}`，20帧/0.2秒。非饱和的纵向积分下，正负脉冲具有**同样终点速度和总位移**，却有不同中间时序。实际执行后仍重新验证 progress 匹配，而不是假定成立。用 oriented SAT、精确 non-ego pair ID、ego 保持物理占用但不进入他车 IDM leader 检测的 ghost response 对照，判断模型内部的疑似传递路径。这是**假设发现器**，不含真实路线、Waymax 原生闭环、学得的人类反应，也没有 H1/H3 保证。新脚本结果不能提前填成正例。

`research_probe/verify_cohort_identity.py` 可从重新抽出的 NPZ 验证 SDC=0、邻车数量和源轨迹索引，也可用原始 TFExample 做逐条 SDC ID/xy 比对。旧上传 manifest 缺失源记录索引，将失败；这是预期行为。

通过的回归测试是 65 条（另有两条 PyTorch 非阻塞 warning）。未测的 `tests/test_core.py` 依赖 `ocrap` 旧包，**源 ZIP 未提供该模块**；因此不能声称全量测试通过。Waymax 真正执行、tensorflow parser、GameFormer/RA-CBF 模型运行因缺 server runtime 和 checkpoint，均未实测。

## 4. H1-H4 分别如何变成可证伪实验

### H1：agent-wise feasible / jointly incompatible

固定车辆物理模型、道路约束与优先权自然根，先找每个 agent 独立响应、再做全局 CSP；每种响应库 × 不同密度/优化覆盖 × 连续优化求补解，并剔除 t=0 重叠。阳性：“通过动力学和环境验证的独立集均非空，但全局问题无解”的命题只能对**明确穷举的离散模型**使用；不能由有限库 CSP 空集推出真实驾驶无解。其科学价值低于现阶段主要 H2，除非跨响应族出现完整的、经过覆盖校验的稳定正例。

### H2：joint response exists / independent selectors collide

同一真 SDC、同一真实 WOMD 历史（只在观测阶段）、相同接受策略和相同 policy noise，先确认 joint witness，再让相互独立的 reactive agents 真实执行；统计 `P_dec(Safe|s,do(a))` 或每一固定 pair 的风险。条件中的“ego action”必须是实际投影/执行控制，不是计划轨迹名义上第一点。变化应在 **至少两个结构不同的反应模型族**（例如 Waymax IDMRoutePolicy 和有 heldout 行为验证的 learned reactive model）中稳定，并存在进展、合法性与舒适度接近的 alternative。区分 baseline hazard（两边都撞）、ego 自撞、ghost-response mediated pair、新 pair、时间/初始重叠。输出误差区间与 actor-by-actor witness trace。

### H3：successor recoverability loss

只有 H2 或可信 physical viability 机制出现后再启动：先 freeze terminal backup 的形式、响应行为、可达集近似；执行**真控制命令**后评估下一步 backup 集合、重复多步检测，而不是把当前 lookup 没 proposal 当成没实际备份；用 backup MPC / verified reachable set 对最初只在有限响应库中出现的倒闭做再求解。预设多阈值敏感性和单独 holdout。连续零检出的情况下不得把 H3 写成主贡献。

### H4：现有方法不足及你的方法改善

只有 H2 在独立响应模型和任务匹配条件下明确成立后才大规模接入：同样场景、同样观测、同样 route、同样动作/CPU budget、同样真正 reactive eval、相同碰撞几何，比较 risk-progress Pareto 和 safety at fixed progress，含 confidence intervals, bootstrap by scenario, ≥3 train seeds for learned baselines。正面收益不能仅来自共同候选池提高 coverage，必须加 hard courtesy、priority-aware social cost、GameFormer/conditional forecasting、RA-CBF/safety filtering、Studt contingency MPC；如 H3 未成立则 Studt 是重要理论相关工作，但不该声称打败它。

## 5. Baseline 的真实就绪度、优先顺序

| 基线 | 原生提供内容 | 未满足条件 | 目前优先级 |
|:--|:--|:--|:--|
| Waymax IDMRoutePolicy | 服务器 Waymax 可用时可运行的一套 route-following reactive pilot | 版本/TFExample 1.3.1 适配、SDC identity、真实执行控制、在同一初始状态复现 | P0，先做小样本 |
| Constant-response CV / IDM + hard courtesy | 可直接在共同动作接口构建可解释参照 | 碰撞物理与任务匹配、约束公平性 | P0 |
| GameFormer (ICCV 2023) | 提供源码 WOMD joint prediction/open-loop planning | 缺官方兼容 checkpoint；WOMD closed-loop planning 作者仓库未发布，须重新适配 shared Waymax，不可标称官方结果 | P1，当 H2 门槛通过后 |
| JFP (CoRL 2023) | 预测论文、joint interaction model 语义 | 不等于提供的 closed-loop planner；无模型权重时只能为预测侧参照，不能当可执行安全控制器 | P1 / prediction-only |
| RA-CBF (ICRA 2023) | 用户 ZIP 带作者责任学习/CBF 部分 | 原作者声明完整 closed-loop sim 不公开；必须移植训练策略与参数，并单独验证 | P1，优先验证小规模 CBF 对照，再全闭环 |
| Studt & Schildbach (2026) | state-only consensual-rule contingency MPC 理论 | 需要保留共同规则、后备不变集、系统动力学的原有假设才叫忠实复现 | P2，H3 重要时上升 P1 |
| R3R (ACC 2026) | 有开源仓库，信息依赖有有限通信 | 无通信版本不是作者原法；其通讯承诺机制不可随意删除后称 faithful baseline | P2 |

**不要直接把 GameFormer 的 open-loop collision 率或 JFP 的 joint prediction overlap 与 RA-CBF 的 closed-loop 风险放在一张“谁更安全”的表中**。先定义三个层次：预测 quality、在同一候选输入上的选择 quality、闭环实际控制 quality。缺 checkpoint/完整适配的内容不填写数字；作者实现未公开的部分报告为“未忠实复现”。

## 6. 下一轮操作顺序（服务器）

解压 `COWP_EICDRR_AUDITED_CODE_20261013.zip`，不要覆盖服务器历史 COWP 项目。以下命令必须在解压后的源码目录内执行：

```bash
# 0. 环境；这些只是你的服务器的示例路径，可按实际情况覆盖
export RAW_WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export EICDRR_DATA_ROOT=/data0/senzeyu2/dataset/COWP/eicdrr_identity_audited_20261013

# 1. 回归与原有 JSON 独立审计
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh tests
export ORIGINAL_SCREEN_JSON=/path/to/reactive_screen.json
export ORIGINAL_MANIFEST_JSON=/path/to/cohort_manifest.json
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh old_audit

# 2. 只跑研究开发用的四分片 cohort（如果结果有希望，再完全独立路径做 full）
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh cohort_dev
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh identity_npz
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh identity_raw12

# 3. 新同进展匹配干预筛查；零正例也是有价值的证伪证据
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh matched_dev
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh ghost

# 4. 只有上一步机制成立才运行官方 Waymax 类行为策略 pilot
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh waymax_idm_pilot
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh baseline_status
```

如需高可信采样，换全新 `EICDRR_DATA_ROOT` 路径后：

```bash
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh cohort_full  # 150 shards 全量扫描，较慢
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh identity_npz
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh identity_raw_all
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh matched_full
```

**注意**：不论全量扫描还是 hash 排序，若看过全体 scene 的 positive/negative 并围绕其修改算法，就不再是 blind test。正式需要把 identity-verified 的集合按 scenario hash 预划为 discovery/lockbox，后者在 threshold、候选、基线代码、checkpoint 和判定准则冻结前不可使用；以不同 salt 反复选择漂亮示例同样不成立。探索阶段的正例只可用于指定候选 high-fidelity eval，不可报为 WOMD 总体发生率。

下一轮请上传：`results/identity_npz.json`、`results/identity_raw12.json`、`results/matched_dev.json`、`results/paired_ghost.json`（如运行过，追加 `results/waymax_idm_pilot.json` 及对应日志），再提供 5~15 个有代表性场景的 **原生 NPZ 和元数据**，以便同一对车辆的时序轨迹复查；如果 Waymax 失败也上传 `errors` 而不是只给 summary。

## 7. 决策树（严格门控）

- **Gate 0：数据来源不一致**（identity 不过、原始 NPZ 不可靠、原始场景已重叠） → 暂停科学解释，修复数据；任何算法结果不纳入 H1-H4。
- **Gate 1：paired IDM timing screen 中没有匹配同一 pair 风险差异** → 结果如实负报告；考虑回到 original COWP same-root burden 及 response coverage 的三值判定，不推销 EICDRR。
- **Gate 2：只有同 family 正例，Waymax 类 policy 不支持** → 减少对“现实失败”的强声明；做不同模型/地图/路径机理消融，谨防 proposer artifact。
- **Gate 3：两个独立已验证 reactive family、冻结 matched-task、对 specific pair 有显著作用且避险有效** → EICDRR 才值得正式升主研究问题；首先尝试 hard courtesy/RA-CBF 处理以免重发明。
- **Gate 4：GameFormer、RA-CBF、contingency MPC 不解决这个 gap（相同可观察信息和操作约束）** → 才考虑发布新学习+verified recourse 方法，且按安全/效率 Pareto、跨模型+跨 seed 评价 H4。

所有受模型、动力学、穷举枚举影响的结论必须加限定，特别是“存在有证明”的 witness 与“未搜索到”的 uncertified 不可混为一谈。
