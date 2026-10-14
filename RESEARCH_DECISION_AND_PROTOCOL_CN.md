# COWP / EICDRR：2026-10-14 全量 validation 实验复核与可证伪实验协议

## 0. 复核依据、分析边界

- 本次实际打开并检查：`results(2).zip` 内 `validation_inventory.json`、`full150_scan_stdout.txt`、`identity_npz.json`、`identity_raw32.json` 和 `matched_dev.json`；`COWP(3).zip` 中 `native_womd.py`、`matched_response_test.py`、`reactive_stress.py`、`waymax_paired_idm.py` 与论文 TeX。
- 未提供服务器上 3,276 个 NPZ、全部原始 WOMD 记录、额外学习响应模型检查点或 Waymax 真正闭环轨迹。因此新增补丁在本地进行单元/合成回归，**不能虚构全量新实验结果**。
- 观察到的 logged WOMD future 只有一个事实轨迹，不存在同一真实驾驶者对两个自变量干预的观察标签。所有干预验证都只能是**明确模型条件下**的证据。

## 1. validation 全扫描是否完成

这次**完成了**：文件名清单 150/150、缺失 0、逐分片日志最后到 00149-of-00150，`record_iteration_uncapped=true`，处理 43,695 条，其中 41,283 初步可用，排除与历史数据集重叠 ID 共 2,200，TFExample 解码失败 0。文件名一致、日志遍历完整，不代表每个记录已经进行反事实执行。保存 cohort 3,276 条；新 `matched_dev` 只有开发集 160 尝试、147 有效，13 个排除（初始定向框重叠 11、速度/航向不一致 2），错误 0。NPZ 自验证 3276/3276，抽查原始记录与 source SDC 32/32。剩余未复核的 NPZ 原始溯源需要 `raw_all`，已新增命令。

注意：已选 3276 个场景属于多种 stratified/hash reservoir 的**非均匀样本**，不能将 2/147 推广成总体 WOMD 失效比例。validation 150 个分片全部扫过 != 41,283 条均做过响应干预。使用 hash 切分 development/sealed_holdout 且经过诸多阈值探索，不得事后将 development 冒充 blind 验证。

## 2. 新结果逐级判定

| 指标 | 本轮 | 可以证明 / 不能证明 |
|---|---:|---|
| 147 个有效 matched 场景 | 是 | 筛选成功，非 150 分片全量对照 |
| Cautious 严格 H2 候选 | 1 scene | 单模型参数下存在候选 |
| Assertive 严格 H2 候选 | 1 scene | 单模型参数下存在候选 |
| Heterogeneous 严格 H2 候选 | 2 scenes | 单模型参数下存在候选 |
| 任一配置严格 H2 候选（去重） | **2/147** | 仍只是纵向、静态航向 IDM 类筛选结果 |
| 同一 ego 动作对 + 同一 non-ego pair 在至少两个参数配置下复现 | **1/147** | 仅同结构 IDM 复现，不是独立模型 |
| 三参数配置同一动作/同一 pair 完全一致 | **0/147** | 无稳健跨三参数证据 |
| 任一配置 burden proxy 变化 | **40/147** | 可供后续机制发现，非现实因果 / 事故 |
| 任一配置 near-pair proxy 变化 | **10/147** | 只有连续距离敏感性，非联合不可行证明 |
| 同一场景且跨独立响应结构模型复现 | **0（未测试）** | 尚无跨模型验证 |
| 真实驾驶者对替代干预的观测 | **0（不可从单个 WOMD log 识别）** | 当前数据库无法直接提供人类反事实 |

候选 1 `e86e7f0149ebaea4`：Cautious 与 Heterogeneous 都给出相同 bad/alt pulse 对和局部 non-ego pair `(1,2)`，实际两组 ego 代理进展差约 -0.487/-1.282 m。候选 2 `e44ce0d62879853c`：Assertive / Heterogeneous 都有候选但动作和 pair 不一致。请将局部 pair 映射为 WOMD 原始 `object_id`，并检查原生地图/车道、实际轨迹与首个碰撞前行为。

2/147 的 Wilson 95% **描述性**区间约 [0.37%, 4.82%]，仅供当前非均匀 development 样本量级感知，不能作统计总体事故概率。

## 3. 实际工程漏洞与本次补丁

**已实证确认的脚本问题**：`locate_shards` 旧逻辑用 `validation_` 前缀匹配数据目录；当同一根目录存在 `validation_interactive` 时，查询 `validation` 可能误纳入 interactive 数据。已改为精确 split 目录（或明确 TFExample 文件名前缀），添加合成目录隔离测试。支持 `validation_interactive` 的独立 `inventory/cohort` 扫描；推荐先检查真实 shards 是否存在，而非猜测已下载。

**科学测量能力缺口**：
1. `matched_response_test` 仅列出非 ego pair 的**局部索引**、未输出原始 actor IDs；现在对新增结果同时输出 `pair_object_ids`。旧 JSON 不应被误说为原始 object_id。
2. 严格终点是 *non-ego--non-ego collision AND ego clear AND ghost 免于此碰撞 AND progress-matched alternative safe*，不等于所有 H2 定义的响应风险；现在 `h2_evidence_gate` 分离这类事件、direct ego collision preventability、他车制动代理和近接代理，**不把后几类偷换成 H2 主终点**。
3. 仅使用固定航向、纵向 IDM、4 秒（20 × 0.2 s）、ghost 排除 ego 为前车、取邻车至多 20 辆：在汇入、转弯、信号灯、强遮挡、高密度路口存在明显模型错设与截断风险；存在正例也可能是仿真产物，负结果也可能是漏检。ghost 不等于真实 `do(ego absent)`，因为 ego 物理位置仍在场。
4. 过去用 1s ADE 描述“模型拟合”；本补丁新增**在观测到的 logged ego future 条件下**的 responder 1s/2s ADE，与常速度 null 比较。它仅判断模型对**一个已观察干预**的描述能力，不测未知 ego 干预下的人类行为。静态非 ego 航向的不适配可以在此暴露。
5. 跨模型 gate 新增 `independent_model_gate`，输入实际执行的成对轨迹评测导出 JSONL；同一场景、动作对、**原始 actor pair ID**、多个种子、各模型的物理安全/路线与进展相同，才得到独立模型内部一致性候选。没有实际第二模型执行记录时永远不会伪报跨模型复现。
6. 新增 `cohort_overlap_audit`，检查 interactive 和普通 validation 是否重叠；即使 ID 不同，时间窗/原始 log segment 仍可能重叠，因此独立性没有自动证明。

## 4. H1 / H2 / H3 / H4 阶段门槛

- **H2 第一优先**：首先必须将 non-ego 真实 object IDs、ego 动作、实际进展、路权、路径、舒适度等预注册；在开发集观察充分、CV 与 learned/rule response 描述能力合格后，针对 `e86...` 和 `e44...` 做原生地图的响应者闭环。至少两类**结构不同**的模型复现相同动作对、相同第三方 pair，使用 common random numbers，并报告置信区间；再把冻结协议在未触碰 holdout 上执行。仍不能据此说“真实驾驶者一定会这样”——必须承认行为转移不确定性。
- **H1**：独立 responder 的 admissible action family 与联合 CSP 需要区分 SAT positive witness、有限集 UNSAT/UNKNOWN（search truncated）。尤其 bad ego action 的 **联合存在性**未由 `matched_dev` 检查；不能凭“safe alternative”证明 bad action 有联合可行方案。
- **H3**：只在 H2 接近成立或出现严格的 post-actuation backup loss 时启用。实测投影控制后的 successor state，和同预算可恢复集合比较，不能使用离线候选 bank 空集替代真正递归可行性。
- **H4**：现阶段没有执行 GameFormer、JFP 或 RA-CBF 的对应方法，没有“其他算法系统性失败”的比较数据。最终应与 courtesy/hard courtesy、FeAR-like action space / responsibility safety、joint predictor + common planner、predictive safety filter、decentralized contingency MPC 比较，统一候选集/信息预算/种子、报告 safety-vs-progress。GameFormer 与 RA-CBF 只有适配并跑通的版本才可称 reproduced baseline。

## 5. 是否下载 validation_interactive / training / testing？

**推荐优先检查并使用官方 `validation_interactive`**，因为原始互动预测 split 面向已筛选交互场景，更有针对性；但预选带来 prevalence selection bias，且和 validation 可能共用原始场景，不能当成独立 i.i.d. 外部验证。严格保证开发和保留场景/原始采集片段不泄漏；测试原始 root 下是否已有 `validation_interactive`，若没有再通过 Waymo 许可途径下载对应 TFExample split。**training 后置**（模型预训练和独立机制开发），**testing 暂不需要**（隐藏 future labels，难以进行本方案观测拟合）。

没有找到更多正例不能推出交通中不存在此类场景：数据是单一实际行为下的历史记录，非由不同 ego 指令重复执行的随机实验。若在 interactive enrich、真实地图、身份/校准、两种独立模型、完全冻结场景和 control budgets 下仍找不到，则应下调该机制作为主贡献的优先级。

## 6. 对 same-root 的定位

**建议移除作为“被发现的全新失效现象”的说法，但不建议删除代码中的 latent intent/root-conditioning。** Same-root 更合适作为一种*条件比较/匹配机制*，用于排除“用完全不同他车意图的响应掩盖不合理的避让要求”。这一思想与 counterfactual matching、intention-conditioned prediction、courtesy/FeAR 有重叠，不能自己命名就视为原创。必须用 root-free、root-conditioned、oracle-root、shuffled-root 消融验证其真实作用，并检验未来标签泄漏及交通优先权一致性。

如果 H2 成立，潜在主贡献是 **Intervention-Conditioned Decentralized Safety Under Response-Model Ambiguity**，随后研究 **Evidence-Calibrated Decentralized Recourse Viability**：从有限样本的正向存在 witness、严格模型域内不可行性证明、和不可判定/证据不足三值判定，构建在模型不确定性与响应缺失下有效的保守决策。这是研究**问题方向**，并非现成 theorem 或已经打赢 contingency MPC 的算法。

如果 H2 不成立，避免靠“same-root false-safe”人为制造新场景。后备更稳健的研究问题是 **Proposal-Limited Behavioral Feasibility: When Can a Planner Trust a Negative Recourse Test?**：有限提议集本来只能给 positive witness，不能给真实不可行结论。需要 coverage/无解界、verified reachable-set 或 search/exhaustion 返回 UNKNOWN，并跟提议规模、算力、任务进展建立新的可测试理论与实证结果。这也不能仅凭命名声称 CCF-A 新颖，必须证明相较覆盖规划、风险控制和预测安全过滤已有研究的精确差异。

## 7. 推荐服务器执行（不可混淆“执行了代码”和“证明了结论”）

```bash
unzip COWP_EICDRR_H2_EVIDENCE_GATE_20261014.zip -d COWP_EICDRR_H2
cd COWP_EICDRR_H2
export RAW_WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export EICDRR_FULL_ROOT=/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014

# 已有 150 分片 cohort，不要再从头扫描，不要覆盖 frozen manifest
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh tests
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh audit_uploaded
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh raw_all
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh observational_fit

# Dev sample size扩大至 320 再做新的风险分层审计；不动 holdout
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh matched_dev_320
MATCHED_JSON="$EICDRR_FULL_ROOT/results/matched_dev_320_v2.json" bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh audit_uploaded
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh triage_deep

# 只是 IDMRoutePolicy 跟驰策略敏感性 pilot，不是完整同一动作对的 H2 验证
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh waymax_pair

# 检查是否有交互专用 split，若不存在按许可途径下载它
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh interactive_inventory
# 仅在 inventory 完整，且为新的非覆盖输出目录时运行
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh interactive_cohort
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh interactive_overlap

# 等你真实接入独立的第二套非 IDM 响应模型，导出本脚本规定的 paired JSONL 后：
export MODEL_PAIR_JSONL=/absolute/path/to/actual_paired_trials.jsonl
bash NEXT_EICDRR_H2_VALIDATION_20261014_CN.sh independent_gate
```

### 必须反馈的文件

- `h2_gate_v2/h2_evidence_gate.json`, `h2_triage.csv`, `h2_priority_dev_ids.txt`
- `results/identity_raw_all.json`
- `h2_gate_v2/observational_fit_dev.json`
- `results/matched_dev_320_v2.json`
- `h2_gate_v2/waymax_priority_pilot.json`
- `h2_gate_v2/validation_interactive_inventory.json`（如下载 interactive，则还有 `interactive_overlap.json`）
- 实际第二结构响应策略 JSONL + `independent_model_gate.json`。

如果 `independent_gate` 看到零输入，它只能报告“未测试”，不能宣称 H2 不成立；如果 positive 也仍然**不等于人类干预因果真值**。
