# COWP v16.8.17：v16.8.16 Smoke 重新分析与 Evidence-Coverage Promotion 修复

## 1. 结论

本轮重新分析完全以重新上传的 v16.8.16 代码和 `formal_v16_8_16_support_smoke` 为依据，不沿用上一轮结论。

v16.8.16 的核心 natural-basis 修复已经**在可审计（mechanism-valid）critical 上生效**，因此不应继续围绕 map threshold、driveway、OBS/NEU/PRIO root 数或 response bank 做无目标迭代。当前 `recommend_strict_probe=false` 的唯一实质 blocker 是 **counterfactual evidence coverage policy**，而不是 natural/response/witness target 退化。

上传 smoke 的精确状态：

- 96 scenes；538 selected critical agents。
- 517/538 critical 为 mechanism-auditable，覆盖率 96.10%；21/538 为 mechanism unknown，3.90%。
- 对 517 个 auditable critical：rootless=0，<2 natural roots=0，0 low-burden roots=0，<2 low-burden roots=0。
- protected auditable critical=404；PRIO coverage=403/404=99.75%。
- natural source：OBS=1800、NEU=2719、PRIO=3999，均非退化。
- relevant response slots：355200/355200 完整；无 relevant pair 缺 32-slot bank。
- root indexed responses=212390；confident affected roots=121645；root recovery 正负均非退化。
- candidate NCF、false-safe、pair relevance、witness、pair-NCF、safe/unsafe、low/high burden、protected candidate feasibility 都非退化。
- proposal/causal smoke screen PASS。
- FASTPATH 12/12 bitwise semantic equivalence PASS。
- model-support 唯一失败项：`auditability_coverage` 和 `certificate_complete_scene_coverage`。

因此，v16.8.16 已经从“natural-basis 构造失败”进入“数据集不能为少量 critical 提供可靠完整 8 秒 counterfactual mechanism target”的阶段。

## 2. FASTPATH PASS 不等于数据集 PASS

FASTPATH A/B 的语义是：打开 exact-result reuse 与关闭该 fast path 时，参考 label tensors 逐项一致。它证明性能优化没有改变标签语义。

它不检查：

- WOMD 是否给每个 selected critical 足够的 8 秒机制证据；
- natural support coverage；
- candidate-level certificate coverage；
- hard/random missingness bias。

因此 `FASTPATH PASS + smoke promotion FAIL` 并不矛盾。

## 3. 上一轮 natural-basis root collapse 是否解决

是，但必须准确表述为：**auditable support 内已经解决**。

v16.8.16 的 `model_support_audit.json` 对以下检查全部 PASS：

- every auditable critical has natural root；
- every auditable critical has multi-root support；
- every auditable critical has low-burden natural root；
- every auditable critical has multi-low-burden support；
- natural weights valid；
- OBS/NEU/PRIO source support；
- full relevant response-bank coverage；
- transport / recovery / witness / burden continuous support。

这说明继续修改 natural generator 来“增加根”并不是当前 blocker。

## 4. 21 个 mechanism-unknown critical 的真实结构

重新从 `fresh_profile.jsonl` 对 21 个 unknown critical 分类：

- 11 个拥有 >=60 个 valid future samples，其中 9 个为完整 80/80，另有 78/80、79/80；但没有 full-horizon routable lane，也没有 non-degenerate factual route geometry。
- 10 个只有 17--56 个 valid future samples，本身不足以支撑当前 8 秒 natural/transport/witness target contract。
- 19/21 是 `EQUAL_OR_NEGOTIATED`，2/21 是 `AGENT_PRIORITY`。
- 19 个由 late finalizer 标记 `no_routable_lane_or_substantial_factual_geometry`，2 个在前置 evidence precheck 就不足。

这里不应再用“只要 future=80 就强行生成 6 roots”的逻辑。对于第一类，empirical segment cloud 在拥有足够 valid timestamps 的情况下仍为空，意味着 factual path 没有可用于定义 route corridor 的非退化运动几何（典型情形是长期近静止）。把单一 hold observation 扩张为 6 条 counterfactual option roots，会把不存在的数据证据伪装成丰富 action space。对于第二类，直接外推到 8 秒更明显会制造未观测 counterfactual truth。

正确语义是：这些 actor 仍保留在 `critical/valid` 物理安全 universe 中，但 `mechanism_valid=0`，其 mechanism/certificate target 为 unknown，而不是 false/non-coercive。

## 5. 为什么 v16.8.16 composite gate 会 FAIL

精确 blocker：

- `critical_unauditable_rate = 21/538 = 3.903%`，旧 smoke cap = 1%。
- `certificate_complete_scene_rate = 80/96 = 83.333%`，旧 smoke floor = 95%。

其余 model-support checks 全部 PASS。

candidate-level `certificate_valid` 是 scene-complete 语义：只要 selected critical 中有一个 mechanism target unknown，该 scene 的所有 candidate NCF/false-safe certificate 都失效。因此 3.9% 的 pair-level missing evidence 会被放大成 16.7% scene-level incomplete，这不是另一个独立的 root failure。

上传 smoke 中 candidate proposal 总数为 4950；certificate-valid candidate NCF targets 为 4074，约 82.3%，与 80/96 complete scenes 一致。当前训练代码仍可在所有 scene 上做 imitation / physical outcome，并在 mechanism-valid pairs 上训练 natural/response/witness；只有需要完整 NCF certificate 的 planner ranking/certificate loss 被 `certificate_valid` mask。

## 6. 旧 coverage gate 的内部不一致

active config `max_critical_agents=6`，上传 smoke 平均 538/96=5.60 selected criticals/scene。

旧 strict 同时要求：

- per-critical unknown <=1%；
- whole-scene certificate complete >=98%。

若只用一个独立缺失参考模型，1% per-critical unknown 在平均 5.60 critical/scene 时只对应约 `0.99^5.60 = 94.5%` complete scenes。反过来，98% complete scenes 隐含 per-critical unknown 约 0.36%。因此旧 scene gate 实际比显式的 1% gate严格约 3 倍，二者不是同一个 evidence contract。

对于当前 max 6 critical 的结构，95% per-critical coverage 的同量级 whole-scene reference 是 `0.95^6 = 73.5%`。因此 v16.8.17 使用：

- per-critical mechanism coverage >=95%（unknown <=5%）；
- whole-scene certificate coverage >=75%；
- auditable rootless=0 和 <2 low-burden roots=0 仍为零容忍。

75% 不是 publication metric，也不是说 25% 缺失可以忽略；它是“planner 有足够完整 certificate supervision 才值得 full build”的最低 support floor。所有最终 NCF/false-safe/OPR/FSR/PBTR mechanism 指标必须报告 certificate coverage，并仅在 certificate-valid support 上计算。

## 7. 防止通过 mask hard scenes 获得 PASS

只放宽 coverage cap 会有 selection-bias 风险，因此 v16.8.17 新增 hard/random strata missingness audit。

上传 smoke 的实际分层：

- hard：12/277 critical unknown = 4.33%；39/48 complete scenes = 81.25%。
- random：9/261 critical unknown = 3.45%；41/48 complete scenes = 85.42%。
- per-critical unknown-rate gap = 0.88 percentage points。
- complete-scene gap = 4.17 percentage points。

这说明当前 missing evidence 没有明显集中在 hard stratum。

v16.8.17 gates：

- smoke：unknown-rate hard/random gap <=3 pp；complete-scene gap <=10 pp；coverage 本身使用 Wilson gross-failure screen。
- strict/train-pilot：上述 coverage 使用 point estimate；unknown gap <=3 pp，complete-scene gap <=8 pp。

如果 1200-scene strict 或 train-pilot 超过这些 gap，说明 mechanism-unknown 与困难交互系统相关，此时必须回到 critical/auditability construction，而不能 full rebuild。

## 8. 为什么不继续生成 stationary empirical roots

官方 WOMD 明确规定 train/validation 是 1 s history + current + 8 s future，且 `valid=false` 表示该时刻没有对象状态测量。对象轨迹可以存在 missing states。因此构造器必须区分“没有测量”“有测量但没有足够 route geometry”“有完整可审计 route”，不能把 padding 或未观测外推当真值。

对于完整 80-step 但近静止、且 map route 不可解析的 actor，数据最多可靠支持 observed hold，而不是论文需要的多根 counterfactual option space。继续生成 synthetic roots 会改善 coverage 数字，却降低 natural-basis 的可论证性。因此 v16.8.17 不改 label-producing natural logic。

## 9. v16.8.17 实际代码修改

本版本是 **policy/data-support contract** 修复，而不是算法标签重写：

1. `65_audit_model_support.py`
   - coverage gate 分离为 `point` 和 `wilson_gross_failure`；
   - 默认 evidence cap 改为 unknown<=5%、complete scenes>=75%；
   - 新增 hard/random auditability/certificate coverage 与 gap gate；
   - 输出 Wilson CI、mean critical density、在 configured pair cap 下的 independent scene-complete reference。
2. `68_summarize_natural_support_diagnostics.py`
   - 明确统计 unknown critical 的 future support、late finalizer、sufficient-future-but-no-substantial-route 与 insufficient-future 两类。
3. `59_gate_fresh_v16_8_9_cache_protocol.py`
   - 新增 label-semantic fingerprint，分离“会改变 label tensor 的代码”和“promotion policy”；
   - 避免以后只改统计 gate 就重建全部 smoke labels。
4. 新增 `70_reaudit_v16_8_16_smoke_policy.py`
   - 验证上传/本机 v16.8.16 source code fingerprint；
   - 验证当前 label-producing semantic fingerprint 与 v16.8.16 完全一致；
   - 在原 96-scene NPZ labels 上重新跑 v16.8.17 model-support policy；无需重新 label build。
5. smoke / strict / train-pilot wrappers
   - composite verdict 不再重复 hardcode 1%/95--98%，而直接使用 `model_support` 的唯一政策结果；
   - smoke 使用 Wilson gross-failure；strict/pilot 使用 point estimate；
   - 加入 hard/random missingness-bias gate。
6. full-core labels 与 post-tensor cache support audits
   - 使用相同的 95% per-critical / 75% complete-scene point coverage contract；
   - 原来的 root/low-burden/response/transport/source/integrity 硬门均保留。

没有修改 `cowp/label/*`、label config 或其他 Scenario->label tensor 语义文件。v16.8.17 label-semantic fingerprint 与 reviewed v16.8.16 完全相同：

`adcea5cb927d4c06c7f667725ce1c5b7b62808d6bd2e84244149d01ab25a1fa0`

因此优先执行 policy re-audit，而不是再次构建 96-scene smoke。

## 10. Promotion 决策

基于上传的 JSON/profile，v16.8.16 已知非-coverage checks 全 PASS；新 coverage policy 对当前点估计与 hard/random gap 也均满足。因此**从现有 artifacts 推断**，v16.8.17 policy re-audit 应授权 strict probe。但由于上传包排除了 NPZ，本分析环境无法重新执行 `65_audit_model_support` 对全部 labels 的读取，不能把“推断会 PASS”写成“实际已 PASS”。应在数据机上运行 `reaudit-smoke` 获得机械 verdict。

如果 re-audit PASS：直接运行 1200-scene validation strict；不要再重建 96-scene smoke。若 strict PASS，再运行 1200-scene training pilot。只有两者在同一 v16.8.17 full code fingerprint 下 PASS，才允许 full-core。

若 strict 失败，下一步按失败类型处理：

- rootless / low-burden / transport/integrity/source 失败：回到标签构造；
- mechanism unknown >5%：说明 WOMD support coverage 确实不足或 critical/auditability selection 仍需重新设计；
- hard/random gap 超阈值：说明 unknown 选择性集中在难场景，不能通过 mask 解决；
- proposal point gates 失败：回到 candidate/proposal 分布，而不是 natural builder；
- 所有上述通过：才值得全量 rebuild。

这一分流就是为了避免后续继续在无关模块反复迭代。
