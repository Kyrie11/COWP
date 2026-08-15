# COWP v16.8.18：Sparse Probe 构建完整性、FASTPATH 误报与 Promotion 链修复

## 1. 本轮重新诊断的结论

本轮只依据重新上传的 v16.8.17 代码包和 `formal_v16_8_17_support_smoke` 产物重新建立证据链。当前无法进入 strict/full rebuild 的**直接原因不是新的 COWP natural/response/witness 科学门失败，而是 v16.8.17 fresh smoke reference build 没有形成一个完整、可审计的 96-scene 数据集，随后 FASTPATH 将“reference scene 缺失”误报成了“可能发生 tensor semantic change”，最后 strict 因为 composite smoke verdict 从未生成而停止。**

因此：

- FASTPATH 报错和 `Missing composite smoke verdict` 属于同一条 promotion pipeline 的上下游故障链；
- 但它们不是同一个科学指标失败；
- 现有上传物不足以证明 v16.8.17 的 natural/model-support 重新失败，因为这些审计根本没有完成；
- 不应据此再次修改 natural generator、PRIO、candidate bank、response bank 或 NCF 阈值。

## 2. 上传产物中的直接证据

### 2.1 FASTPATH 没有比较到任何 tensor

上传的 `fastpath_ab/fastpath_semantic_equivalence.json` 显示：

- `requested_scenes = 12`
- `compared_scenes = 0`
- `missing_reference_scenes = 12`
- `missing_candidate_scenes = 0`
- `mismatches = []`
- `unexpected_extra_keys = []`

所以这份报告**没有发现一个数组发生变化**。旧 v16.8.17 的 interpretation 将“scene/key 缺失”和“tensor 变化”合并成一句 `Treat this as a semantic change`，在本次情况下是不准确的。真正发生的是 FASTPATH 的 reference precondition 不满足。

更具体地说，FASTPATH 请求的 12 个 scene 正好是 `union_scene_ids.txt` 的前 12 个。candidate/no-fastpath 侧都能找到，而当前 v16.8.17 fresh-label reference 侧在比较时全部找不到。

### 2.2 fresh smoke build 没有完成

上传包中的：

- `fresh_profile.jsonl` 为 **0 行**；
- `build_fresh.log` 只记录了 Scenario split 扫描进度，最后停在约第 1873 个 record；
- 没有“96 个请求场景全部 resolved / labels build complete”的终态记录；
- 没有 `fresh_profile_summary.json`；
- 没有 `natural_support_diagnostic.json`；
- 没有 `training_supervision_audit.json`；
- 没有 `model_support_audit.json`；
- 没有 `base_screen_verdict.json`；
- 没有 `v16_8_17_smoke_verdict.json`。

`probe_manifest_audit.json` 本身是 PASS，hard/random/union 数量也正确，因此不是 manifest 选择失败。

现有日志没有 shell return-code / signal / OOM 信息，所以**不能严谨地断言构建究竟因 OOM、SIGKILL、终端中断或其他外部原因结束**。能够确定的是：构建在完成请求场景之前停止，而旧脚本没有把这个事实单独记录成 pipeline-integrity failure。

### 2.3 strict 的 Missing verdict 是下游结果

v16.8.17 `strict()` 第一项前置条件就是要求：

`$SMOKE_ROOT/v16_8_17_smoke_verdict.json`

由于 fresh smoke 没完成，后续审计和 composite verdict writer 没有执行，因此 strict 必然输出 `Missing composite smoke verdict`。

所以这两个错误的关系是：

`fresh sparse smoke 未完成 -> reference NPZ 缺目标 scene -> FASTPATH precondition FAIL -> smoke composite verdict 未生成 -> strict Missing verdict`

这不等价于：

`自然根/模型支持已经科学上 FAIL`。

## 3. 目前最近一次真正完整的科学证据仍是 v16.8.16

因为 v16.8.17 没有生成完整 scientific audit，本轮不能用它否定 v16.8.16 已经得到的结果。v16.8.16 96-scene smoke 的完整审计显示：

- selected critical = 538；
- mechanism-auditable = 517；
- auditable critical 中 rootless = **0**；
- auditable critical 中 `<2 low-burden roots` = **0**；
- protected PRIO coverage = 403/404 = **99.75%**；
- relevant response slots = **355200/355200**；
- response safe / unsafe、low / high burden 均非退化；
- affected-root recovery ≈ **41.48%**；
- witness|relevant ≈ **63.92%**；
- pair-NCF|relevant ≈ **36.08%**；
- OBS / NEU / PRIO 与 PRED / OPT / EMG source 均有监督；
- training supervision PASS；
- causal/RootTransport integrity PASS；
- v16.8.16 FASTPATH A/B 12/12 bitwise equivalent。

因此，前几轮真正的“可审计 critical 上 natural-basis root collapse”已经初步解决。剩余约 3.9% mechanism-unknown 是 evidence coverage 问题，v16.8.17 设计的 coverage policy 本来就是要在更大的 strict/train-pilot 中检验它是否稳定、是否对 hard scenes 有偏置。但这套 policy 还没有被一个完整 v16.8.17 fresh smoke 真正执行到 verdict。

## 4. v16.8.17 数据工程逻辑中的主要缺陷

### 4.1 sparse allow-list 仍按完整 split 顺序扫描

`01_build_labels_from_proto --allow-scenario-ids` 虽然只把目标 scene 送入 worker，但 raw source 仍来自整个 `Scenario` TFRecord split 的顺序/交错扫描。对于只有 96 或 1200 个 scene 的 probe，运行时间和成功与否受到“最后一个目标 scene 在完整 split 中出现在哪里”的影响。

这不是 label semantic 问题，而是 sparse data access 问题。strict/train-pilot 更大，继续依赖全 split 扫描会增加无意义 I/O、恢复成本以及中断风险。

### 4.2 没有“所有请求 scene 已 resolved”的构建后置条件

旧脚本直接从 build 命令进入下游 audit。若 build 中断、目标 scene 没扫描到或只留下 partial label directory，没有一个独立 gate 检查：

- 96/1200 个请求 ID 是否全部在 profile 中出现终态；
- 每个需要 NPZ 的 probe scene 是否真的存在完整 NPZ；
- 是否出现 filtered/error/corrupt artifact；
- profile 本身是否为空/截断。

所以不完整 pipeline 会被误送到 model-support/FASTPATH 层。

### 4.3 FASTPATH 只看目录存在，不验证 reference 完整性

v16.8.17 中，只要 `$SMOKE_ROOT/labels_val_v16_8_17` 目录存在，就优先使用它作为 FASTPATH reference；只有目录不存在才回退到完整的 v16.8.16 reference。

因此一个**部分创建但缺目标 NPZ 的 current-version 目录会遮蔽已经验证过的 reference**，正是本轮 12/12 missing-reference 的直接工程原因。

### 4.4 semantic comparator 将 precondition failure 与 tensor mismatch 合并

missing reference scene、missing candidate scene、missing key 和真实 array mismatch 使用同一个 FAIL 文案，导致本次 `compared_scenes=0` 也被描述成“pre-existing tensor changed”。这会诱导错误的算法迭代。

### 4.5 pipeline 非正常退出没有 stage-aware verdict/status

v16.8.17 smoke 在 composite verdict 之前退出时不会留下明确的：

- `failure_stage`
- `return_code`
- `pipeline_complete=false`

strict 因此只能输出笼统的 `Missing composite smoke verdict`，无法区分“scientific FAIL”与“pipeline 没完成”。

## 5. v16.8.18 的修复原则

v16.8.18 **不改变 label tensor 的算法语义**，也不继续调 natural/NCF/support 阈值。label-semantic fingerprint 保持：

`adcea5cb927d4c06c7f667725ce1c5b7b62808d6bd2e84244149d01ab25a1fa0`

因此本轮是数据访问、构建完整性、promotion orchestration 修复，而不是新的算法版本。

### 5.1 Scenario location index

新增 `72_build_scenario_location_index.py`，建立可复用：

`scenario_id -> TFRecord file -> record_index`

索引同时记录源 shard manifest hash。如果 WOMD shard 文件集合/大小变化，旧索引不会被盲复用。

smoke、strict 和 train-pilot 用 location index 直接读取目标 scene 所在的少量 shards/records，不再靠全 split interleaved scan 找目标 ID。

### 5.2 强制 sparse allow-list resolution

`01_build_labels_from_proto` 新增：

`--require-all-allowed-resolved`

location index 缺任何请求 ID，或 sparse build 结束后还有请求 ID 未解析，立即报明确 RuntimeError。

### 5.3 sparse build integrity audit

新增 `71_validate_sparse_label_build.py`，在任何 scientific audit 前检查：

- 请求 scene 是否全部在 profile 出现；
- terminal status 是 written/existing/filtered/error 中哪一种；
- NPZ 是否存在；
- 必需 tensor keys 是否存在；
- `scenario/id` 是否匹配；
- NPZ 是否可读/是否截断。

failure class 被明确区分为：

- `pipeline_incomplete_or_requested_scene_not_resolved`
- `build_or_artifact_error`
- `terminal_filtered_or_missing_npz`
- `none`

只有 `none` 才进入 model-support/causal/screen。

### 5.4 FASTPATH precondition 与 semantic mismatch 分离

`66_compare_label_semantic_equivalence.py` 升级后：

- reference scene 缺失 -> `reference_build_incomplete`，`semantic_change_detected=false`；
- candidate scene 缺失 -> `candidate_build_incomplete`；
- 两侧 NPZ 都存在后，真实 key/shape/value 差异才 -> `semantic_mismatch`。

v16.8.18 master FASTPATH 也不再按目录是否存在选 reference，而是先运行 sparse integrity validator；current fresh reference 不完整时，可以在 label-semantic fingerprint 一致的前提下安全回退到已验证的 v16.8.16 labels。

### 5.5 stage-aware pipeline status

fresh smoke、strict、train-pilot 都写：

- `smoke_pipeline_status.json`
- `strict_pipeline_status.json`
- `train_pilot_pipeline_status.json`

若中途异常，记录 `failure_stage` 和 return code；若完整运行但科学门失败，则仍写正式 composite FAIL verdict。`strict` 发现 smoke verdict 不存在时，会同时展示 pipeline status / sparse integrity，而不是把它当成 model-support FAIL。

### 5.6 优先复审完整 v16.8.16 labels

v16.8.18 的 `smoke` 模式默认 `auto`：

1. 若 reviewed v16.8.16 smoke labels + fingerprint + verdict 仍在本机，先执行 policy re-audit；
2. label-semantic fingerprint 必须完全一致；
3. re-audit 完整写出新的 v16.8.18 composite smoke verdict；
4. 只有 source artifact 缺失/不可信时才 fallback 到 fresh sparse smoke。

因此仅修改 promotion policy / orchestration 时不再重复构建昂贵的 96-scene label set。

## 6. 当前是否应该 full rebuild

仍然不应该直接 full rebuild，但原因需要精确表述：

- 不是因为 v16.8.17 已经证明 natural-basis 又失败；
- 不是因为 FASTPATH 检测到了 tensor mismatch；
- 是因为 v16.8.17 的 promotion pipeline 没有形成一个完整 smoke verdict，strict/training pilot 因此根本没有运行。

正确下一步是：先用 v16.8.18 对最后一次完整的 v16.8.16 labels 做 policy re-audit；若新 smoke verdict 授权 strict，再真正运行 1200-scene validation strict。strict 是判断 v16.8.17 evidence-coverage policy 是否合理、3.9% unknown 是否稳定的关键实验。只有 strict PASS 后再跑 1200-scene train pilot，两者同 fingerprint PASS 后才能 full-core。

## 7. 什么性质才足以支撑当前算法模型

当前应继续把“质量”和“覆盖率”分开：

**质量硬门（不能靠 mask/调阈值绕过）：** auditable critical rootless=0、至少 2 个 low-burden roots、typed source 非退化、32-slot relevant response 完整、safe/unsafe 与 low/high burden 非退化、RootTransport exact consistency、root recovery 与 witness 非退化、tensor input visibility 与 Scenario/tf.Example 对齐。

**证据覆盖门：** selected critical 中有足够比例拥有 mechanism target；unknown 必须保持显式 unknown，不能伪装成 non-coercive negative；hard/random missingness 不能明显失衡；candidate-level certificate coverage 需要与 per-critical evidence coverage 自洽。

v16.8.16 已经支持前一组质量结论；v16.8.18 不修改这些 label semantics。接下来 1200-scene strict/train-pilot 的任务是验证后一组覆盖率在 validation/train 两个主 split 上是否稳定。

## 8. 本地代码回归

v16.8.18 完整仓库测试：

`224 passed, 5 skipped`

同时通过 `python -m compileall -q cowp` 和 v16.8.18 smoke/strict/train-pilot/full orchestration shell 语法检查。
