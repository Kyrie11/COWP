# COWP v16.8.14：WOMD 1.3.1 split、完整性与 smoke manifest 修复说明

## 1. 当前错误的直接根因

v16.8.13 `NEXT_RUN_COMMANDS_V16_8_13_CAUSAL_AUDIT_SMOKE_CN.sh` 没有从当前 WOMD 或当前 baseline cache 重新生成 smoke hard scenes，而是读取历史目录：

`/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke/hard_scene_ids.txt`

随后执行 `head -n 48` 并断言结果必须正好有 48 行。历史 manifest 少于 48 行时，就会在任何 v16.8.13 label build 之前报：

`smoke hard manifest shorter than HARD_COUNT`

因此该错误不是 WOMD validation 天然少于 48 个交互场景，也不是 v16.8.13 natural-basis 代码已经在新 smoke 上失败；新 smoke 实际根本没有开始构建。

v16.8.14 删除了这一历史依赖。smoke 现在通过 `cowp.scripts.45_diagnose_proposal_ceiling` 从当前 `OLD_VAL_CACHE` 动态定义 hard pool：旧 cache 中“存在 conventional-safe candidate，但不存在 NCF candidate”的场景。48 只是 preferred hard stress count，不再被误认为 WOMD split 的固定性质。smoke 总量固定为 96；若当前 baseline 的 hard pool 少于 preferred hard count，则 representative-random 填满总量，同时保留显式的 minimum-hard stress gate。

## 2. 当前本地 WOMD 暴露出的独立完整性问题

用户 `result.txt` 中 v16.8.13 preflight 报告：

- Scenario training：1000 shards
- Scenario validation：150 shards
- tf.Example validation：150 shards
- tf.Example training：557 shards

旧 preflight 只从“当前存在的文件”中抽样；557 个 training tf.Example 中抽到的 64 个都可解析且含 `sdc_paths`，所以误报 PASS。它没有问“本地是否拥有完整 split”。

Waymax 官方 WOMD 1.3.1 DatasetConfig 使用 training `@1000`、validation `@150`、testing `@150`，并对 1.3.1 开启 45 条 `sdc_paths`、每条 800 points。因此当前 557 个 training tf.Example 不能视为完整 primary training tensor source。

v16.8.14 preflight 升级为 v2：

1. 从本地 shard filename 的 `xxxxx-of-yyyyy` 后缀解析 shard index/总数；
2. 检查缺失 index、重复 index、无法解析的 shard name；
3. primary Scenario/tf.Example training/validation 必须完整；
4. 再抽样验证 128-object tensor、10/1/80 时序、SDC、roadgraph、`sdc_paths` 等 record-level contract。

因此当前机器在补齐 training tf.Example 之前，新 `preflight` 应当 FAIL；这是预期保护，不是新 bug。

## 3. COWP 应使用哪些 WOMD split

### Primary training

- `uncompressed/scenario/training`：COWP label 权威源。
- `uncompressed/tf_example/training`：与 label scenario-ID 对齐后的模型 tensor / Waymax source。

不要用 `tracks_to_predict` 限制 COWP critical universe；它是 challenge/reference target，训练可以选择其他对象。

### Primary validation / promotion

- `uncompressed/scenario/validation`
- `uncompressed/tf_example/validation`

用于 smoke、strict、train/validation data-contract gate 与标准 held-out Waymax/模型验证。最终论文报告建议在 evaluation 阶段从 standard validation 预先固定一个从未参与门控/超参选择的 report manifest；这不要求重新构建 raw labels，可在评估时按 scenario ID 切分。

### `validation_interactive`

把它作为 **secondary interaction stress benchmark**，不要直接替换 standard validation，也不要在未审 scenario-ID overlap 前称其为独立测试集。v16.8.14 新增 `69_audit_womd_split_layout.py`：可盘点本地 Scenario/tf.Example challenge split，并可完整扫描 standard validation 与 validation_interactive 的 scenario-ID 交集/包含关系。

如果它是 standard validation 的子集或高度重叠，只能作为交互子群/压力分层单独报告，不能与 standard validation 混在一起当新增独立样本量。如果本地 release 显示其 scenario IDs 独立，则可作为单独 secondary stress set，但仍需冻结使用协议。

### `testing`

WOMD official test future GT 隐藏。COWP natural roots、root transport、witness、false-safe/NCF mechanism GT 都依赖 future/counterfactual supervision，因此不能从 testing 构建这些离线机制标签。Testing 可用于官方 blind benchmark/submission 或只依赖可见输入/官方 evaluator 的项目，但不能伪造 COWP future-dependent ground truth。

### `testing_interactive`

同理按 blind challenge split 处理：不用于 offline natural/transport/witness supervision。若参与官方 interaction benchmark，按照官方 challenge evaluator/submission 协议使用。

### `training_20s`

当前 COWP pipeline、label engine、response horizon 和 Waymax cache contract 均围绕 9 s WOMD window（10 history + current + 80 future）设计，因此 `training_20s` 不应与 primary training 混用。若未来要利用 20 s source，需要单独定义窗口化、泄漏、重复 segment/group split 与 horizon contract，而不是直接换 glob。

## 4. Scenario proto 与 tf.Example 的职责

COWP 当前正确职责应保持：

- Scenario proto：vector map、全部 proto tracks、traffic control / right-of-way、critical selection、natural/priority/counterfactual label construction。
- tf.Example：固定 tensor input、与 label scenario-ID 对齐、模型训练 cache、Waymax state source、`sdc_paths` route metrics。

原因包括 tf.Example 固定最多 128 objects，以及 traffic-light tensor 只有 16 positions；Waymo 已说明某些仍有效 traffic-light states 可能因 tf.Example 空间限制而丢失，而 Scenario proto 保存全部有效 traffic-light data。因此不能在 tensor-cache 阶段用 tf.Example 重新定义 COWP priority/right-of-way GT。

## 5. v16.8.14 新增的 silent-partial-cache 防线

即使 shard preflight 被绕过，旧 `02_build_tensor_cache` 仍存在“扫描到部分匹配就成功退出”的风险。v16.8.14 新增：

`--require-all-labels-matched`

full-core 的 train/val tensor merge 都默认启用。merge 完成后，对每个 proto-derived label scenario ID 检查是否存在匹配 tf.Example cache；任何缺失立即失败，并打印缺失数量和 ID preview。这样 557/1000 的 training tf.Example 不可能静默生成一个看似可训练但覆盖不完整的 `tensor_cache_train`。

## 6. 推荐执行顺序

1. 先运行 `split-audit`，盘点本地所有 split。
2. 补齐 training tf.Example 到官方 primary contract。
3. 重新运行 strict `preflight`，必须 PASS。
4. fresh v16.8.14 smoke：动态从当前 OLD_VAL_CACHE 生成 96-scene manifest。
5. smoke PASS 后运行 fastpath A/B。
6. strict validation probe。
7. train-pilot。
8. strict + train-pilot 同 fingerprint 均 PASS 后才 full-core。
9. full-core tensor cache 还要通过 all-labels-matched / input-visibility support。
10. 再做 Waymax outcomes。

## 7. 代码验证

v16.8.14 本地回归：206 passed, 5 skipped；`python -m compileall -q cowp` 通过；master/smoke/strict/train-pilot/full-core shell 均通过 `bash -n`。

该结果验证的是代码/契约逻辑，不代表用户机器上的 fresh v16.8.14 smoke 已经通过。真实 promotion 必须以补齐 WOMD 后重新生成的 verdict 为准。
