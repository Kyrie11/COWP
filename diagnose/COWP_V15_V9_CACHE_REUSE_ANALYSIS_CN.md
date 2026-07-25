# COWP v15 复用 transport_v9 数据集的充分性分析

## 1. 结论

本轮不建议从 index、labels、tensor cache、Waymax replay 到 transport overlay 全量重建。

推荐采用两阶段策略：

1. 当前下一轮实验直接复用：
   - `tensor_cache_train_waymax`
   - `tensor_cache_val_waymax`
   - `tensor_cache_train_waymax_transport_v9`
   - `tensor_cache_val_waymax_transport_v9`
2. 将实验协议明确记为 **v15 model + v9 labels**，先验证模型、损失、选择器和真实在线 Waymax 闭环是否改善。
3. 只有当需要在论文中独立证明 v15 新增的 OBS 因果去污染、地图过滤和由此重算的 response/witness 标签贡献时，才需要重新生成 v15 标签数据；不能把 v9 标签实验写成 v15 新标签实验。

因此，“是否足够”的答案需要分开：

- 对可靠训练、自然/transport/planner 阶段验证及真实在线 Waymax 闭环：**足够，但必须通过本次新增的启动前门禁**。
- 对 log-divergence 监督：**不够**。
- 对把稀疏 attached Waymax candidate outcome 当完整离线闭环结论：**不够**。
- 对验证 v15 新标签生成贡献：**不够，因为 v9 没有物化 v15 新标签语义**。

## 2. 原 v15 指令是否从头构建数据集

是。原 `NEXT_RUN_COMMANDS_V15_CN.sh` 调用 `prepare_cowp_v15_data.sh`，其链路包括：

- WOMD index train/val；
- build labels train/val；
- dataset diagnostics；
- build tensor cache train/val；
- Waymax candidate replay train/val；
- attach Waymax outcomes；
- verify Waymax cache；
- augment transport v15；
- raw/overlay alignment diagnostics。

这是一条完整重建链，成本显著高于直接复用 v9 overlay。

## 3. `cache_sufficiency_full` 对原始 Waymax cache 的支持

上传的完整扫描报告记录：

- train：14,640 个文件，全部可读；
- val：5,013 个文件，全部可读；
- train/val scenario ID overlap：0；
- train core natural/response/witness/planner/Waymax 字段完整率：14,639 / 14,640；
- val core 字段完整率：100%；
- train/val 平均有效候选数：50.6385 / 50.6302；
- selected rollout success：98.92% / 100%；
- replay-valid 对全部有效候选的覆盖：23.44% / 23.70%；
- finite log-divergence 覆盖：train=0，val=0。

该报告自己的总决策是 `REUSE_WITH_LOGDIV_DISABLED`：

- 核心 staged training：PASS；
- collision/offroad auxiliary training：PASS；
- logdiv training/evaluation：FAIL；
- learned-offline Waymax outcome metrics：WARN；
- real online Waymax closed loop：PASS_NOT_CACHE_DEPENDENT。

这意味着不需要为了真实在线 Waymax 闭环重新 replay 全训练集。在线闭环直接加载 validation tf.Example 并逐步推进 simulator，attached candidate outcomes 不是在线环境本身。

## 4. 第三个文件是否包含 transport_v9 的性质分析

包含，而且比第二个文件更新。

v14 结果中的 cache alignment 对 train/val 各抽样 2,000 个场景：

- raw 和 transport 文件数一致；
- base payload mismatch = 0；
- critical unmapped = 0；
- critical invisible = 0；
- response root out of range = 0；
- natural nonfinite = 0；
- selected rollout success = 1.0；
- finite logdiv = 0；
- train/val alignment 均 `pass=true`。

其中报告的当前文件数是：

- train raw/transport：20,440 / 20,440；
- val raw/transport：5,013 / 5,013。

模型面对的数据路径预检也通过：

- critical mapping pass；
- first-step anchor pass；
- typed basis 1 s pass；
- typed basis 8 s pass；
- typed basis 8 s mean minADE 约 2.196 m；
- OBS/NEU/PRIO 8 s mean 约 4.568 / 1.134 / 1.294 m。

因此 transport_v9 的结构、索引、anchor 和 v14 typed basis 训练路径有充分证据支持复用。

## 5. 为什么仍必须在服务器启动前重新审计

两个上传报告存在训练集数量差异：

- 较早的 full sufficiency report：14,640；
- 较新的 v14 alignment/training result：20,440。

这很可能表示早期报告扫描时训练 cache 尚未完全构建，或扫描的是同一路径的较早状态。不能仅凭上传的旧 JSON 推断服务器当前目录仍是 20,440 个完整样本。

为此，新的复用脚本在训练前会：

1. 对服务器当前 raw train/val cache 重新做 full scan；
2. 对 raw 与 transport_v9 overlay 做独立数量和字段门禁；
3. 默认要求 train >= 20,000、val >= 5,000；
4. 检查 raw/overlay 文件数一致；
5. 抽样检查 SDC、critical index、natural/response/witness/planner/transport 字段；
6. 检查 response-root 索引范围；
7. 检查 train/val 文件名重叠；
8. 检查 finite logdiv，并阻止把缺失值当真实零监督。

如果服务器上当前 train 仍只有 14,640，脚本会直接失败，而不是静默用缩小的数据集训练。

## 6. v9 数据能否代表真正的 v15 标签协议

不能。

v15 的标签生成新增：

- OBS pressure-contamination 评分、降权和拒绝；
- map-aware natural-option filtering；
- 当前状态连续积分与重算速度/航向；
- 新字段：
  - `cowp/natural/obs_contamination`
  - `cowp/natural/map_compliant`
  - `cowp/natural/map_distance_max`
  - `cowp/natural/map_verified`

这些变化会改变 natural root，继而级联改变 response、witness、candidate NCF 和 transport supervision。

`26_augment_transport_labels.py` 只从已有 natural/response 标签派生 transport 标签，不能把旧 v9 raw cache 自动升级成真正的 v15 natural/response/witness 语义。因此简单对旧 cache 再跑一次 transport augment 也不等价于 v15 重标注。

不过当前 v15 模型并不要求这四个新字段作为网络输入；它们属于标签生成与审计语义。因此 v9 仍可训练 v15 模型结构，前提是实验名称和结论保持诚实。

## 7. 推荐的当前实验定义

建议输出目录：

`outputs/cowp_v15_model_v9labels_seed2026`

数据协议：

`DATA_PROTOCOL=v9_reuse`

可以验证：

- v15 decoder/loss/gate 是否优于 v14；
- OBS 分支是否从约 4.57 m 明显下降；
- transport/planner 是否改善；
- 真实在线 Waymax CR/offroad/progress；
- v15 工程修复是否稳定。

不能据此声称：

- OBS decontamination 标签贡献有效；
- map-filtered natural labels 有效；
- v15 标签协议的完整消融已经完成。

## 8. logdiv 与 Waymax outcome 使用限制

现有安全 replay 的 finite logdiv 为 0。因此：

- `outcome_logdiv` loss 必须为 0；
- 不能报告 SelectedWaymaxMeanLogDivergence 为有效证据；
- 不能把缺失 logdiv 当作真实的 0；
- collision/offroad outcome 可作为稀疏辅助监督；
- learned-offline selected outcome 必须同步报告 replay coverage；
- 最终 CR/offroad 必须来自真实在线 Waymax evaluator。

若论文最终必须包含 logdiv，只建议：

1. 先 replay 完整 validation；
2. 再对训练集做目标化子集 replay；
3. 不需要重放全部 20k 训练场景。

## 9. 修改后的执行方式

默认复用：

```bash
cd COWP_v15_reuse_v9
bash NEXT_RUN_COMMANDS_V15_CN.sh
```

只做 natural 阶段：

```bash
STOP_AFTER_STAGE=natural bash NEXT_RUN_COMMANDS_V15_CN.sh
```

强制重新扫描当前服务器 cache：

```bash
FORCE_CACHE_AUDIT=1 bash NEXT_RUN_COMMANDS_V15_CN.sh
```

只检查 cache，不训练：

```bash
bash CHECK_V9_CACHE_ONLY.sh
```

确实需要完整 v15 重标注时：

```bash
bash NEXT_RUN_COMMANDS_V15_REBUILD_FULL_CN.sh
```

## 10. 最终建议

下一步先不重建数据集。用新增门禁确认服务器当前的 20,440/5,013 raw 与 v9 overlay 完整对齐，然后运行 v15 model + v9 labels。

如果 natural gate 和真实在线闭环没有改善，重建 v15 labels 的投入价值较低，应先继续修模型。如果模型结果明显改善，再针对论文标签 novelty 构建真正的 v15 数据，并将“模型增益”和“新标签增益”做独立消融。
