# COWP v15 失败诊断、结果判定与 v16 执行方案

## 一、结论摘要

本轮上传的 `cowp_v15_model_v9labels_seed2026` **没有进入 natural 训练**。
执行在 `35_diagnose_model_anchor.py` 的配置白名单检查处终止：模型配置为
`typed_causal_residual`，但诊断脚本仍只认可旧 v14 decoder 名称。该错误是
**工程版本漂移**，不是数据错位，也不是 model-anchor 数值未达标。

因此，当前结果不能验证以下任何模型结论：

- v15 decoder 是否有效；
- 新 loss 是否有效；
- OBS residual capacity 是否有效；
- natural gate 是否改善；
- planner/selector 是否改善；
- 在线 Waymax CR/offroad/progress 是否改善。

当前结果只证明：v9 raw/transport cache 的结构与映射检查通过，解析 typed
natural oracle 具有可用覆盖；它们不是学习结果。

v16 的默认决策仍是：**先复用 v9 数据验证模型，再决定是否构建真正 v15
数据集**。真正 v15 数据集对“OBS 去污染和地图过滤标签贡献”的论文论证是必需
的，但不是验证 v16 decoder/loss/planner 的前置条件。

## 二、`diagnose_model_anchor` 错误根因

日志中的异常为：

```text
ValueError: Preflight expects typed natural decoder, got 'typed_causal_residual'.
Use configs/train_cowp_v14.yaml.
```

代码对齐后可确认：

1. `NaturalDecoder` 已接受 `typed_causal_residual`；
2. `35_diagnose_model_anchor.py` 维护了另一份过期字符串白名单；
3. 诊断在创建 DataLoader 和计算 anchor 指标之前就抛错；
4. 所以这不是 anchor、critical index、坐标变换或数据数值本身失败。

v16 删除了独立白名单。诊断脚本和协议审计统一读取
`model.natural_decoder.uses_typed_basis` / `uses_dynamic_residual`，避免同类错误。

## 三、当前产物实际包含什么

`cowp_v15_model_v9labels_seed2026` 中有：

- cache sufficiency；
- raw/transport alignment；
- transport label diagnostics；
- natural analytic oracle；
- 失败的 model-anchor 日志。

没有：

- `checkpoints/natural/cowp_natural_best.pt`；
- `history_natural.json`；
- `natural_basis_gate.json`；
- learned-natural effectiveness；
- transport/planner checkpoint；
- selector/mechanism verification；
- online Waymax probe/full result。

因此后续所有 learned claim 都必须标记为 `NOT_EVALUATED`，而不是“未提升”。

## 四、当前可使用的数据证据

### 4.1 v9 cache/overlay 工程可靠性

抽样 2000 场景的 train/val alignment 均通过：

- train raw/overlay：20440 / 20440；
- val raw/overlay：5013 / 5013；
- base payload mismatch：0；
- critical unmapped/invisible：0；
- response root 越界：0；
- selected Waymax rollout success：1.0；
- finite logdiv：0。

transport diagnostics 也通过，train/val mode-conflict rate 约为 0.3527/0.3531。

### 4.2 analytic typed basis 不是 learned decoder 结果

2000 个 validation 场景上的解析 kinematic bank oracle：

| 指标 | mean minADE |
|---|---:|
| all@1s | 0.2834 m |
| all@3s | 0.6075 m |
| all@5s | 1.2041 m |
| all@8s | 2.4655 m |
| OBS@8s | 3.0104 m |
| NEU@8s | 2.1133 m |
| PRIO@8s | 2.5943 m |

这只说明标签空间中存在可用解析覆盖，不能回答残差有没有学习、loss 是否有效。

### 4.3 cache 报告中的隐藏文件假异常

旧 sufficiency 报告显示 train `files_total=20441`，但训练 Dataset 和 alignment
均为 20440。原因是隐藏的 sampler/metadata `.npz` 被诊断脚本误计为场景，并被
判为“缺失全部标签”。v16 让诊断脚本与 Dataset 使用相同过滤规则，隐藏文件不再
计入场景。

## 五、用户要求的七项验证状态

| 验证项 | 当前状态 | 原因 | v16 验证方式 |
|---|---|---|---|
| decoder 是否有效 | 未验证 | 无 checkpoint | learned-vs-analytic gain + physical gate |
| 新 loss 是否有效 | 未验证 | 无训练，且单模型无法归因 | same-decoder/no-new-loss 消融 |
| OBS capacity 是否有效 | 未验证 | 无训练，且无容量对照 | `obs_capacity_scale=0` 消融 |
| natural gate 是否改善 | 未验证 | 无 history/gate | 两级 natural gate，与 v14 同 seed 比较 |
| planner 是否改善 | 未验证 | 无 planner checkpoint | learned-offline mechanism gate + baseline delta |
| selector 是否改善 | 未验证 | 无 selector eval | BCOT vs pairmax/Pareto 消融 |
| online CR/offroad/progress | 未验证 | 未进入 Waymax | paired probe，再 full validation |

## 六、算法层面主要缺陷与 v16 修复

### 6.1 v15 decoder 名称变化但机制没有真正变化

`typed_causal_residual` 实质上仍走旧 typed residual 路径。它不能成为有力 novelty。

**v16：CNOB dynamics decoder**

- 预测 bounded longitudinal/lateral acceleration；
- 预测 jerk 和 yaw-rate correction；
- 通过积分生成 velocity/position；
- 运动时 yaw 从 velocity 方向得到；
- length/width 不允许被自然轨迹 decoder 修改；
- 零初始化严格等于解析 typed basis。

论文表述可收敛为：typed causal option prior + dynamics-consistent learned correction
+ same-root counterfactual transport，而不是简单的“多模态残差预测”。

### 6.2 旧残差可能生成非物理解

旧实现独立修改 position、yaw、velocity、size，可能出现位置曲线、速度方向、航向
互相矛盾。v16 从控制量积分，增加 finite-difference velocity、velocity-heading、
control smoothness 检查。

### 6.3 旧 gate 允许残差完全不工作

只要解析 basis 本身达到绝对 minADE，residual 为零也可能通过。v16 新增：

- overall learned improvement；
- OBS learned improvement；
- NEU/PRIO preservation；
- effective mode usage；
- residual bound；
- kinematic consistency。

因此不能再把解析先验性能误写成 learned decoder 性能。

### 6.4 新 loss 与 OBS capacity 原来不可归因

v16 增加两个严格控制的自然阶段消融：

1. 相同 dynamics decoder 和容量，去掉新 effectiveness/preservation/physical loss；
2. 相同 decoder 和 loss，仅把 OBS 专属容量增益设为 0。

`41_compare_natural_ablations.py` 要求主模型分别显著优于两者，否则对应组件 claim
不成立。

### 6.5 natural 权重未完整进入 branch minADE

真正 v15 标签会通过 contamination 调整 natural weight。旧 branch minADE 不带权，
会削弱去污染的意义。v16 改为 weighted source-restricted matching。

### 6.6 planner/selector 可能被通用分类器替代

generic candidate certificate 能在不学 same-root option transport 的情况下取得较好
分类指标。v16：

- planner checkpoint 更偏重 set-transport、candidate budget、root recovery；
- generic certificate 权重下降；
- `candidate_cert_allow_hybrid_fallback=false` 现在被真正执行；
- mechanism gate 仍要求 transport-specific AUPRC、NCF recall、accepted rate、false-safe
  improvement。

### 6.7 稀疏 Waymax outcome 有选择偏差

只有约 23.7% valid candidates 有 replay outcome，且不是随机缺失。collision/offroad
只能作为辅助监督；v16 下调相关权重。finite logdiv 为 0，因此 logdiv loss/report 保持
关闭。

### 6.8 在线 Waymax 协议仍非 reactive-agent

当前 evaluator 控制 SDC，其他 agent 使用 logged replay。它能比较 SDC 的 CR、
offroad、progress，但不能证明“其他交通参与者真实响应下 burden 降低”。最终论文
还需要独立 reactive-agent/sim-agent 协议；v16 不伪称已实现。

## 七、是否现在需要构建 v15 数据集

### 现在不需要完整重建

v9 数据足以验证：

- CNOB dynamics decoder；
- 新 loss；
- OBS capacity；
- natural gate；
- transport/planner；
- BCOT selector；
- SDC logged-replay Waymax CR/offroad/progress。

先用同一 v9 数据做 v15/v16 对比，可以把收益归因到模型而不是数据变化。

### 最终论文需要真正 v15 数据

v9 中没有经过新规则生成的 natural roots/weights，也没有物化：

- `obs_contamination`；
- `map_compliant`；
- `map_distance_max`；
- `map_verified`。

虽然这些字段主要用于审计，真正的学习差异已经写入 root validity 和 natural weight，
因此只有重新生成标签/cache，模型才会学习去污染与地图过滤后的目标。

建议顺序：

1. v16 model + v9 labels；
2. 如果模型 gate/闭环不提升，继续改模型，不重建；
3. 如果模型稳定提升，构建 interaction/OBS-heavy 的小规模 v15 pilot；
4. pilot 显示新标签额外增益后再做完整 v15 rebuild；
5. 最终将“模型贡献”和“标签贡献”分别消融。

## 八、下一步命令

### 8.1 主模型：先只跑 natural

```bash
cd COWP_v16
STOP_AFTER_STAGE=natural bash NEXT_RUN_COMMANDS_V16_CN.sh
```

必须得到并通过：

```text
.../natural_basis_gate.json
.../learned_natural_effectiveness.json
.../natural_effectiveness_gate.json
```

### 8.2 组件归因消融

主 natural 通过后：

```bash
MAIN_OUT_ROOT=outputs/cowp_v16_cnob_dynamics_v9labels_seed2026 \
  bash RUN_NATURAL_ABLATIONS_V16_CN.sh
```

必须检查：

```text
outputs/cowp_v16_natural_ablations_v9labels_seed2026/
  natural_component_attribution_gate.json
```

### 8.3 transport/planner/selector/Waymax

前两类 gate 通过后：

```bash
bash NEXT_RUN_COMMANDS_V16_FULL_CN.sh
```

默认先进行 learned-offline mechanism verification，再允许 online probe。完整
`RUN_FULL=1` 时会额外生成：

```text
eval/waymax/delta_conventional_vs_cowp.json
eval/waymax/delta_planner_vs_cowp.json
```

如需正式 1000-scene full run：

```bash
RUN_FULL=1 FULL_SCENARIOS=1000 bash NEXT_RUN_COMMANDS_V16_FULL_CN.sh
```

### 8.4 通过标准

- natural absolute gate 与 effectiveness gate 均通过；
- 新 loss/OBS capacity attribution gate 通过；
- learned-offline mechanism verification 通过；
- probe 中 CR/offroad 不恶化且 progress 不显著下降；
- full validation 至少 3 seeds，报告 paired CI；
- 然后才进入 v15 label pilot/full rebuild。

## 九、尚未解决、不得隐瞒的问题

- 本地环境不能运行完整 WOMD/Waymax 训练，因此 v16 尚无真实提升数据；
- v9-reuse 不能验证 v15 标签贡献；
- logged-replay 不能证明 reactive burden reduction；
- validation 上调 threshold 再在同一 validation 报告结果存在选择偏差，正式论文应保留
 独立 calibration split 或使用 test/leaderboard；
- 在获得多 seed、full validation、置信区间之前不得声称 SOTA。
