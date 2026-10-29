# RIDE v13：模型证据 → 事前选择 → 独立响应后果（2026-10-29）

## 0. 判读边界

本版本不是 GameFormer、RACP、HYPE 的官方复现，也不是新规划器；是一个 **可在新算法训练前执行、严格按时间冻结选择** 的机制诊断器。

- 两个 CARLA family 使用相同引擎与地图，**只能称为不同响应策略**，不能称为两个独立的人类驾驶真值。
- `SOURCE_THIRD_PARTY` 不是实际在线网络：它使用单一来源响应模型下预先模拟的完整候选结果，因而是信息特别充分的 **source-model lookahead oracle diagnostic**。如果它在另一响应策略下仍出错，说明具有继续验证的模型依赖风险；**不能推出任何官方强基线失败**。
- `PROGRESS_ONLY` 是透明参考，不是 hard courtesy/RACP 的作者实现。
- 早期 v12 全部属于历史数据，**新定义的指标和参数选择是事后探索，不构成 v13 前瞻性独立确认**。
- 危险 TTC（1.5s）与持续 C 急刹（<-4m/s² 约 0.2s）沿用主要风险代理；C 连续负担为**新增次端点**，先冻结再运行 v13。特别区分强碰撞、安全、一般舒适外部性。
- C 受影响并不自动证明 B 介导了 C；若声称 B→C 因果链，必须额外对 B 响应实施介入控制。

## 1. 科学问题与证据

> 在无共享响应协议的异质交通中，有限或失准的他车响应证据，会在什么条件下让 ego 对**特定第三方 C** 的风险/负担估计缺乏分辨力或判断错误，并在**任务进展匹配的动作之间造成真实的选择损失**？该损失是否能在行为模型相异的独立评测环境中复核，并在公平强规划器中仍存在？

对应三个分层假设：

**RQ1' Evidence decision-relevance：** 固定 scene、C/B 身份、合法的 ego 任务匹配动作对，判断 source-model 对动作的风险差异是否为不正确反向排序，或将确有明显后果差异的两个动作错误判为平局。R1 的结论是 “model misses a consequential distinction”，**不等于**选择错误。

**RQ2 Avoidable selection harm：** source-model 评分器先锁定选择；另一响应评测环境随后对两个动作完整执行。若选择的第三方损失高于任务匹配替代动作，且初态、ego 执行、合法性全部通过，则发现 simulator-conditional 决策损失。要求严谨的 forward choice commitment，不能事后拿 target 结果挑选动作。需要独立道路场景的持久复现，避免局部偶然。对 C 仅舒适负担而非碰撞的结果，必须用 burden/comfort 命名，不可称安全事故。

**RQ3 Residual strong-baseline gap：** 同候选池、同观测信息、调参独立、无停车投机，让作者 GameFormer-Planner、RACP/HYPE 与经过调参 hard courtesy/risk-aware selector 提前选择；如在相同 target environment 下还有稳定第三方后悔值，才有值得设计算法的实证空白。当前不应宣称任何强 baseline 已失败。

形式化：令 source 模型的候选风险/负担估计为 \(\widehat L_M(s,a)\)，独立 response evaluator 观测到的相应仿真损失为 \(L_E(s,a)\)。对动作对 \((a,b)\)，定义 `blind`：source 分数近似平局但 evaluator 差异明显；`misrank`：source 与 evaluator 次序严格不一致。仅当事前选择 \(\hat a_M\) 且 evaluator 中有符合合法性/进展要求的更优候选 \(b\)，报告 \(L_E(s,\hat a_M)-L_E(s,b)>\delta\)。这个差异是在**指定响应策略下的模拟反事实**，不是从 WOMD 观察数据识别出的人类因果效应。

注意 PDF 对比：Courteous 提出他车负担，RACP/HYPE 考虑交互和响应概率，PDT 和 iDriveVLA 研究普适选择/评价偏差，SIGMA 研究决策关键不确定性。RIDE 的独特主张需始终限定到**具体第三方、独立响应证据的模型错配、任务匹配决策后果及有限响应证据边界**。该独特性是待检验的候选 gap，非保证创新。

## 2. V13 机制诊断的新协议

实验具有两个阶段：

1. **Source 模型评估：** 首先冻结 CARLA 场景、速度、道路身份、候选对。仅运行一个 source family（推荐 `behavior`）；该模型预模拟全部动作。若有异常、同源动作不匹配、缺 C 速度/加速度/合法性，则**UNKNOWN**，不能给动作安全标签。对每对动作，运行 `SOURCE_THIRD_PARTY`（先最小化原主风险，再最小化 C 负担；预测平局由进展/词典序裁决）和 `PROGRESS_ONLY`（同候选池、只依据任务进展的透明负对照），输出 choices 和 SHA256 锁定。
2. **Independent target evaluation：** 在 choices + commitment **创建之后**，执行目标响应 family（推荐 `tm`）。使用严格配对要求：两条动作合法完成、真实 progress 差 <=3m、同初态匹配、**同一 ego 动作跨 family 执行轨迹每步位置偏差 <=0.5m、速度差 <=0.75m/s**。不满足则记录 UNKNOWN。用不依赖 source 评分的 target 结果判断是否存在安全优先级损失或 C 连续负担损失。

来源和目标的仿真都使用现实可见的 CARLA 物理环境。Source **不等于部署时拥有未来 ground truth**：这里模拟器的完整 source response trajectory 是一个诊断型 oracle，可用于先论证“模型假设错配在原则上导致选择损失”；后续必须以事前的学习式模型分数/概率取代这个 oracle，才能研究真实强基线缺口。

### 预注册评价规则（本轮开始前固定）

- `C_comfort_cost = sum_{t>=1.0} max(0, -a_C(t) - 2.0) * dt`，单位 m/s（超舒适制动加速度累计积分）。
- 风险原端点优先：B–C 物理碰撞、C 持续急刹（>=0.2s 且 a<-4m/s²）、B–C 短 TTC (<1.5s)。不得用舒适负担抵消碰撞。
- `source` 差额 <=0.05 m/s 视为**候选证据近似平局**；`target` 差额 >0.25 m/s 才记为本研究协议下有实质意义的舒适负担差异。这两个数值是 v13 *pilot* 的预注册决策规则，尚未通过控制重复性标定；若估计的测量噪声接近或超过阈值，应重新冻结另一版协议，不能对 v13 target 后验调参。
- 预注册的动作对：`early_slow` vs `late_slow`，`mild_early` vs `mild_late`，`speed_early` vs `speed_late`。各组必须满足实际 ego progress 差和动作物理不同，不与 `cruise` 视为任务匹配（`cruise` 仅负对照）。
- 不把多个动作对、同 seed 下多个 trial、同一走廊的不同 seed 当成独立样本。最终至少需多个空间去重走廊、独立确认、针对审稿人关心的自选择场景偏差报告覆盖和无效分母。

`ride.v13_model_transfer`：`lock`→`evaluate`、`retrospective`。`ride.v13_aggregate`：90m 空间走廊聚类并拒绝将 posthoc 与 prospective 数据混合。

注意：`v13_model_transfer` 是一个 **新证据层**，没有修改 CARLA 0.9.16 的控制器/物理交互，沿用 v12 的 `ride_lane_corridor_v12_20261028` 场景协议。每个阶段的输入、文件 SHA256、地图、种子、冻结 anchor 均记录。软件 SHA 不等于外部可信时间戳；应保留原始 shell / server log。

## 3. 服务器运行——开发性试验

将 `RIDE_V13_EVIDENCE_TRANSFER_20261029.zip` 上传到服务器后：

```bash
unzip -q RIDE_V13_EVIDENCE_TRANSFER_20261029.zip -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V13_EVIDENCE_TRANSFER_20261029
conda activate mfrp
python -m pip install -e '.[test]'
python -m pytest -q
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
export CARLA_TOWN=Town04
RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04
bash scripts/29_v9_carla_preflight.sh Town04
```

**开发性两阶段模型迁移探测：**

```bash
export RIDE_V13_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v13_dev_behavior_to_tm_seed17
export RIDE_V13_SOURCE_FAMILY=behavior
export RIDE_V13_TARGET_FAMILY=tm
export RIDE_SEED=17
export RIDE_ANCHORS=4
export RIDE_WORLD_RESET=actors
export RIDE_WARMUP_TICKS=20
export RIDE_CRUISE_MPS=7.5
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_CHAIN_BC_GAP_M=20
export RIDE_CHAIN_C_SPEED_DELTA_MPS=0.5
bash scripts/40_v13_two_stage_probe.sh
cat "$RIDE_V13_ROOT/v13_target_decision_audit.json"
```

脚本按顺序执行：冻场景 → 只运行 BehaviorAgent → 事前选择与锁定 → 运行 TM → 独立结果审计。若 M0 对 source 无完整候选、不能找到独立 corridor、CARLA 服务器不健康，脚本立即失败并保留部分诊断；**不能跳过锁定直接运行 target**。

`RIDE_V13_ROOT` 必须是全新目录。若要检查反方向模型是否也可能犯错，新开一个目录，并交换 `RIDE_V13_SOURCE_FAMILY=tm`、`RIDE_V13_TARGET_FAMILY=behavior`；将其作为**模型方向敏感性**，不要与主方向直接合并统计。

## 4. 确认实验：新的物理走廊/seed，避免与全部历史开发集重叠

本对话已经查看过多版 v11/v12 数据，所以这些结果**不再是新的独立确认集**。做新确认前，列出全部曾用于发现/调参的 frozen manifests，用空格分隔传递给排斥参数。示例：

```bash
export RIDE_EXCLUDE_ANCHORS_FROM="\
/data0/senzeyu2/dataset/RIDE/evidence/v12_dev_bc20_rear35_seed17/frozen/chain_cut_in_anchors.jsonl \
/data0/senzeyu2/dataset/RIDE/evidence/v12_confirm_bc20_rear35/seed_23/frozen/chain_cut_in_anchors.jsonl \
/data0/senzeyu2/dataset/RIDE/evidence/v12_confirm_bc20_rear35/seed_41/frozen/chain_cut_in_anchors.jsonl \
/data0/senzeyu2/dataset/RIDE/evidence/v12_confirm_bc20_rear35/seed_73/frozen/chain_cut_in_anchors.jsonl"
export RIDE_EXCLUDE_RADIUS_M=90
export RIDE_SEED=101
export RIDE_ANCHORS=6
export RIDE_V13_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v13_holdout_behavior_to_tm_seed101
bash scripts/40_v13_two_stage_probe.sh
```

上面路径是根据以往运行命令的目录布局示例，执行前应先 `ls` 验证。对于尚未纳入排斥清单的其他历史 development manifests，也应加入；**不能只排除四个文件便宣称所有既有道路均未看过**。

如果严格道路排斥使 Town04 找不到足够的 corridor，须在新的独立 CARLA Town 上重复准备+preflight；不要降低排斥距离来寻找论文正例，也不要在旧 Confirm 目录重新运行。

第二、第三确认 seed 同理，但应继续排除前面确认 seed 的 frozen 场景以获得更多独立物理走廊，或显式将 seed 仅当同一 corridor 的随机重复。建议在算力允许时至少覆盖 ~20 条不同物理走廊、多个随机种子。这个数字是设计目标，不是保证显著性的阈值。

多个正式选择报告（同源模型方向及相同冻结参数）生成空间去重审计：

```bash
python -m ride.v13_aggregate \
  --reports /ABS/PATH/v13_holdout_seed101/v13_target_decision_audit.json \
            /ABS/PATH/v13_holdout_seed103/v13_target_decision_audit.json \
  --radius-m 90 \
  --output /ABS/PATH/v13_holdout_physical_clusters.json
```

聚类结果不能代表现实道路总体的出现频率。显著性需要按物理走廊、预注册差异方向/幅度、独立模型重现进行区间估计；不要用候选数直接 bootstrap。

## 5. 只读历史材料 / 科学含义

对既有 v12 已执行完毕的双 family 文件可以使用：

```bash
bash scripts/41_v13_readonly_historical.sh \
  /data0/senzeyu2/dataset/RIDE/evidence/v12_confirm_bc20_rear35/seed_41/carla/chain_cut_in.jsonl \
  /data0/senzeyu2/dataset/RIDE/evidence/v12_exploration_readonly_seed41.json
```

历史报告固定标注 `POSTHOC_RETROSPECTIVE_DIAGNOSTIC_NOT_PRECOMMITTED`；**永远不能从该结果反推“已经有事前选择错误”**。

## 6. 如果有正例 / 没有正例，下一步路线

- **有源模型模拟下的选择错误，且独立 target 再现：** 记录完整案例 trace、来源 model 在各候选的分数、目标执行差异、ego progress 和初态；用反向模型、不同地图、复现次数验证。随后使用 SMART/MTR/Bosch 作为**真正不同架构的反应环境**验证；但它们也不是人类真值。尚不能声称强算法系统性失败。
- **只有源证据平局、目标有差异，但选择了较好的动作：** 支持 RQ1 的“错误无法区分”，**不支持 RQ2 的实际选错**。
- **目标 C 负担存在，却没有危险 TTC/碰撞：** 如实称为 *third-party comfort burden*，不可称为严重安全问题。可以研究可避免舒适外部性，但与 courtesy 已有工作交叠，必须补强模型错配/证据充分性价值。
- **仍然没有可辨识差异，或大部分被初态和 ego 执行检查排除：** 不无限收缩车距寻正例。下一步优先修复严格同 ego 干预控制/轨迹复现，并用自然化 WOMD/nuPlan 场景交互分层验证；如充分有效样本仍呈零结果，收缩或转向论文中心。
- **如果要证明“有限 response bank 导致错误”，** 需要与模型错配分开做：保持同一响应策略、同一候选、同一验证器，使用预先固定的嵌套 K=1/2/4/8... 响应库、独立高覆盖参考；只记录“找到的正向 witness”和 UNKNOWN，不能把 K 次没有找到误写成 UNSAT。当前 v13 只针对 **模型失准** 的证据链实施了前瞻性测试；不宣称 finite-bank 假设已经被证明。

## 7. RQ3 / 算法设计的启动条件

最先用透明、充分调参的 hard courtesy 与 risk-aware scoring（保留通用安全与 ego progress）在相同有限响应证据、相同 action pool 上选动作；其次作者原始 GameFormer-Planner/HYPE/RACP。在独立 target 中执行选择结果，记录决策前 candidate scores、checkpoint SHA、信息时序、预算、invalid/UNKNOWN。若 hard courtesy 使用正确响应模型已排除问题，应诚实说明 gap 依赖模型失准；若强 planner 同样能检测并阻止，则不要声称他们系统失败。

只有发现至少一个稳健、可理解的失败机制，且强 baseline 在其真实适用假设下不能充分解决，才值得将 **decision-critical third-party response evidence** 提升为算法创新中心。算法候选是“对候选特异性的第三方响应证据进行验证/可信度控制，并在不完整证据下保持 UNKNOWN、主动分配验证预算”，但此阶段不训练网络。

## 8. 必须进行 SAME-family placebo，校准重置与响应随机性的噪声下限

因 actor-reset 不保证 Unreal 隐状态完全重置，建议在正式异模型迁移之前或并行运行 **BehaviorAgent→BehaviorAgent 同模型负对照**。仅两次执行的差异可估计重放不稳定性，**绝不能作为响应模型错配错误统计**。

```bash
export RIDE_V13_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v13_placebo_behavior_to_behavior_seed89
export RIDE_V13_SOURCE_FAMILY=behavior
export RIDE_V13_TARGET_FAMILY=behavior
export RIDE_V13_PLACEBO=1
export RIDE_SEED=89
export RIDE_ANCHORS=4
# 其余配置与试验保持一致
bash scripts/40_v13_two_stage_probe.sh
```

报告包含 `same_model_placebo=true` 且 `diagnostic_model_induced_choice_errors=0`（这个字段按定义不用于同模型）。对同模型两次的 C 负担差异另行检查：若大量同模型重复具有大于 0.25m/s 的实际差额，v13 异模型结果不具有足够归因力，应先修正 reset 和轨迹一致性，**不能把模型随机漂移误认为系统性错误**。

Placebo 应尽可能在相同地图、但不占用独立 confirm holdout 的预注册开发场景进行；不要因此读取新的确认结果后调节阈值。
