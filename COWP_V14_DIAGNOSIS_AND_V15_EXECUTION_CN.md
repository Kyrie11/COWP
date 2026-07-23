# COWP v14 实验诊断、v15 工程修复与下一阶段算法计划

## 0. 最终结论

本次上传结果中所看到的“natural gate 没有通过”，首先是一个**执行链和产物完整性问题**，并不是 v14 natural 模型按原门槛发生了数值失败：

1. `NEXT_RUN_COMMANDS_V14_CN.txt` 同时包含 Bash 命令和中文说明，却被整体交给 `bash`，因此日志中出现大量 `command not found` 和最终语法错误。
2. v14 驱动发现 `cowp_natural_best.pt` 已存在后跳过训练，但 `history_natural.json` 不存在；natural gate 必须读取 history，所以没有生成 `natural_basis_gate.json`。
3. 从 `train_natural_ddp.log` 中恢复验证记录后，按原 v14 门槛，最佳 epoch 15 的 natural gate **实际通过**。

但从 CCF-A 投稿和论文核心论点所需的质量来看，v14 仍有实质算法缺陷：总体 set minADE 很好，但 OBS 分支明显弱，且学习残差几乎不起作用。按 v15 的严格门槛，v14 应当被拒绝：

- typed set minADE@8s：**1.8974 m**；
- 1/3/5s：**0.2289 / 0.4480 / 0.8951 m**；
- branch minADE：**2.9117 m**；
- OBS / NEU / PRIO：**4.6060 / 1.1351 / 1.2998 m**；
- branch spread：**3.4708 m**；
- neutral consistency：**1.8647 m**；
- priority BCE：**0.3495**。

v15 对 v14 的严格复核结果在：

`/mnt/data/results_v14/eval/v15_strict_gate_on_v14.json`

失败项只有：

- OBS minADE 4.6060 m > 4.0 m；
- branch spread 3.4708 m > 3.0 m。

因此，正确的定性是：

> **表面上的 natural gate 失败是工程/执行问题；更严格的论文级 gate 失败则是 OBS 建模和分支不均衡的算法问题。**

---

## 1. 我对论文核心 idea 的理解

论文提出的关键问题不是普通的碰撞风险，而是 **false-safe planning**：自车轨迹在 rollout 中没有碰撞，但这一“安全”依赖其他交通参与者通过急刹、突然让行、放弃合理路权或交出间隙来承担冲突。

论文的核心目标是把“礼貌”从一个软成本提升为一个可验证的可行性条件：

> 一个自车候选只有在每个关键交通参与者仍保留足够多的低负担安全响应时，才是 non-coercively feasible。

对应 pipeline 为：

1. 构建 burden-oriented interaction graph，定位潜在的“谁将为谁承担冲突”；
2. 生成 ego candidate trajectories；
3. 为每个关键他车构造 natural alternatives：
   - OBS：观测未来及其合理扰动；
   - NEU：在自车不施压的干预条件下的行为；
   - PRIO：保持路权/优先权的规则约束行为；
4. 在每个 ego candidate 条件下构造他车安全响应集合；
5. 判断自然选项是否被压缩、剩余响应是否高负担；
6. 生成 coercion witness；
7. 对确认依赖他车高负担让行的候选进行 hard rejection；
8. 在剩余 non-coercively feasible 候选中按安全、进度和舒适性排序。

论文真正需要证明的不是“模型能预测他车会让”，而是：

- 自车计划不会把冲突负担转移给他人；
- natural option set 在自车干预后没有被压缩到高负担逃生动作；
- coercion witness 与独立闭环反事实证据一致；
- 降低 FSR/CBS/HBCR 的同时不牺牲 CR、offroad、progress 和 comfort。

这意味着 natural basis 不是普通预测附属模块，而是整个论文论证链的地基。其三个分支必须有明确、稳定、因果上可信的语义。

---

## 2. 数据集和评测链路理解

数据构建采用 WOMD 1.3.1 的两种表示：

- `uncompressed/scenario/...`：用于 proto 标签构建、地图和交互语义；
- `uncompressed/tf_example/...`：用于固定张量输入和 Waymax-ready cache。

当前数据链路是：

1. scenario proto 建索引；
2. 从 proto 构建 COWP 标签；
3. 进行标签诊断；
4. 将 tf.Example 与标签对齐并构建 tensor cache；
5. 在 Waymax 中对平衡采样的 ego candidates 做 replay；
6. 将 candidate outcome 附加到 cache；
7. 构建 transport overlay；
8. 训练 natural、transport/witness、planner；
9. learned-offline 机制验证；
10. Waymax probe/full validation。

v14 的 cache alignment 和 exact model-anchor preflight 是通过的：

- critical unmapped/invisible = 0；
- response root out of range = 0；
- Waymax selected rollout success = 1.0；
- first-step anchor p90 约 0.30 m；
- typed basis 8s minADE 约 2.20 m；
- `waymax_logdiv_finite = 0`，所以 log-divergence 监督必须继续禁用。

日志中 `input_index_vs_original_track_index_anchor_delta_m` 很大，不代表当前映射错误。它是在比较正确的 input-index 行与错误的 legacy original-track-index 行，数值大反而说明若退回 legacy 索引会发生严重错位。当前真正使用的 track-ID 到 input-index 映射在诊断中是完整的。

---

## 3. 为什么 natural gate 没有通过

### 3.1 工程原因：真实的直接原因

`run.log` 明确记录：

- 中文说明被 Bash 当成命令；
- natural checkpoint 被保留；
- 随后提示缺少 `history_natural.json`；
- 最后读取 `natural_basis_gate.json` 时抛出 `FileNotFoundError`。

也就是说，运行失败发生在“产物管理和脚本执行层”，不是 gate 计算后输出 `pass=false`。

v15 已修复：

- 中文说明和可执行脚本分离；
- `checkpoint + history` 被视为一个原子产物；
- 只有 checkpoint 而没有 history 时，不能跳过训练；
- gate 文件必须真实生成后才能进入 transport/planner。

### 3.2 算法原因：按论文级质量仍然存在

恢复训练记录后可以看出：

- epoch 0–1 整体误差很高；
- epoch 2 激活正确的 typed basis/图路径后，8s set minADE 突然降到约 1.90 m；
- epoch 2 到 epoch 15 几乎不再改善；
- 最终 base deviation 约 0.017 m，residual L2 约 8e-4。

这说明 v14 natural 模块的性能主要来自固定运动学原型，不是场景条件下学习出的行为分布。

其中：

- NEU 和 PRIO 分支较好，说明解析运动学先验适合作为约束分支；
- OBS 为 4.606 m，远差于其他分支，说明固定 OBS 原型不能拟合真实的转弯、交互减速、车道曲率和局部行为；
- source CE 接近 0 主要因为 mode source 是结构性固定的，不能作为“模型学会三种语义”的证据。

---

## 4. 工程审计与修复

### 4.1 未来信息泄露

#### 问题 A：未来 natural label 反推编码器当前状态

原模型在缺少真实 `state/history` 或 WOMD current/history 时，会从 `cowp/natural/traj` 的第一个未来点重建 agent input。这使预测标签进入输入编码器，是直接的未来泄露。

#### 修复

v15 默认：

```yaml
allow_label_only_state_fallback: false
```

缺少真实历史/当前状态时直接抛错，要求修复 cache。保留的 legacy fallback 只能用于显式 opt-in 的 toy/migration 测试，禁止用于报告结果。

### 4.2 Waymax logged future 泄露

#### 问题

online policy 的 trajectory 提取函数可以在 simulated trajectory 不可用时静默退回 `log_trajectory`。Waymax 的 `log_trajectory` 包含未来真值，会把闭环评测变成 privileged oracle。

#### 修复

- 主路径只能读取 simulator/current/history；
- `log_trajectory` fallback 默认关闭；
- 只有明确命名的 `logged_oracle` 消融允许访问；
- 主模型缺少反应式预测器时使用基于当前状态的 causal constant-velocity extrapolation，而不是 logged future。

### 4.3 SDC 身份和坐标转换

#### 问题

若 `state/is_sdc` 缺失，旧路径可能默认第 0 个 agent 是自车。WOMD 中 agent 行顺序不能被当作稳定 ego identity；一旦错误，ego-centric 原点、旋转角、critical index、candidate conditioning 和碰撞关系都会错。

#### 修复

v15 报告配置：

```yaml
require_explicit_sdc_index: true
```

缺失 SDC marker 或某个 batch row 没有合法 SDC 时硬失败，不再静默假设 row 0。

坐标处理保持：

- 神经编码输入做 ego-centric 平移和旋转；
-绝对监督标签和 decoder anchor 保留在原 WOMD frame；
-位置、航向、速度矢量使用同一个旋转；
- critical agent 使用显式 `input_index`，并检查 in-range + visible。

### 4.4 OBS 扰动的非物理运动

#### 问题

旧 `resample_logged` 对未来位置做时间/速度/横向偏移，但没有始终从当前状态连续积分，也没有根据新路径完整重算 yaw 和 velocity。结果可能出现：

- 第一帧跳变；
- 位置横移但速度方向仍沿原轨迹；
- 不连续航向；
- 对自然轨迹 minADE 和闭环碰撞造成伪信号。

#### 修复

v15：

- 所有变换从 current state 连续起步；
- lateral offset 使用起点为 0 的 quintic smoothstep；
- 根据新路径差分重建速度和航向；
- 检查 finite、速度和几何连续性；
- 新单测验证首步连续及 velocity/yaw 一致性。

### 4.5 地图合规性

#### 问题

旧代码在自然替代筛选中等价于 `map_ok=True`，论文所声称的 road/map adherence 没有真实执行。

#### 修复

v15 从可用 lane/road point cloud 构造 map corridor，按 agent type 设置距离阈值，对采样点检查：

- compliant fraction；
- maximum distance；
- hard maximum distance；
- map 是否实际可验证。

新增 label 字段：

- `map_compliant`；
- `map_distance_max`；
- `map_verified`。

### 4.6 OBS 不是纯自然行为：因果污染

#### 问题

logged OBS future 是真实场景中“在 logged ego 行为下”的他车未来。若 logged ego 已经逼迫他车减速/让行，直接把它当 natural behavior 会把被胁迫行为写入自然集合，核心 idea 会自相矛盾。

#### 修复：OBS decontamination

v15 估计：

- 他车是否在 ego proximity 内明显减速或损失进度；
- logged ego 与 neutral ego continuation 对他车 clearance 的变化；
- logged 行为是否具有“通过让行来消除 ego conflict”的特征。

得到 `obs_contamination`：

- 高污染 OBS 被降权；
- 超阈值 OBS 被剔除；
- 保留最低权重避免标签完全空集；
- NEU/PRIO 作为相对更干净的反事实支撑。

这项修改直接服务论文核心命题：自然选项必须尽可能表示“没有被当前 ego candidate 施压时”的行为，而不是把已发生的让行当自然基准。

### 4.7 闭环指标计算和命名

#### 问题

旧离线指标将 candidate conventional-safe 的补集称为 `CR`。它只是 label-space proxy，不是 Waymax 实际 rollout 中的 collision/offroad episode rate。

#### 修复

v15 输出：

- `OfflineConventionalUnsafeRate`：标签空间候选安全 proxy；
- `CR_proxy_deprecated`：只为旧结果读取兼容；
- `ClosedLoopCollisionAvailable=0`：明确没有闭环真值；
- 真正 CR/offroad 只接受 Waymax standard metric accumulator 的逐步 episode 聚合。

Waymax 标准指标在每个 simulator step 后更新，并以 episode 内 any-collision / any-offroad 的方式聚合，不再只看末帧。

### 4.8 非 ego 闭环协议

当前 evaluator 实际只控制 SDC，其他 agent 是 logged replay。logged replay 不会根据自车偏离 logged trajectory 做真实反应，因此：

- 可以用来比较自车碰撞、offroad、进度和运动学；
- 不能单独证明“他车负担降低”或“他车保留更多响应”；
- 不能称为论文中写的 logged + learned reactive + rule reactive mixture。

v15 不再虚构该协议：配置明确写明 `actual_non_ego_policy: logged_replay` 和 `reactive_mixture_implemented: false`。

投稿前必须补充至少两个独立协议：

1. Waymax IDM/rule-reactive；
2. 一个冻结的 learned sim-agent/reactive model。

主结论应要求在两类 reactive agents 下方向一致，避免 COWP 只利用某个响应模型的偏差。

---

## 5. 算法层面：哪些有效，哪些无效

### 5.1 应保留并增强的部分

#### Typed OBS/NEU/PRIO root identity

这是 v14 相比 v13 的关键有效修改。它消除了跨 source 全局匹配造成的语义置换，使 same-root transport 有可识别含义。必须保留。

#### Exact model-anchor preflight

它真实覆盖 `dataset -> model history -> critical input index -> anchor -> natural basis` 的生产路径，能在训练前定位 frame/index 错误。必须保留并作为硬门禁。

#### Source-restricted matching

同一 source 内匹配比 global nearest matching 更符合论文的自然分支定义，也为 root-indexed transport 提供稳定监督。必须保留。

#### NEU/PRIO 解析先验

两分支在 v14 已达到约 1.14/1.30 m，说明受约束的运动学基底对反事实/规则分支有效。应保留，并避免过大的自由残差破坏语义。

#### Hard witness rejection 与 option preservation

这两项是论文区别于 courtesy cost、普通风险代价的核心。不能退回单纯 soft burden penalty。后续应强化“校准后硬拒绝 + fallback coverage”，而不是取消。

### 5.2 当前无效或证据不足的部分

#### v14 learned residual

几乎没有移动 analytic basis，未形成真正的 scene-conditioned correction。当前权重/容量设计使其更多是装饰，不能支撑论文中复杂预测模型的叙述。

#### source CE

typed mode source 是固定结构，source CE 接近 0 是预设结果，不是 learned semantic separation。v15 主损失将其权重置 0，只保留诊断。

#### OBS 固定原型

真实观测分支包含道路曲率、转向、交互减速和多种时序，固定 speed/time/lateral 小扰动难以覆盖。它是 v14 natural 的主要算法瓶颈。

#### logged replay 下的在线 FSR/CBS/OPR 因果解释

这些值若来自模型内部预测，只能称为 model-estimated mechanism metrics，不能作为独立因果 ground truth。必须用 reactive rollout 或 label-side counterfactual oracle 交叉验证。

#### 稀疏 attached Waymax outcomes 作为主 selector

平衡采样的每场少量候选 outcome 只能作为辅助 supervision。若直接成为主排序依据，模型容易学习采样偏差。v15 继续要求主机制证据来自 transport/witness，outcome 头只辅助。

---

## 6. v15 算法修改

v15 将自然模块定义为 **Causal Natural Option Basis (CNOB)**：

### 6.1 结构不变的部分

- 仍然是 8/8/8 的 OBS/NEU/PRIO typed roots；
- 保持 source-restricted assignment；
- 保持 analytic kinematic basis；
- 保持 same-root transport 所需的稳定 root identity。

### 6.2 针对 OBS 的增强

- OBS 残差位移/航向/速度容量高于 NEU/PRIO；
- OBS residual gate 增加正偏置，使其在训练初期不再被关闭；
- OBS analytic prior deviation penalty 降低；
- OBS loss 权重提高；
- 通过 decontamination weight 避免网络拟合已被 ego 逼迫的 logged future；
- 通过 map filter 避免用几何上不合理的扰动训练。

### 6.3 针对 NEU/PRIO 的保护

- 更小的 residual bound；
- 更强 base-prior regularization；
- 保留 priority BCE、neutral consistency；
- 防止为了改善 OBS 而破坏两个已经有效的分支。

### 6.4 新 gate

除了总体 set minADE，还必须检查：

- OBS absolute minADE；
- 三分支 max-min spread；
- NEU/PRIO absolute quality；
- typed-untyped gap；
- priority semantics；
- neutral consistency。

原 v14 gate 只看 aggregate/宽松 branch 指标，容易让较强 NEU/PRIO 掩盖 OBS 失败。

---

## 7. 论文与代码不一致

当前 TeX 中存在需要在投稿前解决的重大一致性问题：

- 论文部分描述 OBS transformer multi-modal predictor；
- 描述 NEU conditional diffusion under ego-neutral intervention；
- 描述 learned + rule-based reactive-agent mixture；
- 当前代码实际是 typed analytic kinematic basis + bounded residual，在线 non-ego 主要是 logged replay。

有两种合法路线：

### 路线 A：以当前代码为准修改论文

把自然模块表述为：

- typed causal option basis；
- observationally decontaminated logged proposals；
- ego-neutral and priority-preserving analytic counterfactual primitives；
- scene-conditioned bounded residual refinement。

优点是可复现、实现与论文一致；核心 novelty 放在 false-safe definition、option preservation、same-root transport 和 coercion witness，而不是声称新 diffusion predictor。

### 路线 B：真正实现论文中的 transformer/diffusion

需要额外完成：

- scene/map-conditioned OBS forecasting backbone；
- explicit ego intervention-conditioned NEU diffusion；
- calibrated multi-modal likelihood；
- 与 typed roots 的稳定对应或可微 transport；
- 大量训练和消融。

以当前阶段和结果看，建议先采用路线 A，确保主论点成立。除非已有足够算力和时间，否则临时加入 diffusion 会增加不可控变量，未必提升闭环。

---

## 8. CCF-A / SOTA 的指标门槛估计

不存在一个统一的“CCF-A 数值门槛”。审稿人会同时判断：

- novelty 是否独立成立；
- 实现和论文是否一致；
- 评测是否因果、无泄露、可复现；
- 是否与强 baseline 公平比较；
- 主结果是否有统计显著性；
- 是否在标准指标上不退化，并在新指标上取得稳定优势。

Waymo 官方 2025 Sim Agents 评测以 32 个 8 秒 joint futures 为输入，并同时评价 motion、interaction 和 map adherence；主榜指标是 Realism Meta-Metric，minADE 为 tie-breaker。它说明仅有 displacement 或 collision 单指标不够。COWP 是 planner/certifier，不应把 WOSAC 分数直接当主目标，但应借鉴其闭环分布、交互、地图和运动学完整性要求。

建议把投稿门槛分为四层。

### 8.1 工程可信度硬门槛

必须全部满足：

- 0 个 future-label encoder fallback；
- 0 个非 oracle `log_trajectory` 访问；
- 100% 样本显式 SDC identity；
- critical mapping/unmapped rate 近 0，至少 <0.1%；
- response root out-of-range = 0；
- finite/kinematic/map diagnostics 通过；
- closed-loop CR 只来自 simulator；
- 3 个以上 seed；
- full validation，不以 100/500 scene probe 代替主表。

### 8.2 Natural basis 投稿目标

v15 当前 hard gate：

- typed set minADE@8s <= 8.5 m；
- branch <= 3.0 m；
- OBS <= 4.0 m；
- branch spread <= 3.0 m；
- NEU/PRIO <= 2.0 m。

这些是“允许继续训练”的门槛，不是 SOTA 目标。建议论文主实验目标：

- typed set minADE@8s：**<= 1.5–1.7 m**；
- OBS：**<= 3.0–3.5 m**；
- NEU：**<= 1.2 m**；
- PRIO：**<= 1.2–1.3 m**；
- branch spread：**<= 2.0 m**；
- typed-untyped gap：**<= 0.5–0.8 m**；
- 1/3/5s 不劣于 v14；
- decontamination/map filter 消融能显示机制质量提升，而不是只降低样本数。

### 8.3 Waymax 标准闭环门槛

相对最强 baseline：

- Collision + Offroad 不得显著变差；建议绝对差 <=0.2 percentage point，或相对退化 <=5%；
- progress 差异控制在 1–2% 内；
- comfort/kinematic infeasibility 不退化；
- 多 seed paired bootstrap 95% CI 覆盖清楚；
- 所有方法使用同一候选库、同一 scenario、同一非 ego policy 和同一 rollout horizon。

### 8.4 核心新指标门槛

为了让 false-safe 核心 idea 有足够说服力，建议相对 strongest non-COWP baseline 达到：

- FSR 相对下降 **>=30%**，理想 **40–50%**；
- high-burden ceding rate / HBCR 相对下降 **>=25–30%**；
- CBS 或 max transferred burden 下降 **>=20%**；
- OPR 提升 **>=10–20%**；
- fallback rate 不能大幅上升；
- progress 几乎持平；
- logged replay、IDM reactive、learned reactive 三种协议下方向一致；
- witness precision/recall 或 AUROC/AUPRC 必须校准，尤其在 interaction-heavy stress set。

这些不是官方阈值，而是根据 CCF-A 审稿常见证据强度做的研究目标估计。

---

## 9. 下一步实验顺序

### 阶段 0：本地完整性

```bash
cd /path/to/COWP_v15
pytest -q
bash -n prepare_cowp_v15_data.sh
bash -n run_cowp_v15_dual_gpu.sh
bash -n NEXT_RUN_COMMANDS_V15_CN.sh
```

预期：81 tests passed。

### 阶段 1：重建 v15 数据

v15 改变了 natural label，不能继续用旧 labels/tensor cache 作为论文结果。

```bash
cd /path/to/COWP_v15
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v15
export CUDA_VISIBLE_DEVICES=0
bash prepare_cowp_v15_data.sh
```

这个脚本会正确执行：

- index；
- v15 labels；
- label diagnostics；
- tensor cache；
- Waymax candidate replay；
- outcome attach/verify；
- transport overlay；
- train/val alignment。

### 阶段 2：只训练 natural 并看 gate

第一次建议只跑 natural，避免在 OBS 仍失败时浪费 transport/planner GPU。

```bash
cd /path/to/COWP_v15
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v15
export RAW_TRAIN_CACHE="$COWP_ROOT/tensor_cache_train_waymax"
export RAW_VAL_CACHE="$COWP_ROOT/tensor_cache_val_waymax"
export TRAIN_CACHE="$COWP_ROOT/tensor_cache_train_waymax_transport_v15"
export VAL_CACHE="$COWP_ROOT/tensor_cache_val_waymax_transport_v15"
export OUT_ROOT=outputs/cowp_v15_causal_natural_seed2026
export TRAIN_VISIBLE_DEVICES=0,1
export TRAIN_NPROC=2
export RUN_DIAGNOSE=1
export RUN_NATURAL=1
export RUN_TRANSPORT=0
export RUN_PLANNER=0
export RUN_OFFLINE=0
export RUN_PROBE=0
export STOP_AFTER_STAGE=natural
bash run_cowp_v15_dual_gpu.sh
```

重点检查：

- `eval/causal_protocol_audit.json` 必须 pass；
- `eval/model_anchor_preflight_val.json` 必须 pass；
- `eval/learned_offline/natural_basis_gate.json` 必须 pass；
- OBS 是否从 4.606 m 降到 <4.0 m；
- branch spread 是否 <3.0 m；
- NEU/PRIO 是否保持；
- OBS contamination 分布和 map rejection rate 是否合理；
- residual/base deviation 是否不再近零，但也没有爆炸。

### 阶段 3：完整 v15

natural gate 通过后：

```bash
cd /path/to/COWP_v15
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v15
export OUT_ROOT=outputs/cowp_v15_causal_natural_seed2026
export RUN_AUGMENT=0
export RUN_DIAGNOSE=1
export RUN_NATURAL=1
export RUN_TRANSPORT=1
export RUN_PLANNER=1
export RUN_OFFLINE=1
export RUN_PROBE=1
export RUN_FULL=0
bash run_cowp_v15_dual_gpu.sh
```

或直接：

```bash
bash NEXT_RUN_COMMANDS_V15_CN.sh
```

### 阶段 4：主表前的必要实验

1. seeds：2026、2027、2028；
2. full validation；
3. same scenarios paired evaluation；
4. Pareto / pairmax / no-option-preservation / soft-only / no-decontamination / no-map-filter / untyped/global-match ablation；
5. logged replay + IDM reactive + learned reactive；
6. bootstrap 95% CI；
7. witness calibration curve、AUPRC、reliability diagram；
8. case study 显示：普通 planner collision-free，但 forcing agent 急刹；COWP 选择另一个 progress 接近且保留自然响应的轨迹。

---

## 10. 交付文件

优化代码目录：`COWP_v15`

关键入口：

- `NEXT_RUN_COMMANDS_V15_CN.sh`
- `prepare_cowp_v15_data.sh`
- `run_cowp_v15_dual_gpu.sh`

审计：

- `cowp/scripts/36_audit_causal_protocol.py`
- `V15_CAUSAL_AUDIT_SAMPLE.json`

说明：

- `ALGORITHM_CHANGELOG.md`
- `V15_MODIFICATION_MANIFEST.md`
- 本文档。

恢复结果：

- `/mnt/data/results_v14/eval/reconstructed_history_natural.json`
- `/mnt/data/results_v14/eval/reconstructed_natural_basis_gate.json`
- `/mnt/data/results_v14/eval/v15_strict_gate_on_v14.json`

---

## 11. 当前不能声称的结论

虽然 v15 已经修复已知因果、坐标和指标协议问题，并通过 81 项本地测试，但本环境没有完整 WOMD cache、Waymax runtime 和 GPU 训练，因此当前不能诚实声称：

- v15 已经提升 OBS；
- v15 闭环优于 v14；
- COWP 已达到 SOTA；
- reactive burden reduction 已被独立验证。

可以确认的是：

- v14 的“gate 未生成”根因已定位；
- v14 原 natural 指标已恢复；
- v14 的真实算法短板已被严格 gate 暴露；
- 已知未来泄露、SDC 假设、非物理扰动、地图伪检查和 CR 命名问题已在代码中封堵；
- 下一轮实验将更接近论文核心 idea 的有效验证，而不是继续在错误协议上调参。
