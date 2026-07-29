# OC-RAP 外部 Baseline 复现、数据接入与闭环加速说明

## 1. 对论文方法与实验设计的理解

论文提出 **OC-RAP（Observation-Consistent Recovery-Affordance Planner）**。它不是直接预测一个“最可能未来”后做风险最小化，而是把每个短时可执行候选前缀放入如下流程：

1. 编码场景历史、地图/路线、候选前缀与可见性；
2. 生成只保留恢复判定所需信息的 recovery-sufficient latent roots；
3. 预测前缀执行后的 observation embedding，并将部署时不可区分的 latent roots 归入同一 observation-equivalence class；
4. 对每个 root–recovery-option 预测 signed recovery margin；
5. 使用 OC-MERO，要求同一观测等价类共享一个可部署恢复选项，再进行 lower-tail 聚合；
6. 用校准后的 CRISP admission rule，在硬约束、伤害阈值和 deployable recoverability 条件下尽量保留 nominal utility。

论文的核心比较对象不是单纯的 collision risk，而是 **oracle recoverability 与 deployable recoverability 的差异**。主要指标为：

- FRA：被选中/准入动作实际不可部署恢复的比例；
- ODG：oracle recovery 与 observation-consistent deployable recovery 的差距；
- DRS：执行候选后，仅依据可观察信息选择共享恢复动作时的成功率；
- NUP：相对 nominal prefix 的效用保留；
- contact/post-contact：secondary collision、stable stop、yaw-rate violation、route rejoin、harm/impact severity 等。

上传的数据构建命令把 safe、near-contact、contact 分开采样。near-contact 和 contact 使用 targeted futures、隐藏根、可见扰动根以及 oracle-artifact 配对；contact 还加入 contact impulse surrogate 与 secondary-collision approach。该结构决定了 baseline 必须在动作选择时只使用部署可见量，而教师反事实张量只能用于训练标签或选择后的评估审计。

## 2. 外部 Baseline 与原论文的对应关系

### 2.1 Safe regime

| 代码名 | 对应思想 | 本项目中的诚实实现 |
|---|---|---|
| `nominal_replay` / `log_replay` | logged nominal planner | 直接选择 nominal prefix，不训练 |
| `wayformer_bc` | Wayformer attention encoder/decoder 与 multimodal scene fusion | scene/history/map/route/candidate token Transformer，候选 logits，logged nominal imitation target |
| `gameformer_lite` | GameFormer hierarchical level-k interaction reasoning | scene encoder + 多层 level-k decoder；每层读取上一层 ego/neighbor future，而不是把候选动作误当成不同 agent |
| `betopnet_lite` | BeTop behavioral topology prior + topology-guided planning | actor/map topology token、topology supervision、top-k topology attention和最终候选策略头 |

这些实现是 **OC-RAP candidate-lattice adapter**，不是直接复制官方完整代码、官方数据预处理和官方 checkpoint。论文中应使用 `Wayformer-style BC adapter`、`GameFormer level-k adapter`、`BeTopNet-lite topology adapter` 之类的准确名称，避免声称是官方逐行复现。

### 2.2 Near-contact regime

| 代码名 | 对应思想 | 本项目中的实现 |
|---|---|---|
| `marc_lite` | policy-conditioned critical futures、semantic multipolicy、dynamic branch point、risk-aware contingency | 按 semantic macro 分组，在每个 macro 内选代表候选；使用公共前缀长度、观测未来风险、backup margin 和效用进行 policy-level 比较 |
| `racp_lite` | multimodal intent belief、long-term contingent plans、belief-weighted risk | 从当前 agent history 构造多模态行为假设和先验权重；使用 prior-predictive expected/CVaR risk，不用未来教师 margin 伪造 Bayesian posterior |
| `expected_risk_filter` | expected risk planner | 对观测生成的多模态 future loss 求期望并过滤 |
| `cvar_risk_filter` | tail-risk planner | 对相同 future loss 求 upper CVaR |
| `dro_cvar_filter` | Wasserstein/DRO-CVaR 思想 | CVaR 加 ambiguity-radius × loss dispersion 的有限场景近似 |
| `predictive_safety_filter` | predictive safety filter / backup control barrier | 检查 control bounds、stopping/backup margin 与 barrier decrease 条件；不满足时投影到最接近的可行候选 |
| `oracle_recovery_filter` | 论文定义的 branch-wise oracle recovery upper bound | **唯一允许在动作选择前读取 `m_star/r_orc_star` 等教师量的非部署上界** |
| `gameformer_lite` | learned interactive planner | 与 safe 版本共用部署特征契约，在 near-contact 数据上重新训练 |

### 2.3 Contact/post-contact regime

| 代码名 | 对应思想 | 本项目中的实现 |
|---|---|---|
| `postimpact_mpc_lite` | planning-integrated post-impact MPC，稳定恢复与 secondary-collision avoidance 协同 | 在已有有限候选轨迹上计算稳定性、adhesion、safe-braking-distance、观测碰撞风险和 route-rejoin 代价 |
| `post_crash_braking` | post-crash braking / stable-stop rule | 优先 brake/yield/pull-over/stabilize 宏动作，联合 terminal speed、yaw rate、jerk 与 secondary-risk |
| `post_collision_restoration` | post-collision Ackermann trajectory restoration heuristic | 两阶段 steering/tractive-force shape proxy + yaw/lateral restoration + speed preservation + collision risk |
| `severity_minimization` | unavoidable-collision mitigation | 最小化 collision probability、relative-speed severity、delta-v proxy、residual energy 与 instability |

contact 组同样是有限 candidate lattice 上的 planning/control adapter。它保留论文的目标函数和约束顺序，但没有声称替代原文的高保真车辆动力学、连续 MPC 求解器、轮胎模型或控制分配器。

## 3. 原实现中会被审稿人质疑的问题

### 3.1 教师/未来信息泄漏

原实现的多个 baseline 在动作选择时读取：

- `m_star`；
- `r_orc_star` / `r_dep_star`；
- `hard_violation` / `harm_proxy` 的反事实教师值；
- GameFormer branch context 中的 root/margin tensor。

这些字段由 OC-RAP recovery teacher 和反事实 rollout 产生，部署时不可获得。若 baseline 用这些字段选动作，再与 OC-RAP 比较，会把 baseline 变成 hindsight/oracle diagnostic，而不是公平的 deployable planner。

本次修改后：

- 非 oracle baseline 的动作选择只使用 history、map/route/candidate prefix、candidate kinematics、模型 logits 和观测生成的 future-risk surrogate；
- 教师张量仅在选定动作之后用于统一计算 FRA/ODG/DRS 等；
- `oracle_recovery_filter` 明确标记为 teacher-only upper bound；
- checkpoint 写入 `input_contract.version=2`，旧的 teacher-conditioned checkpoint 会被脚本拒绝并要求重训。

### 3.2 Learned baseline 的训练目标不忠实

原始 learned selector 把 OC-RAP utility、hard/harm、oracle/deploy recovery heads 混入选动作分数，使 Wayformer/GameFormer/BeTop 变成 OC-RAP 标签蒸馏器。

修改后：

- 默认 supervision target 为 `logged_nominal`；
- `allow_teacher_supervision=false`；
- Wayformer 使用候选策略分类损失；
- GameFormer 使用策略损失、逐 level-k 响应损失和 best-of-M Gaussian trajectory NLL；
- BeTop 使用策略损失与 topology supervision；
- teacher recovery auxiliary losses 默认全部为 0。

### 3.3 GameFormer 交互对象错误

原实现把不同 candidate prefix 当成不同 interacting agents。修改后，level-k decoder 使用 observed neighbor tokens 和上一层预测 future；ego candidate 仍是动作集合，不再冒充 agent 集合。轨迹输出改为相对当前位置的累计增量并带 log-scale，用于多模态 NLL。

### 3.4 BeTop invalid topology 未屏蔽

原 top-k topology attention 可能把 padding/invalid topology 送入 decoder。现已在 top-k 和 attention 前统一 mask。

### 3.5 contact yaw-rate 指标读取错误

OC-RAP `prefix_states` 的约定是：

`[x, y, vx, vy, heading, yaw_rate, speed, length, width]`。

旧代码把第 3 列 `vx` 当作 heading 计算 yaw-rate violation，导致 contact 指标被系统性污染。现在优先读取第 6 列 `yaw_rate`；只有兼容旧样本时才由第 5 列 heading 差分回退。

## 4. 新增的 observation-only 多模态风险模块

新增文件：

`src/ocrap/external_baselines/observed_risk.py`

它从当前可见 agent history 构造以下假设：

- constant velocity；
- yield；
- accelerate；
- hard brake；
- left/right drift；
- delay/noise。

对每个 ego candidate 计算：

- predicted minimum clearance；
- collision probability surrogate；
- expected loss；
- upper-CVaR loss；
- worst loss；
- TTC proxy；
- stopping/backup margin；
- relative-speed severity proxy。

这组量同时供 MARC、RACP、expected/CVaR/DRO、predictive safety filter 和 post-impact planners 使用，保证不同 baseline 基于同一 observation-only future set 比较，而不是各自偷偷读取教师真值。

## 5. 数据集、训练与评估接入

### 5.1 数据输入

所有 learned baseline 继续直接读取 OC-RAP `.npz` 样本和 manifest，无需重新构建另一套数据集。输入适配做了以下修复：

- WOMD 16-D agent state 正确映射到模型使用字段；
- scene/history 使用 current-ego-relative 坐标；
- candidate target 同样相对当前 ego；
- padding mask、candidate validity、actor validity 保持一致；
- 默认标签为 logged nominal candidate；
- 可选 teacher-rank supervision 必须显式同时设置 `supervision_target=teacher_rank` 和 `allow_teacher_supervision=true`，防止误开。

### 5.2 统一输出指标

`evaluate-baseline` 会对每个 baseline 输出与 OC-RAP 对齐的：

- FRA / admitted FRA；
- ODG；
- DRS；
- NUP；
- hard/harm、artifact selection；
- selected observed expected/CVaR risk、collision probability、minimum clearance、backup margin；
- contact/post-contact 相关指标。

闭环默认 `label_mode=selected`：先用部署量选动作，再只给实际执行 candidate 生成 recovery teacher 标签。因此被审计步上的 selected-action FRA、ODG、DRS 是教师评估值，同时避免给所有 24 candidates 做昂贵的 root × option rollout。

注意：

- `closed_loop_FRA_exec`、selected ODG、selected DRS 可在 selected audit 下比较；
- `closed_loop_FRA_cand`、全候选 best-regret 等需要 `label_mode=all` 或 coverage audit；
- 最终论文主表若需要“全候选精确审计”，应另跑小规模 `label_mode=all` 验证集；不要把 selected-only 结果描述成全候选 exhaustive result。

## 6. 闭环速度优化

### 6.1 最大加速来源：从全候选教师 rollout 中移除决策

旧流程对每一步的 24 candidates 都生成 8 roots × 12 recovery options，再由 baseline 使用教师量选动作。代价近似随：

`steps × candidates × roots × recovery_options`

增长。

新流程：

1. feature-only 构建 24 candidates；
2. baseline 使用 observation/model output 选 1 个 candidate；
3. 只对该 candidate 做教师审计。

严格 oracle baseline 仍使用 `label_mode=all`，因为其定义就是需要每个 candidate 的 branch-wise teacher recovery；不能为了速度把 oracle upper bound 近似掉。

### 6.2 Waymax/JAX 优化

三个脚本统一加入：

- 两 GPU 独立进程分配；
- `JAX_COMPILATION_CACHE_DIR` persistent cache；
- `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`；
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`；
- `waymax.use_jit_scan_rollouts=true`；
- 外部 baseline 默认关闭重复的 online future metrics；
- 限制每进程 OMP/MKL/OpenBLAS/NumExpr 线程，避免两个 GPU 任务争抢 CPU；
- 输出 `timing` 分解，定位 state history、candidate feature、teacher label、policy selection、audit 和 Waymax step 的耗时。

### 6.3 不改变主协议的默认值

默认仍为：

- 24 candidates；
- replan interval = 1；
- audit every step = 1；
- 40 closed-loop steps；
- 非 oracle selected audit；
- oracle full audit。

脚本暴露了：

- `CL_NUM_CANDIDATES`；
- `CL_REPLAN_INTERVAL_STEPS`；
- `CL_AUDIT_EVERY_N_STEPS`；
- `CL_MAX_SCENARIOS`；
- `CL_MAX_STEPS`。

改变前三项会改变评估协议或审计密度，只适合快速调试/消融，不能与默认主表混用。

## 7. 双 GPU 并行脚本

### 7.1 Safe

```bash
CUDA_DEVICES=0,1 bash scripts/run_safe_regime_external_baselines.sh
```

行为：

- nominal replay 在 CPU 离线评估；
- GPU0/GPU1 同时训练并离线评估两个 learned baseline；
- 空闲 slot 继续第三个；
- 训练完成后 Wayformer、GameFormer、BeTopNet closed-loop 也按两个 GPU 并行；
- 输出 offline + closed-loop 汇总 JSON。

仅复用已有 checkpoint：

```bash
DO_TRAIN=false CUDA_DEVICES=0,1 bash scripts/run_safe_regime_external_baselines.sh
```

### 7.2 Near-contact

```bash
CUDA_DEVICES=0,1 bash scripts/run_near_contact_external_baselines.sh
```

- 若缺少 observation-only GameFormer checkpoint，自动训练；
- 非 learned baseline 与 GameFormer 离线评估并行；
- closed-loop 每批同时运行两个方法；
- `oracle_recovery_filter` 自动切换到 `label_mode=all`；其余为 selected audit。

### 7.3 Contact

```bash
CUDA_DEVICES=0,1 bash scripts/run_contact_external_baselines.sh
```

- 离线统一评估；
- 四个 post-contact baseline 两两并行闭环；
- 不再先给 24 个候选全部生成教师标签。

### 7.4 快速调试示例（不用于论文主表）

```bash
CUDA_DEVICES=0,1 \
CL_MAX_SCENARIOS=10 \
CL_MAX_STEPS=20 \
CL_REPLAN_INTERVAL_STEPS=2 \
CL_AUDIT_EVERY_N_STEPS=2 \
bash scripts/run_contact_external_baselines.sh
```

## 8. 验证结果

已执行：

```bash
PYTHONPATH=src pytest -q
```

结果：

- 85 passed；
- 2 个 PyTorch Transformer nested-tensor warning；
- 无失败；
- 三个 shell script 均通过 `bash -n`。

新增回归测试覆盖：

- 非 oracle selector 对教师字段扰动保持不变；
- learned selector 只使用模型策略 logits；
- observation-risk 对碰撞候选给出更高风险；
- GameFormer teacher context 只能显式 opt-in；
- yaw-rate 使用正确 schema channel。

## 9. 必须诚实披露的限制

1. 当前环境没有挂载完整 WOMD、Waymax CUDA 环境和你的服务器 GPU，因此没有在这里完成全规模训练或 50-scene Waymax 数值验证；交付前验证是静态检查、单元/回归测试和接口检查。
2. 上传内容中没有找到 `.bib` 文件；论文末尾引用 `post-collision.bib`，但本次只能依据 TeX citation keys 和公开论文核对 baseline。请把 bib 补入最终仓库。
3. `*_lite` 是与 OC-RAP candidate lattice 对齐的忠实算法 adapter，不是官方源代码的逐行复刻。论文实验部分必须准确命名并列出架构/训练超参数。
4. post-impact MPC 和 restoration 在 WOMD/Waymax bounding-box planner 层只能做 finite-lattice surrogate；不能声称复现原论文的高保真轮胎、执行器或硬件在环控制结果。
5. 所有最终表格应保存 scenario IDs、随机种子、checkpoint hash、config、label mode、审计密度与 timing，防止不同协议结果被误合并。

## 10. 主要公开依据

- Wayformer: Motion Forecasting via Simple & Efficient Attention Networks, ICRA 2023.
- GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving, ICCV 2023.
- Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving (BeTop/BeTopNet), NeurIPS 2024.
- MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving, IEEE RA-L 2023.
- RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions, IEEE T-IV.
- A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems.
- Predictive Control Barrier Functions: Enhanced Safety Mechanisms for Learning-Based Control.
- Integrated Post-Impact Planning and Active Safety Control for Autonomous Vehicles, IEEE T-IV 2023.
- Post-Impact Motion Planning and Tracking Control for Autonomous Vehicles, CJME 2022.
- Motion Planning for Autonomous Vehicles with the Inclusion of Post-Impact Motions for Minimising Collision Risk, Vehicle System Dynamics 2022.
- Post-Collision Trajectory Restoration for a Single-track Ackermann Vehicle using Heuristic Steering and Tractive Force Functions, arXiv 2026 (preprint).
