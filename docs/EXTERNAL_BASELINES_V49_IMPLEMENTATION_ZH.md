# OC-RAP v48.34 / external baselines v49 实现与评测说明

## 1. 结论先行

本次实现没有把“论文启发的启发式函数”冒充成论文官方实现，而是为每个方法建立了明确的 provenance/fidelity 合同：

- 能在 OC-RAP 的统一可执行 candidate-prefix 动作空间中保留论文核心设计的，标为 **paper-core adaptation**；
- 有公开代码且主要网络结构可在当前张量协议中重建的，标为 **high paper-core fidelity**；
- 只实现快速风险代理、没有完成原论文内层优化的，明确标为 **surrogate**；
- teacher oracle 单独列为 **non-deployable audit upper bound**，不进入外部 baseline 主排名。

完整机器可读审计见：

- `external_baseline_fidelity_audit_v49.json`
- `docs/EXTERNAL_BASELINE_FIDELITY_AUDIT_V49.md`

## 2. 论文、开源代码与当前复现保真度

| Regime | 方法 | 论文/来源 | 开源状态 | 本仓库实现结论 |
|---|---|---|---|---|
| Safe | nominal replay | 数据集控制 | 不适用 | 精确重放 logged nominal candidate |
| Safe | Wayformer-style BC | Wayformer, arXiv:2207.05844 | 未识别到可直接使用的官方 planner checkpoint | 保留 factorized attention、latent query、scene fusion；原方法是预测模型，本实现是 candidate-policy adaptation |
| Safe/Near | GameFormer | arXiv:2303.05760；`MCZhi/GameFormer-Planner` | 有官方代码 | 重建 agent/history encoder、initial multimodal decoder、level-k interaction、previous-level future encoding、deep supervision；数据管线和输出动作空间改为 WOMD/OC-RAP，因此不与官方 checkpoint 兼容 |
| Safe | BeTop | arXiv:2409.18031；`OpenDriveLab/BeTop` | WOMD prediction 已公开，nuPlan planning 在公开仓库中仍为 TODO | 保留 actor/map topology、iterative topology decoder、topology-guided sparse attention 和辅助监督；不能声称完整官方 planning 复现 |
| Near | MARC | arXiv:2308.12021 | 未识别到作者公开实现 | 保留 semantic multi-policy、动态共享前缀、non-anticipativity、scenario-tail branching、expected/CVaR 约束；连续优化替换为可执行候选格枚举 |
| Near | RACP | arXiv:2402.17387；作者公开仓库 | 有代码，但不与本项目 API/checkpoint 兼容 | 保留多模态 belief、共享前缀、branch expected/CVaR、chance constraint、utility-risk optimization；continuous branch MPC 替换为 candidate lattice |
| Near | Expected/CVaR filters | 通用风险规划基线 | 不适用 | 按本评测协议精确定义 |
| Near | DRO-CVaR | 通用分布鲁棒基线 | 不适用 | 使用快速 Wasserstein-inspired dispersion penalty；明确不是完整内层 ambiguity optimization |
| Near | Predictive Safety Filter | arXiv:1812.05506 | 无直接可用的本项目实现 | 保留 minimal intervention、stage/terminal backup set、predictive barrier；非线性 MPC 替换为候选投影 |
| Contact | Post-impact MPC | DOI:10.1109/TIV.2023.3236150 | 未识别到公开代码 | 保留二次碰撞风险、yaw/adhesion stability、terminal motion、control effort；没有冲量/损伤状态估计和连续 vehicle-dynamics MPC |
| Contact | Post-crash stable stop | 规则控制 | 不适用 | 精确规则基线 |
| Contact | Post-collision restoration | arXiv:2602.08444 | 未识别到公开代码 | 保留轨迹回正、yaw 稳定、progress、clearance、Ackermann feasibility proxy；连续恢复优化替换为候选选择 |
| Contact | Severity minimization | DOI:10.1080/00423114.2022.2088396 | 未识别到公开代码 | 保留 relative-speed/severity、collision probability、penetration/clearance、post-impact controllability；WOMD 无 occupant injury/damage model |

这里的“完整复现”是指：**在所有方法共享同一组可执行候选、同一 observation history、同一 Waymax 闭环后端和同一场景 target key 的前提下，尽可能完整保留论文核心决策结构**。对于没有公开代码、planning 部分尚未发布，或原论文依赖连续动力学优化的方法，无法诚实地宣称 bit-exact 官方复现。

## 3. 主要代码修正

### 3.1 Observation-only 风险模型

`src/ocrap/external_baselines/observed_risk.py`

- 对同一 candidate group 的周车未来只预测一次；所有候选共享缓存的 7 模态 future bank。
- 周车预测和候选风险按 `[mode, actor, time]` 向量化。
- `min_ttc` 改为“signed clearance 首次进入阈值的时间”；旧实现错误地使用最近距离发生时间。
- 同时输出：mode-wise TTC、closest-approach time、collision probability、severity、clearance/loss/backup-margin curves。
- backup margin 使用逐时刻 stopping-distance margin，而不是单点几何距离。

### 3.2 MARC / RACP / PSF

`src/ocrap/external_baselines/policies.py`

- MARC：semantic policy family、动态 shared-prefix branch point、共享段 upper-tail risk、分支段 expected/CVaR、chance constraint、family representative selection。
- RACP：belief-weighted multimodal risk、non-anticipative prefix、branch risk、chance constraint、information-preserving branch term。
- PSF：输入可行性、stage backup margin、terminal backup margin 和 predictive barrier 条件。
- 所有 deployable baseline 只使用 observation/candidate/model quantities；仅 `oracle_recovery_filter` 可读取 teacher tensor。

### 3.3 训练和推理效率

`src/ocrap/external_baselines/train.py`

- CUDA 默认启用 BF16 autocast、TF32、cuDNN benchmark、fused AdamW。
- FP16 时使用 GradScaler；BF16 不做不必要缩放。
- `torch.compile` 保持 opt-in，避免一次性短训练中编译开销反而增加总时长。
- DDP validation 使用不补齐、不复制样本的 sampler，保证 best-checkpoint 统计语义不变。
- 两 GPU 脚本采用动态回填调度；JAX compilation cache、禁止预分配、限制 CPU 线程，降低 PyTorch/JAX 并行争用。

### 3.4 统一闭环协议

旧 external-baseline 脚本只限制 WOMD 场景数量，没有绑定 safe/near/contact 数据集 target key。现在：

- `closed_loop.bucket_dataset` 必须指向对应 `test_safe`、`test_near_contact`、`test_contact`；
- `closed_loop.require_bucket_targets=true`，不允许目标加载失败后静默回退到任意 WOMD 场景；
- `MAX_SCENARIOS=0` 在 bucket-targeted 模式下表示运行全部加载目标；
- master 脚本不再默认加 `@150`，避免只扫描前 150 条 WOMD record；
- 默认每个 scene 选一个确定性 target，统计单位为独立 scene；需要同场景多时刻评测时可提高 `CL_MAX_TARGETS_PER_SCENE`；
- 比较表默认要求每个方法的 `.scenes.jsonl` target set 完全相同，否则直接失败。

### 3.5 bucket-specific calibration

上传的 v48.34 RC=20 calibration 文件给出的 frozen threshold 为：

- Safe: `1.2729634046554565`
- Near-contact: `0.19746489822864532`
- Contact: `0.16214445233345032`

`run_ocrap_three_regime_closed_loop.sh` 现在从所选 variant 的 `gamma_rec_by_bucket_v48.json` 自动读取三个阈值。禁止再把一个标量 `GAMMA_REC` 无差别用于三个 regime。

## 4. 指标

### Safe

- collision/offroad scene rate
- minimum clearance / minimum TTC
- bounded NUP
- intervention rate
- decision latency

### Near-contact

- collision/offroad scene rate
- scene-level minimum-clearance p05、TTC p05
- terminal clearance / terminal TTC
- critical-TTC exposure duration
- DRS、ODG、FRA-exec、bounded NUP
- decision latency

### Contact

Contact 已发生 initiating collision，因此主表只报告物理恢复，不报告 FRA/DRS/ODG：

- post-contact terminal clearance
- normalized free-space AUC
- post-contact clearance gain
- escape scene rate
- re-contact / secondary-overlap scene rate
- stable-stop-quality scene rate
- offroad scene rate
- post-contact overlap duration
- decision latency

离线 external baseline 另外输出 selected-risk、正确 TTC、severity proxy、Brier score、ECE 和吞吐率。Brier/ECE 是选中动作执行后相对 teacher hard-violation 的诊断，不参与 baseline 选动作。

## 5. 视频选择

`tools/select_critical_scenes_v48_34.py` 和 `scripts/build_top10_recovery_videos.sh`：

- Near 选 5 条，Contact 选 5 条，总计严格 10 条 MP4。
- Near 主要按 TTC、clearance、critical exposure 改善评分。
- Contact 主要按 terminal clearance、free-space AUC、clearance gain、escape、stable stop、re-contact 评分。
- 必须不引入 overlap/offroad/re-contact 明显回归；默认每个 scene 最多一条，避免同一场景刷屏。
- 自动与多指标综合最强的 deployable external baseline 配对；teacher oracle 不参与。
- 每条视频保存选择分数、指标 delta、target key 和可审计索引。

由于当前上传包没有 checkpoint，本环境不能实际生成可信的闭环数值和 MP4；脚本会在你的训练服务器上直接消费现有 checkpoint 和完整 WOMD/OC-RAP 数据。

## 6. 执行顺序

完整命令见仓库根目录 `OC-RAP-v48.34-external-baselines-v49-run-commands-ZH.txt`。

建议先 `MAX_SCENARIOS=20` 做 smoke test，再删除该覆盖或设为 `0` 跑全部独立 scene。RC=20 结果必须标为 **post-gate exploratory diagnostic**；如果使用 held-out test 做选择或挑视频，该 test 集不能再被声称为后续调参完全未触碰的最终测试集。
