# OC-RAP v37→v38 结果诊断与 v39（OC-RAC）优化方案

## 0. 结论先行

1. **Safe 已达到“nominal-preserving”目标。** v37/v38 均为 1120/1120 nominal，intervention=0，bounded NUP=1.0。
2. **v38 对 near-contact 完全没有产生作用。** v37 主实验、v37 near-margin、v38 主实验、v38 near-margin 的 near 结果逐项一致：240/240 nominal、intervention=0、PCD=0.548116、FRA=0.116667、DRS=0.883333、paper-PCD miss=0.05。
3. **v38 对 contact 有真实但不充分的收益。** selected-topk paper-PCD miss 从 0.08333 降到 0.05，PCD 从 0.47240 提升到 0.49403，FRA 从 0.23333 降到 0.20，DRS 从 0.76667 提升到 0.80；但 intervention 从 0.0125 增到 0.025，并出现 4 步连续 brake。0.05 恰好等于当前门槛，只对应 60 个审计点中的 3 个 miss，不具备安全余量。
4. **near 的根因不是 gap window，而是模型排序反转。** 3 个 near miss 中，失败 nominal 均被预测为 DRS=1、gap≈0.025、R_dep≈0.34–0.36；真正 teacher-best brake 的预测 DRS≈0.94、gap≈0.173/0.342、R_dep≈0.05–0.07。模型因此把 nominal 判断为更可靠，stress nominal anchor 会在 challenge 之前锁定 nominal。
5. **继续放宽 selector 会把问题变成 audit-specific heuristic。** 根治路径应当是让模型学习 scene-time 内的反事实恢复优势，而不是在推理时绕过错误预测。

---

## 1. 论文 idea、pipeline 与当前代码映射

论文的核心问题是 **oracle-to-deployable gap**：对多个潜在隐状态分别选择最优恢复动作，会得到一个不可部署的 branch-wise oracle；真实系统在后续观测仍不可区分时，只能执行同一个恢复动作。OC-RAP 将这种“观测一致的可恢复性”作为规划原语。

算法链路：

1. 场景与候选动作前缀编码；
2. 预测 recovery-sufficient latent roots，而不是完整未来轨迹；
3. 预测执行前缀后的 observation embedding，构建观测兼容关系；
4. 对语义恢复选项预测 root-conditioned recovery margins；
5. OC-MERO 在观测不可区分 roots 上要求共享恢复动作，得到 deployable recoverability，同时计算 oracle recoverability 与 gap；
6. CRISP/校准 selector 优先保留可准入 nominal，否则在准入集合中按效用选择，必要时 recovery-first fallback。

代码中的主要对应关系：

- `src/ocrap/models/ocrap.py`：模型输出 root、observation、margin/option heads；
- `src/ocrap/planning/selector.py`：校准准入、nominal preservation、rescue certificates、budget/cooldown；
- `src/ocrap/models/losses.py`：anti-oracle、shared-option、DDC、teacher-PCD 等训练目标；
- `src/ocrap/data/build/`：safe/near/contact regime 划分、隐变量 roots、恢复选项和 teacher labels；
- `src/ocrap/simulation/closed_loop_runner.py`：闭环选择、审计和指标聚合。

数据构建的三类差异是合理的：safe 不生成 targeted futures；near 加入 hidden yield/accelerate、低附着和控制延迟；contact 再加入 contact impulse surrogate 和 secondary-collision approach。near/contact 都使用 augmented hidden roots、可见扰动和 artifact-pair mining。

---

## 2. 三个 regime 的目标

### 2.1 Safe：不扰动正常驾驶，同时保持校准安全

**现实目标**：系统不应因为“可能存在恢复空间”而无故制动或偏离正常轨迹；风险、舒适性、通行效率不能劣于 nominal。

**建议主指标**：

- intervention rate = 0；intervention episode rate = 0；
- bounded NUP ≥ 0.999（理想为 1.0）；
- collision/overlap、offroad、kinematic infeasibility 对 nominal 非劣；
- calibration coverage 与 FRA 在预设 delta 下满足承诺。

**解释性指标**：log divergence、加速度/jerk、宏动作切换率。Safe 不需要把 PCD 做得越高越好，否则会诱导保守动作。

### 2.2 Near-contact：低扰动地增加“避免进入接触”的物理余量

**现实目标**：还没有接触时，优先扩大空间与时间余量，减少进入 contact 的概率；只在确有恢复优势时短促干预。

**建议主指标**：

- paper-PCD miss ≤ 0.033（当前 60 个审计点意味着最多 2 miss；正式实验应扩展到约 240 个审计点）；
- PCD ≥ 0.558、FRA ≤ 0.10、DRS ≥ 0.90；
- bounded NUP ≥ 0.995，intervention ≤ 0.02；
- scene-level minimum clearance 的 median/p05；
- finite TTC 的 p05、TTC<3s exposure；
- clearance<2m exposure、contact transition rate。

**必须增加的指标**：仅报告 PCD/FRA/DRS 不能证明 near-contact 的现实收益。应报告每个 rollout 的最小车体间距，再跨场景统计 median/p05；TTC=99 的无风险样本应作为右删失或单独统计，不应直接求普通均值。

### 2.3 Contact/post-contact：阻断二次事故并进入稳定可控状态

**现实目标**：允许稀疏恢复动作，但不能把策略退化成持续刹车；重点是避免二次碰撞、降低伤害、稳定停车或安全重返路线。

**建议主指标**：

- selected-topk paper-PCD miss ≤ 0.033，PCD ≥ 0.50；
- FRA ≤ 0.185、DRS ≥ 0.815；
- bounded NUP ≥ 0.985、decision intervention ≤ 0.04；
- intervention episode rate ≤ 0.02、max run length ≤ 2；
- secondary overlap event、stable-stop rate、time-to-stable-stop；
- impact severity/Delta-v、peak yaw rate、offroad duration、route-rejoin success（后续需 teacher/Waymax 暴露更完整物理量）。

**为什么要 episode 指标**：v38 的 4 个连续 brake 是一个持续恢复 episode，但旧指标把它计成 4 次独立干预，既无法区分合理保持与高频切换，也无法暴露 cooldown bypass 的连续触发问题。应同时报告 decision rate、episode rate、run length 和 macro switch rate。

---

## 3. v37 与 v38 的定量对比

| Regime/metric | v37 | v38 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Safe intervention | 0 | 0 | 0 | 已稳定达标 |
| Safe NUP | 1.0000 | 1.0000 | 0 | 已稳定达标 |
| Near intervention | 0 | 0 | 0 | 没有主动余量恢复 |
| Near PCD | 0.548116 | 0.548116 | 0 | 无收益 |
| Near FRA | 0.116667 | 0.116667 | 0 | 无收益 |
| Near DRS | 0.883333 | 0.883333 | 0 | 无收益 |
| Near paper-PCD miss | 0.050000 | 0.050000 | 0 | 无收益 |
| Contact intervention | 0.0125 | 0.0250 | +0.0125 | 干预翻倍 |
| Contact NUP | 0.992711 | 0.991639 | -0.001072 | 小幅下降 |
| Contact PCD | 0.472399 | 0.494034 | +0.021635 | 有效改善 |
| Contact FRA | 0.233333 | 0.200000 | -0.033333 | 有效改善 |
| Contact DRS | 0.766667 | 0.800000 | +0.033333 | 有效改善 |
| Contact paper-PCD miss | 0.083333 | 0.050000 | -0.033333 | 达到边界，无缓冲 |
| Offline contact PCD | 0.561200 | 0.564198 | +0.002998 | 小幅改善 |
| Offline contact FRA | 0.093541 | 0.086860 | -0.006682 | 改善 |
| Offline contact DRS | 0.904499 | 0.911626 | +0.007127 | 改善 |
| Offline contact NUP | 0.985154 | 0.985232 | +0.000079 | 基本不变 |

v37 的 main/near-margin 两组结果完全相同；v38 的 main/near-margin 两组也完全相同。因此 near-margin 命令没有改变实际决策路径。

### v38 contact 的剩余问题

- `waymax_14b64c...` step 8：nominal PCD=0.58786，paper-best brake=0.62246，regret=0.03460；这是较小优势，可能不值得强制干预。
- `waymax_0c0141...` step 12：regret=0.06389；step 16：nominal PCD=0、brake=0.62246，是真正严重 miss。
- `waymax_14b64c...` step 11–14 连续 4 次 brake，单步 NUP≈0.868。
- `waymax_0c0141...` step 7–8 连续 2 次 brake，NUP≈0.26，说明 selector 的 tail bypass 有明显效用代价。

### near-contact 的关键反例

3 个 miss 全部具有同样的预测反转：

- selected nominal：pred DRS=1.0、R_dep≈0.34–0.36、gap≈0.025，但 teacher PCD=0；
- teacher-best brake：pred DRS≈0.94、R_dep≈0.05–0.07、gap≈0.173/0.342，但 teacher PCD=0.62246。

这说明 v38 的“扩大 gap band”只覆盖了候选形状，却没有修复 nominal 的错误高置信度。由于 `stress_nominal_anchor` 在 rescue challenge 之前执行，且 brake 在预测的 DRS/R_dep/gap 三个轴上均不优于 nominal，challenge 不会触发。

---

## 4. v39 优化：OC-RAC + 有界恢复 episode

### 4.1 新增 OC-RAC 训练目标

新增 `observation_consistent_recovery_advantage_loss`：对同一 `(scene, time)` 候选组计算

`Delta_PCD = PCD(best shared recovery) - PCD(nominal)`。

当 teacher 显示恢复动作显著优于 nominal 时，损失包含：

1. 预测 Delta_PCD 对 teacher Delta_PCD 的回归；
2. 恢复动作相对 nominal 的 pairwise margin ranking；
3. 恢复动作预测 PCD floor；
4. teacher nominal 失败时的 nominal overconfidence cap；
5. DRS、R_dep、gap 三个分量的 anti-inversion 配对监督。

当 teacher nominal 更好时，加入 anti-false-positive ranking，避免重新产生 v35 一类广泛 brake。

与原 direct teacher-PCD 的区别：原损失以绝对候选标定为主，v39 明确监督**部署决策真正依赖的组内反事实优势**，并为 near/contact 设置不同权重。训练脚本把 direct PCD、DDC、macro-DRS、protective macro 的 bucket 从 contact-only 扩展到 near+contact。

### 4.2 有界 residual-tail episode

新增 `brake_tail_challenge_max_consecutive`，runner 向 selector 传入上一宏动作及连续长度。严格 tail cooldown bypass 最多连续 2 步（near 默认 1 步），防止 v38 的 4 步重复 brake，同时保留普通 admission/fallback。

### 4.3 新增闭环现实指标

`closed_loop_runner.py` 新增：

- `min_clearance_m` 的 mean/min/p05；
- `ttc_s` 的 mean/min/p05；
- near/contact/critical-TTC exposure；
- overlap episode count、secondary overlap event；
- stable-stop event；
- intervention episode count/rate、mean/max run length、macro switch rate。

注意：目前 clearance 使用代码中已有的近似 oriented-box radius distance，适合相对比较。正式论文可进一步换成精确多边形距离；impact Delta-v、route rejoin、time-to-stop 仍建议继续实现。

---

## 5. 实验策略与 CCF-A 级证据要求

1. **小规模 audit 只用于开发，不用于最终结论。** 当前 60 个审计点下，miss rate 只能以 1/60=0.01667 跳变；v38 的 0.05 只是 3/60。正式结果至少约 240 个审计点，并报告 bootstrap 95% CI。
2. **至少 3 个随机种子**，按 scene bootstrap，而不是把时间步当独立样本。
3. **主表同时报告安全与效率**：PCD/FRA/DRS/miss + NUP + decision/episode intervention + clearance/TTC/secondary overlap/stable stop。
4. **做关键消融**：无 observation consistency、branch-wise oracle、无 OC-RAC、无 anti-inversion、无 episode cap、selector-only v38。
5. **避免把 selector hack 写成核心 novelty。** 论文的强贡献应是 observation-consistent deployability；OC-RAC 可作为学习层创新，证明该理论量能在 near/contact 上被正确排序和校准。
6. **外部基线需覆盖**：nominal/backup、CVaR/DRO、predictive safety filter、post-crash braking/stable-stop、post-impact MPC、branch-wise oracle upper bound；使用相同候选池和物理预算。

---

## 6. 交付代码变更

- `src/ocrap/models/losses.py`：OC-RAC 反事实优势损失；
- `src/ocrap/cli/train.py`、`config/defaults.py`：训练接入与配置；
- `src/ocrap/planning/selector.py`、`evaluation/baselines.py`：连续 tail challenge cap；
- `src/ocrap/simulation/closed_loop_runner.py`：selector state、episode 与物理指标；
- `scripts/train_ocrap_v39_ocrac.sh`：v33→v39 fine-tune；
- `scripts/run_ocrap_v39_ocrac.sh`：双 GPU 评测；
- `tools/check_v39_regime_targets.py`：带安全余量的目标检查；
- `tests/test_v39_recovery_advantage.py`：新损失单元测试。

本地完成：`compileall`、两个 shell 的 `bash -n`、`PYTHONPATH=src pytest -q`，结果 **49 passed**。未在当前环境执行 GPU 训练或 Waymax 闭环，因此 v39 的数值收益必须按命令实际验证。
