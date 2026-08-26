# ALGORITHM_CHANGELOG V16.8.33 — Recovery Option Spectrum + Dominance Hysteresis

## Status

V16.8.32 上传结果通过完整性/因果实现审计，可以做算法归因；没有 repair-only 阻断。按 V16.8.32 预注册 gate，THOP 与 SOV Recovery Commitment 都 **不 promotion**，因此 V16.8.33 不跑 fresh37/exact200 去补救失败结果，而是只围绕 V16.8.32 新暴露的两个结构问题做可证伪分解。

V16.8.33 新增：

- `cowp_sov_dominance_hysteresis`（SDH）：state-machine-only diagnostic；
- `cowp_recovery_option_spectrum_hysteresis`（ROSH）：main physical-option probe。

二者仍只作用于 `full conventional set == empty && valid candidate exists`；COWP certified/common-conventional path 完全冻结。

---

## V16.8.32 可靠性结论

- 上传代码 `sanity`: 34/34 passed；
- equivalence16 COWP vs V16.8.29 reference：1120 fields / 0 mismatch；
- counterfactual48 两个新方法均 24+24 disjoint shards，union 精确等于 48-ID manifest；
- merged CR/Collision/Offroad/Kinematics/EP 可由 48 scenario rows 零误差重算；
- manifest logical SHA256：`ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0`；
- checkpoint 相同；online `mechanism_ground_truth_available_online=false`；
- fresh37/exact200 没有运行，符合预注册纪律，因为两个 Stage-1 gate 都失败。

结论：**可以可靠算法归因。**

---

## V16.8.32 预注册 GO 结果

### THOP — fail / archive

相对 COWP：

- collision：9 rescue / 5 induced，net -4；
- old RVR rescue retained：9/10；
- old RVR induced avoided：4/9（要求 >=7/9，失败）；
- kinematics：+3 scenes（要求 <=+1，失败）；
- paired EP delta ≈ -0.15070，bootstrap 95% CI ≈ [-0.3230, -0.0209]（要求 >=-0.05，失败）；
- nonzero intervention：通过。

THOP 同时失败 3 个 gate。把 one-step successor 继续堆到 second-successor 并没有解决 false-positive recovery，而且 48-scene wall time 约 1243 s，高于 commitment 约 975 s。**禁止继续 V3/V4 horizon stacking。**

### SOV Recovery Commitment — near-positive diagnostic, but fail promotion

相对 COWP：

- collision：8 rescue / 3 induced，net -5；
- old RVR rescue retained：7/10；
- old RVR induced avoided：6/9（唯一 promotion 硬 gate 失败；要求 >=7/9）；
- kinematics：+1 scene；
- paired EP delta ≈ -0.02301，CI ≈ [-0.0600, +0.00619]；
- intervention nonzero。

因此 **unconditional commitment 不 promotion**，但 mode consistency 信号值得保留。

---

## V16.8.32 新的关键机制证据

### 1. SOV -> Commitment 清楚隔离了 chattering 与 over-commitment

相对 V16.8.30 SOV：

- commitment 新救回 6 个 SOV collision scenes；
- 但新诱发 2 个 SOV 原本安全 scenes：`3919ccd73c0fabd7`, `c34fe8e79cdf1161`。

这说明：

- strict SOV 每步重新 gate 的确过于容易退出 recovery，mode continuity 有价值；
- 但“进入后一直 RVR 到 conventional 恢复”的无条件 commitment 又过于 sticky。

因此下一步不应使用固定 dwell time / 手调 hysteresis margin，而应使用 **同一 viability relation 的 strict-entry / weak-continue / dominance-loss-exit** 状态机。

### 2. 当前 successor signature 在 zero-conventional regime 结构上不完整

V16.8.30 SOV 对 9 个 old RVR-induced collisions 避免 8/9；唯一 false positive 是：

`7721ff4800156886`

该 scene 后续 first collision 前进入 `no_valid_candidate` bounded emergency。现有 successor signature：

`(conventional_exists, conventional_macro_types, conventional_candidates, max_prefix)`

在 successor conventional 仍为空时，前三项全部归零，比较几乎退化为 **单个 max-prefix**。这与 V16.8.29 已被否定的 brittle statistic 本质相同：一个很长的单一路径可以掩盖其它 recovery alternatives 已经消失。

所以物理 feasibility 不应继续用更多 rollout depths 修 max-prefix，而应该重新表示 **整个 recovery option set 如何随 causal horizon 消失**。

---

## 冻结层 / 禁止伤害

继续 Freeze：

- compact-5k data/label contract；
- natural roots；
- RCOT same-root transport；
- BCOT structured certificate；
- protected-priority hard feasibility；
- certificate-compatible set-preservation frontier；
- 8 s conventional collision-screen contract；
- V16.8.27 conventional integrity；
- V16.8.28 no-valid bounded execution integrity；
- outcome head diagnostic-only。

Accepted-path kinematics 仍是独立 secondary bottleneck，本轮禁止与 recovery 同时修改。

---

## 新增禁止方向

除 V16.8.32 已列出的禁止项外，V16.8.33 进一步禁止：

1. 继续 THOP V3/V4/V5 horizon stacking；
2. unconditional recovery commitment 直接 promotion；
3. 固定 N-step dwell time；
4. hysteresis epsilon/margin 搜索；
5. 用 option-profile AUC / 加权和做 selection；
6. 用 candidate-count / prefix / risk 手工线性组合修 false positives；
7. 在 recovery-option representation 未被 analytic probe 验证前训练新的 viability head。

---

# 新机制 A — SOV Dominance Hysteresis (SDH)

Method: `cowp_sov_dominance_hysteresis`

这是 **state-machine-only diagnostic**，保持 V16.8.30 的 successor signature 不变，用来单独回答 V16.8.32 commitment 的两个新增 collision 是否来自 over-commitment。

在 zero-conventional+valid 且 RVR/base emitted action 不同的时候：

### Entry

`V_RVR >lex V_COWP`

才进入 recovery mode，完全复用 strict SOV 的高精度 entry。

### Continue

已在 recovery mode 时，只要求：

`V_RVR >=lex V_COWP`

精确 tie 继续 RVR，因此不会像 stateless SOV 那样在 equality 上抖回 COWP。

### Exit

一旦：

`V_RVR <lex V_COWP`

立即退出 recovery；certificate/conventional 恢复或 no-valid emergency 同样清除状态。

不存在 dwell time、epsilon 或权重。

---

# 新机制 B — Recovery Option-Spectrum Hysteresis (ROSH)

Method: `cowp_recovery_option_spectrum_hysteresis`

这是 V16.8.33 主机制 probe。

## Recovery Option Persistence Profile

对执行 base/RVR 的 **actual jerk/yaw-rate-limited emitted action** 后得到的 causal successor state，重新生成完全相同 online physical proposal bank。

对每个 causal horizon `h=1..H` 定义：

`P_s(h) = number of distinct non-PAD macro types with >=1 valid + roadgraph-safe candidate whose collision-safe prefix >= h`

因此 `P_s` 不是单个 longest prefix，而是一条完整的 **semantic recovery-option survival curve**：

- 近端值表示还有多少独立 recovery modes；
- 远端值表示这些 modes 中有多少能长期存活；
- full-horizon 端自然包含 conventionally collision-safe macro diversity；
- 同一 macro 内重复 trajectory 不会虚增 option diversity。

周围 agent 只使用与 conventional screen 相同的 causal constant-velocity model；不读取 Waymax/logged future，无 GT leakage。

## Dominance

RVR profile 对 COWP profile strict-dominates 当且仅当：

`P_RVR(h) >= P_COWP(h) for every h`

且至少一个 horizon 严格更大。

选择不使用 profile area、权重或阈值。area/min-margin 只记录诊断。

## Mode state

ROSH 使用与 SDH 完全相同的 parameter-free hysteresis：

- inactive -> active：strict profile dominance；
- active -> active：weak pointwise dominance，包括 equality；
- active -> inactive：任一 horizon option support regression；
- conventional/certificate restore 或 no-valid：clear。

因此 V16.8.33 把两个纠缠因素干净拆开：

- SDH 好、ROSH 无额外收益：主要是 mode semantics；
- ROSH > SDH：当前 successor representation 确实缺少 option diversity/persistence；
- 两者都失败：停止 SOV/BHOV/commitment family，转向 proposal/reachable-set construction 或更高保真 dynamics，而不是继续调 gate。

---

## 预注册实验协议

### Stage 0 — sanity + equivalence16

common COWP path 必须 0 mismatch。

### Stage 1 — counterfactual48

SDH 与 ROSH 同时跑。每个方法都必须：

- retain >= 5/10 old RVR rescues；
- avoid >= 7/9 old RVR induced；
- net COWP collision reduction >=3 scenes；
- kinematics regression <=1 scene；
- paired mean EP delta >= -0.05；
- intervention >0。

任一失败即 archive；禁止调 threshold/margin。

### Stage 2 — fresh37

V16.8.32 因 Stage-1 失败从未运行 fresh37，因此该 37-ID panel 对 V16.8.33 新方法仍未见。它是 exact200 历史 development set 的剩余 ID，不是 publication holdout。

GO：

- no net collision harm；
- no net CR harm；
- offroad regression <=1；
- kinematics regression <=1；
- paired mean EP delta >= -0.03；
- intervention >0。

### Stage 3 — exact200

只有 Stage 1+2 都通过的方法运行；只跑 promoted method，复用 immutable COWP/RVR reference。仍只作为 development confirmation。

---

## 论文主线

当前最值得维护的统一主线仍为 **Orthogonal Option-Set Feasibility**：

- social axis：判断 ego 是否靠压缩其他 critical actor 的 natural low-burden option set 获得“安全”；
- physical-temporal axis：判断 uncertified recovery 是否压缩 ego 自己未来的 executable/recovery option set。

V16.8.33 的 ROSH 不是单独宣称“hysteresis”或“backup planning”新颖，而是在验证 physical axis 的正确可观测对象是否应从 longest trajectory horizon 升级成 **semantic option-set persistence profile**。

## 交付前 promotion-integrity 加固

最终 launcher 将预注册 promotion 从“文档约定”升级成 **fail-closed runtime gate**：

- `fresh37_parallel2` 会强制读取 `counterfactual48_v33_recovery_option_spectrum_analysis.json`；
- `confirm200_parallel2` 会强制读取 `fresh37_v33_recovery_option_spectrum_analysis.json`；
- 指定的 `PROMOTED_METHODS` 只有对应 `preregistered_gate.pass == true` 才允许继续；
- 缺 analyzer JSON、未知 method 或任一 gate 为 false 都直接终止。

该修改不改变 planner/action 语义，只防止人为越过预注册 gate。最终 focused `sanity` 仍为 **39/39 passed**。
