# V16.8.32 结果分析与 V16.8.33 设计

> 用户消息写作 V12.8.32/V12.8.33；本次上传代码与结果包实际为 V16.8.32，因此版本链按 V16.8.32 -> V16.8.33 处理。

## 0. 总结

**V16.8.32 结果可靠，可以做算法归因，没有 repair-only 工程阻断。**

但按 V16.8.32 预注册 GO 条件：

- THOP：明确失败，不 promotion；
- SOV Recovery Commitment：有强机制信号、接近通过，但仍因 old-RVR-induced avoidance=6/9 < 7/9 而失败，不 promotion；
- 因两个方法 Stage-1 都失败，未运行 fresh37/exact200 是正确科研行为。

本轮最关键的新结论不是“再多一个 horizon 还不够”，而是把此前耦合的 recovery 问题拆成了两个更具体的结构缺陷：

1. **stateless strict SOV 会过早退出 recovery，但 unconditional commitment 又会过度持续；需要由同一个 viability relation 定义 entry/continue/exit，而不是时间阈值。**
2. **当前 successor signature 在 zero-conventional regime 中会退化为单个 max-prefix，无法表示 recovery-option diversity，因此即使 one-step SOV 很高精度，仍留下一个结构性 false positive。**

V16.8.33 因此新增：

- SDH：严格 entry / 弱 continue / dominance-loss exit，单独验证 state-machine semantics；
- ROSH：将 successor physical viability 表示成完整 semantic recovery-option persistence profile，再使用相同 hysteresis。

---

# 1. 可靠性审计

## 1.1 代码/语义回归

在用户上传的 V16.8.32 代码上重新执行：

`bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh sanity`

结果：**34/34 passed**。

manifest：

- exact200: 200 unique, SHA `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`；
- equivalence16: 16 unique, SHA `81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760`；
- counterfactual48: 48 unique, SHA `ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0`；
- fresh37: 37 unique, SHA `ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481`。

## 1.2 Common-path equivalence

V16.8.32 COWP equivalence16 vs immutable V16.8.29 COWP reference：

- 16 scenes；
- 1120 fields；
- tolerance 1e-7；
- **0 mismatch**。

因此 THOP/commitment 并未偷偷改变 COWP common path。

## 1.3 Counterfactual48 paired integrity

THOP 与 commitment 均：

- shard0=24；
- shard1=24；
- shard overlap=0；
- union=48 unique；
- union 精确等于 counterfactual48 manifest；
- merged rows=48 unique；
- merged hash 与 manifest 一致；
- checkpoint 相同：`outputs/v16_8_24_compact5k_all/cowp_all_best.pt`；
- `mechanism_ground_truth_available_online=false`。

从 48 个 scenario rows 独立重算 CR / Collision / Offroad / Kinematics / EP，与 merged summary 精确一致。

## 1.4 因果/泄漏检查

THOP V1/V2 与 commitment entry：

- ego 使用实际 jerk/acceleration/yaw-rate limited emitted action；
- surrounding agents 使用现有 conventional screen 同一 constant-velocity causal model；
- online physical candidate generator 重新生成 bank；
- 不读取 Waymax future/logged GT；
- 当前配置不启用 oracle future。

因此本轮没有发现信息泄漏。

## 1.5 为什么没有 fresh37/exact200 不是缺结果

V16.8.32 明确预注册：counterfactual48 gate 失败即 archive，不允许继续 fresh37/exact200。实际两个方法都 fail，因此用户停止在 Stage-1 是正确的。

**可靠性结论：PASS，可以算法归因。**

机器审计：`V16_8_32_RESULT_INTEGRITY_AND_MECHANISM_AUDIT.json`。

---

# 2. 按预注册 GO 条件判断算法成败

## 2.1 Counterfactual48 aggregate

| method | Collision | Kinematics | EP | switch rate |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 6/48 | 1.00251 | - |
| RVR | 33/48 | 9/48 | 0.82362 | - |
| SOV | 33/48 | 6/48 | 0.99863 | 1.93% |
| BHOV | 29/48 | 9/48 | 0.85246 | 13.52% |
| **THOP** | **30/48** | **9/48** | **0.85181** | **12.03%** |
| **Commitment** | **29/48** | **7/48** | **0.97951** | **8.85%** |

## 2.2 THOP：FAIL

相对 COWP：

- collision：9 rescue / 5 induced，net -4；
- McNemar exact p≈0.424；
- old RVR rescue retained=9/10；
- old RVR induced avoided=4/9；
- kinematics：0 rescue / 3 induced；
- EP delta=-0.15070；bootstrap95≈[-0.3230,-0.0209]。

预注册检查：

- retain>=5/10：PASS；
- avoid>=7/9：**FAIL**；
- net collision>=3：PASS；
- kine regression<=1：**FAIL**；
- EP>=-0.05：**FAIL**；
- nonzero intervention：PASS。

**结论：THOP archive。**

更重要的是，THOP 不是简单“V2 还不够远”。它已经证明继续将 successor 深度做成 V3/V4/V5 会走向高成本 horizon stacking，而没有改变 option-set representation 本身。48 scenes wall time THOP≈1243 s，commitment≈975 s；继续堆 horizon 同时恶化科学性和迭代速度。

## 2.3 SOV Recovery Commitment：机制有价值，但仍 FAIL promotion

相对 COWP：

- collision：8 rescue / 3 induced，net -5；
- old RVR rescue retained=7/10；
- old RVR induced avoided=6/9；
- kine：+1；
- EP delta=-0.02301；CI≈[-0.0600,+0.00619]。

预注册检查只有一项失败：

- avoid old induced >=7/9：**6/9，FAIL**。

因此不能因为 aggregate collision 29/48 就 promotion。上轮已经规定，old induced avoidance 是 recovery precision 的硬 gate。

**结论：unconditional commitment 算法失败；“mode consistency”机制信号成功。**

---

# 3. V16.8.32 给出的更深机制证据

## 3.1 SOV -> Commitment：明确看到“退出过早”和“持续过头”同时存在

将 commitment 与 V16.8.30 strict SOV 直接 paired：

- SOV safe -> commitment collision：2 scenes
  - `3919ccd73c0fabd7`
  - `c34fe8e79cdf1161`
- SOV collision -> commitment safe：6 scenes
  - `9e3e5f19ee38f2e3`
  - `ad7d72d8adca3e25`
  - `9ccf60966ec93c20`
  - `2c2395ec28c6a158`
  - `7c6ac47c0deee2af`
  - `84196f1d9198b616`

这是一条很强的机制证据：

- stateless SOV strict `>` 每步重新 gate，确实会太容易退出 recovery；
- unconditional commitment 可以把其中 6 个 case 救回来；
- 但它又把两个 SOV 原本安全的 case 做坏，说明持续到 conventional restore 太 sticky。

所以 V16.8.33 不应该引入固定 5/10/20-step commitment，也不应该调 release threshold；应该让 **同一个 viability relation 决定状态机语义**：

- strict better 才 entry；
- equal/non-worse 可以 continue；
- worse 立即 exit。

## 3.2 Commitment 的 3 个 old-RVR-induced failure 不是同一种错误

Commitment 仍失败的 old induced：

- `3919ccd73c0fabd7`：commitment active≈18.75%，first collision step 53，collision 前仍为 uncertified `STOP_BEFORE_CONFLICT`；
- `c34fe8e79cdf1161`：commitment active≈70%，典型 over-commitment candidate；
- `7721ff4800156886`：commitment active≈43.75%，first collision step 78，collision 前已进入 `no_valid_candidate` bounded emergency。

其中前两个是 **SOV 本来安全、commitment 新诱发**，主要指向 continuation semantics；`7721...` 则 SOV 本身已经 false-positive，不能仅靠 state machine 修复。

## 3.3 `7721...` 暴露 successor representation 缺陷

V16.8.30 SOV 对 9 个 old RVR induced avoidance=8/9；唯一错误就是 `7721...`。

当前 successor signature：

`(conventional_exists, conventional_macro_types, conventional_candidates, max_collision_safe_prefix)`

但是算法工作的正是 zero-conventional regime。当 base/RVR successor 都仍没有 conventional candidate 时：

`(0,0,0,max_prefix)`

于是“successor option-set”会重新退化成 **单条 trajectory 的 longest prefix**。

这与 V16.8.29 的核心负结果冲突：最长 prefix 有信息，但不是 sufficient viability statistic。

因此当前 physical representation 真正没学到的是：

> 不是“最佳 recovery trajectory 还能撑多久”，而是“还有多少语义不同的 recovery choices 能分别撑到多远”。

---

# 4. V16.8.25 -> V16.8.32 完整证据链

1. **CTU negative**：certificate 后直接 planner-score argmin 伤害 offline/Waymax，证明 set-preservation frontier 有真实价值；RCOT/BCOT 不是只用于 hard threshold。
2. **RCOT/BCOT strong**：Root low-safe existence AUPRC≈0.897；BCOT priority/global false-safe≈0.837/0.928；generic candidate false-safe classifier≈0.354。前半条 semantic certificate 应保护，而不是继续替换成 generic classifier。
3. **Outcome fallback clean negative**：小幅 progress regression、无可信 physical gain；不再调 outcome weight，不升级 hard shield。
4. **V27/V28 engineering repair**：先修 conventional audit、再修 no-valid PAD execution，之后 physical attribution 才可信。
5. **V28 clean exact200**：collision first-event 主要发生在 no-conventional / valid-but-uncertified action，而不是 conventional-safe fallback 或 accepted COWP path。
6. **Conventional collapse decomposition**：主要是 `collision_empty`，roadgraph_empty 极少；所以 route/Frenet/map repair 不是当前 P0。
7. **RVR**：同一个 bank 下确实能救 10 个 collision，证明 proposal support 不是当前唯一 actionable bottleneck；但又诱发 9 个 collision，longest prefix 不充分。
8. **SOV**：只改变约1.9% steps，却挡住8/9 RVR induced，证明 emitted-action successor option support 是独立有效物理信号；但只保留2/10 rescues，high-precision/low-recall。
9. **BHOV**：放宽成 current prefix + successor non-regression，召回上升到10/10，但 precision 崩溃，只避3/9 induced，并在 disjoint holdout64 诱发一个 pure COWP/RVR 都没有的 hybrid collision。
10. **THOP**：增加 second successor 仍无法恢复 precision，且 kinematics/EP 显著变差，否定“继续堆 horizon”路线。
11. **Commitment**：mode continuity 可救回6个 SOV collision，但 unconditional continuation 又新造2个 collision；说明状态机要做 dominance-consistent hysteresis，而不是 stateless 或 sticky 二选一。
12. **Current signature structural defect**：zero-conventional successor 中 conventional-only tuple 退化成 max-prefix，需要升级到 recovery-option set persistence。

当前证据链因此收紧为：

**false-safe semantic feasibility 已较成熟 -> zero-conventional dynamic support collapse 是 collision operating regime -> recovery action 确有可挽救空间 -> single-prefix 不足 -> successor option support 有高精度信息 -> state switching semantics 与 option-set representation 是当前两个 P0 子问题。**

---

# 5. 当前 dominant bottleneck

不再写成泛泛的：

- fallback 很差；
- proposal 不够；
- safety head 不够强。

当前最精确的 P0 是：

## **Recovery Option-Set Representation + Dominance-Consistent Mode Dynamics**

即在 full conventional set 已经为空的情况下，planner 必须回答：

1. 哪个 actual emitted recovery action 保留了**更多独立语义 recovery modes**；
2. 这些 modes 在 causal horizon 上是否**持续存活**；
3. 一旦进入 recovery，何时保持/何时退出才能避免 hybrid chattering 与 over-commitment。

长期 global ceiling 仍然是 proposal support；但目前同一 bank 已有真实 rescue evidence，因此当前不应先扩 proposal。

Accepted-path kinematics 仍是 secondary independent issue：历史 clean COWP 25 个 kinematic failures 中 16 个 first event 来自 accepted_priority_ncf，17/25 前一动作 conventional-safe。它后续需要 execution-viability axis，但本轮继续解耦。

---

# 6. 模型每层成熟度

| layer | maturity | action |
|---|---|---|
| compact-5k data/labels | mature | Freeze |
| natural roots | mature | Freeze |
| RCOT same-root transport | mature/strong | Freeze |
| BCOT | mature/strong | Freeze |
| protected-priority hard certificate | mature | Freeze |
| certificate-compatible frontier | supported by CTU negative | Freeze |
| outcome head | diagnostic only | Freeze |
| 8s conventional contract | semantic baseline | Freeze |
| conventional/no-valid execution integrity | solved | Freeze |
| RVR max-prefix | useful alternative generator, failed selector | keep only controlled alternative |
| SOV successor signal | real/high precision, incomplete | absorb signal |
| BHOV | failed | archive |
| THOP | failed | archive; no more horizon stacking |
| unconditional commitment | mode signal positive, policy failed | replace state semantics |
| successor recovery-option representation | immature | V33 P0 |
| recovery state machine | immature | V33 P0 |
| accepted-path kinematics | secondary unresolved | later isolated branch |
| proposal support | long-term ceiling | do not change now |

---

# 7. 当前模型下一步真正应该“学”的内容

暂时仍不训练新 neural head。当前应该先把 analytic physical target 定义正确。

Social axis 已经在学习：

`ego plan -> other actors' natural low-burden option preservation`

Physical axis 应学习/刻画：

`(s_t, actual emitted a_t) -> successor recovery-option persistence profile`

其中不应只用 collision probability 或 max prefix，而应体现：

- distinct semantic recovery modes；
- 每个 causal horizon 上仍有多少 mode 存活；
- mode 是否在 closed loop 中持续保持 non-regression；
- recovery entry/continue/exit 的一致语义。

只有这个 analytic target 被实验验证后，才有资格考虑 distill/learn 一个 fast viability head；否则训练只是在拟合未经证实的标签。

---

# 8. 后续明确禁止的算法修改方向

继续禁止：

- CTU / planner-score replacement；
- outcome fallback weight search / hard outcome shield；
- 缩短 8 s conventional horizon；
- RVR max-prefix 直接 promotion；
- Pareto tolerance/weight search；
- BHOV epsilon/comparator 放宽；
- social/physical/utility scalarization；
- 当前阶段扩 map/Frenet/proposal primitive；
- RCOT/BCOT threshold/budget 调参；
- generic candidate safety classifier replacement；
- analytic physical target 未验证前训练 successor head；
- accepted-path kinematics 与 recovery 同轮修改。

V16.8.32 新增禁止：

- V3/V4/V5 successor horizon stacking；
- unconditional commitment promotion；
- fixed-N dwell time；
- hysteresis margin/epsilon 搜索；
- option-profile AUC/加权和作为 selection objective。

---

# 9. V16.8.33 设计

## 9.1 Diagnostic: SOV Dominance Hysteresis (SDH)

Method: `cowp_sov_dominance_hysteresis`

完全保持 V16.8.30 successor signature，只改 state transition：

- entry：`V_RVR >lex V_COWP`；
- continue：`V_RVR >=lex V_COWP`；
- exit：`V_RVR <lex V_COWP`；
- conventional/certificate restore 或 no-valid：clear。

这是一种 **parameter-free dominance hysteresis**：

- inactive tie 不进入；
- active tie 保持，不产生 strict-SOV equality chatter；
- dominance loss 立即退出，不会像 V32 commitment 一直 sticky。

如果 SDH 显著优于 commitment，则 mode semantics 是主要问题。

## 9.2 Main probe: Recovery Option-Spectrum Hysteresis (ROSH)

Method: `cowp_recovery_option_spectrum_hysteresis`

### Profile definition

在 base/RVR actual emitted action 的 causal successor state，生成原始同一 physical bank。对每个 horizon `h=1..H`：

`P(h) = # distinct non-PAD macro types that have >=1 valid + roadgraph-safe candidate with collision-safe prefix >= h`

这条曲线同时表示：

- recovery diversity；
- temporal persistence；
- full-horizon macro support。

同一 macro 的多个 timing/speed variant 不会把“option diversity”重复计数。

### Dominance

strict：

`P_RVR(h) >= P_COWP(h) for all h`, 且至少一处 `>`。

weak：

`P_RVR(h) >= P_COWP(h) for all h`，允许全相等。

profile area / min margin 只 diagnostic，不用于 selection。

### Mode transition

- strict profile dominance -> enter；
- weak profile dominance -> continue；
- 任意 horizon deficit -> exit；
- state restoration/no-valid -> clear。

因此 ROSH 不是 `max-prefix + another penalty`，而是把 physical viability object 从“最佳轨迹”改为“整个 semantic recovery option set 的 survival profile”。

---

# 10. CCF-A novelty 边界

必须保持克制：

- recursive feasibility / safety filter 已有成熟文献；
- backup-plan MPC 已经研究 alternative-plan feasibility 与 multi-horizon inputs；
- generalized backup-plan MPC 2026 仍在扩展 multistep feasibility；
- robust MPC safety architectures 已有 supervisor/backup takeover；
- 因此“多看两步”“加 commitment/hysteresis”“有 backup trajectories”本身都不够 CCF-A novelty。

当前真正有潜力形成论文主机制的是 **Orthogonal Option-Set Feasibility**：

### Social option-set feasibility

用 natural roots + same-root RCOT + BCOT 判断 ego 是否通过压缩其他 critical actor 的 natural low-burden option set 获得 safety。

### Physical-temporal option-set feasibility

用 actual emitted recovery action 后的 causal recovery-option persistence 判断 ego 是否通过一个当前看起来不错的动作压缩自己的 future executable choices。

两者的共同数学对象都是“option-set preservation”，但作用主体与语义正交；这比 risk/cost scalar 堆叠更符合当前证据链，也更有机会形成 CCF-A 级统一结构。

V16.8.33 只是验证 physical half 的正确 observable，不提前声称 ROSH 本身就是最终 contribution。

---

# 11. V16.8.33 预注册实验协议

## Stage 0

`sanity -> make_ids -> equivalence16`

COWP common path 必须继续 0 mismatch。

## Stage 1 counterfactual48

同时跑 SDH + ROSH。

每个方法 GO：

- retain >=5/10 old RVR rescues；
- avoid >=7/9 old RVR induced；
- net COWP collision reduction >=3；
- kinematics regression <=1；
- paired mean EP delta >=-0.05；
- nonzero intervention。

任何一项失败即 archive，不能调 margin/threshold。

## Stage 2 fresh37

V16.8.32 因 Stage-1 fail **从未运行** fresh37，因此 V16.8.33 新方法对这 37 scenes 未见。

GO：

- no net collision harm；
- no net CR harm；
- offroad <= +1 scene；
- kinematics <= +1 scene；
- EP delta >= -0.03；
- intervention >0。

## Stage 3 exact200

只有 Stage 1+2 pass 的方法运行；不重跑 COWP/RVR。

exact200 仍然只是 lineage development confirmation。最终论文必须另冻结从未参与机制选择的新 final evaluation scenes，并执行 multi-seed/paired CI。

---

# 12. 下一步命令

```bash
cd COWP_v16_8_33_RECOVERY_OPTION_SPECTRUM

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/你的旧COWP目录/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh make_ids

# 只有 index 缺失才运行：
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh base_equivalence16_parallel2

bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh analyze_counterfactual48
```

到这里停止。

只有 JSON 中某方法 `preregistered_gate.pass=true` 才运行 fresh37。例如 ROSH 通过：

```bash
PROMOTED_METHODS=cowp_recovery_option_spectrum_hysteresis \
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh fresh37_parallel2
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh analyze_fresh37
```

fresh37 再通过才运行 exact200。

---

# 13. V16.8.33 本地代码验证

- 新 V16.8.33 helper/state/profile tests：5/5 passed；
- V16.8.25→33 focused semantic/integrity set：**39/39 passed**；
- `py_compile`: passed；
- launcher `bash -n`: passed；
- V16.8.33 `sanity`: **39/39 passed** + all four manifest hashes passed；
- full repository `pytest -x` 首个失败仍为历史缺失 `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`；首个 failure 前 **124 passed / 5 skipped**。没有观察到 V16.8.33 新功能 regression。



## 14. 交付前实验协议加固

最终 launcher 已加入 **fail-closed promotion**：`fresh37_parallel2` 与 `confirm200_parallel2` 不再只依赖人工遵守说明，而会读取前一阶段 analyzer JSON 并校验指定方法的 `preregistered_gate.pass=true`。任何失败、缺少分析 JSON 或未知 method 都直接终止。这只加强实验纪律，不改变任何规划或控制语义。最终 focused sanity 仍为 **39/39 passed**。
