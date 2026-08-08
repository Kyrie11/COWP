# COWP v16.8.9 数据/算法/实验联合审计与优化报告

版本：**v16.8.9 — Candidate-Conditioned Causal Audit + Affected-Root Transport**

## 1. 本轮结论

**当前仍不建议 full rebuild。** v16.8.8 的 96-scene smoke 已经证明 stable-critical 修复成功，且 protected-priority/PBTR 出现明确正向信号；但 universal NCF 与 false-safe ceiling 仍失败。继续扩大 RMR/PSY/PCHR proposal grid 不足以解释或解决这一现象，因为 proposal-source ablation 中删除这些 optional proposal 后 scene-level ceiling 基本不变。

本轮定位到的主要缺口是：旧数据 contract 只知道“global critical agent”和“geometric conflict root”，却没有完整描述**某一个 ego candidate 是否真的因果影响某一个 critical agent，以及其 natural root 是否虽然没有碰撞、却已经被 ego 迫使进入高负担状态**。结果是数据标签、RootTransport 和模型 loss 之间存在语义缺口。

因此 v16.8.9 的目标不是放宽 NCF gate，而是让：

`candidate -> causal relevance -> affected roots -> response/recovery -> witness -> pair blocker -> candidate NCF`

成为一条完全一致、可训练、可审计的数据链。

---

## 2. v16.8.8 smoke 的真实信号

96-scene smoke：

- `AnyValid = 1.0`：通过；没有零候选/构建错误。
- stable critical：96/96 为 `fixed_anchor_v1`；通过。
- proposal-union monotonicity：AnyNCF / false-safe / PBTR 三项均通过。
- `AnyNCF = 0.2083`：失败（smoke 要求 >=0.30）。
- false-safe lower bound `=0.7083`：失败（要求 <=0.65）。
- PBTR lower bound `=0.4419`：通过（要求 <=0.50）。
- hard-scene recovery `=0.125`：通过 smoke threshold。
- PSY `64/64` accepted；确实生成了 protected priority-NCF pair，但对 scene-level AnyNCF/false-safe/PBTR 的增量均为 0。

48-scene paired representative subset：

| 指标 | old v16.8 bank | v16.8.8 fresh |
|---|---:|---:|
| conventional-safe scene rate | 0.8958 | **0.9167** |
| AnyNCF | **0.3750** | 0.2083 |
| false-safe floor | **0.5208** | 0.7083 |
| priority-NCF scene rate | 0.3750 | **0.5000** |
| PBTR floor | 0.5814 | **0.4419** |
| mean valid candidates | 50.77 | **54.58** |

旧 18 个 NCF scene 中，新 bank 仅保留 8 个、丢失 10 个，同时只新获得 2 个。这说明新 proposal 不是简单“物理轨迹更差”：conventional-safe 和 protected-priority 都改善，但 universal NCF 被别的 pair blocker 拖累。

proposal-source ablation（96 scenes）进一步显示：去掉 PSY、RMR 或 legacy timing 会改变 candidate count，却不改变 scene-level AnyNCF / false-safe / PBTR ceiling。因此当前瓶颈已经不能主要归因于“再多生成几条 proposal”。

---

## 3. 数据 contract 的根本缺口

### 3.1 Global critical 不等于 candidate-specific causal relevance

stable critical universe 是必要的，因为 proposal refinement 不应该反过来改变“谁属于场景 critical set”。但修复以后仍不能把：

`valid candidate × every global critical agent`

无差别当成同等强度的 non-coercion 审计对象。

一个 agent 可以对场景整体很 critical，但某个具体 ego candidate 可能根本没有扰动其低负担 natural roots。此前数据没有显式 `pair_relevant`，模型 witness/response/transport loss 也无法区分：

- genuinely affected candidate-agent pair；
- global-critical but causally irrelevant pair。

这会造成 unrelated pair 成为 universal NCF blocker，且模型被迫把大量 unrelated pair 当作普通负样本学习。

### 3.2 `mode_conflict` 不能覆盖 burden-only perturbation

COWP 的核心不是只检测 collision。存在一类重要 natural root：

- root 与 ego candidate 仍几何安全；
- 但为了应对 ego，agent 的 direct burden 已超过其 adaptive beta；
- 因而该 root 对 non-coercion certificate 已经是“被影响”的。

旧数据只提供 `transport/mode_conflict`。于是 RootTransport 可以学习“碰撞 root 怎么恢复”，却无法完整学习“安全但被迫减速/让行的 root 怎么恢复”。这正是 protected burden / PBTR 与 universal NCF 之间需要补齐的桥梁。

### 3.3 Silent blocker

旧语义下可能出现：pair 最终是 non-NCF（tail burden / OPR 不通过），但没有一致的 causal relevance、affected-root 和 positive witness support。这类 pair 对模型而言是一个“必须拒绝 candidate，但不知道为什么”的 silent blocker。

对 CCF-A 级实验，这类监督不完整会同时伤害：

- witness attribution；
- RootTransport；
- planner certificate features；
- ablation 可解释性；
- full rebuild 后的理论归因。

---

## 4. v16.8.9 数据语义修改

### 4.1 Candidate-conditioned causal audit

新增 `cowp/label/audit_relevance.py`。对每个 `(candidate k, critical agent a, natural root m)` 计算：

- `root_unsafe[k,a,m]`：几何 unsafe；
- `root_direct_burden[k,a,m]`：该 natural root 在 ego candidate 下的 direct burden；
- `root_affected[k,a,m]`：`unsafe OR burden > beta`；
- `canonical_root_weight[a,m]`：与 SetTransport/OPR 完全一致的 floor-smoothed root distribution；
- `relevance_mass[k,a]`：affected low-burden root probability mass；
- `pair_relevant[k,a]`：relevance mass 是否达到现有 witness-support 语义的 0.10 支持阈值。

这不是降低 NCF/burden threshold，而是定义**该 candidate 是否因果影响这个 global-critical agent，因而是否需要被该 pair certificate 审计**。

### 4.2 Irrelevant pair 的语义

对 candidate-specific irrelevant pair：

- 不生成 expensive safe-response search；
- OPR 设为 neutral value `1`；
- 不允许 blocker；
- pair 对该 ego intervention 是 vacuously non-coercive。

但 global critical universe 本身仍固定；因此 proposal union 不会再次改变 scene semantics。

### 4.3 Affected-root transport

RootTransport 从 conflict-only 扩展为 affected-root support：

`affected = geometric conflict OR burden-only perturbation`。

新增/保证：

- `transport/mode_affected`；
- affected-root same-root recovery；
- root minimum safe burden；
- root recovery mass；
- transported OPR；
- tail/CVaR burden support。

几何 `mode_conflict` 仍保留，是 affected 的物理安全子集，而不是被删除。

### 4.4 Pair/candidate blocker contract

新增：

- `witness/pair_noncoercive_feasible`；
- `witness/blocker_code`（burden / OPR / witness）；
- `candidates/audited_pair_count`；
- `candidates/ncf_blocker_count`；
- `candidates/min_audited_opr`；
- `candidates/max_audited_tail_burden_excess`。

Fresh schema 强制：

- relevant non-NCF pair 必须有可解释 witness/blocker；
- irrelevant pair 不能出现 blocker；
- irrelevant pair 不能存在 response search result；
- audit `root_affected` 与 transport `mode_affected` 必须一致。

---

## 5. 模型训练语义同步

仅增加标签而不改 loss 会再次出现数据/模型错位。因此 v16.8.9 同步修改：

### WitnessDecoder

新增 `relevance_logit`。relevance 对所有 valid global-critical pairs 训练；witness/OPR/burden 主监督默认只施加在 causally relevant pairs。

### SetTransport

从 conflict-only transport 扩展到 affected-root transport，并显式学习：

- conflict probability；
- affected probability；
- retain probability；
- root recovery probability；
- root minimum safe burden；
- uncertainty。

约束 `P(affected) >= P(conflict)`，并使用 affected support 计算 transported root feasibility。

### Planner

irrelevant pair 的 witness feature 被 gate 为 0，OPR feature neutralize 为 1；防止模型再次从无关 pair 学到拒绝 candidate 的 shortcut。

### paper-aligned supervision

Fresh v16.8.9 cache 不能被 silent fallback 成 v16.8.8 的 all-critical/conflict-only target。Legacy cache 仍可读，但不会被冒充 v16.8.9 paper-grade 数据。

---

## 6. 新数据集必须包含的完整监督字段

正式 full fresh cache 至少必须自包含以下信息：

1. **Candidate**：trajectory/valid/macro/proposal provenance/conventional-safe/NCF/false-safe/audited pair count/blocker count。
2. **Stable critical universe**：track index/base priority/relation/valid/reference mode。
3. **Natural roots**：trajectory/valid/source/root weight/neutral burden/adaptive beta/map validity。
4. **Causal audit**：pair relevance/relevance mass/root affected/root unsafe/root direct burden。
5. **Safe responses**：root-indexed response trajectory/valid/safe/burden/components/source/root affinity。
6. **Witness/certificate**：exists/token/rho/OPR/tail burden/pair NCF/blocker code/causal relevance support。
7. **RootTransport**：mode valid/conflict/affected/retain/recovery/root minimum safe burden/root target confidence/canonical root weights/transported OPR。
8. **Waymax-ready state**：state/is_sdc 等在线 rollout 必要 state。Waymax outcome labels 本身是可选辅助监督，不再作为训练数据构建前置条件。

v16.8.9 full cache 是**单目录、实体 NPZ、自包含 transport**，不再依赖 symlink overlay/backing Waymax cache。

---

## 7. 为什么下一轮仍然只跑 96-scene smoke

v16.8.9 改变了 label semantics，因此 v16.8.8 已生成的 labels 不能直接 retrofit。下一轮必须 fresh 重算相同 48 hard + 48 random scenes，但只需 96 个场景。

Smoke 除原 proposal ceiling 外，还必须证明：

- relevance rate 非退化（不是几乎全 0 或全 1）；
- 存在 measurable burden-only affected roots；
- silent blocker = 0；
- irrelevant blocker = 0；
- irrelevant response = 0；
- audit/transport affected 完全一致；
- stable critical 仍成立；
- proposal union monotonic 仍成立。

若 smoke 失败，**不运行 strict，不 full rebuild**。

只有 smoke 通过，才跑 400 hard + 800 random strict probe；只有 strict 输出 `recommend_full_rebuild=true`，才允许 full rebuild。

---

## 8. 构建速度判断

v16.8.8 profile 的平均耗时：

- candidate generation ~0.95 s/scene；
- critical ~20.56 s；
- safe responses ~202.31 s；
- witness ~113.45 s；
- label engine ~338.46 s。

因此主要成本在 safe-response + witness，而不是 PSY/RMR generation。

v16.8.9 对 causally irrelevant pair **直接跳过 safe-response search**，理论上会减少最昂贵阶段的无用搜索；但具体加速幅度必须由下一轮 `fresh_profile_summary.json` 实测，当前不宣称固定倍数。

Full build 仍保留不会降低数据质量的加速：

- producer-side scene allowlist；
- 精确复用旧 tensor-cache filename scene set；
- BLAS/MKL/TF 每 worker 单线程；
- self-contained inline transport；
- 不重复 transport augmentation；
- 默认不做 full-train cached Waymax replay；
- fingerprint-safe resume。

---

## 9. 训练前最后一道数据 gate

即使 strict 1200-scene PASS，full rebuild 完成后仍不会立刻训练。完整 validation cache 必须重新通过：

- AnyValid >= 0.99；
- AnyNCF >= 0.40；
- false-safe floor <= 0.55；
- PBTR floor <= 0.45；
- causal-audit read/integrity 全部通过；
- relevance 非退化；
- burden-only affected roots 存在；
- proposal-union monotonic。

任何一项失败即 `DO NOT TRAIN`。

---

## 10. 正式实验设计

数据 full gate 通过后，实验顺序为：

1. seed 2026 main mechanism；
2. fresh-data natural-decoder revalidation；
3. transport/planner + calibration + held-out mechanism gate；
4. cheap selection ablation；
5. label/proposal oracle ablation；
6. **真实独立重训**的 causal ablation：
   - w/o candidate-conditioned causal relevance；
   - conflict-only RootTransport；
7. 100-scene Waymax probe；
8. probe promotion 后 1000-scene Waymax full；
9. pipeline 稳定后 seeds 2026/2027/2028；
10. 同一 fresh proposal bank 上的 matched GameFormer/DTPP baseline。

注意：repository 内 GameFormer/DTPP 是 matched implementation baseline；除非另外验证官方实现/权重，论文中不应写成 official reproduction。

---

## 11. 当前算法取舍

**继续保留/深化**：stable critical universe、natural roots、counterfactual response、protected priority、BCOT/RootTransport、BCS-RMR、explicit fallback、certificate-guided proposal refinement。

**本轮新增且必须验证**：candidate-conditioned causal relevance、burden-only affected-root transport。这两项对应 v16.8.8 smoke 暴露的监督缺口，并提供可独立重训的 ablation。

**不要重复投入**：threshold-only repair、继续增加 BCOT budget、flat all-critical veto、继续扩大 PCHR/PSY fixed grid、把 stop/yield 默认当 non-coercive。

如果 v16.8.9 在数据 contract 完整后仍无法恢复 AnyNCF/false-safe ceiling，那么下一步才应明确归因于 action-space/proposal optimization，并进入 certificate-guided continuous/adaptive trajectory refinement，而不是继续修改标签语义。

---

## 12. 本地验证

最终 v16.8.9：

- `python -m compileall -q cowp`：PASS
- `pytest -q`：**175 passed**
- v16.8.9/core shell entrypoints `bash -n`：PASS
- new diagnostic/gate CLI import smoke：PASS
