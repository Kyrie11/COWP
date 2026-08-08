# COWP v16.8.9 smoke 数据契约修复审计

## 结论

本轮 96-scene smoke 的 `screen_pass=false` **不是由 proposal ceiling 的点估计失败导致，也不是简单的 96 场景样本太小造成的偶发工程失败**。在 48 个无偏 representative scenes 上，fresh v16.8.9 已达到：

- AnyValid = 1.0000
- AnyNCF = 0.4167
- best-case false-safe floor = 0.5000
- best-case PBTR floor = 0.4419
- hard-scene NCF recovery = 0.2083

这些点估计均通过 smoke proposal gate。相对同一批 v16.8.8 fresh scenes，AnyNCF 从 0.2083 提高到 0.4167，旧 NCF 10/10 全保留并新增 10 个 NCF scene，说明 candidate-conditioned causal relevance 是明确正向信号。

最终 smoke 只被两个数据-contract 条件阻断：

1. `transport_affected_matches_audit=false`：112231 个 audit affected roots 中存在 1258 个 transport mismatch。这是代码级 serialization/control-flow bug，不能靠增加样本消失。
2. `burden_only_affected_signal_present=false`：burden-only affected roots 只有 40 个，占 affected roots 的 0.000356。代码审计表明该稀疏性在当前 `unsafe_between` 定义下是自然结果；将固定 prevalence 作为硬 gate 会诱导人为修改语义，因此改为 advisory，而不改变 NCF/PBTR/burden threshold。

`recommend_full_rebuild=false` 在非 strict smoke 中本来就是正确设计：smoke 最多只允许 `recommend_strict_probe=true`。当前 48 个 random scenes 的 Wilson 95% 区间仍较宽，无法支持直接投入约四天 full rebuild。

## transport mismatch 根因

`audit_relevance` 已经为每个 valid candidate/global-critical pair 计算 root-level `root_unsafe/root_affected`。但旧 `witness.py` 在 pair 被判为 audit-irrelevant 后立即 `continue`，导致该 pair 的 transport root tensors 保持零值，没有复制 audit 已计算的 root support。

因此出现：

- candidate/pair relevance、response、witness、NCF 语义本身正确；
- 但 root-level `transport/mode_affected` 对 irrelevant pair 与 `audit/root_affected` 不一致。

修复后 root-level transport 在 relevance early-exit 之前填充：

- `mode_conflict = root_unsafe`
- `mode_affected = root_affected`
- `mode_retained = mode_valid & ~root_affected`

pair relevance 只控制 response/witness search 与 learned pair loss，不再改变 root-level data contract。

## burden-only affected signal 为什么不能做硬 prevalence gate

新定义显式保存：

- `root_budget_crossed`
- `root_burden_only_affected`
- `root_affected = root_unsafe OR root_budget_crossed`
- `root_burden_only_affected = root_budget_crossed AND NOT root_unsafe`

但 `unsafe_between` 已覆盖 collision、near-miss、dangerous TTC、RSS gap violation。natural trajectory 本身保持不变时，candidate-induced risk pressure 很容易同时进入 unsafe 集，因此 “burden affected but not unsafe” 本来就是很窄的补集。

因此正确的数据性质要求应是：

- 定义可计算；
- tensor 非缺失；
- root identity 严格一致；
- audit/transport 使用同一 canonical root probability measure；

而不是强制任意数据集必须有 >=0.1% burden-only roots。

## 当前 96-scene 结果是否可能只是小样本偶然性

点估计确实仍有明显统计不确定性。约 95% Wilson interval：

- AnyNCF 20/48：约 [0.289, 0.557]
- false-safe 24/48：约 [0.364, 0.636]
- PBTR 19/43：约 [0.304, 0.589]
- hard recovery 10/48：约 [0.117, 0.343]

因此不能从 48 random scenes 直接推断 full validation 一定满足 strict gate，必须运行 400 hard + 800 random strict probe。

但是这不能解释 1258 个 transport mismatch：后者是确定性的代码契约错误。也不能把 burden-only prevalence 不足解释为普通抽样噪声，因为它来自定义结构和极低基率；扩大样本只会更精确地估计这个稀疏率，不会把它自然推到人为的 0.1% gate。

## proposal/action-space 当前判断

本轮不建议继续修改 RMR/PSY geometry：

- proposal AnyValid=1.0；
- conventional-safe coverage 正常；
- v16.8.9 causal relevance 将 representative AnyNCF 从 v16.8.8 的 0.2083 提升到 0.4167；
- PBTR 仍保持 0.4419；
- proposal-source ablation 在当前 96-scene union 上 RMR/PSY 没有 scene-level ceiling 增量，但它们产生了有效 NCF candidates，可继续作为 diversity source 与论文消融。

当前应先验证 repaired supervision contract。若 1200-scene strict 在 contract 完整后仍因 proposal ceiling 失败，再进入 certificate-guided continuous refinement，而不是继续盲目扩大 primitive grid。

## 新的数据 contract

fresh label/cache 必须同时具备且一致：

1. stable global critical universe；
2. candidate-conditioned pair relevance；
3. root unsafe / direct burden / budget crossed / burden-only affected / affected；
4. relevant-pair response、witness、pair NCF、blocker reason；
5. root-level transport conflict/affected/retained；
6. canonical root weights 在 audit/transport 完全一致；
7. candidate NCF/blocker summary；
8. self-contained NPZ，无 transport symlink overlay；
9. Waymax-ready state；
10. 每个 learned head 有非退化的正/负监督。

新增 `62_audit_training_supervision.py` 会检查 candidate NCF、pair relevance、witness、pair NCF、mode conflict、mode affected、affected-root recovery、protected priority 等监督是否真实存在。full train/val cache 如某个核心 head 退化为全 0/全 1，会在 GPU training 前停止。

## 构建性能

当前 96-scene profile 平均每 scene：

- label engine: 243.66 s
- safe response: 108.50 s
- witness: 99.96 s
- critical: 18.79 s
- audit relevance: 14.48 s
- candidate generation: 0.87 s

因此构建慢的主要来源仍是 safe-response + witness，而非新 proposal generation。v16.8.9 已跳过 causally irrelevant pair 的 response search；后续只有在 strict/full profile 确认仍有必要时，才应做不改变结果的缓存/向量化优化。

## 当前 promotion 决策

1. 不要重新从 WOMD 构建当前 96-scene smoke；先使用 contract repair 工具原位修复现有 NPZ。
2. repaired smoke 通过后，只进入 400+800 strict probe。
3. strict probe 只有在 proposal gates、causal contract、training-supervision audit 全部通过且 fingerprint 一致时，才允许 full rebuild。
4. full validation cache 完成后再次全量 gate；任何失败均 `DO NOT TRAIN`。
