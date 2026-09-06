# V16.8.45R2 结果可靠性审计与 V16.8.45R3 Stage-0 工程修复

## 0. 判决

本轮严格执行用户预先规定的科研纪律：**先判断结果是否足以做算法归因；不满足时只做工程修复，不用不完整结果修改算法。**

当前结论是：

```text
V16.8.45R2 sidecar/build/train engineering evidence = broadly coherent
V16.8.45R2 Stage-0 scientific evidence             = MISSING / INCOMPLETE
RCRSO scientific GO/STOP                            = NOT AVAILABLE
algorithm attribution                               = NOT ALLOWED
new scientific algorithm / V46                      = NOT AUTHORIZED
required action                                     = repair Stage-0 runtime/observability and rerun frozen Stage-0
```

上传的结果包中没有 `stage0_val_support_audit.json`，也没有 Stage-0 finalize 后的 `rcrso_stage0_selected.pt`。用户明确说明 `stage0_support` 没有跑完，这和包内容一致。因此现在既不能判 GO，也不能判 STOP。

这不是“V45R2 已经失败”的算法结论，而是**决定 V45 scientific fate 的 preregistered measurement 尚未产生**。

---

## 1. 我实际读取到的论文与代码主线存在 provenance mismatch

当前 `COWP.zip` 中实际只有一个 TeX：

```text
paper/post-collision-v48.36-OCAF.tex
```

标题为：

> Observation-Consistent Recoverability as a Calibrated Planning Primitive for Autonomous Driving

其主方法是 **OC-RAP**，核心问题是 oracle-to-deployable recoverability gap：不同 latent futures 即使 branch-wise 各自存在 recovery，也可能在执行 prefix 后对 ego observation 不可区分，因此部署系统不能依赖隐藏 future identity 为每个 branch 选择不同 recovery。Method pipeline 是：

```text
scene/candidate prefix
→ recovery-sufficient latent roots
→ post-prefix observation-equivalence kernel
→ affordance-conditioned signed recovery margins
→ oracle recoverability vs deployable recoverability
→ OC-MERO shared-recovery aggregation
→ CRISP calibrated hard admission / nominal utility preservation
```

实验主 benchmark 写的是 WOMD + Waymax，另有 MetaDrive stress 和可选 CARLA；主指标包括 FRA、ODG、DRS、NUP、post-contact metrics 与 calibration。正文的主结果表仍是 placeholder `--`。

这与 `大模型建议.md` / `ALGORITHM_CHANGELOG.md` / 当前 V25→V45 代码链的 **COWP / safety-by-coercion / natural roots / RCOT / BCOT / protected-priority / exact blocker / verified recourse-set completeness** 并不是同一篇方法论文。

因此这是当前 artifact provenance 的一个重要问题：**我不能默认为 OC-RAP TeX 就是当前 RCRSO 应继续修改的 paper。** 本轮又处于 repair-only，所以没有修改 TeX。后续在 scientific evidence 恢复后，必须先把“ intended paper TeX 与 current COWP/RCRSO code lineage ”对齐，再做 CCF-A 主线文字更新。

---

## 2. compact-5k 数据性质：当前继续 Freeze

上传的 `formal_v16_8_24_compact_full_5k.zip` 更准确地说是 compact-5k 的 manifest/profile/natural-support/cache-verifier **性质审计包**，不是服务器完整 `tensor_cache_train/val/heldout` NPZ payload。

独立核对得到：

| 性质 | Train | Val | Heldout |
|---|---:|---:|---:|
| scenes | 5000 | 1000 | 1200 |
| audit-relevant pair rate | 0.42970 | 0.42863 | 0.42949 |
| protected PRIO-root coverage | 99.453% | 99.363% | 99.465% |
| rootless | 0 | 0 | 0 |
| `<2 low-burden roots` | 0 | 0 | 0 |
| mechanism unauditable | 4.07% | 4.34% | 4.46% |
| critical-agent mean | 5.385 | 5.387 | 5.338 |
| selected-cap saturation | 95.78% | 95.50% | 95.42% |

三个 split 的关键统计高度稳定。当前没有证据把 RCRSO Stage-0 的问题归因于 split drift、rootless、low-burden-root sparsity 或整体数据质量，因此：

> **本轮继续 Freeze compact-5k，不重建数据。**

长期 watch items 保留：

1. critical-agent cap=6 的 selected-cap saturation 已经约 95%+；
2. `PRIORITY_SMOOTH_YIELD` proposal acceptance 约 20–22%，明显偏低；
3. `TERMINAL` acceptance 约 54–55%；
4. `verify_cache_train.json` 仍写有 `pass=false`，原因是 `irrelevant pair blockers=58243`，虽然 runtime supervision 会按 `audit_target` 重 mask，投稿 artifact 前仍应修 serialization/verifier accounting 或给 cache→runtime semantic-equivalence proof。

这些是**模型收敛后判断 support ceiling 的 watch items**，不是本轮 Stage-0 未完成的解释。

---

## 3. 已有 R2 sidecar/build/train 结果能信到什么程度

### 3.1 Sidecar build 日志完整

Train 8 shards 均扫描 625/625 scenes：

```text
5000 train scenes
111,016 sidecar examples
6,748 hypothesis groups
24,445 positive examples
launcher wall = 11,031 s ≈ 3.06 h
```

Validation 2 shards 均扫描 500/500 scenes：

```text
1000 val scenes
25,235 sidecar examples
1,612 hypothesis groups
6,044 positive examples
launcher wall = 12,839 s ≈ 3.57 h
```

Val shard timing也直接说明 authoritative hard verification 是主要成本：

```text
shard0 wall             10,369.97 s
  analytic completion    3,468.27 s
  proposal verify        6,413.42 s

shard1 wall             12,835.84 s
  analytic completion    3,948.88 s
  proposal verify        8,369.98 s
```

所以 Stage-0 很慢并不奇怪：它又会重新运行 hard verifier / V44 analytic / exact CSP。

### 3.2 RCRSO training 完整跑了 30 epochs

`training_history.json` 含 30 个完整 epoch，每个 epoch 的 validation examples 都是 **25,235**，与 sidecar val 总量一致。

训练脚本以 `val_stats["set"]` 选择 `rcrso_best_unselected.pt`。最优 epoch 是：

```text
epoch 9
val set loss   = 0.08545546
val total loss = 0.72250204
```

最后 epoch 30：

```text
val set loss   = 0.08851700
val total loss = 0.76022567
```

这说明优化过程完成且 checkpoint-selection trace 自洽；后期存在轻微 validation degradation，但**不能据此判 RCRSO 成败**。预注册成功条件是 verified support completeness，不是 set-loss 是否更低。

### 3.3 当前结果包的 provenance 边界

结果包没有：

- full train/val sidecar NPZ payload；
- `rcrso_best_unselected.pt` bytes；
- Stage-0 audit；
- Stage-0 selected checkpoint。

因此我能审计 build/train logs 与 training history 的自洽性，但不能只靠这个结果 zip 重新 hash/replay server checkpoint 与完整 sidecar。

---

## 4. `stage0_support` 为什么“没有任何 hint”

这不是 Waymax hang，也不是 GPU deadlock。V16.8.45R2 的 `cowp/scripts/106_eval_rcrso_support.py` 在结构上就是**直到全部结束才输出最终 JSON**。

旧实现的主要执行路径：

```text
25,235 val examples
for each example, sequentially:
    batch-1 RCRSO neural forward
    hard verify max K=16 learned proposals
    frozen fixed-static support
    frozen fixed candidate-specific support
    V44 analytic completion
    save detailed trajectories/profiles in Python records[]

all examples finish
→ group by 1,612 hypothesis groups
→ fixed baseline exact CSP
→ V44 analytic baseline exact CSP
→ K=2 exact CSP
→ K=4 exact CSP
→ K=8 exact CSP
→ K=16 exact CSP
→ choose 95%-plateau K
→ calculate frozen Stage-0 gate
→ only now print/write final JSON
```

### 4.1 旧实现有四个工程问题

#### A. 完全无进度可观测性

没有 `status_every` / progress callback。即使正常运行数小时，terminal 也会像“卡死”。

#### B. Stage-0 是单进程

用户有两张 GPU，但旧 `stage0_support` 只启动一个 process。GPU 只负责小型 RCRSO forward；hard verifier 与 CSP 主要是 CPU/NumPy/Python，因此第二张 GPU 和额外 CPU process 没被利用。

#### C. 每个 item 重复构造 fixed static bank

旧代码先：

```python
fixed_static_profiles = _fixed_static_bank_profiles(item, cfg)
```

紧接着 `_fixed_bank_profiles(item,cfg)` 内部又重新调用一次 `_fixed_static_bank_profiles`。这是完全相同的 authoritative frozen computation，可安全 work-reuse。

#### D. CSP 的 pairwise trajectory compatibility 被重复计算

每个 hypothesis group 上会分别运行 fixed/V44/K2/K4/K8/K16 六套 exact CSP。很多 responder-trajectory pairs 在这些 nested domains 中相同，旧代码每次都重新计算 4-direction current/shift collision predicate。

此外旧实现把所有详细 trajectory/profile records 留到最后才聚合，造成不必要的内存驻留。

---

## 5. V16.8.45R3 工程修复：不改变任何 scientific boolean

我没有设计 V46。新版本命名为：

> **V16.8.45R3 — Stage-0 Runtime / Observability Fidelity Repair**

scientific method 仍然是：

```text
V16.8.45 RCRSO unchanged
```

### 5.1 实时进度

每个 Stage-0 shard 默认每 30 s 输出：

```text
examples=done/total
groups=...
verifier=...
elapsed=...
rate=... examples/s
eta=...
timing[load/model_forward/learned_verify/fixed/analytic/csp percentages]
```

所以服务器上不再出现“几个小时没有任何 hint”。

### 5.2 scenario-disjoint parallel2

Stage-0 validation 按 `scenario_hash % 2` 拆成两个 process。一个 scenario 的所有 hypothesis groups/roots 永远在同一个 shard，避免跨 shard 拆 CSP。

两个 shard 完成后只合并**整数 raw counts 与 burden sums**，再统一执行完全相同的：

- baseline coverage；
- K curves；
- 95% plateau K selection；
- 3 pp lift Gate。

Merge 脚本强制检查：

- shard index 完整；
- scenario hashes 无 overlap；
- assigned examples 全部完成；
- checkpoint SHA256 相同；
- sidecar summary SHA256 相同。

### 5.3 streaming group aggregation

每个 hypothesis group 的 exact metrics 算完即丢弃 trajectory profiles，不再持有完整 25k examples 的 detailed profile records 到结束。

### 5.4 fixed static exact work reuse

同 item 的 fixed static bank只算一次，再提供给 candidate-specific fixed screen。

跨 hypothesis 的任何静态 cache 复用都要求下面输入的 semantic namespace 完全一致：

```text
agent/root/object_type/beta
root trajectory + blocker state
candidate-local roadgraph subset
environment current/shift trajectories
```

这点很重要：profile8 审计发现**同 scene 同 agent/root 在不同 ego hypotheses 下 candidate-local roadgraph subset 可能不同**，所以简单使用 `(agent,root)` 作为 Stage-0 static-cache key 是不安全的。R3 最终版已经将 roadgraph/environment/root-state 都绑定到 cache namespace，避免 stale hard predicate。

### 5.5 exact CSP pair memoization

只对**字节级相同**的 current/shift responder trajectory pair 复用 `_pair_ok` 结果。它不近似几何，不改变 threshold，也不跳过新的 trajectory pair。

---

## 6. R3 semantic fidelity 验证

### 6.1 Legacy vs R3 smoke

使用同一个 deterministic synthetic RCRSO checkpoint 和 R2 sidecar smoke，比较以下 17 个 Stage-0 科学字段：

```text
examples / eligible_examples
hypothesis_groups
oracle positives
fixed baseline
V44 baseline
K curves
plateau / selected K
selected metrics
coverage lift
frozen minimum lift
stage0 gate
verifier calls
```

结果：

> **17/17 scientific fields exact match.**

### 6.2 Cross-hypothesis semantic reuse regression

另外抽取同一 scenario 的 3 个 hypothesis groups、24 sidecar examples，legacy R2 与最终 R3：

> **17/17 scientific fields exact match.**

这专门验证 candidate-local roadgraph 不同情况下的新 semantic namespace 没有改变 hard admission。

### 6.3 Parallel merge regression

把 smoke 按 scenario-disjoint 2 shards 执行并 merge，和 legacy single-process：

> **17/17 scientific fields exact match.**

### 6.4 Focused repository sanity

最终重新运行 V25→V45R3 focused semantic/integrity suite：

```text
143 / 143 passed
```

Python compile 和 launcher `bash -n` 均通过。

---

## 7. 本轮为什么绝对不能做 V45 GO/STOP

RCRSO 在前一轮已经预注册：

```text
K ∈ {2,4,8,16}
K: validation only, smallest K reaching 95% of FullHypothesisRootCoverage plateau

Stage-0 GO iff:
selected FullHypothesisRootCoverage
  - max(fixed baseline FHR, V44 analytic baseline FHR)
>= 3 percentage points
AND selected VerifiedRootRecall > 0
```

而当前上传包根本没有 Stage-0 final metrics。

因此以下全部没有证据基础，本轮不做：

- RCRSO = GO/STOP；
- 解释哪个新机制生效；
- promotion RCRSO；
- 关闭 learned-recourse family；
- 更新 dominant scientific bottleneck；
- 设计 V46；
- 修改论文 Method/Abstract/Contributions；
- 用训练 loss 猜 Stage-0 support coverage。

当前 scientific question 仍保持冻结：

> **learned set-valued RCRSO proposal 是否在完全不放宽 hard verifier 的前提下，真正提高 universal retained-root verified support completeness？**

---

## 8. 内部迭代收敛条件：保持原预注册，不在本轮触发

当前 broader learned-recourse family 的关闭协议保持不变：

1. 连续 **2 个** preregistered architectures 都既没有 `FullHypothesisRootCoverage ≥ +3 pp`，也没有增加至少 `+1 lost7 rescue` → 关闭 learned-recourse family；
2. 最多 **3 个** preregistered architectures 仍没有一个达到 lost7 `≥2/7` → 关闭 learned-recourse proposal，转 `natural-root validity / interaction-model uncertainty`；
3. 任何 gain 只有通过放宽 beta/root/CSP/horizon/hard verifier 才能得到 → 立即 STOP。

当前 RCRSO 连**第一个合法 Stage-0 verdict 都还没有**，所以不应把 R2 的“没跑完”算成一次 architecture failure。

离“内部迭代结束、可以和外部 baseline 做正式比较”最大的差距也暂时不是 headline Waymax collision，而是：

> **尚未得到第一个可审计的 RCRSO support-completeness gate。**

在这之前讨论 final baseline comparison 会前置证据顺序。

---

## 9. 下一步命令：不重建 sidecar、不重新训练

R2 的 train/val sidecar build 和 30-epoch training 已完成。现在只应该复用：

```text
recourse_sidecar_v16_8_45r2
rcrso_best_unselected.pt
```

然后跑 repaired Stage-0。

```bash
cd COWP_V16_8_45R3_STAGE0_RUNTIME_OBSERVABILITY

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k

# 指到你服务器上 R2 真正训练出来的 checkpoint；不要用结果 zip，因为上传包没包含 pt bytes。
export RCRSO_UNSELECTED=/你的V16.8.45R2运行目录/outputs/v16_8_45r2_rcrso_operator/rcrso_best_unselected.pt

# R2 sidecar 默认已经是：
export SIDECAR_ROOT="$COWP_ROOT/recourse_sidecar_v16_8_45r2"

bash NEXT_RUN_COMMANDS_V16_8_45R3_STAGE0_RUNTIME_OBSERVABILITY_CN.sh sanity

# 推荐：两张 A30 对应两个 scenario-disjoint Stage-0 process；terminal 每 30s 会有 hint。
bash NEXT_RUN_COMMANDS_V16_8_45R3_STAGE0_RUNTIME_OBSERVABILITY_CN.sh stage0_support
```

`stage0_support` 在 R3 中就是 recommended `parallel2`。如果只想诊断单进程：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R3_STAGE0_RUNTIME_OBSERVABILITY_CN.sh stage0_support_single
```

Stage-0 如果 FAIL，merge/finalizer 按原协议会写出审计 JSON 后以 **exit code 4** 退出；这表示 preregistered scientific STOP signal，而不是脚本 crash。

第一轮只把下面文件给我即可：

```text
outputs/v16_8_45r3_stage0_runtime_observability/stage0_val_support_audit.json
outputs/v16_8_45r3_stage0_runtime_observability/stage0_parallel2_wall_seconds.txt
outputs/v16_8_45r3_stage0_runtime_observability/stage0_partials/stage0_val_s0.log
outputs/v16_8_45r3_stage0_runtime_observability/stage0_partials/stage0_val_s1.log
```

只有 `stage0_support_gate.pass=true` 才进入 `base_equivalence16_parallel2`。在这之前不要跑 lost7。

---

## 10. 本轮 Changelog 决策

本轮**不关闭新的 scientific algorithm family**。

新增记录：

- `V16.8.45R2 Stage-0 incomplete → scientific status unresolved`；
- 旧 Stage-0 single-process/no-progress 路径归档为 engineering runtime/observability defect；
- V16.8.45R3 仅做 exact semantic work reuse / scenario-disjoint parallel / streaming/progress；
- RCRSO scientific method、Stage-0 threshold、K selection、lost7/CF48 protocol全部冻结；
- 不授权 V46。

