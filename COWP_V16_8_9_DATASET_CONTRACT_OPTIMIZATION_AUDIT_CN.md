# COWP v16.8.9：算法—数据契约审计、WOMD 1.3.1 使用边界与全量重构方案

> 审计对象：用户上传的 `interactive_planning_v16_7_revised.tex`、当前 `COWP.zip`（以 `ALGORITHM_CHANGELOG.md` 和 v16.8.9 当前代码为算法真值）、`构建数据集指令.txt`、`formal_v16_8_9_causal_audit_strict_probe_contract_fixed.zip`。
>
> 结论先行：**当前上传的 strict-probe ZIP 不是“1200-scene strict 数据性质失败”的证据，而是一个在 old-ceiling/manifest 阶段就中断的不完整产物。不要据此启动 full rebuild，也不要再改 NCF/PBTR 阈值。应先用本次补丁重新跑 raw-WOMD preflight → 96 smoke → 400+800 distinct strict；只有 strict verdict 明确 `recommend_full_rebuild=true` 且 fingerprint 一致，才运行 full rebuild。**

---

## 1. 我对当前代码算法的理解（以 v16.8.9 代码为准，不强行要求与论文 v16.7 一致）

论文的核心论点是：传统“ego 自身无碰撞”不等价于交互安全；如果 ego 的计划只有在其他参与者承担高 burden 的让行/急刹/高风险响应时才能成立，则它是 false-safe。当前代码把这个思想扩展成一套 candidate-conditioned 的证书化标签系统。

### 1.1 当前标签生成链路

1. **Ego candidate bank**：先生成有限、确定性的 ego 轨迹候选，带 proposal source/provenance。候选空间是证书的动作域；如果动作空间本身没有 NCF 解，后续网络不可能学习出不存在的安全解。
2. **Global critical universe**：对场景先建立 candidate-independent 的关键交通参与者集合，避免候选改变 critical set 引入标签漂移。
3. **Natural-root alternatives**：对每个 critical agent 生成 OBS / NEU / PRIO 等 natural alternatives；它们是“在 ego 不施加特定干预时，该 agent 可自然采取的低 burden 选择”的离散根集合。根权重使用 floor-smoothed canonical probability measure。
4. **Candidate-conditioned causal audit**：对 `(ego candidate, critical agent, natural root)` 计算：
   - `root_unsafe`；
   - candidate 对该 root 诱发的 direct burden；
   - `root_affected = root_unsafe OR direct_burden > beta`；
   - 按 canonical root weights 聚合 affected mass，并与 relevance threshold 比较，得到 `pair_relevant`。
5. **Irrelevant pair**：该 candidate 没有实质改变这个 critical agent 的自然低 burden 支持，因此该 pair 对该干预按定义 vacuously non-coercive，不做 response/witness 搜索，OPR 取中性值 1。
6. **Relevant pair**：搜索 same-root / safe-response 集，要求物理安全并检查 burden；进一步得到 root recovery、minimum safe burden、tail burden/CVaR、OPR 等。
7. **Affected-Root Transport / RootTransport**：v16.8.9 不再只把 geometric conflict 当 transport 支持，而是使用 `affected`。所以 burden-only affected root 与 unsafe root 进入同一 root-preservation 监督体系；`conflict` 仍是 `affected` 的物理安全子集。
8. **Witness / pair NCF**：对 relevant 且非 NCF 的 pair，必须有可解释 witness（机制 token、冲突/事件 interval、burden、OPR、tail 等），禁止 silent blocker。
9. **Candidate NCF**：候选首先必须 conventional-safe，然后对所有 **causally relevant** 的 critical pairs 满足 pair-level non-coercive feasibility；false-safe 是 conventional-safe 但非 NCF 的候选。
10. **Planner / learned surrogate**：网络并不是只学一个 NCF bit，而是同时学习 natural/source、response、relevance、witness、burden、RootTransport、candidate certificate/ranking 等多组目标；因此“cache 能打开、AnyNCF 非零”远远不等于数据足够训练当前模型。

### 1.2 与论文 v16.7 的重要差异（是合理演化，不视为 bug）

- 论文强调 protected-priority critical agents；当前 v16.8.9 代码改成 **stable global critical universe + candidate-conditioned causal relevance**。
- 论文/旧代码主要从 geometric conflict 理解 root transport；v16.8.9 扩为 `unsafe OR burden-budget-crossing` 的 affected-root transport。
- 当前代码把 irrelevant pair 从 response/witness 搜索和 learned pair loss 中排除，但 root-level audit/transport tensor 仍必须保持一致。

因此，后续数据合同必须服务 **当前 v16.8.9 算法**，而不是机械复刻 v16.7 文稿。

---

## 2. 什么样的数据才能完整支撑训练、验证和测试/最终评测

我把要求分成五层。任何一层失败，都可能出现“模型能跑，但论文结论站不住”的问题。

### 2.1 L0：WOMD 原始输入和时间语义必须正确

对当前 COWP 8 s 证书，训练/开发离线标签必须使用 **future-visible 的 training/validation**：

- WOMD Motion 是 10 Hz；train/val 一个 window 有 `10 past + 1 current + 80 future = 91` 个时刻；`current_time_index=10`。
- 官方 test 隐藏 future，只保留 `10 past + 1 current = 11`，不能用公开 test 直接构造当前 COWP 依赖 80-step future 的离线 counterfactual labels。
- Scenario proto 的每个 track 与 `timestamps_seconds` 对齐；`valid=false` 表示该时刻没有测量。必须尊重 valid mask，不能把 padding/未来才出现的 agent 当成真实历史。
- tf.Example 的状态维度固定到 128 个 objects；Scenario proto 的 track 数可以更大，所以 Scenario 生成的 critical index 必须在合并时通过 object ID 映射到 tf.Example 128-row state，并屏蔽不在模型输入中的 critical agents。
- 坐标是每场景独立 origin 的 global East/North/Up；不能跨 scene 直接比较绝对坐标。
- training Scenario 与 training tf.Example、validation Scenario 与 validation tf.Example 必须来自同一 WOMD release/split，并按 `scenario_id` 对齐。

### 2.2 L1：WOMD 1.3.1 / Waymax route 数据合同

WOMD 1.3.1 新增 `sdc_paths`。官方 tf.Example 字段是：

- `path_samples/xyz`
- `path_samples/valid`
- `path_samples/id`
- `path_samples/arc_length`
- `path_samples/on_route`

Waymax 1.3.1+ 可以据此计算包含 wrong-way / route-progression 在内的 route 相关指标。

**一个容易误用的点：当前官方 `Scenario` proto 定义本身没有 `sdc_paths` 字段；`path_samples/*` 是 tf.Example 侧的数据。** 因此你现在“Scenario proto 构 COWP counterfactual label + tf.Example 合模型输入/Waymax route data”的两阶段路线是合理的，不能期待 `uncompressed/scenario/training` 里的 Scenario message 自己带 `path_samples/*`。

本次代码将 `--require-waymax-ready` 与 `--require-sdc-paths` 分开：

- safety-only Waymax replay 不因 route path 缺失而伪装成 route-complete；
- paper-grade 全量 tensor cache 用 `--require-sdc-paths` 逐场景强制 1.3.1 字段合同；
- **只验证字段/shape/有限性，不因某个官方场景没有 valid on-route path 就删掉该场景**。Waymax 对“没有 valid on-route path”有定义，因此把它当坏数据会造成选择偏差。

### 2.3 L2：COWP 标签语义必须闭环

每个 fresh label 至少应完整包含并满足：

- stable global critical set；
- candidate valid/conventional-safe/NCF/false-safe + proposal provenance；
- natural valid/source/weight/traj/priority/burden-neutral/beta；
- candidate-conditioned `pair_relevant`；
- root `unsafe / direct burden / budget crossed / affected`；
- canonical root weights 在 audit 与 transport 中一致；
- relevant pair 才搜索 response/witness；irrelevant pair 不产生 blocker；
- relevant non-NCF 必须有 witness；
- root-conditioned response 有 root identity；
- mode conflict 是 affected 的子集；
- affected root 有 recovery / root min-safe-burden / confidence / retained support；
- unsafe root 的事件 interval 完整；
- candidate NCF 与 pair NCF、blocker counts 一致。

历史 repaired smoke 中曾出现 `audit/root_affected` 与 `transport/mode_affected` 在 irrelevant pair 上不一致，这正是“标签语义看似可用但 learned transport supervision 不闭环”的典型问题。

### 2.4 L3：当前网络的 active supervision 必须有统计支持

不能只检查 5~6 个 binary head。至少要检查：

- candidate NCF / false-safe 有正负样本；
- pair relevance / pair NCF / witness 有正负样本；
- response valid/safe/low-burden/min-burden 非退化；
- transport conflict/affected/retained/recovery 非退化；
- OBS/NEU/PRIO natural sources 有支持；
- PRED/OPT/EMG response sources 有支持；
- witness positive mechanism token 不应只剩一个类别；
- OPR、tail burden、min-safe-burden、response burden、root recovery score 等连续目标必须 finite 且有方差；
- 每个 critical 至少有 natural root，当前严格审计要求 multi-root support；
- affected/conflict/root interval 等完整性关系成立。

本次新增 `65_audit_model_support.py` 就是做这一层，strict probe 和 full labels/cache 都会执行。

### 2.5 L4：训练 / 工程验证 / 最终评测的场景角色必须隔离

这是目前流程里最容易被忽视的科学性问题。

- 你原始流程的 `--limit 22000` / `--limit 5000` 是“写够 N 个就停”，不是显式 hash-random sample；所以旧 22k/5k cohort **不能仅从代码证明对完整官方 split 无偏代表**。
- v16.8.x 反复在旧 val cache 上做 smoke/strict/gate，这一批数据属于 **engineering/development validation**，不应再称为完全 untouched test。
- official WOMD test future 被隐藏，不能直接做当前 8 s COWP offline certificate ground truth。

因此论文级建议是三角色：

1. `train`: fresh training cohort；
2. `dev/validation`: 当前 full val + smoke/strict/gates，用于算法/阈值/超参选择；
3. `frozen publication holdout`: 从**完整 validation index** 中按确定性 hash 抽取，并排除所有旧 val cache / strict / development IDs。只有算法和超参冻结后才构建/打开。

本次新增 `67_make_hash_holdout_manifest.py` 来做这一点。这样不破坏你当前 paired strict gate 的可比性，又能为最终论文结果补一个真正未参与开发的 future-visible evaluation cohort。

---

## 3. 上传的 `formal_v16_8_9_causal_audit_strict_probe_contract_fixed.zip` 为什么没有通过

### 3.1 这个 ZIP 实际不是一个完成的 strict probe

我逐项解包后发现只有 5 个文件：

- `current_proposal_ceiling.json`
- `logs/old_ceiling.log`
- `v16_8_9_code_fingerprint.sha256`
- `hard_scene_ids.txt`
- `probe_union_scene_ids.txt`

而一个真正完成的 strict probe 至少还应有：

- `representative_random_scene_ids.txt`
- fresh `labels_val_v16_8_9/*.npz`
- `fresh_probe_profile*.json*`
- `paired_proposal_probe.json`
- source ablation
- causal audit
- training/model support audit
- `v16_8_9_strict_verdict.json`

这些在上传 ZIP 中全部不存在。

### 3.2 manifest 本身已证明中断在 random-ID 阶段

上传包中：

- hard IDs = **400**；
- union IDs = **400**；
- `union == hard`；
- random manifest 不存在；
- 但 `current_proposal_ceiling.json` 明确记录 `representative_random_scene_count = 800`。

更关键的是，它记录的 random ID 路径是：

`/home/senzeyu2/code/COWP/20034`

这不是合理的 manifest 文件名。它非常符合 Bash 特殊变量 `RANDOM` 被误当普通路径变量时的表现：读取 `$RANDOM` 会得到 0~32767 的伪随机整数。因此我把它判断为**高概率的 shell path-variable collision / historical command bug**，而不是算法数据性质失败。

当前上传的 `COWP.zip` 里的 strict wrapper 已使用 `RANDOM_IDS_FILE`，说明代码包本身比这次失败产物的生成命令更新；本次补丁又增加了 manifest fail-fast，对 numeric basename、数量、路径、union、重复等显式检查。

### 3.3 `full_rebuild=false` 的语义也被误读了

`current_proposal_ceiling.json` 是**旧 cache ceiling 诊断**，其中设计上就写着：

- old cache alone 不能 justify full rebuild；
- 必须先做 paired fresh proposal probe。

所以在 old-ceiling 阶段看到 full rebuild 未获授权是正确行为，不是 fresh smoke/strict 数据性质失败。

另外，v16.8.9 的 96-scene smoke 本来也**不允许**直接授权 full rebuild；smoke 只能授权进入 strict。只有 400 hard + 800 random strict 全部通过才能令最终 verdict 的 `recommend_full_rebuild=true`。

### 3.4 已有 repaired 96-smoke 的算法点估计其实是通过的

仓库的 `COWP_V16_8_9_SMOKE_CONTRACT_REPAIR_AUDIT_CN.md` 记录了 repaired 48 representative scenes：

- AnyValid = 1.0000
- AnyNCF = 0.4167
- false-safe floor = 0.5000
- PBTR floor = 0.4419
- hard recovery = 0.2083

这些都通过 smoke proposal point gate。历史 smoke 真正的阻断项之一是 `transport_affected_matches_audit=false`，根因是 irrelevant pair 的 transport tensor 在 early-continue 前没复制 audit root support；当前代码已经修复。

但是 48 random 的 Wilson interval 很宽，所以 96 smoke 仍不能替代 1200 strict。

---

## 4. 本次代码优化做了什么

### 4.1 不改变算法定义的构建加速

#### A. Safe-budget trajectory bank 复用

旧逻辑对每个 `(candidate, critical-agent)` 都重复生成相同的 candidate-independent safe-budget response trajectory primitives；真正 candidate-conditioned 的是后续 collision/risk/burden 评价。

现在：

- 每个 critical agent 只生成一次 trajectory bank；
- 每个 candidate 仍逐一执行原始 safety + burden 评价；
- NCF、beta、OPR、response budget、trajectory primitive 参数全部不改。

#### B. Same-root recovery trajectory bank 复用

root-conditioned recovery trajectory 只依赖 `(natural root, config)`，不依赖 ego candidate。现在按 `(agent, root)` 生成一次，再对每个 candidate 做原始 unsafe/burden 评价。

#### C. 复用 audit 已计算的 unsafe event interval

causal audit 已经算过 root unsafe event mask；旧 witness 为了解释 interval 又跑一次 geometry。现在 audit 直接写 `root_event_interval`，witness 读取它；如果旧数据缺该字段仍有 fallback。

这三项均不改变候选集合、root 集、response search budget、阈值、NCF/OPR/CVaR 定义。

### 4.2 strict 400+800 manifest 修复

新增/修改：

- strict representative random **显式排除已选的 400 hard probe IDs**，保证 400+800 真的是 1200 个 distinct scenes；
- `63_validate_probe_manifest.py` 检查 hard/random exact count、唯一性、disjoint、union exact 1200、ceiling 路径/数量一致，以及 numeric random-path bug；
- smoke 同样自动从 source random 中剔除与 48 hard 的重叠，保证 96 distinct scenes。

这是我在最终审计时发现的另一个重要问题：旧 `45_diagnose_proposal_ceiling.py` 的 random 是从全体 old-cache IDs 抽样，理论上可能与 hard probe 重叠；如果强称“400+800=1200”，就必须让二者 disjoint。现在 strict wrapper 通过 `--random-exclude-hard-probe` 明确做到这一点。

### 4.3 WOMD 1.3.1 fail-fast 与逐场景强制

新增 `64_validate_womd_v131_contract.py`：在耗时 labels 前抽样审计 train/val 的 Scenario + tf.Example：

- Scenario: 91 timestamps、current index 10、合法 SDC、SDC current valid；
- tf.Example: state/id=128、唯一 SDC、10/1/80 状态 shape、roadgraph、Waymax core keys、1.3.1 path tensor contract。

full tensor-cache merge 再逐 matched scene 执行：

- `--require-waymax-ready`
- `--require-sdc-paths`

并把结果写到 cache meta。`60_verify_fresh_v16_8_9_cache.py` 对现有/resume cache 也检查 `sdc_paths_ready`，防止旧缓存借 `--skip-existing` 混入新数据。

### 4.4 从“核心二分类头审计”扩展为“完整 learned supervision 审计”

新增 `65_audit_model_support.py`，strict/full 都执行。它不是为了“把所有数调到能过”，而是让真实缺失/塌缩在 full rebuild 前暴露。

### 4.5 性能优化必须做 semantic-equivalence 守门

新增 `66_compare_label_semantic_equivalence.py`：

- 对相同 scenario 的公共 label tensors 做 exact/bitwise 比较；
- 默认只允许新增的 `cowp/audit/root_event_interval` 不在旧版本中；
- 用于确认本次 trajectory-bank/interval reuse 没改变既有 label semantics。

本地 synthetic primitive equivalence 已验证相同输入时 trajectory bank 路径与旧生成路径 exact 一致；完整 WOMD semantic parity 仍必须在你的服务器用同一 smoke scene IDs 运行 `66` 才能完成最终证明。

### 4.6 full rebuild 在耗时阶段前/后都有双重 gate

`PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh` 现在：

1. 先要求 strict verdict `recommend_full_rebuild=true` 且 fingerprint 相同；
2. raw WOMD 1.3.1 preflight；
3. fresh labels；
4. **labels 层 model support strict audit**；
5. tensor merge，逐场景 Waymax + sdc_paths contract；
6. fresh-cache integrity；
7. **实际 tensor cache 层再跑 model support**（因为 128-object mask 可能改变 supervised support）；
8. full validation proposal/causal gate；
9. 任何一步失败都不进入训练。

### 4.7 实验 split hygiene

新增 `67_make_hash_holdout_manifest.py`，从完整 index 按 `SHA256(seed, scenario_id)` 确定性抽样，并可重复 `--exclude` 已开发 cache/manifest，生成 frozen final-eval IDs + digest manifest。

---

## 5. 构建速度：哪些能安全优化，哪些不应碰

仓库已有 repaired 96-scene profile：

- label engine：243.66 s/scene
- safe response：108.50 s/scene
- witness：99.96 s/scene
- critical：18.79 s/scene
- audit relevance：14.48 s/scene
- candidate generation：0.87 s/scene

即 safe-response + witness 约占 208.46/243.66 ≈ **85.6%**。因此优化 proposal generation 几乎没有意义；主要目标必须是 response/witness。

### 本次已经做的“严格等价”优化

- safe-budget trajectory primitive bank 复用；
- same-root recovery primitive bank 复用；
- audit unsafe interval 复用；
- sparse probe allowlist 在主进程过滤；
- multiprocessing `FIRST_COMPLETED` 动态补任务；
- resume 完整性检查；
- BLAS/TF 单线程抑制 worker oversubscription；
- `--no-compress`；
- full build 默认不做 optional full-train Waymax candidate replay；
- 不做重复 post-hoc transport overlay。

### 不建议为了速度做的事

以下都会改变标签/训练分布，除非另做 controlled equivalence/ablation，否则不要做：

- 减少 natural roots；
- 减少 response profiles；
- 降低 rollout horizon；
- 放松 unsafe / burden / OPR / relevance threshold；
- 减少 critical agents；
- 只保留“容易出 NCF”的场景；
- 删除没有 on-route path 的合法 WOMD 1.3.1 场景。

### 下一阶段可能的大优化（本补丁未冒险实现）

candidate-conditioned 的 collision / TTC / RSS / burden geometry 仍是主要算力。如果 fresh smoke profile 证明 bank reuse 后这里仍占大头，下一步应做 **candidate×response batched geometry/vectorization**，但必须以 `66_compare_label_semantic_equivalence.py` 做逐 tensor exact/容差审计。这个优化空间可能比继续增加 worker 更大，但代码风险也更高，不适合在当前 strict 尚未完成时直接上线。

### worker 数不要盲目增大

你的标签生成是 CPU+内存带宽+NUMA 混合瓶颈，32 workers 不一定比 24 快。建议只在固定 16~24 scene allowlist 上对 16/24/32 做一次 microbenchmark，并比较 `profile` 的 wall time / p90，而不是拿 full rebuild 调参。

按旧 243.66 s/scene 的均值做理想化计算，22k train /32 workers + 5k val /24 workers 仅 labels 就约 60.6 小时；这是假定完美负载均衡、无 I/O/straggler 的理论量级，不是对你机器的实际承诺。历史“约四天”完全可能由长尾、CPU oversubscription、内存/NUMA 和其它阶段拉长。因此先用 96 smoke 的**新 profile 实测**本补丁收益，再决定 worker 数。

---

## 6. 推荐执行顺序（严格按这个顺序，不要直接 full rebuild）

假设代码目录是 `/home/senzeyu2/code/COWP`。

### Step 0：安装补丁并本地测试

```bash
cd /home/senzeyu2/code/COWP
cp -a . ../COWP_before_dataset_contract_patch
patch -p1 < /path/to/COWP_v16_8_9_dataset_contract_optimized.patch

python -m compileall -q cowp
bash -n NEXT_RUN_COMMANDS_V16_8_9_CAUSAL_AUDIT_SMOKE_CN.sh \
        NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh \
        PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
pytest -q
```

本审计环境结果：**182 passed, 3 skipped**。

### Step 1：先检查你机器上的 raw WOMD 1.3.1，不生成任何 expensive COWP labels

```bash
cd /home/senzeyu2/code/COWP

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
export SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
export TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"
export TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"

python -m cowp.scripts.64_validate_womd_v131_contract \
  --tfexample-train-glob "$TFEXAMPLE_TRAIN" \
  --tfexample-val-glob "$TFEXAMPLE_VAL" \
  --scenario-train-glob "$SCENARIO_TRAIN" \
  --scenario-val-glob "$SCENARIO_VAL" \
  --sample-shards 64 \
  --scenario-sample-shards 32 \
  --require-sdc-paths \
  --output /data0/senzeyu2/dataset/COWP/womd_v1_3_1_preflight_before_v16_8_9.json
```

**只有 `pass=true` 才继续。**

### Step 2：用全新目录重新跑 96-scene smoke

不要在旧失败目录上混 fingerprint。

```bash
cd /home/senzeyu2/code/COWP

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE="$COWP_ROOT/tensor_cache_val"
export SOURCE_PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke
export SMOKE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_smoke_optimized
export HARD_COUNT=48
export RANDOM_COUNT=48
export LABEL_WORKERS=24
export FORCE_REBUILD_SMOKE=1

bash NEXT_RUN_COMMANDS_V16_8_9_CAUSAL_AUDIT_SMOKE_CN.sh
```

必须检查：

- `$SMOKE_ROOT/v16_8_9_smoke_verdict.json`
- `$SMOKE_ROOT/causal_audit_diagnostic.json`
- `$SMOKE_ROOT/training_supervision_audit.json`
- `$SMOKE_ROOT/model_support_audit.json`
- `$SMOKE_ROOT/fresh_profile_summary.json`

smoke 的 model-support audit 是诊断性（96 scenes 对 rare class 太小），真正 strict 的 model-support gate 在下一步。

### Step 2b：如果你保留了“修复前/本次优化前”的同一批 fresh smoke labels，做 semantic parity

```bash
python -m cowp.scripts.66_compare_label_semantic_equivalence \
  --reference-dir /path/to/previous_repaired_v16_8_9_smoke_labels \
  --candidate-dir "$SMOKE_ROOT/labels_val_v16_8_9" \
  --max-scenes 0 \
  --output "$SMOKE_ROOT/optimization_semantic_equivalence.json"
```

理想结果是 `pass=true`；公共 tensors 全等，只允许本次新增的 event-interval 字段出现在 candidate 侧。

### Step 3：只有 smoke pass 后，跑真正的 400 hard + 800 distinct random strict

**使用全新 PROBE_ROOT**，不要复用上传 ZIP 对应的旧目录。

```bash
cd /home/senzeyu2/code/COWP

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE="$COWP_ROOT/tensor_cache_val"
export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe_optimized
export HARD_COUNT=400
export RANDOM_COUNT=800
export SEED=2026
export LABEL_WORKERS=24
export FORCE_REBUILD_PROBE=0

bash NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh
```

本次 strict wrapper 会在 build 前检查：

- hard=400；
- random=800；
- overlap=0；
- union=1200；
- random manifest 路径非数字；
- old ceiling 与 manifest 路径/数量一致。

然后执行 fresh labels、paired proposal、source ablation、causal audit、training supervision、完整 model support、strict screen。

### Step 4：明确读取 strict verdict，不要凭 console 印象判断

```bash
python - <<'PY'
import json, os
p = os.path.join(os.environ['PROBE_ROOT'], 'v16_8_9_strict_verdict.json')
r = json.load(open(p, encoding='utf-8'))
print(json.dumps(r, indent=2, ensure_ascii=False))
assert r.get('recommend_full_rebuild') is True, 'DO NOT FULL REBUILD'
PY
```

只有这里通过，才进入下一步。

### Step 5：full rebuild，用新的 COWP_ROOT

```bash
cd /home/senzeyu2/code/COWP

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_full_optimized
export STRICT_VERDICT="$PROBE_ROOT/v16_8_9_strict_verdict.json"

# 保留 paired comparability：只复用旧 cache 的 scenario IDs，不复用旧 COWP labels。
export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE="$SOURCE_DATA_ROOT/tensor_cache_train"
export OLD_SCENESET_VAL_CACHE="$SOURCE_DATA_ROOT/tensor_cache_val"

export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8

# 当前 COWP core train 不需要预先给全 train cache 附 Waymax outcome；先省掉这个极贵阶段。
export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0

bash PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
```

这个脚本会自己再次做 raw preflight、labels model-support、per-scene sdc_paths、cache integrity、cache model-support 和 full-val gate。任一失败都会停止。

### Step 6：full rebuild 完成后再做耗时 diagnostics/Waymax evaluation

不要把图片 diagnostics 放在 critical path。先训练前确认 full cache gate 文件全部 pass，再单独运行 diagnostics。

如果训练命令显式启用了 `--with-waymax-outcome-labels`，才需要构建对应 Waymax outcome cache；否则当前 loss 对缺失 outcome 分支按未启用处理，没必要为了“数据看起来更全”先花大量 GPU 时间做 full-train replay。

### Step 7：算法和超参冻结后，建立真正 publication holdout

这里不要再用于调阈值。

```bash
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export FINAL_HOLDOUT=/data0/senzeyu2/dataset/COWP/frozen_publication_holdout_v16_8_9
mkdir -p "$FINAL_HOLDOUT"

python -m cowp.scripts.67_make_hash_holdout_manifest \
  --index-jsonl "$SOURCE_DATA_ROOT/index_val.jsonl" \
  --exclude "$SOURCE_DATA_ROOT/tensor_cache_val" \
  --exclude "$PROBE_ROOT/probe_union_scene_ids.txt" \
  --count 1000 \
  --seed cowp-v16-8-9-publication-holdout-2026 \
  --output-ids "$FINAL_HOLDOUT/scene_ids.txt" \
  --output-manifest "$FINAL_HOLDOUT/holdout_manifest.json"
```

然后用 validation Scenario/tf.Example、`--allow-scenario-ids "$FINAL_HOLDOUT/scene_ids.txt"` 构建该 holdout 的 fresh labels/tensor cache。**只有最终模型和所有阈值冻结后才打开结果。**

---

## 7. 对你原始 `构建数据集指令.txt` 的几个具体修正

1. Scenario→label、tf.Example→tensor-cache 这条总体路线是对的。
2. 原 `--require-waymax-ready` 对 1.3.1 full-route 评测不够严格；本次增加 `--require-sdc-paths`。
3. 原 attach 输出目录和 verify 目录名字不一致：例如 attach 写 `tensor_cache_train_waymax`，verify 却读 `tensor_cache_train_waymax_bal12_safety`；val 也有同类不一致。这是执行脚本级风险，新的 full builder 统一了路径。
4. `--limit 22000/5000` 是规模限制，不是随机抽样保证。当前 full rebuild 为了严格 paired comparison 继续复用旧 scene IDs，但最终论文评测应使用独立 frozen hash holdout。
5. `metric-set safety` 不等价于“Waymax 所有 route metrics 都被验证”；若论文声称 wrong-way/route-progression，需要用 1.3.1 sdc_paths 并在最终 online evaluation 中实际报告对应 metric。

---

## 8. 什么时候可以确信“不白费 full rebuild 时间”

在启动 full rebuild 前，我建议把下面条件当成硬 checklist：

- [ ] raw WOMD preflight `pass=true`；
- [ ] smoke screen pass；
- [ ] smoke 无 audit/transport contract mismatch；
- [ ] 本次优化与前一 repaired smoke 的公共 label tensors semantic-equivalent；
- [ ] strict manifest = 400 hard + 800 random + 0 overlap + 1200 union；
- [ ] strict training supervision pass；
- [ ] strict full model-support audit pass；
- [ ] strict proposal gates：AnyValid ≥ 0.99、AnyNCF ≥ 0.40、false-safe floor ≤ 0.55、PBTR floor ≤ 0.45、hard recovery ≥ 0.20；
- [ ] `v16_8_9_strict_verdict.json` 明确 `recommend_full_rebuild=true`；
- [ ] strict verdict fingerprint 与当前代码完全一致。

任何一项不满足，都不要通过降低 threshold、删 rare scenes 或缩 response/root search budget 来“让数据过”。应该先判断是：动作空间 ceiling、标签 contract、supervision support、raw WOMD version/split，还是统计波动。

---

## 9. 当前无法在离线审计环境替你完成的两件事

1. 我没有你服务器上的 WOMD 原始 TFRecords，因此不能替你实际证明 `/data0/.../waymo_open_dataset_motion_v_1_3_1` 每个 shard 都满足 1.3.1；我已把可执行的 preflight 和逐场景 enforcement 放进代码，必须在你的机器上跑。
2. 我无法在没有真实 96/1200 WOMD label build 的情况下给出本补丁的实际加速百分比。能确定的是优化没有故意减少算法搜索空间；实际 wall-time 改善必须以新的 `fresh_profile_summary.json` 为准。

---

## 10. 本次交付验证

- 全量 Python tests：**182 passed, 3 skipped**；
- 新/改 Python scripts：`py_compile` pass；
- smoke / strict / full shell wrappers：`bash -n` pass；
- patch 已在一份原始 `COWP.zip` 解包副本上执行 `patch -p1 --dry-run` + 实际 apply 验证；
- 本地 synthetic trajectory-bank equivalence：相同输入下旧路径/复用路径输出一致；
- 上传 strict artifact 的结构化审计另附 `uploaded_strict_probe_artifact_audit.json`。

