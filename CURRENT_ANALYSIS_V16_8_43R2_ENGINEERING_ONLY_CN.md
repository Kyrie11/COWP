# V16.8.43 上传结果审计与 V16.8.43R2 工程修复

## 0. 结论先行

本轮**不能进入 V16.8.43 算法归因**，也不能根据当前上传结果判断 BC-IARE 成功/失败、promotion、dominant bottleneck 是否变化，或设计新的科学机制。

原因有两层：

1. 上传结果包没有上一轮预注册所要求的 `counterfactual48` shard / merged / analyzer；只有 `equivalence16 + profile8`。因此六项 Stage-1 GO 条件没有可判定的数据。
2. `profile8` 暴露出足以阻断 Stage-1 实际执行的性能问题：8 个 profile 场景两卡并行 wall-clock 为 **26,242 s = 7.289 h**；两个 shard 的 selection mean 分别为 **81.499 s/step** 和 **76.115 s/step**，selection 占 policy total **99.511% / 99.470%**。这不是 model forward 或 candidate construction 的瓶颈，而是 V43 interaction selection 本身。

按照项目已经使用过的可靠性纪律：**结果不具备预注册 Stage-1 完整性时，只做工程修复，不做算法解释。**

因此本轮交付定义为：

> **V16.8.43R2 — BC-IARE Runtime-Fidelity Repair**
>
> 工程版本，不是 V16.8.44，不引入新的论文算法，不改变任何 frozen Gate。

---

## 1. 我对论文与当前代码主线的理解

论文真正的核心不是增加一个 social cost，而是把传统 ego-centric collision-free planning 会漏掉的 **false-safe / safety-by-coercion** 定义为 hard feasibility defect：如果 ego 的“安全”依赖 protected road user hard braking、abrupt yielding、priority surrender 或 gap surrender，那么这个 ego plan 不应因为 collision-free 就被当作可行。

当前主链条是：

```text
natural alternatives
→ stable natural roots
→ same-root RCOT
→ BCOT
→ protected-priority hard non-coercive certificate
→ certificate-compatible set preservation
→ hard-first selection
→ explicit uncertified fallback
```

其中 proposal/support sufficiency 与 certificate/selector quality 必须分离审计。当前 compact-5k 数据边界稳定，用户也明确要求本阶段不重建数据；本轮没有发现需要推翻这个边界的新证据。

V42 已经把 physical-interactive recovery 的科学对象推进到 root-conditioned interactive hard support；V43 的唯一科学变量是：**recovery support indexing 从 scene-level fixed critical support 改为 exact-blocker-conditioned late binding，同时严禁扩大 social NCF protected/critical set。**

本轮 R2 不改变这个科学对象。

---

## 2. 上传结果包可靠性审计

### 2.1 通过的部分

当前结果包中可验证的内容是内部一致的：

- `equivalence16_cowp_vs_v16_8_29.json`：**16 scenes / 1120 fields / 0 mismatch / passed=true**；
- profile8 manifest：8 IDs、唯一、logical SHA256 正确；
- profile8 两个 shard：4+4、无重叠、并集精确等于 manifest；
- merged profile8 覆盖同一 8 IDs；
- CR / Collision / Offroad / Kinematics / EP 可由 8 个 scenario rows 零误差重算；
- 方法 ID、checkpoint logical path、online mechanism-GT flag 等协议一致；
- 上传 V43 code tree 的 release-file hashes 与其 release manifest 一致。

因此当前不是“已有结果损坏”或“common path 被改坏”。

### 2.2 阻断项

结果包**没有**：

```text
counterfactual48_v43_*_s0.json
counterfactual48_v43_*_s1.json
counterfactual48_v43_*_merged.json
counterfactual48_v43_*_analysis.json
```

上一轮冻结的 Stage-1 是 conjunction Gate：

```text
old RVR rescues retained >= 5/10
old RVR induced avoided  >= 7/9
net COWP collisions removed >= 3
Kinematics net regression <= 1 scene
paired mean EP delta >= -0.05
nonzero action-changing intervention
```

由于 Stage-1 数据不存在，**V43 的 algorithm attribution = NOT ALLOWED**。

这和“V43 Gate fail”是两件完全不同的事：目前不能说 fail，也不能说 pass。

完整机器审计：`V16_8_43_RESULT_RELIABILITY_AND_ENGINEERING_BLOCKER_AUDIT.json`。

---

## 3. profile8 暴露的工程阻断

### 3.1 Runtime

profile8：

```text
parallel wall time = 26,242 s = 7.289 h

shard 0:
  selection mean        = 81.499 s/step
  policy total mean     = 81.900 s/step
  selection fraction    = 99.511%
  candidate build mean  = 0.364 s/step
  model forward mean    = 0.0268 s/step

shard 1:
  selection mean        = 76.115 s/step
  policy total mean     = 76.521 s/step
  selection fraction    = 99.470%
  candidate build mean  = 0.367 s/step
  model forward mean    = 0.0278 s/step
```

所以当前再优化 Torch forward、H2D 或普通 candidate generation 几乎没有意义。

### 3.2 profile8 是极端 zero-conventional stress operating regime

8 个场景的 `zero_conventional_candidate_step_rate` 都是 1.0。描述性宏平均显示：

```text
blocker-query attempt rate                 ≈ 0.950
old V43 queried candidate agents/attempt   ≈ 16.740
old V43 ready query agents/attempt         ≈ 13.134
base V42 interaction hypotheses/attempt    ≈ 199.466
base unsupported rejects/attempt           ≈ 117.223
base root-unrecoverable rejects/attempt    ≈ 68.170
expanded V43 hypotheses/attempt            ≈ 210.693
expanded root-unrecoverable rejects        ≈ 167.882
expanded environment cache hits/attempt    ≈ 10,154.8
```

这些数字不能用于 V43 outcome promotion，但足以定位工程工作量。

---

## 4. 代码根因：实现工作域大于科学变量的真实依赖域

V43 的科学命题是：

> **谁是当前 recovery tube 的 exact collision blocker，谁才需要 late-bound natural-response support。**

但 released V43 的 `__call__` 实际先从 frozen collision context 中取出几乎所有 model-visible、valid、non-SDC、non-original-critical actors，然后：

1. 对整个 query list 运行 NaturalDecoder；
2. exact V42 返回 empty 后，将这些 query roots 全部拼入 support domain；
3. 对完整 V42 interaction hypothesis family 再跑一遍 hard certificate。

这在 admission 语义上仍然由 exact blocker 决定，但计算域远大于真正可能改变结果的域。

### 关键逻辑不变量

一个 V42 hypothesis 只有在原始失败原因是：

```text
unsupported_collision_blocker
```

时，新增 blocker natural support 才可能改变它的可行性。

以下原始失败类型不会仅因为“给另一个 actor 多解码 natural roots”而翻转：

```text
no_collision_blocker
residual_physical_certificate_failed
retained_root_has_no_ego_safe_response
retained_root_has_no_environment_safe_response
invalid_environment_actor_prediction
no_jointly_compatible_response_envelope
```

因此第二遍重放所有 hypotheses 是纯冗余；对所有 nearby query actors 提前 NaturalDecoder 也是纯冗余。

---

## 5. V16.8.43R2 工程修复

### 5.1 冻结 exact V42 first pass

V42 constructor 增加 internal-only trace：

```text
unsupported_hypothesis_indices
unsupported_blocker_union
```

它只记录原来已经计算出的 reject 信息，不改变任何 hard predicate。

### 5.2 exact-blocker deferred decode

V43R2 不再在 `__call__` 里提前 decode 全部 late-bound candidate agents。

流程变为：

```text
exact V42 first pass
→ 得到 unsupported exact blocker union
→ 与 frozen model-visible query pool 取交
→ 只对这些 exact blockers 调用 frozen NaturalDecoder
```

NaturalDecoder 仍使用 root-scene latent；没有 candidate-conditioned natural baseline 污染。

### 5.3 只 replay repairable hypotheses

第二遍 RC-IARE 只评估第一遍记录的：

```text
unsupported_collision_blocker hypotheses
```

并跳过 nested V39，因为同一 policy step 的 exact V42 first pass 已经证明 V39 hard set empty。

### 5.4 保留所有原 hard semantics

R2 没有修改：

```text
p_min / probability floor / mass coverage / min roots / dedup
adaptive beta
same-root response bank
roadgraph
Waymax kinematics
responder-environment bidirectional safety
multi-blocker CSP
V39 tube / conflict-window schedules / shift closure
8 s conventional horizon
controller limits
selector ordering
execution override
dataset / checkpoint / loss / proposal geometry
social protected/critical set
```

所以这是 **work elimination**，不是放松 certificate 或新的 policy。

### 5.5 新增 runtime attribution diagnostics

```text
blocker_conditioned_query_candidate_agents_before_exact_filter
blocker_conditioned_query_exact_blocker_agent_count
blocker_conditioned_query_replayed_hypothesis_count
```

下一次 profile8 会直接告诉我们：旧的 16–20 个 query agents 实际被过滤到多少 exact blockers，以及约 200 个 hypotheses 中有多少真正需要第二遍重放。

---

## 6. 代码验证

当前本地不具备服务器 Waymax dataset/checkpoint/GPU runtime，因此没有伪造 end-to-end speedup 或 Stage-1 outcome。

已完成：

```text
V42 + V43 dedicated tests                    21/21 PASS
V16.8.25 → V16.8.43 focused sanity          107/107 PASS
Python compile                               PASS
new R2 launcher bash syntax                  PASS
all frozen manifest logical SHA256           PASS
exact-blocker filter regression               PASS
deferred NaturalDecoder exact-blocker test   PASS
no-unsupported-failure second-pass skip       PASS
```

同时新增 `97_verify_v43r2_runtime_fidelity.py`：服务器 rerun profile8 后，它会比较原 V43 与 R2 的同一 8-scene behavioral evidence，并把 query/cache work counters 排除在 equivalence 判据之外。

这一步是为了防止“为了快而悄悄改变动作/结果”。

---

## 7. 本轮明确不做的算法判断

因为缺少可靠 Stage-1，本轮**不回答**：

- BC-IARE 是否通过 5/10 rescue retention；
- late-bound indexing 是否应 promotion；
- unsupported blocker 是否已经从 P0 下降；
- root-unrecoverable 是否已成为新 dominant bottleneck；
- 是否应进入 Root-Conditioned Control-Reachable Responder Support；
- `fccd...` 是否应该触发 multi-step invariance；
- V43 是否提升 CCF-A 投稿状态。

这些结论都必须等 R2 在**完全不变的 V43 science + Gate**下重新完成 counterfactual48。

---

## 8. 下一步运行协议

使用：

```text
NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh
```

顺序：

```bash
cd COWP_V16_8_43R2_BLOCKER_CONDITIONED_RUNTIME_FIDELITY_REPAIR

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh make_ids

# 只有 index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh build_tfindex

# common path 必须继续等价
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh base_equivalence16_parallel2

# 先验证 engineering repair，不作为 promotion evidence
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh profile8_parallel2
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh verify_profile8_repair
```

只有 `runtime_profile8_v43r2_runtime_fidelity_equivalence.json` 中：

```text
pass == true
```

并且服务器 wall-clock 确实获得有意义改善，才执行：

```bash
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_43R2_RUNTIME_FIDELITY_REPAIR_CN.sh analyze_counterfactual48
```

到这里停止。

只有 analyzer：

```text
preregistered_gate.blocker_conditioned_interaction_aware_reachable_response_envelope.pass == true
```

才允许 fresh37。任何失败都不能改 Gate。

---

## 9. 防止无限内部优化：预定义停止目标

这不是根据不完整 V43 结果事后设阈值，而是给整个后续项目一个 **algorithm-freeze / external-baseline trigger**。

### Development promotion chain

```text
1. counterfactual48：通过现有六项 frozen Gate
2. fresh37：通过已有 no-net-harm Gate
   - no net Collision harm
   - no net CR harm
   - Offroad/Kinematics regression <= 1 scene
   - mean EP delta >= -0.03
   - intervention > 0
3. historical exact200：只做一次 development confirmation，不再用它调算法
```

### Internal stop target（建议冻结）

以历史 exact200 COWP 约：

```text
Collision 17.0% (34/200)
CR        19.5%
Offroad    3.0%
Kinematics 12.5%
EP         1.046
```

为开发参照，我建议将“停止内部机制迭代、转外部 baseline/final evaluation”的目标预定义为：

```text
Collision <= 14.0%   (<=28/200)
CR        <= 17.0%
Offroad   <= 3.5%
Kinematics<= 13.0%
EP        >= 1.00
```

同时必须没有明显 failure conversion/concentrated catastrophic regression。

一旦满足这组目标，就**停止在 exact200 上继续内部调参/改机制**；冻结算法，转入：

- 从未参与 V25→V43+ 机制选择的 final unseen scenes；
- >=3 independent seeds；
- paired scene bootstrap/CI；
- 外部 planning baselines；
- reactive-agent evaluation；
- human-audited false-safe stress set；
- late-bound natural-root calibration/readiness audit。

这组 stop target 是工程/科研管理边界，不是论文里预先宣称的 SOTA 门槛；最终 CCF-A 投稿状态应由外部 baseline 和真正 unseen evidence 决定。

---

## 10. 本轮结论压缩成一句话

> **当前 V16.8.43 上传结果没有 counterfactual48，因此不允许做算法归因；profile8 又显示 BC-IARE selection 已成为 99.5% 的严重工程瓶颈。V43R2 不改任何科学机制，而把实现严格收缩到“V42 中真正因 unsupported exact blocker 失败的 hypotheses + exact blockers”这一最小依赖域，先恢复可运行性与结果忠实性，再用原封不动的 Stage-1 Gate 判断 V43。**
