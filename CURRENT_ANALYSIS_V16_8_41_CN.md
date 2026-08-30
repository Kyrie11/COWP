# V16.8.40 可靠性审计与 V16.8.41 Shift-Closure Semantic Fidelity Repair

## 0. 决策摘要

### 可靠性结论

V16.8.40 的数据、分片、汇总、运行协议和 analyzer 都是完整且内部一致的；但其核心 hard shift-closure certificate 存在一处与冻结 V39 语义不一致的实现偏差。

独立审计结果：

```text
hard checks              = 243
passed                   = 242
failed                   = 1
code integrity           = PASS
data/statistical integrity = PASS
mechanism fidelity       = FAIL
overall reliability      = FAIL
algorithm attribution    = NOT ALLOWED
required decision        = REPAIR ONLY AND RERUN
```

唯一 hard failure：

```text
V39 LOWER_ALL / UPPER_ALL shifted witness:
    appended terminal schedule = -1 / +1

uploaded V40 interval-completion shifted witness:
    appended terminal schedule = 0 for every policy
```

V40 声称复用“不变的 V39 one-step shift closure”，但实现没有保持 all-horizon envelope 的末端控制语义。由于 shift closure 是候选能否进入 hard set 的必要条件，该偏差可以制造 interval support false negative。

因此，本轮不能执行以下工作：

- 不能归因 V40 interval mechanism 成功或失败；
- 不能依据 V40 结果收紧 dominant bottleneck；
- 不能决定下一条研究机制分支；
- 不能 promotion V40；
- 不能把 V40 的 provisional Gate failure 写成算法负结论；
- 不能运行 fresh37。

### V16.8.41 的角色

V16.8.41 不是新算法版本，而是：

# **Shift-Closure Semantic Fidelity Repair，SC-FAVI-R**

它只修复上述 hard-certificate semantic mismatch，并要求在完全相同的 frozen manifests、checkpoint、方法名和六项预注册 Gate 下重新运行 `equivalence16 + counterfactual48`。

---

# 1. 审计输入与 provenance

## 1.1 代码包

上传 V40 代码 ZIP SHA256：

```text
b7f4191b7b10853f951e1924d6cb23d814c6246361735b829856e0d3e0c6b6a5
```

它与上一轮交付的冻结 V40 ZIP hash 完全一致，因此结果归因审计针对的是预期版本，不存在“用户运行了另一个未识别代码树”的问题。

V40 release hash 文件中的关键源代码、脚本、tests、manifests 和文档均与上传代码树一致。

## 1.2 结果包

结果包包含：

- `equivalence16` 两个 shards、merged 和 equivalence report；
- `counterfactual48` 两个 shards、merged 和 analyzer output；
- exact200 / equivalence16 / counterfactual48 / fresh37 四组冻结 manifest；
- wall-time 文件。

没有 fresh37 / exact200 新 rollout，符合 provisional Stage-1 fail 后停止的执行纪律。

## 1.3 冻结 manifest

| Manifest | Count | Logical SHA256 |
|---|---:|---|
| exact200 | 200 | `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f` |
| equivalence16 | 16 | `81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760` |
| counterfactual48 | 48 | `ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0` |
| fresh37 | 37 | `ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481` |

所有 count、ID 唯一性和 logical SHA256 均通过。

---

# 2. 通过的可靠性检查

## 2.1 Shard 与 manifest

`counterfactual48`：

```text
shard0 = 24 unique scenes
shard1 = 24 unique scenes
intersection = 0
union = exact frozen 48-ID manifest
```

`equivalence16`：

```text
shard0 = 8
shard1 = 8
intersection = 0
union = exact frozen 16-ID manifest
```

## 2.2 Shard-to-merged 与指标重算

48 个 merged scenario rows 可从两个 shards 精确重建。以下标准结果均可从逐场景 rows 零误差复算：

- CR；
- Collision；
- Offroad；
- Kinematics；
- EP；
- fallback rate；
- method-specific tube / interval counts。

V16.8.28 的 execution invariant 对所有场景继续成立：

```text
emergency_action_step_rate
=
zero_valid_candidate_step_rate
=
no_valid_step_rate
```

## 2.3 Common-path equivalence

```text
scenarios checked = 16
fields checked    = 1120
mismatches        = 0
passed            = true
```

因此 accepted/conventional/no-valid common path 没有被 V40 意外改变。

## 2.4 运行协议

两个 counterfactual shards 使用一致的：

- method；
- checkpoint logical path；
- 80-step rollout；
- Waymax standard metrics；
- validation split；
- priority NCF gate；
- witness threshold；
- BCOT budget；
- JIT / reuse / prefilter settings；
- logged-replay surrounding-agent protocol；
- `mechanism_ground_truth_available_online=false`。

结果包没有 checkpoint bytes，因此可以验证 logical provenance 与 shard consistency，但不能重算 `.pt` 内容 hash。这不是当前 blocker。

## 2.5 Analyzer 独立重放

使用 pristine 上传 V40 源码重新执行 V40 analyzer，独立输出与上传 analyzer JSON 的 byte-level SHA256 完全一致：

```text
026727655557dd33a3e88d0eb9f0eac14ac55ae5970c49b0711c0b29d06ee7ca
```

所以 provisional summary 不是 analyzer 随机错误或人工抄录错误。

---

# 3. 唯一但决定性的 hard blocker

## 3.1 冻结 V39 shift semantics

V39 对 schedule 左移一拍时执行：

```python
shifted_schedule[:-1] = schedule[1:]

if policy_id in {-1, +1}:
    shifted_schedule[-1] = sign(policy_id)
```

含义：

- finite event-release policy 已经回到 nominal，因此新增 terminal edge 为 `0`；
- `LOWER_ALL` 是完整 horizon 均取 lower reachable endpoint 的 policy，左移后新增 terminal edge 仍应为 `-1`；
- `UPPER_ALL` 同理应为 `+1`。

这不是 preference，而是被 shift closure 审计的 control-policy identity。

## 3.2 上传 V40 的偏差

上传 V40 interval-completion branch 使用：

```python
shifted_schedule = zeros(H)
shifted_schedule[:-1] = schedule[1:]
```

没有根据 basis 的 `policy_id` 恢复 all-horizon endpoint，因此所有 basis 的新增 terminal edge 都变成 nominal。

这与 V40 changelog 中的两项声明不一致：

```text
unchanged V39 future schedule
unchanged one-step shift closure
```

## 3.3 为什么它阻断算法归因

V40 的 interval candidate 只有同时通过：

```text
current full physical certificate
AND
successor shifted full physical certificate
```

才可能成为可执行 intervention。

末端 schedule 会改变 successor 上最后一条 controller-projected edge，进而可能改变：

- terminal acceleration；
- realized position / speed；
- causal-CV collision margin；
- Waymax-aligned inverse kinematics；
- full shifted certificate boolean。

所以它不是 diagnostics-only 差异，而是 hard set membership 差异。

V40 的 interval support 本身极稀疏：

```text
interval hypotheses evaluated = 1,551,139
full physically safe          = 98
shift closed                  = 19
new certified first actions   = 5
selected new first actions    = 3 steps
```

在如此稀疏的边界上，即使只造成少量 false negatives，也可能改变唯一失败的 rescue-recall Gate。因此不能把当前 3/10 recall 归因于“区间机制本身无效”。

---

# 4. V40 原始数值只能作为 provisional non-attributable record

以下结果可以证明日志完整，但不能证明算法机制：

| Method | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 1/48 | 6/48 | 1.002512 |
| V39 | 30/48 | 1/48 | 6/48 | 0.991434 |
| uploaded V40 | 30/48 | 1/48 | 6/48 | 0.991657 |

uploaded V40 的 provisional Gate：

| Frozen condition | Required | Observed | Raw status |
|---|---:|---:|---|
| old RVR rescues retained | >=5/10 | 3/10 | FAIL |
| old RVR induced avoided | >=7/9 | 9/9 | PASS |
| net COWP collision removed | >=3 | 4 | PASS |
| Kinematics net regression | <=1 | 0 | PASS |
| paired mean EP delta | >=-0.05 | about -0.010855 | PASS |
| action-changing intervention | >0 | 90 steps | PASS |

形式上为 `5/6`，但这个 Gate **不能用于算法 verdict**，因为其输入来自未通过 mechanism-fidelity audit 的实现。

同样不能把以下观察写成结论：

- “first-action interval 没有扩大 support”；
- “只选 3 步所以机制无效”；
- “下一步应立即进入 interaction-aware reachable envelope”；
- “V39 已经是当前最佳 endpoint”。

这些判断都必须等待 repaired V41 rerun。

---

# 5. V16.8.41 修复设计

## 5.1 单一 shared helper

新增：

```python
_shift_longitudinal_envelope_schedule_np(schedule, policy_id)
```

规则：

```text
shift schedule left by one edge

policy_id == -1 (LOWER_ALL):
    append -1

policy_id == +1 (UPPER_ALL):
    append +1

all finite event-release / nominal policies:
    append 0
```

## 5.2 两个 constructor 统一调用

以下两个路径都必须调用同一个 helper：

```text
_construct_conflict_window_control_reachable_tube_np       # V39 nested hard set
_construct_shift_closed_first_action_viability_interval_np # V40 interval completion
```

这样不会再出现 baseline constructor 与 interval constructor 各自维护一份不同 terminal semantics。

## 5.3 不变项

V41 不改变：

- compact-5k dataset / labels / splits；
- checkpoint / training；
- Natural roots；
- same-root RCOT；
- BCOT；
- protected-priority non-coercive certificate；
- certificate-compatible set preservation；
- outcome head/settings；
- 8 s conventional-safety contract；
- proposal geometry / semantic macros；
- current/shifted physical certificate；
- collision model；
- roadgraph screen；
- Waymax-aligned kinematics adapter；
- acceleration/deceleration/jerk/yaw/controller limits；
- interval seeds / secant / quadratic proposals；
- hard-set ordering；
- actual execution override；
- method alias。

因此 V41 是 reliability repair，不是一次新的 scientific hypothesis test。

## 5.4 无未来信息泄漏

修复只读取：

- 当前 schedule；
- 当前 policy identity。

它不读取：

- Waymax logged future；
- future outcome；
- online ground truth；
- 未执行策略的真实反事实结果。

---

# 6. 新增回归保护

新增 dedicated tests：

1. `LOWER_ALL/UPPER_ALL` 左移后保持 endpoint；
2. finite event-release 左移后末端回 nominal；
3. V39 与 V40 constructors 都调用 shared helper；
4. shared helper 对所有 policy IDs、多个 horizons、随机 schedules 与 literal frozen V39 rule 精确一致；
5. helper 的输出始终属于 `{-1, 0, +1}`，不可能扩大 controller limits。

同时把 fail-closed launcher 修成真实的 exit code 4，而不是依赖 Python 默认的 exit code 1。

---

# 7. 代码验证

| Check | Result |
|---|---:|
| V41 dedicated repair tests | 4/4 PASS |
| V16.8.25→41 focused semantic/integrity suite | 86/86 PASS |
| Python compile | PASS |
| launcher `bash -n` | PASS |
| exact200 manifest | PASS |
| equivalence16 manifest | PASS |
| counterfactual48 manifest | PASS |
| fresh37 manifest | PASS |
| analyzer smoke on 48 rows | PASS |
| analyzer Gate field | PASS |
| failed-Gate fresh37 attempt | exit code 4 |
| rollout artifact created after failed Gate | 0 |
| conventional-safety bypass | 0 occurrences |
| future-GT use introduced | none |

完整 repository `pytest -q` 仍在 collection 阶段遇到历史测试：

```text
tests/test_v16_8_29_recovery_viability.py
```

其导入已经不存在的：

```text
_recovery_bridge_viability_mask
```

相同错误在 pristine 上传 V40 tree 中独立复现，因此不是 V41 regression。没有为了制造 full-suite green 而恢复已归档 API。

本地 artifact 环境没有用户服务器上的 Waymax dataset/index/checkpoint/GPU runtime，因此没有生成或伪造 V41 closed-loop outcome。

---

# 8. 下一步实验：只能做 repaired rerun

```bash
cd COWP_v16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh make_ids

# 仅当 index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh analyze_counterfactual48
```

到这里停止。

只有 repaired analyzer 中：

```text
preregistered_gate.shift_closure_semantic_fidelity_repair.pass == true
```

才允许：

```bash
PROMOTED_METHODS=cowp_shift_closed_first_action_viability_interval \
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh analyze_fresh37
```

fresh37 再通过，才允许 historical exact200 development confirmation。

---

# 9. repaired result 回传后才允许的判断

下一轮必须再次先做完整 reliability audit，然后才有两种有效分叉：

## A. V41 可靠并通过 frozen Gate

才可以讨论：

- first-action interval completion 是否真正提高 historical rescue recall；
- new-first-action support/selection 是否非零；
- gain 是否可归因于 repaired interval mechanism；
- 是否进入 fresh37。

## B. V41 可靠但仍失败 frozen Gate

才可以正式 archive SC-FAVI，并依据修复后的：

- interval attempts；
- full-safe / shift-closed support；
- interval-only parent support；
- selected new first actions；
- rescued/induced scene decomposition；

决定下一版是 constrained reachable-set construction、interaction-aware response envelope，还是另一种更基本的 proposal-support问题。

在 repaired outcome 到来以前，不预注册新的 V42/V41 scientific mechanism，避免把 code repair 与 algorithm change 混在一次 rollout 中。

---

# 10. CCF-A 主线的处理

本轮不改变论文主线，也不把 code repair 包装成 novelty。

继续冻结的 social feasibility axis：

```text
natural roots
→ same-root RCOT
→ BCOT
→ protected-priority non-coercive certificate
→ certificate-compatible set preservation
```

physical axis 的 scientific verdict 暂停，直到 repaired V41 结果通过可靠性审计。

统一论文原则仍是：

```text
Safety must not be obtained through critical option-set collapse.
```

但 V41 的 shared helper 只是保证实验忠实于已经定义的 hard certificate，不是论文贡献。

---

# 11. 本轮明确禁止

1. 不用 uploaded V40 的 5/6 provisional Gate 做机制归因。
2. 不把 rescue threshold 从 5/10 改成 3/10。
3. 不在 repair rerun 中增加新 interval seeds、action grid、switch times 或 schedules。
4. 不放宽 full physical certificate 或 shift closure。
5. 不缩短 8 s conventional horizon。
6. 不修改 common controller limits。
7. 不增加 collision/progress/risk/kinematics scalar weights。
8. 不修改 RCOT、BCOT、social certificate 或 compact-5k。
9. 不同时加入 accepted-path kinematics repair。
10. 不在 repaired analytic target 得到结果前训练 viability head。
11. 不越过 Stage-1 直接运行 fresh37。
12. 不把本轮 repair 命名为新的 CCF-A algorithm contribution。

完整机器审计见：

```text
V16_8_40_RESULT_RELIABILITY_AUDIT_INDEPENDENT.json
```

原始暂定数值的隔离记录见：

```text
V16_8_40_PROVISIONAL_NON_ATTRIBUTABLE_RESULT_SUMMARY.json
```
