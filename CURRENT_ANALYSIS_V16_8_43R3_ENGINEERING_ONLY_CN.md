# V16.8.43 当前结果审计与 V16.8.43R3 工程修复

## 结论先行

当前上传的 `v16_8_43r2_blocker_conditioned_runtime_fidelity_repair.zip` **可以可靠解释 engineering fidelity/runtime，但不能进行 V16.8.43 科学算法归因**。原因不是结果损坏，而是结果包只包含 `equivalence16` 和 `profile8`，没有冻结的 `counterfactual48` Stage-1 rollout/analyzer。因此按“可靠性不够则只修代码”的规则，本轮不发布 V43 GO/STOP，不改变 dominant scientific bottleneck，不关闭新的科学算法族，也不设计 V44。

## 1. 当前结果可靠性

独立核查：

- equivalence16 两 shard 为 8+8、无重叠、merged 精确覆盖 16 scenes；对 V16.8.29 reference 为 **16 scenes / 1120 fields / 0 mismatch**。
- profile8 两 shard 为 4+4、无重叠、merged 精确覆盖冻结 8-scene manifest；标准指标可从逐场景 row 复算。
- online mechanism GT 标志为 false。
- V43R2 与修复前 V43 在完全相同 profile8 上通过行为 fidelity verifier：标准指标和应保持的行为 diagnostics **0 mismatch**。
- pre-R2 V43 wall = **26,242 s**；R2 wall = **20,649 s**，speedup = **1.27086x**，wall reduction = **21.31%**。
- R2 两 shard selection mean = **63.94 / 57.74 s per policy step**，selection fraction = **99.271% / 99.193%**。因此当前主要工程瓶颈仍在 selection 内部的重复 RC-IARE work，而非 model forward。

完整机器审计：`V16_8_43R2_RESULT_RELIABILITY_AND_STAGE1_AVAILABILITY_AUDIT.json`。

## 2. 为什么不能判 V43 GO/STOP

V43 冻结的 Stage-1 是完整 48 scenes 上的六项 conjunction：

```text
old RVR rescues retained >= 5/10
old RVR induced avoided  >= 7/9
net COWP collisions removed >= 3
Kinematics net regression <= 1 scene
paired mean EP delta >= -0.05
nonzero action-changing intervention
```

profile8 既不是这 48 scenes，也不覆盖全部 10+9 historical counterexamples。因此任何从 profile8 推导 V43 GO/STOP、promotion、dominant bottleneck 更新或 V44 机制，都会违反预注册逻辑。

## 3. compact-5k 数据性质复核

我重新展开数据包，而不是沿用历史结论。train/val/heldout_test 分别为 5000/1000/1200 scenes；audit-relevant pair rate 为 0.429696/0.428630/0.429492；protected-priority PRIO-root coverage 为 0.994533/0.993632/0.994651；rootless rate 与 `<2 low-burden roots` rate 三个 split 均为 0；mechanism-unauditable rate 为 4.07%/4.34%/4.46%。这些量跨 split 稳定，目前没有支持“重建 compact-5k”作为 P0 的证据。

值得持续监控但不能误判为立即重构理由的性质：critical-agent count max=6，冲突区域 selected-cap saturation 约 95.4%-95.8%；`PRIORITY_SMOOTH_YIELD` 与 `TERMINAL` proposal acceptance 明显低于多数 source。它们可能构成长远 support ceiling，但 V42→V43 当前科学问题是 online recovery support indexing，不能把二者混为同一个问题。

完整复核：`V16_8_24_COMPACT5K_DATASET_CHARACTERIZATION_RECHECK.json`。

## 4. 论文/历史证据链冻结到哪里

截至 V42，科学链条仍是：false-safe/safety-by-coercion 作为 hard feasibility defect；Natural roots → same-root RCOT → BCOT → protected-priority certificate → set preservation → hard-first inference；physical recovery 侧已经从 fixed-path support 逐步走到 interaction-aware root-conditioned recourse。V42 policy 因 frozen historical-rescue gate 失败而不能 promotion，但 RC-IARE interaction support 是 strong mechanism-positive；V43 的唯一科学变量是 exact-blocker-conditioned late-bound support indexing。

因此本轮不能用“没有 CF48”的空白去覆盖 V42 的预注册分叉。V43 的科学问题仍然是：**真正阻塞 recovery tube 的 exact blocker 是否因为被 scene-level fixed critical support 排除而被错误判 unsupported；late-bound natural-root support 能否修复这一 coverage mismatch，同时保留原 hard response semantics。**

## 5. V43R3 工程修复

科学方法仍是 V16.8.43 BC-IARE。R3 只做 semantics-preserving work reuse：

```text
第一次 V42/RC-IARE pass
  ├─ 保存原 critical agents 已准备好的 natural-root/response support
  ├─ 保存 immutable ego hypothesis workspace
  └─ 对 solely-unsupported hypotheses 保存 successor/shift replay record

V43 late-bound exact blockers
  ├─ 只为新增 exact blockers 准备 response support
  ├─ 与第一 pass 原 support 合并
  └─ 只对 repairable unsupported hypotheses 重算 interaction certificate
```

不重复 semantic representative/schedule/controller projection/successor-shift construction；不修改 hard membership semantics、logical reject counters、selector 或 execution override。

focused regression **109/109 passed**。

## 6. 更快、但仍有可靠归因力的实验协议

### Engineering fidelity

默认只跑 `profile4_parallel2` + `verify_profile4_r3`。这 4 scenes 用来检查 R3 与 R2 行为完全一致，不作为算法 evidence。只有需要服务器端 wall-clock 数字时再跑 profile8。

### Scientific Stage-1: 19 → 29，无重复

六项 Gate 中前两项只由历史 10 个 RVR rescue + 9 个 RVR-induced 场景决定。因此先跑冻结 `mandatory gate19`：

```text
retain >=5/10 AND avoid >=7/9 ?
    no  -> full Gate 数学上不可能通过，立即停止
    yes -> 跑与其完全不重叠的 remaining29
           -> stitch 19+29 = 原冻结 counterfactual48
           -> 运行原六项 analyzer
```

失败分支由 48 scenes 降到 19 scenes，减少 **60.4%** Stage-1 closed-loop rollouts；通过分支不会重复跑 19 scenes，因此最终仍恰好是原 48 个完整闭环结果。

### 为什么不能再缩 rollout horizon / 用 open-loop 代替

历史 failure 存在很晚才发生的闭环事件，且整个 V29→V41 证据链反复显示 successor state、mode switching、replanning 和 long-horizon mismatch 会改变归因。因此 promotion Gate 的 80-step closed-loop 不能在当前阶段缩短。open-loop/analytic probe 可用于单元测试和性能验证，不能替代 Stage-1 physical outcome。

## 7. 下一步指令

```bash
cd COWP_V16_8_43R3_RUNTIME_WORK_REUSE
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh make_ids

# 工程等价性：推荐；不参与 promotion
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh profile4_parallel2
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh verify_profile4_r3

# Stage-1 必经反例 gate
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh gate19_parallel2
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh analyze_gate19
```

若 `mandatory_collision_counterexample_gate.pass != true`：停止并回传 gate19 文件，不跑 remaining29。

若为 true：

```bash
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh remaining29_parallel2
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh stitch_counterfactual48
bash NEXT_RUN_COMMANDS_V16_8_43R3_RUNTIME_WORK_REUSE_CN.sh analyze_counterfactual48
```

只有完整 analyzer 中原冻结 `preregistered_gate.blocker_conditioned_interaction_aware_reachable_response_envelope.pass == true`，才允许 fresh37。

## 8. 本轮不关闭新的科学算法族

因为没有可归因的 V43 Stage-1，本轮唯一关闭的是**工程实现上的重复 full-pass work pattern**，不是 scientific family。历史已经归档的 RVR/BHOV/THOP/first-action interval 等继续保持归档；V43 BC-IARE 的 science verdict 暂停，等待 gate19/CF48。

同理，本轮不设计 V44。若 V43 reliable CF48 最终 STOP，再严格按 V42 的预注册分叉决定是 Root-Conditioned Control-Reachable Responder Support、multi-step invariance/interaction uncertainty，还是其他新层；不能提前选分支。
