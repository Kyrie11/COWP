# COWP v16.8.28 当前分析：No-Valid Execution Integrity Repair

## 0. 本轮结论先行

**v16.8.27 当前 strict-Waymax 结果不能用于完整、可靠的算法归因。**

这次不是 exact-ID、分片、merge、metadata 或上一轮 conventional-safe 修复再次出错；这些部分我都核过，基本自洽。真正的问题在更底层的 **online candidate → Waymax action execution interface**：当某一步不存在任何 dynamically valid candidate 时，v16.8.27 会把 `selected=0` 当作哨兵，但后面又把 slot 0 的零填充 `PAD` trajectory 当成真实轨迹，转换成 `valid=True` 的 Waymax ego action。

这个错误直接落在本轮大量 offroad / kinematics first-event 路径上，而且一次错误动作会改变后续闭环状态。因此按照本项目一直采用的归因纪律，本轮必须停在 **repair-only**，不能继续回答“V16.8.27 算法成功还是失败”“dominant bottleneck 应收紧到哪里”“下一版论文算法该做 Recovery Certificate 还是 Execution-Viability Certificate”等后半段算法问题。

我已经把修复做成 **v16.8.28 EXECUTION_INTEGRITY**。不改训练、不改数据、不改 certificate、不改 selector objective、不改 proposal family，只修执行语义与诊断。

---

## 1. 我对论文与前序研究主线的理解

论文的核心不是“再做一个 social cost”，而是定义并解决 **false-safe planning**：ego 自身闭环看起来 collision-free，但安全成立的条件是其他道路使用者必须高负担让行、急刹或放弃合法 gap。COWP 将这种 safety-by-coercion 从软成本提升为 **non-coercive feasibility defect**。

目前代码/论文主线我理解为：

1. 从 interaction-heavy scene 中构造 burden-oriented relation；
2. 对关键 agent 构造 counterfactual natural roots，而不是把被 ego 压迫后的 yielding 当作“自然行为”；
3. 用 same-root RCOT 描述 ego intervention 对自然 option 的保留/恢复；
4. 用 protected-priority 作为主 hard-feasibility 语义，同时保留 all-critical diagnostic；
5. 用 BCOT 将 root-level transport/burden 证据压成结构化 certificate；
6. certificate 后使用 set-preservation / robustness-compatible selection，而不是简单 planner-score argmin；
7. fixed-bank proposal floor 与 certificate/selector 能力分开审计。

前几轮已经明确的研究纪律也继续保留：CTU 不能替代原 frontier；generic candidate classifier 不能取代 RCOT/BCOT mechanism；outcome head 目前只能做 probe；固定 bank proposal support 是长期 ceiling，但在没有把 online execution attribution 做干净前，不应先重建数据或继续堆 proposal primitive。

本轮没有改变上述任何算法判断，只是发现现有 strict physical evidence 仍被执行层实现污染。

---

# 2. 当前结果包哪些完整性检查是通过的

## 2.1 exact 200 IDs 是干净的

四个方法：

- `cowp`
- `cowp_fallback_outcome`
- `conventional_safety`
- `planner_score_only`

都使用同一组 200 个 unique scenario IDs，逻辑 SHA256：

`3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`

四个 merged 文件与 reference manifest 的 ID set 一致。

## 2.2 merge / aggregate 可复算

我从 200 个 per-scenario rows 重新计算 CR / collision / offroad / kinematics / EP，和 merged `standard_metric_summary` 一致。四方法 EP 都是同样 196 个 finite scenes，缺失的 4 个 ID 也相同，因此 paired EP 比较本身没有 ID 错位。

## 2.3 v16.8.27 上一轮两个 repair 确实生效

- `NEUTRAL_EGO` 不再通过 `conventional_check=False` 绕过 conventional audit；
- 普通 `cowp` 和 `cowp_fallback_outcome` 的 OutcomeHead metadata 已经 method-local，不再 stale。

所以这次不是上一轮 repair 没落地。

---

# 3. v16.8.27 实际跑出的结果

这些数字是 **buggy v16.8.27 controller 的真实 simulator outcome**，可以用来定位代码问题，但不能直接升级为算法结论。

| Method | CR | Collision | Offroad | Kinematics | EP | Fallback step rate |
|---|---:|---:|---:|---:|---:|---:|
| COWP | 24.5% | 17.0% | 10.0% | 15.0% | 1.1577 | 73.50% |
| COWP + fallback outcome | 24.0% | 16.0% | 10.5% | 14.0% | 1.1367 | 74.05% |
| Conventional safety | 31.5% | 25.0% | 10.0% | 12.0% | 0.8329 | 60.54% |
| Planner-score-only | 32.5% | 25.5% | 9.5% | 14.0% | 0.9876 | 2.98% |

COWP 当前 step-level reason：

- accepted priority-NCF: 26.50%
- `no_certificate_use_least_coercive_conventional`: 15.781%
- `no_conventional_use_least_coercive_valid`: 53.531%
- `no_valid_candidate`: 4.188%

表面看，这会把研究问题强烈推向“no-conventional / fallback recovery”。但本轮最关键的新发现是：**`no_valid_candidate` 的执行本身是错的，而且它对 offroad / kinematics 的污染非常集中。**

---

# 4. 致命实现错误：no-valid 时执行了 PAD trajectory

## 4.1 代码路径

online candidate bank 固定分配：

```python
traj = np.zeros((K, H, 7), dtype=np.float32)
valid = np.zeros(K, dtype=bool)
macro = np.full(K, int(MacroType.PAD), dtype=np.int64)
```

只有真正生成的 candidate 才覆盖前面的 slot 并置 `valid=True`。

但 v16.8.27 selector 在没有任何 valid candidate 时：

```python
selected = ... if has_valid else 0
```

随后无条件：

```python
traj = batch_np["cowp/candidates/trajectory"][0, selected]
return self._trajectory_to_action(..., traj)
```

因此 `selected=0` 并没有只作为内部 sentinel，而是把全零 PAD 轨迹真的送入了 action projection。

## 4.2 为什么 PAD 并不是“无动作”

`_trajectory_to_action()` 会：

```python
valid[sdc_index, 0] = True
```

也就是说 Waymax 会把它当成有效 ego action。

而 `_consistent_one_step_target()` 在 `desired_vel≈0` 时，会用：

```python
desired_speed = ||desired_xy - current_xy|| / dt
```

PAD 的 `desired_xy=(0,0)`。因此只要当前车辆不在世界坐标原点，它就会把“去原点的巨大位移”解释成 desired speed，然后再经 accel/jerk/yaw clip 变成一个真实控制指令。

这不是 neutral stop，也不是 invalid/no-op，而是从 padding 数值派生出来的错误 action。

## 4.3 它还会污染下一步 selector

旧代码还会：

```python
self._previous_selected_traj = PAD_zero_traj
```

于是下一 replanning step 的 continuity risk 会拿真实 candidates 和世界原点附近的 zero trajectory 比较。

所以污染不仅是“当前一步 offroad”，还可能改变之后的 selector ranking 与闭环状态。

---

# 5. 这个 bug 在当前 first-event 中有多严重

## COWP

- first collision: 34 episodes，其中 PAD/no-valid 2/34 = **5.9%**
- first offroad: 20 episodes，其中 PAD/no-valid 11/20 = **55.0%**
- first kinematics: 30 episodes，其中 PAD/no-valid 13/30 = **43.3%**

## COWP + fallback outcome

- collision: 1/32 = **3.1%**
- offroad: 14/21 = **66.7%**
- kinematics: 11/28 = **39.3%**

## Conventional safety

旧 baseline diagnostics 把 zero-valid 状态也写成 `baseline_use_valid`，但 selected macro 已经暴露为 PAD：

- collision: PAD 3/50 = **6.0%**
- offroad: PAD 10/20 = **50.0%**
- kinematics: PAD 12/24 = **50.0%**

## Planner-score-only

- collision: PAD 2/51 = **3.9%**
- offroad: PAD 9/19 = **47.4%**
- kinematics: PAD 15/28 = **53.6%**

这说明不是 COWP-specific bug，而是 **四方法共享的 online execution-interface bug**。

尤其 offroad / kinematics 的 first-event 有大约一半直接落在 PAD execution 上，因此当前 physical comparison 不能被解释为 selector/certificate/fallback mechanism 的干净因果证据。

---

# 6. 为什么连 collision-only 也不建议现在直接做算法归因

COWP collision 的 PAD 比例确实只有 2/34；其余 32/34 first collision 前一个 action 是 `no_conventional_use_least_coercive_valid`，且 selected candidate 是 dynamically valid 但 non-conventional。这个信号值得下一轮重点看。

但是我仍然不把它升级成算法结论，原因有两个：

1. 一个 episode 在 first collision 之前可能早已经历 no-valid/PAD 错误 action，闭环状态已经被改变；
2. conventional/planner baselines 也共享同一 execution bug，paired scene outcome 不再是“只差 selector”的严格比较。

所以当前最严谨的处理是：**记录这个 signal，但不据此设计下一算法。**

---

# 7. v16.8.28 的 repair

## 7.1 Selection 与 execution 解耦

新增：

```python
_resolve_execution_trajectory(...)
```

行为：

### 有 valid candidate

返回 exact selected candidate，bit-exact，不改变任何正常 selection 语义。

### 没有 valid candidate

不执行 candidate slot，不把任何 PAD 当真实 proposal。

而是从当前 ego state 调用已有：

```python
smooth_stop_trajectory(..., decel=fallback_decel_mps2)
```

生成 execution-only bounded stop trajectory，再走原有 `_consistent_one_step_target()` 的 jerk/yaw-rate limited projection。

它：

- 不加入 candidate bank；
- 不经过 RCOT/BCOT；
- 不标 conventional-safe；
- 不标 NCF；
- 不参与 frontier；
- 不是新的 learned mechanism；
- 只是修复“无 candidate 时不能执行 padding”这一最基本执行契约。

## 7.2 no-valid diagnostics 不再伪装成 selected candidate

新字段：

```text
selected_candidate = -1
selected_candidate_valid = false
selected_candidate_conventional_safe = false
selected_macro_name = EMERGENCY_BOUNDED_STOP
emergency_action_used = true
execution_trajectory_source = bounded_smooth_stop
```

baseline 的 no-valid 也不再和 `baseline_use_valid` 混在一起，而是：

```text
baseline_no_valid_emergency_stop
```

## 7.3 first-event attribution 继续增强

episode summary 新增：

- emergency action step rate / episode flag；
- actual zero-valid-candidate step rate；
- actual zero-conventional-candidate step rate；
- first emergency step；
- first zero-valid / zero-conventional step；
- first physical event 前一个 action 是否是 emergency；
- execution trajectory source。

这能避免下一轮再次把“selection 失败”和“execution emergency”混为一谈。

## 7.4 删除 dead branch

原 fallback flags 顺序：

```text
selection exists
conventional exists
valid exists
valid & stop_like exists
```

第四个分支永远不可能到达，因为 `valid & stop_like` 成立必然已经在第三个 `valid exists` 被截获。

因此 `emergency_stop_like` 是 dead code。我删除它，但没有重排有效 candidate 的选择顺序，避免 repair-only 偷偷变成算法修改。

---

# 8. 回归

`NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh sanity`：

**15 / 15 passed**

额外 focused：

**17 / 17 passed**

完整 repository：

**265 passed / 5 skipped / 8 failed**

8 个失败仍是历史类问题：

- 6 个测试引用当前压缩包中不存在的旧 launcher；
- 2 个测试硬编码旧 semantic fingerprint。

没有新增 functional regression。

新增核心测试直接验证：

1. no-valid + 全零 padding 时，execution trajectory 仍锚定在当前 world pose，不会跑到原点；
2. emergency trajectory finite 且速度单调下降；
3. valid candidate 路径返回 exact selected trajectory；
4. first-event emergency provenance 能落到 compact diagnostics；
5. dead `emergency_stop_like` branch 不存在。

---

# 9. 当前结果中仍然可以保留的工程结论

12-scene fine-grained profiler 本身不依赖论文算法归因，可作为工程优化依据：

- `candidate_build_cpu`: **87.8% -- 88.6%** policy time
- model forward: 约 **5.8%**
- selection: 约 **4.2% -- 4.8%**
- H2D: <1%
- action projection: <0.2%

所以运行慢的主要工程瓶颈已经很明确是 **CPU online candidate construction**，不是 Torch model、BCOT selector 或 Waymax env.step。

但这不是当前要改的内容；本轮我没有做性能重构，以免把 execution repair 与工程 fast-path 混在一个版本里。

---

# 10. 下一步只需要重新建立 clean physical evidence

不训练、不重建 dataset、不重建 label/cache、不调 BCOT、不改 proposal、不加 Recovery Certificate。

首先：

```bash
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh make_ids
```

如果 TFExample index 不存在才运行：

```bash
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh build_tfindex
```

然后重跑四方法同一 exact 200 IDs：

```bash
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh waymax_diag200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh analyze_parallel2
```

如果两张 A30 的 co-location OOM：

```bash
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh waymax_diag200_split
bash NEXT_RUN_COMMANDS_V16_8_28_EXECUTION_INTEGRITY_CN.sh analyze_split
```

`offline_metadata_check` 不需要为这次 repair 强制重跑，因为 metadata 代码没改；可以作为可选自检。

`profile_parallel2` 同样不是 clean attribution 的必要步骤，除非需要重新测 wall-clock。

---

# 11. 下一轮我会回答什么

等 v16.8.28 exact-200 回来后，再恢复你原本要求的完整算法分析，包括：

- V16.8.27/28 机制层面到底成功还是失败；
- 每一层学到了什么、没学到什么；
- repaired first-event 是 certified path、non-conventional valid fallback、zero-conventional transition，还是 common proposal/action interface；
- dominant bottleneck 如何收紧；
- 上轮“最该回答的问题”是否真正得到答案；
- Recovery Certificate vs Execution-Viability Certificate vs proposal/action consistency 哪个才是论文级下一机制；
- 基于 `ALGORITHM_CHANGELOG.md` 排除已证伪方向后，设计下一版算法与机制消融。

在这组 clean evidence 之前继续做算法设计，会违反你要求的 CCF-A 级归因标准，也容易再次围绕实现 bug 堆机制。
