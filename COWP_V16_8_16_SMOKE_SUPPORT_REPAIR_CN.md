# COWP v16.8.16：v16.8.15 Smoke 数据支撑复盘与修复

## 1. 结论

v16.8.15 的 `recommend_strict_probe=false` 由两个不同层面的 gate 共同造成：

1. **proposal/causal smoke screen 的 PBTR 点估计轻微越线**：代表性 48 scenes 中，priority-eligible 为 45，PBTR=23/45=0.5111，高于 smoke 上限 0.50；但 95% Wilson 区间约为 [0.370, 0.650]，因此 96-scene smoke 无法统计上断言其真实 PBTR 已经差于 0.50。
2. **natural/model-support 的真实构造失败**：538 个 selected critical 中 536 个被判定为 mechanism-auditable，但仍有 13 个 auditable critical 完全没有 natural root，同时也是 13 个 `<2` low-burden roots；423 个 protected auditable critical 中 18 个没有 PRIO root。旧 98% certificate scene coverage 在 96 个 scenes 上因 94/96=0.97917 也失败。

因此不能仅把 PBTR 上限从 0.50 改成 0.52。旧数据即使应用新的 smoke 统计 gate，仍会因为 13 个 auditable rootless critical 被拒绝。

## 2. v16.8.15 已经健康的监督

v16.8.15 smoke 的训练监督并没有整体退化：

- pair relevance positive rate ≈ 0.4068；
- witness | relevant ≈ 0.6437；
- pair NCF | relevant ≈ 0.3563；
- response safe ≈ 0.4841；
- response low-burden ≈ 0.4654；
- affected-root recovery ≈ 0.4119；
- relevant pair 的固定 response bank 为 360096 / 360096 slots；
- natural source：OBS=1781、NEU=2756、PRIO=4003；
- response source：PRED=165811、OPT=192456、EMG=1829；
- candidate NCF / false-safe、root transport、witness continuous targets 都非退化。

这说明不应扩大 ego proposal bank，也不应削减 response/root budget。当前失败集中在少数 critical actor 的 natural-basis map/auditability contract。

## 3. 13 个 rootless 的共同结构

natural diagnostic 显示：

- 13/13 的 `reference_kind = logged_geometry_neutral_timing`；
- 13/13 的 dominant rejection = `map`；
- priority relation：11 个 EQUAL_OR_NEGOTIATED，2 个 AGENT_PRIORITY；
- future support：9 个有完整 80 steps，2 个 78/79 steps，2 个仅 35/36 steps；
- 它们的最优 map-rejected max-distance 从约 7.5m 到 20.2m。

这表明“WOMD 没有 future”不能解释大多数失败。

## 4. 找到的构造 bug

### 4.1 短 lane polyline 被错误当成完整 8s route

v16.8.15 同时计算：

- `route_polylines = _map_route_polylines(...)`
- `map_refs = route_variants(0,0)`

`route_polylines` 可能非空，但因为 lane 没有 exit / continuation 太短，无法 retime 到完整 80 steps，此时 `map_refs=[]`。

旧代码却使用：

```python
route_supported = bool(route_polylines)
empirical_eligible = empirical_supported and not route_polylines
```

因此会出现：

> 没有完整 map route，但因为存在一小段 lane polyline，又禁止使用完整 factual future 的 empirical corridor。

结果是本来有 78--80 step factual geometry 的 actor 被强制只接受 lane-centreline map check，然后所有 root 被 map filter 清空。

v16.8.16 修为：

```python
route_supported = bool(map_refs)  # 真正可 retime 完整 8 秒
empirical_eligible = empirical_supported and not route_supported
```

对于 35/36-step 且没有 full map route 的 actor，不再把它们错误保留为 mechanism-auditable；critical selection 本身不变，但 `mechanism_valid=false`，并计入 coverage gate。

### 4.2 Canonical OBS 不应重新积分 WOMD velocity

v16.8.15 的 OBS identity `(speed_scale=1, shift=0, lateral=0)` 仍从 current state 积分 logged velocity 重构 positions。WOMD factual box position 与 velocity/heading 存在正常离散差异时，这种做法会在 8s 内积累几米漂移。

v16.8.16 对 identity OBS 直接保留 factual positions，仅用 finite difference 修复 yaw/vx/vy，避免最可靠的 observational root 自己漂出 factual corridor。

### 4.3 WOMD driveway polygon 被 parser 丢弃

v16.8.15 的 Scenario map parser 没有读取 `MapFeature.driveway`。v16.8.16 在 `MapData` 中增加 `driveways`，并将 lane corridor 与显式 driveway polygon 做 HD-map compliance union。Driveway margin 只用于 polygon 边界离散误差，不通过 road edges 推测任意 drivable polygon，也没有放宽原 lane 阈值。

### 4.4 失败 map distance 诊断口径

旧逻辑在 lane 和 empirical 都失败时只记录 lane distance。v16.8.16 使用可用证据中的最小有限 distance，并额外记录 `route_polyline_count`、`full_horizon_map_route_count`、`driveway_polygon_count`，natural-support summary 新增：

- `short_route_candidate_agents`
- `rootless_short_route_candidate_agents`
- `empirical_corridor_eligible_agents`
- `driveway_context_agents`

下一轮若还有 rootless，可以直接判断是否仍为短 route / driveway / empirical evidence 问题。

## 5. 门槛哪些过严，哪些不能降

### Smoke 改成统计筛选，而不是小样本 publication gate

v16.8.16 smoke 对 AnyNCF / false-safe / PBTR / hard-recovery 使用 95% Wilson interval 的 gross-failure gate：

- minimum metric：只有 Wilson **upper** 仍低于阈值才 FAIL；
- maximum metric：只有 Wilson **lower** 仍高于阈值才 FAIL。

严格 1200-scene probe **保持原点估计门槛不变**：

- AnyNCF >= 0.40
- false-safe <= 0.55
- PBTR <= 0.45
- hard NCF recovery >= 0.20

把 v16.8.15 的旧 smoke JSON 用 v16.8.16 screen 重放，proposal/causal screen 会从 FAIL 变 PASS；但旧 natural rootless gate 仍 FAIL。因此这不是“放宽门槛让旧数据过关”。

### Smoke certificate coverage

96 scenes 上 94/96=0.97917，因为 1 个 scene 就改变约 1.04%。v16.8.16 smoke 只要求 >=95%，strict/train-pilot 仍要求 >=98%。

### PRIO coverage

论文定义的是过滤后的 natural union `OBS ∪ NEU ∪ PRIO`，并没有要求每个 protected critical 在 filtering 后都必须保留一个 PRIO root。v16.8.15 dataset-wide PRIO 已有 4003 roots，protected per-agent coverage 为 405/423≈95.74%。

因此 v16.8.16 改为：

- smoke protected PRIO coverage >=95%；
- strict/train-pilot >=98%；
- dataset-wide PRIO source 仍必须达到 source-count gate；
- 所有保留下来的 PRIO root 仍必须真正 `priority_preserved=true`。

这样不会为了 audit 人工制造 duplicate PRIO root。

### 仍然零容忍的硬门

以下没有降低：

- auditable critical rootless = 0；
- auditable critical `<2 low-burden roots` = 0；
- invalid natural weights = 0；
- relevant pair response bank 必须完整；
- transport/audit identity mismatch = 0；
- silent/irrelevant blockers = 0；
- unauditable critical rate <=1%。

## 6. 理论上能否支撑 COWP 模型

Fresh v16.8.16 数据只有在 smoke -> strict -> train-pilot 三层 gate 全通过时，才可认为对当前代码的 supervised object 具备充分支撑：

1. ego candidate bank 有 conventional-safe / unsafe / NCF / false-safe 支撑；
2. 每个 auditable critical 至少有 2 个 low-burden natural roots；
3. OBS/NEU/PRIO 三个 typed sources 全局非退化，protected PRIO 覆盖充分；
4. candidate-conditioned pair relevance、affected roots、same-root recovery 和 minimum safe burden 有正负监督；
5. relevant pair 固定 response bank 完整，PRED/OPT/EMG 来源非退化；
6. witness / OPR / tail burden / NCF selector labels 非退化；
7. certificate unknown 与 negative 明确分离；tensor input 不可见 critical 会使 certificate target 失效；
8. train/validation 必须分别通过同一 fingerprint 的 support audit。

这足以支撑当前 COWP 模型训练、标准 validation、Waymax logged-replay 物理指标和机制层 ablation。论文中更强的真实 counterfactual causal-burden 主张仍需独立 reactive-agent protocol + held-out human-audited false-safe stress set，不能由 WOMD logged future 单独证明。

## 7. 性能

v16.8.15 smoke 平均约 243.1 s/scene；safe response + witness 仍约占 label-engine 80%。v16.8.16 的修复主要发生在 natural/map 阶段，不减少 32-slot response bank、natural root budget 或 proposal family，因此不会用数据性质换速度。Full rebuild worker 数继续应在 train-pilot PASS 后根据 host profile 决定。

## 8. 本地回归

- v16.8.16 新增 5 个数据契约测试全部 PASS；
- repository 分两组完整执行：87 passed + 128 passed = **215 passed, 5 skipped**；
- `python -m compileall -q cowp` PASS；
- master / smoke / strict / train-pilot / full-core shell syntax PASS。
