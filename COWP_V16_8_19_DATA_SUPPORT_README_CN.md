# COWP v16.8.19 数据支持补丁说明

## 结论

v16.8.18 strict 的失败不是一个单点 bug，而是两个互相独立的 gate：

1. **Natural-basis / certificate contract**：`mechanism_valid` 在“发现 lane route 或 substantial factual geometry”时过早置真，但 root 经过 map/burden/priority/dedup 后仍可能只剩 0/1 个。模型的 root-indexed transport supervision 需要至少两个低负担 root，因此 evidence availability 不能等价于 constructive auditability。
2. **Ego proposal support**：strict representative 800 scenes 中 `any_ncf=0.33875 < 0.40`、`false_safe=0.61125 > 0.55`。全 1200-scene source ablation 又显示 RMR-BCTE/PSY 对 scene-level NCF 的增量几乎为零；这不是增加 WOMD scene 数量能自动修好的问题，而是候选几何/多 agent compatibility 的覆盖问题。

v16.8.19 不降低任何 strict threshold；它修复 label 语义并增加 causal route-topology/group pass-after proposal support。

## 代码变化

`cowp/label/natural_alternatives.py`

- factual-geometry retiming 增加 `extrapolate_after_path=False` 路径：empirical-only fallback 不再越过最后观测路径端点外推“虚构几何”。
- 所有 natural root filter 完成后增加 **constructive-auditability closure**。只有 `root_count >= 2` 且 `low_burden_root_count >= 2` 才保留 `mechanism_valid=1`。
- 若 closure 失败，critical actor 仍保持 `critical/valid=1`，但 `mechanism_valid=0`，并清空其 partial natural roots；这样 inference-time critical universe 不变，offline mechanism/certificate target 变为 unknown，而不是把 unknown 当 negative。

`cowp/label/ego_candidates.py`

- BCTE timing-profile 去重 key 从 `(region, side, accel_bin)` 改成 `(region, agent, side, accel_bin)`，避免不同 agent 的候选互相误去重。
- 增加 `_project_progress_profile_to_route`：把纵向 timing profile 投影到当前状态可见的 WOMD lane-topology route，不使用 logged future。
- 增加 RT-NCF bank：少量 route-following keep/yield + **group protected pass-after**。group candidate 的 target time 绑定同一 conflict region 中 protected agents 的最晚 causal arrival envelope，目标是避免“对 A yield 但又 coercive B”的 pairwise-only 症状。
- 新 route proposals 仍复用原 ProposalSource，不改 cache/model enum schema。

`configs/label_cowp_v16_8.yaml`

- 新增 constructive certificate minimum（2 root / 2 low-burden roots）。
- 新增 route-topology proposal bank 参数，默认总预算 10 个，route 数 2，group yield 最多 8 个；固定 `K=64` 不变。

`NEXT_EXECUTION_V16_8_19_CN.sh`

- 使用新的 smoke/strict/train/full 输出目录。
- `smoke` 强制映射到 v16.8.18 gate harness 的 `fresh-smoke`；显式禁止 `reaudit-smoke`，因为 label semantic fingerprint 已改变。
- v16.8.18 的 gate schema/threshold 保持不变，避免“为了通过而改考试规则”。

## 为什么原 smoke 能过而 strict 不能过

上传的 v16.8.18 smoke 实际是对 reviewed v16.8.16 标签的 policy re-audit。proposal screen 的 point estimate 已经出现坏信号：`any_ncf=13/48=0.2708`、`false_safe=33/48=0.6875`；只是 smoke 使用 Wilson gross-failure policy，置信区间尚跨过阈值，因此被视为“不足以否决 strict”，而不是证明 proposal 性质合格。

strict 的 representative sample 扩到 800 后，`any_ncf=271/800=0.33875`，95% Wilson 上界约 0.372，仍低于 0.40；`false_safe=489/800=0.61125`，95% Wilson 下界约 0.577，仍高于 0.55。此时统计上已经不是 smoke sampling noise。

## Natural support 的具体缺口

strict 1200 scenes / 6663 selected critical 中，6422 个原先被标为 auditable，241 个被 mask，整体 coverage 本身合格；真正导致 exact support gate 失败的是 2 个 rootless actor 和 5 个 `<2 low-burden roots` actor（含那 2 个 rootless）。两个 rootless 都只有 77/80 future-valid、没有可覆盖完整 8s 的 lane route、走 empirical corridor；旧代码在 factual geometry 末端继续外推，导致 map/empirical corridor 拒绝。三个 singleton actor 主要是 priority comfort/progress filter 后只剩一个 root。

因此 v16.8.19 的核心不是“想办法硬造 6 个 root”，而是：先尝试更正确的 bounded geometry；如果真实 constructible basis 仍不足，必须把 mechanism target 标成 unknown。按当前 strict 计数做保守上界，即便这 5 个 actor 全部转为 unauditable，unauditable rate 仍约 246/6663=3.69%（低于 5% gate），certificate-complete scene rate 最坏也约 1043/1200=86.9%（高于 75% gate）。这只是基于旧 strict 计数的理论界，不替代 fresh rerun。

## WOMD / Waymax 数据合同

训练/验证 label 构建应继续以 Scenario proto 提供的 vector map、tracks、dynamic map states 和 91-step temporal contract 为基础；tf.Example 用于 tensor/cache/Waymax 对齐。不要把 `tracks_to_predict` 当 COWP critical universe，它是 WOMD prediction benchmark 的 target selection，而不是交互规划中的因果 critical set。

Waymax 侧应保持整个 scenario batching，`aggregate_timesteps=True`；`max_num_objects` 在 DatasetConfig 与 EnvironmentConfig 一致。WOMD 1.3.1 的 `sdc_paths` 可以用于 Waymax route/wrong-way/progression metrics，也可以在未来做 planner route conditioning；但若它参与 label-time proposal geometry，就必须保证同等 route 信息在线 planner/model input 中可见，不能形成 label-only future-route shortcut。本补丁没有用 `sdc_paths` 生成 COWP label candidate，而是只用 current state + vector-map topology。

## 测试结果

新增的 v16.8.19 constructive-support / evidence-bounded-retime / curved-route projection tests 全通过；natural support、proposal ceiling、stable critical/PSY 关键回归也通过。全体 `tests/test_v16_8*.py` 结果为 88 passed / 2 failed；两个失败在原上传包中同样存在，原因是测试引用了上传包缺失的旧 `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh` 与 `NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh`，不是本补丁回归。

当前 label semantic fingerprint：`51844462540c083592280a7a8c24da962aba9d743a92b41d1a7f27095f0c2452`。

## 执行

直接按 `NEXT_EXECUTION_V16_8_19_COMMANDS_CN.txt` 执行。最重要的约束是：**strict 不授权就不要手动调用 train-pilot；train-pilot 不授权就不要 full-core。** 如果新 strict 的 natural gate 已通过但 proposal gate 仍失败，应先查看 `paired_proposal_probe.json` 与 `proposal_source_ablation.json` 中 RT-NCF/PSY 所在 source 的 scene-level NCF 增量；不要先放宽 0.40/0.55 gate。
