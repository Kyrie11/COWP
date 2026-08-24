# COWP v16.8.27 当前分析：Strict-Waymax Conventional-Safety Integrity Repair

## 结论

v16.8.26 的 learned-offline 结果可以继续使用，但 strict Waymax 的 fallback/physical-failure **机制归因不能继续使用**。原因是在 online candidate generator 中，`NEUTRAL_EGO` 两处调用通过 `conventional_check=False` 绕过了 roadgraph 与 causal collision screen，却被默认写成 `conventional_safe=True`。该候选随后可以进入 `no_certificate -> least_coercive conventional` fallback pool。

因此 v16.8.26 中“44/45 collision 在 first collision 前最后一个 action 是 fallback”是真实运行现象，但不能区分：是 fallback objective 本身错误，还是 fallback pool 被未经 conventional audit 的 neutral candidate 污染。

## 哪些结果仍然有效

1. val/held-out RCOT、BCOT、candidate classifier、outcome-head AUPRC/Brier/ECE/low-FPR recall；
2. CTU 的 learned-offline 与此前 strict paired 负结果（CTU 不是本轮 bug 的来源）；
3. fixed-bank proposal floor；
4. v16.8.26 Waymax runtime profile 对工程瓶颈的判断：policy 内部占主要计算时间，Waymax env.step 本身很小。

## 哪些结果需要作废重跑

1. v16.8.26 exact-200 COWP / fallback-outcome / conventional / planner 的最终 physical comparison，作为“当前正确算法”的性能数字；
2. first-event fallback localization 的算法原因解释；
3. fallback-only outcome guard 是否有效的 strict-online结论。

## v16.8.27 修复

- 完全移除 conventional-safety bypass API；
- neutral 仍可作为 dynamically-valid 最终候选，但只有通过 conventional audit 才能进入 conventional pool；
- conventional fallback selection 加 runtime integrity assertion；
- selected macro/conventional/fallback provenance 进入 first-event compact diagnostics；
- 修复 learned-offline outcome-head selection metadata stale-variable bug；
- 增加 fine-grained policy profiler。

## 下一轮只回答一个算法问题

在修复后的同一 200 exact IDs 上重新回答：

> certificate 为空时的物理失败，究竟来自 uncertified recovery selection，还是来自共同的 online proposal/action execution interface？

只有这个问题回答后，才允许设计下一版论文级机制。若失败仍高度集中在“真正 conventional-safe 的 fallback action”上，则可设计 Feasibility-Preserving Recovery；若 accepted COWP action 也大量失败，则应设计 execution-viability certificate/online proposal consistency；若 conventional/planner 同样差，则应先修 common online proposal/action interface。
