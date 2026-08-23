# COWP v16.8.25 当前证据、瓶颈与下一步

## 结论状态

当前 v16.8.24 结果足以做**一阶瓶颈归因**：主要瓶颈是 proposal-bank 的 non-coercive feasibility coverage，而不是 BCOT 判别能力。当前证据不足以宣称最终 closed-loop 改进，因为严格 exact-ID Waymax 输出缺失，且当前 Waymax 主协议为 logged replay non-ego，不能单独验证反事实 burden-transfer 因果主张。

从 v16.8.25 起，已经检查并用于算法设计的 1,200-scene held-out 集只作为 development/diagnostic heldout；最终论文需要重新冻结一个 content-blind final holdout。

## 当前论文主线（以代码为准）

1. **False-safe planning**：ego collision-free 可能依赖他车高负担让行。
2. **Non-Coercive Feasibility (NCF)**：对 protected-priority agent，不能仅最小化 soft social cost，而要保留足够的 low-burden safe option mass。
3. **RCOT**：以 natural root 为身份锚点，对同一 root 做 ego intervention 后的 safe-response transport/recovery。
4. **BCOT**：将 unrecovered conflict mass、conflict-conditioned burden excess、OPR shortfall 以结构化单调方式聚合成 protected-priority candidate certificate；all-critical 仅作更严格诊断。
5. **Hard-first selection**：conventional safety -> protected BCOT gate -> bounded utility ranking；空证书集进入显式 uncertified fallback。

Generic candidate classifier 只能作为 diagnostic/ablation，不能成为主机制。Candidate proposal family 是 solver support，不应抢占 RCOT/BCOT 的主贡献叙事。

## v16.8.24 的关键数值归因

### Proposal ceiling

- train certificate-complete scene rate: 0.8654；AnyNCF: 0.35845；best-case selected global false-safe floor: 0.59187。
- val certificate-complete: 0.8630；AnyNCF: 0.35110；floor: 0.60371。
- held-out certificate-complete: 0.8575；AnyNCF: 0.36346；floor: 0.59475。

因此在当前固定 proposal bank 上，约 59--60% 的场景级 global false-safe 是 selector/certificate 无法通过阈值优化消除的结构性下界。

### Certificate / selector

held-out：

- BCOT Priority FalseSafe AUPRC ~= 0.8374；Global FalseSafe AUPRC ~= 0.9281。
- Generic candidate classifier NCF AUPRC ~= 0.1756；FalseSafe AUPRC ~= 0.3544。
- 当场景存在 NCF candidate 时，selector 选中 NCF 的 recall ~= 0.7888。
- selected global false-safe ~= 0.6715；proposal oracle floor ~= 0.5948；selector excess ~= 0.0768。

解释：证书已经显著强于 flat classifier；selector 仍有约 7.7 percentage points 的次级改进空间，但 proposal support 是第一瓶颈。

### 当前 v16.8.24 operating point

val calibration 在 budget=0.50 时仍返回 `proposal_infeasible`，不是 constraint-satisfied calibration。关键原因是要求 selected global false-safe <=0.55，而 fixed-bank oracle floor 已约 0.6037。故不能把 0.50 写成“满足验证约束的 calibrated operating point”。

held-out learned-offline COWP（budget=0.50）：CR~=0.045, EP~=0.6155, Fallback~=0.3267, PBTR~=0.4711, OPR~=0.8488, SelectedNCF~=0.2867, SelectedFalseSafe~=0.6715。缓存 Waymax candidate-outcome coverage 只有约 0.519，不能替代严格 online Waymax rollout。

## 当前最可靠 / 最不可靠的机制结论

### 可以较可靠支持

- fixed proposal bank 是当前主要性能上限；继续只调 BCOT 阈值不可能跨过 proposal floor。
- structured RCOT/BCOT 比 generic flat candidate safety classifier 更适合当前 false-safe 机制目标。
- protected-priority gate 比 universal all-critical hard veto 更符合当前目标；universal veto 会显著增加 fallback/conservatism。
- hard feasibility 与 soft burden ranking 需要在相同 bank/checkpoint 下继续做 paired evidence，但当前结果方向与主假设一致。

### 目前不能强 claim

- affected-root formulation 优于 conflict-only：当前 train/val/held-out 的 burden-only affected roots 为 0，缺少关键正例支持。
- logged-replay Waymax 能验证 causal burden transfer：不能；论文自身也应将它限定为 conventional SDC closed-loop safety/progress evidence。
- v16.8.25 MCFC 有效：尚未跑 fresh paired proposal probe，不可声称有效。
- 当前 1,200-scene heldout 是未来新算法的 blind final test：已经被检查并影响算法设计，因此不再成立。

## v16.8.25 实验性算法：MCFC

**Multi-Conflict Feasibility Corridor (MCFC)** 针对 JR-NCF 的单一恒加速度限制：

- 从当前可见 lane topology / protected conflict timing 构造多个 “不得早于 t 进入 conflict” 的约束；
- 将约束变成 piecewise time-progress corridor；
- 使用分段 quintic Hermite progress trajectory，允许在不同 conflict 前采用不同 delay profile；
- 最后一个 binding conflict 后允许受限的 speed recovery；
- 全程保留 hard geometry/timing/kinematics validation；
- 不把 proposal source 本身当成 NCF 证据，最终仍必须经过 RCOT/BCOT certificate。

MCFC 与已失败的 PCHR stop-hold-release 不同，也不是简单复刻 JR constant-acceleration。当前默认旧配置不启用 MCFC；只有 `configs/label_cowp_v16_8_25_mcfc.yaml` 显式启用。

## Promotion rule

先在 validation 上做 600-scene exact paired fresh-label/cache probe。除了原 proposal-bank gate，还必须通过 `85_screen_v16_8_25_mcfc_probe.py` 的 source-attributed gate。默认要求 MCFC 自身产生足够 NCF/priority-NCF，并带来 >=3pp AnyNCF gain、>=2pp priority-NCF gain、>=3pp global floor drop、>=2pp PBTR-floor drop，同时 old-NCF loss <=2%。

这只是是否值得 full rebuild/retrain 的工程门槛，不是 publication statistical significance。

## 严格 Waymax 修复

原 `训练和测试.txt` 的 `--scenario-ids-file` 在旧 evaluator 中没有 argparse/rollout plumbing，因此 strict exact-ID command 会失败。v16.8.25 已实现：

- exact scenario-ID allowlist；
- 可选 TFExample shard index；
- deterministic exact-ID sharding；
- requested/covered/missing ID accounting；
- scenario-ID SHA256；
- 默认缺一个 requested ID 就失败；
- 每个 rollout item 写入 scenario_id。

## 运行顺序

使用 `NEXT_RUN_COMMANDS_V16_8_25_MCFC_CN.sh`：

1. `freeze_final_split`（在查看任何新实验输出前先冻结 ID）
2. `strict_v24_dev`
3. `learned_v24_diagnostics`
4. `proposal_probe`
5. 仅当 `promote_mcfc_to_full_rebuild=true`：`full_rebuild -> attach_waymax -> train_v25 -> eval_v25_dev`
6. 冻结算法/超参/校准规则/checkpoint selection 后：`build_final_blind -> eval_final_blind`

若 MCFC probe 不通过，停止 v16.8.25 全量重建；回到 proposal mechanism 继续设计，不要通过重训噪声“救”一个没有 source-attributed coverage effect 的 proposal。
