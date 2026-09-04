# V16.8.43R3 可靠性归因、V43 科学结论与 V16.8.44 设计

## 0. 最终判决

**V16.8.43R3 当前结果通过可靠性审计，可以进行算法归因。V16.8.43 BC-IARE 按冻结预注册条件必须 STOP。**

这不是“因为没有 remaining29 所以实验不完整”。R3 的 mandatory gate19 由历史固定的 10 个 RVR rescue + 9 个 RVR-induced counterexample 组成，而冻结六项 Gate 的前两项恰好只依赖这 19 个场景。V43 在 10 个历史 rescue 中只保留 **3/10**，低于预注册 **>=5/10**；因此无论 remaining29 发生什么，完整 conjunction Gate 都不可能 PASS。按 R3 的 fail-fast protocol 停止 remaining29 是正确的科研早停。

独立机器审计文件：`V16_8_43R3_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT.json`。

---

## 1. 对论文 idea / motivation / pipeline 的理解

论文的核心不是“给 planner 加一个 social burden cost”，而是指出传统 ego-centric collision-free safety 会接受一种 **false-safe / safety-by-coercion**：ego 轨迹表面无碰撞，但成立的前提是别的交通参与者 hard brake、abrupt yield、priority abandonment 或 gap surrender。有限 soft penalty 无法保证这类轨迹被排除，因此应把它定义成 **hard feasibility defect**。

当前成熟主链：

```text
Natural alternatives
→ stable natural roots
→ same-root RCOT
→ BCOT
→ protected-priority hard non-coercive certificate
→ certificate-compatible set preservation
→ hard-first selection
→ explicit uncertified fallback
```

论文最重要的结构纪律有两条：

1. certificate 与 ego utility 不再重新 scalarize；
2. proposal/support sufficiency 与 certificate/selector quality 分开审计。固定 bank 本身缺失 NCF/recovery support 时，调 threshold/ranking/fallback 无法消除 support-dependent floor。

V28→V43 后，可以把论文主线进一步收紧为 **Orthogonal Option-Set Feasibility**：

- **Social option-set feasibility**：ego 不能通过压垮 protected agents 的 natural low-burden options 获得所谓安全；
- **Physical-interactive recourse feasibility**：uncertified ego recovery 也不能依赖“真实 blocker 没有合理 recourse”，并且 ego 自己的 recovery tube 必须 control/shift closed。

统一命题：

> **Safety must not be obtained through critical option-set collapse.**

---

## 2. benchmark / compact-5k 数据性质

稳定统计：

| Split | scenes | audit-relevant pair rate | protected PRIO-root coverage | rootless |
|---|---:|---:|---:|---:|
| train | 5000 | 0.42970 | 0.99453 | 0 |
| val | 1000 | 0.42863 | 0.99363 | 0 |
| heldout | 1200 | 0.42949 | 0.99465 | 0 |

跨 split 性质稳定；历史统计还显示 `<2 low-burden roots=0`，mechanism unauditable 约 4.1%–4.5%。因此目前没有证据支持“模型已被 split/data quality 卡死”，更没有依据现在重构 compact-5k。

长期 watch items：

- critical-agent cap=6，conflict-region selected-cap saturation 约 95.4%–95.8%；
- `PRIORITY_SMOOTH_YIELD`、`TERMINAL` proposal acceptance 偏弱。

它们可能是后期 support ceiling，但不是 V43 的当前 dominant bottleneck。

### 2.1 新发现的数据 provenance caveat

重新检查上传数据包时，发现 `verify_cache_train.json` 自身为：

```text
pass = false
files/inspected = 5000/5000
valid_scene_rate = 1.0
silent_blocker_count = 0
irrelevant_blocker_count = 58,243
reason = "irrelevant pair blockers=58243"
```

这是一个必须在投稿前处理的 **cache/provenance integrity defect**。但当前代码的 fresh witness 对 irrelevant pair 明确采用 vacuous non-coercive 语义；training-time `paper_aligned_supervision_batch` 又用 `pair = base_pair & audit_target` 重算 relevance，并令非审计 pair 在 `pair_good` 中为真。因此它**目前没有证据可解释 V43 online root-unrecoverable failure**。

科研决策：**当前不重建数据**；但在最终 retrain/publication 前，单独修复 cache serialization 或做 cache→runtime semantic equivalence 证明，不能把现有 verifier `pass=false` 当作“完全干净的数据 provenance”。

---

## 3. V16.8.43R3 可靠性审计

Blocking audit 全部 PASS：

- gate19 = 19 unique IDs；remaining29 = 29 unique IDs；二者不重叠且并集精确等于冻结 CF48；
- gate19 两 shard = 10+9，无 overlap，union 精确等于 manifest；
- merged scenario rows 可由 shard 精确重构；
- CR / Collision / Offroad / Kinematics / EP 等逐场景重算误差为 0；
- checkpoint / method identity 一致；
- online mechanism GT disabled；
- R3 profile4 与 R2 同场景行为 0 mismatch；
- 历史 V29 10 rescue + 9 induced set 可由 immutable COWP/RVR reference 独立重建；
- mandatory Gate 独立复算仍为 3/10 与 9/9；
- V43 vs V42 在 gate19 的核心场景结果 **0 mismatch**。

因此当前不是代码 bug 造成的假失败，可以进入算法归因。

---

## 4. 严格按预注册 Gate：V43 = STOP

冻结六项 Stage-1 条件：

```text
historical RVR rescues retained >= 5/10
historical RVR induced avoided >= 7/9
net COWP collision removed >= 3
Kinematics net regression <= 1 scene
paired mean EP delta >= -0.05
nonzero action-changing intervention
```

mandatory19 已经得到：

- historical rescue retained = **3/10 → FAIL**
- historical induced avoided = **9/9 → PASS**

第一项已经数学上不可被 remaining29 修复，所以：

> **V16.8.43 BC-IARE policy = STOP / Archive。**

不得为了 headline 调整 5/10 阈值，也不运行 remaining29/fresh37 去“碰运气”。

---

## 5. V43 真正告诉我们的机制是什么

V42 的 dominant reject 曾是：unsupported blocker 57.08%、root-unrecoverable 34.95%。V43 的目标是 exact-blocker-conditioned late-bound natural support，验证 top-4 fixed indexing 是否漏掉真实 blocker。

R3 gate19 机制结果：

- late-bound query 在 **18/19 scenes** 实际运行；
- BC query-specific selection scene = **0/19**；
- V43 与 V42 gate19 core outcomes = **0 mismatch**；
- 7 个仍丢失的历史 RVR rescue 中，6 个 query-active 场景平均：
  - exact blocker agents / attempt ≈ 8.22；
  - ready agents / attempt ≈ 7.67；
  - unsupported rejects / attempt ≈ **8.56**；
  - root-unrecoverable rejects / attempt ≈ **126.13**。

因此 V43 的结果不是“late-binding 根本没执行”；而是它执行后把问题揭露得更干净：**blocker 大多已被找到且 natural support 已 ready，但冻结的同-root responder support 仍无法为 retained roots 找到满足完整 hard certificate 的低负担 response。**

所以 V43 的科学价值是 **falsify indexing as dominant bottleneck**，而不是一个可 promotion 的 policy improvement。

### 算法族关闭

正式关闭：

> **fixed scene-level support / blocker-indexing enlargement family**

后续禁止：

- 继续扩大 `max_online_critical_agents`；
- 把更多 nearby actors 塞进 protected social set；
- 再做 blocker query scope/grid/ranking patch；
- 在 CF48 上调 p_min / root mass / beta / dedup；
- 放宽 universal root/environment/joint CSP certificate。

V43 late-bound exact-blocker query 可以保留为 diagnostic/support completion infrastructure，但不再作为下一轮 scientific variable。

---

## 6. dominant bottleneck 收紧

当前 P0：

# **Root-Conditioned Control-Reachable Responder Support**

更完整地说：

> 给定 `(ego recovery tube, exact blocker, retained natural root)`，是否存在一个 **same-root、low-burden、controller-realizable、road/kinematic-feasible、current+shift safe、environment-compatible、jointly realizable** 的 responder recourse？

这与“预测 blocker 最可能怎么走”不是同一个问题。模型当前真正缺的是 **recourse existence/support completeness**，不是 generic collision probability。

要回答的算法问题是：

```text
(τ_ego, blocker, natural root, current control state)
→ exists low-burden control-reachable recourse ?
```

如果 analytic support completion 能显著恢复 lost rescues，下一步才有理由学习一个 dedicated **root-conditioned recourse viability representation/head**；在 analytic target 未被证明前，不应先训练新 classifier。

### 仍需保持解耦的两个问题

1. `fccd9a25...`：已有 local certificate 但后期仍 collision，属于 P1 multi-step invariance / interaction-model uncertainty；
2. accepted-path Kinematics：属于 certified path execution viability secondary bottleneck。

两者都不能与 V44 同轮混合，否则无法归因。

---

## 7. V16.8.44：RC-CRRS 的具体设计

方法 ID：

```text
cowp_root_conditioned_control_reachable_responder_support
```

V44 **只改一个变量**：当 V43 exact blocker support 已 ready，但某 retained natural root 的冻结有限 response bank 没有任何 ego+environment-safe profile 时，允许在**同一个 root 几何上**做 bounded control-reachable response completion。

### 7.1 不做什么

不新增：

- 新 natural roots；
- 新 route geometry；
- 新 social critical agents；
- p_min / mass / beta tolerance；
- safety horizon；
- scalar risk weight；
- response-duration grid；
- controller relaxation。

### 7.2 做什么

对失败 retained root：

1. 使用与 offline same-root transport 相同的 arc-length residual geometry；
2. 用 identity root 与 ego current/shift tube、frozen environment 的最后 unsafe event 确定必要响应持续时间；
3. acceleration/deceleration 范围继承 frozen physical limits；
4. 每个 control candidate 必须重新通过：
   - adaptive low-burden beta；
   - roadgraph；
   - Waymax inverse-dynamics current + shifted；
   - ego current + shifted interaction safety；
   - responder↔environment 双向 safety；
5. 在每个正/负控制方向使用 deterministic dyadic bracketing + fixed bisection，寻找最小幅度可行 residual；
6. 找到的新 profile 最终仍进入原 V42/V43 exact multi-blocker CSP，任何 hard constraint 不变。

这不是“更密的 primitive grid”，而是测试：

> **root-unrecoverable 是否主要是 finite response support discretization / control-reachability miss，而非 natural-root semantics 本身错误。**

### 7.3 Promotion 纪律

V44 当前仍是 probe，不应该直接包装成 CCF-A contribution。Generic safety filter、recursive feasibility、backup-plan MPC 都已经成熟；2024 的 Safety Filter review 已系统总结运行时安全过滤框架，2026 的 Generalized Backup Plan-Constrained MPC 也直接研究 alternative-plan feasibility/multi-horizon backup support。FeAR 2023/2025 又已从 causal responsibility 角度研究一个 agent 对另一个 agent feasible action space 的压缩。

所以真正有 CCF-A 潜力的不是“bisection 找一个 responder acceleration”，而是：

> **natural-root-based social non-coercive feasibility × exact-blocker/root-conditioned physical recourse feasibility 的统一 option-set formulation，及其可解释 blocker→root→response witness。**

V44 如果成功，只说明 physical half 的 analytic support object 成立；之后还需要 learned representation、unseen final holdout、reactive agents 和 human-audited causal stress 共同把论文故事闭合。

---

## 8. 下一轮实验：从 48 scenes 压到最坏先 7 scenes

### Stage A — lost7（唯一立即要跑的科学实验）

V43 已有 3/10 rescue retained，另外 7 个丢失。要达到 >=5/10，V44 必须在 lost7 中至少新增 2 个 rescue。

```text
lost7 newly rescued < 2/7 → STOP
lost7 newly rescued >=2/7 → continue
```

这是数学必要条件，不是降低标准。

### Stage B — retained3

确认 V44 没伤掉 V43 原先保留的 3 个，计算 rescue10 总保留：

```text
historical rescue retained <5/10 → STOP
>=5/10 → continue
```

### Stage C — induced9

```text
historical RVR induced avoided <7/9 → STOP
>=7/9 → continue
```

### Stage D — remaining29

只有前三阶段都 PASS，才跑 remaining29；将 `7+3+9+29=48` stitch 回原冻结 CF48，再运行**原六项完整 Gate**。

### 为什么不缩 80-step horizon

不建议暂时改成 30/40/60 steps，也不建议用 open-loop 替代上述 7/10/19-scene Gate。历史 V29 onward 的 first collision / switching / certificate→outcome failure 可在几十步后出现，最晚接近 rollout 末端；缩 horizon 会把“没有撞”误判成 rescue。

### 计算量

若 V44 在最常见的早期失败分支被 falsify：

```text
full CF48: 48 scene rollouts
V43 gate19: 19 scene rollouts
V44 lost7: 7 scene rollouts
```

相对 48 降 **85.4%**；相对 V43 的 19 再降 **63.2%**。并且无需重复 COWP/RVR/V42/V43 baseline，继续使用 immutable paired reference。

---

## 9. 下一步运行命令

```bash
cd COWP_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT_CN.sh sanity

# 只有 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT_CN.sh build_tfindex

# 第一阶段：只跑 7 scenes
bash NEXT_RUN_COMMANDS_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT_CN.sh lost7_parallel2
bash NEXT_RUN_COMMANDS_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT_CN.sh analyze_lost7
```

此时检查：

```text
outputs/v16_8_44_root_conditioned_control_reachable_responder_support/lost7_failfast_gate.json
→ lost7_gate.pass
```

如果不是 true，立即停止并把 `lost7_*merged.json + lost7_failfast_gate.json` 给我。

如果 true，再执行：

```bash
bash NEXT_RUN_COMMANDS_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT_CN.sh retained3_parallel2
bash NEXT_RUN_COMMANDS_V16_8_44_ROOT_CONDITIONED_CONTROL_REACHABLE_RESPONDER_SUPPORT_CN.sh analyze_rescue10
```

`rescue10_gate.pass=true` 后才运行 induced9；`induced9_gate.pass=true` 后才运行 remaining29/stitch/analyze_counterfactual48。launcher 内置 fail-closed 检查。

---

## 10. 代码验证

本地已完成：

- V44 dedicated tests：2/2 PASS；
- V42/V43/shift-closure selected regression：27/27 PASS；
- 原 V16.8.25→43R3 focused semantic/integrity suite：109/109 PASS；
- V44 `sanity` aggregate selected suite：29/29 PASS；
- Python compile / shell syntax：PASS；
- no `conventional_check=False` safety bypass：PASS。

本环境没有服务器 Waymax 数据/checkpoint runtime，因此没有虚报 V44 Waymax outcome。下一步唯一必要科学结果是 lost7。

---

## 11. 当前论文级结论

到 V43 为止，证据链可以写成：

```text
false-safe ≠ merely high social cost
→ RCOT/BCOT + protected-priority hard certificate 已形成强 social feasibility axis
→ physical collision 主要发生于 uncertified / zero-conventional regime
→ same bank 中存在可恢复动作，但 current-prefix / greedy temporal selectors 不可靠
→ V39 证明 control/shift-closed ego tube 是高精度 positive support
→ V42 证明允许 root-consistent interactive responses 能额外恢复 physical support，但 policy recall 不足
→ V43 排除 exact blocker indexing 是主要 recall 瓶颈
→ 当前瓶颈收紧到 retained natural root 上的 low-burden control-reachable responder recourse completeness
```

这使论文主线比“不断加 recovery heuristic”更清楚：

> **The planner should certify not only that ego preserves protected agents' natural low-burden options, but also that any interactive recovery it relies on is backed by root-consistent, control-reachable recourse for the exact physical blockers.**

如果 V44 lost7 / CF48 支持这一点，后续最值得做的不是继续扩 analytic search，而是把该 analytic certificate 变成可学习的 **root-conditioned recourse viability representation**，并在全新 final holdout + reactive-agent stress 中验证其 generalization 和 causal burden validity。
