# V16.8.45R1：RCRSO Schema / Support-Semantics Fidelity Repair

## 0. 最终判决

这次用户遇到的 `sidecar_smoke` 崩溃是 **V16.8.45 新增 sidecar builder 的数据 schema 使用错误**，不是 compact-5k 数据损坏，也不构成重建数据集的证据。

更重要的是，全面审计发现原 V45 除显式 `IndexError` 外还有数个 silent semantic-fidelity 问题，其中最严重的是：**online RCRSO 在 retained root 的 frozen static response domain 为空时，会在 learned proposal callback 之前被历史 support-preparation 逻辑直接判 `not ready`**。这会使 RCRSO 无法补齐它本来要解决的 proposal-completeness hole。

因此当前版本定义为：

> **V16.8.45R1 — engineering + semantic-fidelity repair**
>
> scientific hypothesis 仍然是 V45 RCRSO，不是 V46。

原 V45 没有产生可用科学结果：它在 sidecar 构建阶段就失败；因此无需保留任何 V45 Stage-0/lost7 结论，必须从 sidecar 开始重跑。

---

## 1. 用户报错的直接根因

错误：

```text
heading=np.arctan2(dd[:len(xy),1],dd[:len(xy),0])
IndexError: too many indices for array: array is 1-dimensional
```

formal tensor cache 的设计是保留 raw WOMD `tf.Example` 数组；`roadgraph_samples/xyz` / `roadgraph_samples/dir` 在 raw cache 中允许是 flat `[P*3]`。项目自己的 Waymax dataloader 已明确在 simulator 使用前把它们 reshape 为 `[P,3]`。

原 V45 `104_build_rcrso_sidecar.py` 绕过 Waymax dataloader直接读 NPZ，却只做：

```python
dd=np.asarray(d)
while dd.ndim>2: dd=dd[0]
heading=np.arctan2(dd[:,1],dd[:,0])
```

对 `[P*3]` 没有 reshape，因此服务器上的真实 raw cache 触发 1-D 二维索引错误。

### R1 修复

新增 `_reshape_roadgraph_vector_field`：

- 支持 `[P*3]`、`[P,3]`、`[1,P,3]`；
- `xyz/dir` point 数必须一致；
- valid/type 长度不足直接 fail loudly；
- 不对 malformed field 静默猜测。

我用与 formal cache contract 一致的 flat raw roadgraph fake cache 重新执行 `sidecar_smoke`，修复版成功生成 sidecar，未再出现该异常。

---

## 2. 全面审计发现并修复的其他工程/语义错误

### 2.1 Critical actor 的 track index / input row 混用

Scenario proto 的 `cowp/critical/track_index` 不保证等于 raw WOMD state tensor 的 row index。项目已有 `cowp/critical/input_index` 明确处理二者映射，model-facing 路径也应优先使用它。

原 V45 sidecar 直接用 `track_index` 读取 current state，可能不崩但监督对象错位。

R1：sidecar 优先 `input_index`；dataset alignment 在 object-id 不可重映射时保留已有 authoritative `input_index`，而不是回退覆盖成 track index。

### 2.2 Blocker heading 取错 state slot

项目 state layout 中 yaw/heading 使用 `state[6]`；原 RCRSO feature builder 使用 `state[2]`。

因为 train/online 可能同时错，所以属于 silent representation bug。R1 改为 `state[6]` 并加 regression。

### 2.3 Shifted environment 与 online certificate 不一致

原 sidecar current/shift environment 没完整复现 online causal successor：non-SDC actor 应按 CV 推进一个 `dt`。

R1 增加 one-step CV shifted environment，和 online current/shift verifier 对齐。

### 2.4 Sidecar roadgraph crop 可能制造 vacuous drivable truth

原 sidecar 为节省存储裁图时可能把有效 lane center 全裁掉。空 roadgraph 进入某些 predicate 会产生与 full map 不同的行为。

R1：保留 root/current/shift ego 周围足够大的局部 roadgraph；若局部没有 valid vehicle lane，则额外保留 nearest global valid lane，避免空图改变 roadgraph truth。

### 2.5 Teacher positive 未充分 replay hard verifier

V45 设计要求 network 只负责 proposal completeness，所有 positive 的 hard feasibility 必须由冻结 verifier 定义。原 sidecar 对 cached/V44 teacher controls 的重放语义不够严格，存在 label drift 风险。

R1：response label/V44 analytic control 都只视为 proposal seed；最终每个 control sequence 重新通过冻结：

```text
burden beta
+ roadgraph
+ Waymax kinematics
+ ego current safety
+ ego shift safety
+ environment current/shift safety
```

通过才是 `target_valid=True`；失败作为 hard negative，用于 ordering，不获得 certificate 权力。

### 2.6 Canonical root probability/mass 对齐

R1 优先使用 cache 中的 canonical root weight；缺失时调用项目 shared `canonical_root_weights`。retained-root contract保持：

- p_min/floor 语义冻结；
- minimum roots = 2；
- retained canonical mass >= 0.75；
- root dedup 与 online 一致。

### 2.7 一个 hypothesis 不能被 sidecar cap 截断半个 root set

`FullHypothesisRootCoverage` 是 universal retained-root object。若 sidecar max examples 在一个 hypothesis 的 root group 中途截断，会虚假改变 coverage。

R1 只允许在完整 hypothesis group 之间截断；不会留下 partial root set。

### 2.8 Transformer padding 未完整 mask

Environment set 有 padding。原 query cross-attention 的 memory padding 不完整，pad token 可能影响 learned query output。

R1 对 root/ego/environment/global memory 建立 padding mask，并加 padding-invariance regression。

### 2.9 Stage-0 baseline 与真实 online extension semantics 不一致

原 V45 Stage-0 更接近“RCRSO 替代 fixed bank”来计算 coverage，但真实目标应该是：

```text
先 exact-nest frozen V43；
V43 hard set empty 后：fixed verified profiles ∪ RCRSO hard-verified profiles
```

同时 V44 analytic baseline 必须遵守它历史上的 callback 条件，不能给 V44 credit 到当时线上根本无法进入 completion 的 static-support hole。

R1 Stage-0 已修成：

- baseline A = historical fixed bank；
- baseline B = historical V44 analytic extension semantics；
- V45R1 = fixed + learned verified union（仅在 nested V43 失败后的 extension pass）；
- ExactCSPCompletionRate 使用对应实际 profile domains。

### 2.10 原 V45 online implementation under-implements RCRSO proposal completeness

这是全面审计中最重要的 silent bug。

历史 `_prepare_interaction_response_support_np` 规定：一个 retained root 必须先从 frozen static response bank 找到至少一个 low-burden/roadgraph/kinematic profile，否则整个 actor support `ready=False`。然后 interaction certificate 在 learned callback 之前就 reject。

这对 V42/V43 是正确的历史语义，但对 V45 RCRSO 是错误的结构前置条件：**RCRSO 的科学目的正是补 proposal hole。**

R1 的正确结构：

```text
1. 完整运行 frozen V43 BC-IARE
2. V43 certificate 非空 -> 原样返回，RCRSO 不运行
3. V43 hard set 为空 -> V45R1 extension pass
4. retained root fixed static domain 为空时保留为 proposal hole
5. proposal hole 本身不是 feasible
6. RCRSO proposal 必须重新通过全部 frozen hard verifier 才能填入 root domain
7. 对已有 fixed profile 的 root，RCRSO verified profiles 也可以作为额外 domain members，帮助 exact CSP joint compatibility
8. exact CSP / beta / root mass / roadgraph / kinematics / current+shift / environment 均不放宽
```

所以 R1 修的是 **implementation fidelity / completeness access**，不是事后放宽 certificate。

### 2.11 Dynamic learned profile cache identity

RCRSO output 几何随 control knots 改变，不能只用一个循环复用的 `profile_index` 做 compatibility cache identity。

R1 的 RCRSO identity 绑定真实 knot/acceleration sequence；失败时以 trajectory bytes 做 exact fallback，避免类似 V44 dynamic-profile stale-cache bug 重现。

### 2.12 输出隔离与 fail-closed

R1 默认路径改为：

```text
recourse_sidecar_v16_8_45r1
outputs/v16_8_45r1_rcrso_operator
outputs/v16_8_45r1_rcrso_closed_loop
```

不要复用原 V45 sidecar/checkpoint/output。

closed-loop commands 现在同时要求：

```text
Stage-0 support gate PASS
AND
base equivalence PASS
```

否则 launcher 在 Waymax rollout 前停止。

---

## 3. 数据集重新审计：不是这次错误的根因

上传的 `formal_v16_8_24_compact_full_5k(1).zip` 是数据性质/manifest/profile/verifier审计包，不含服务器上的实际 tensor-cache NPZ bytes。因此无法仅从该 zip 读取发生 crash 的那一个 `roadgraph_samples/dir` 数组；但：

1. 用户服务器 traceback 已直接证明该数组在实际 cache 中是 1-D；
2. 项目 Waymax dataloader 源码明确把 flat roadgraph xyz/dir reshape 为 `[-1,3]`；
3. cache contract 本身保存 raw WOMD arrays。

因此 root cause 可以可靠归为 **V45 sidecar consumer schema bug**，不是 compact-5k corruption。

关键 split 性质：

| metric | train | val | heldout |
|---|---:|---:|---:|
| scenes | 5000 | 1000 | 1200 |
| audit-relevant pair rate | 0.429696 | 0.428630 | 0.429492 |
| protected PRIO-root coverage | 0.994533 | 0.993632 | 0.994651 |
| rootless | 0 | 0 | 0 |
| `<2 low-burden roots` | 0 | 0 | 0 |
| mechanism unauditable | 4.067% | 4.344% | 4.465% |
| critical-agent mean | 5.3846 | 5.3870 | 5.3383 |
| selected-cap saturation | 95.78% | 95.50% | 95.42% |
| PRIORITY_SMOOTH_YIELD acceptance | 20.35% | 20.65% | 21.87% |
| TERMINAL acceptance | 54.15% | 54.55% | 55.10% |

判断：**V45R1 继续 Freeze compact-5k，不重建。**

模型真正收敛后再检查的数据/support ceiling：critical-agent cap=6 的高饱和、PRIORITY_SMOOTH_YIELD / TERMINAL 较低 acceptance。当前不能用这些解释 sidecar crash 或 RCRSO proposal-completeness failure。

### 投稿前 provenance caveat

`verify_cache_train.json` 仍为 `pass=false`，唯一主要 reason 是 `irrelevant pair blockers=58243`；同时：5000/5000 inspected、valid scene rate=1、silent blockers=0、affected/conflict/retained root mismatch=0、canonical root weight mismatch=0。

因此现在不重建数据，但投稿 artifact 前必须：修 serialization/verifier accounting，或给出严格 cache→runtime semantic-equivalence proof。

---

## 4. V45R1 的算法对象保持不变

RCRSO 定义的是 set-valued recourse proposal correspondence：

```text
(exact blocker, retained natural root, blocker/control state,
 ego current/shift tube, frozen environment)
        -> K same-root longitudinal control-sequence proposals
        -> frozen hard verifier
        -> verified root recourse set
        -> original exact multi-root / multi-blocker CSP
```

网络永远不能：

- 改 root mass；
- 改 beta；
- 改 roadgraph threshold；
- 改 Waymax kinematics；
- 改 current/shift safety；
- 忽略 environment；
- 放宽 joint CSP；
- 把 predicted feasible probability 当 certificate。

这保持了 V45 最重要的理论结构：**learned proposal completeness 与 hard-certificate soundness 解耦。**

---

## 5. 所有预注册分支和 GO / STOP 条件

### Gate E0 — Engineering / schema fidelity

必须：

```text
sanity PASS
sidecar_smoke PASS
```

否则：**REPAIR ONLY**，禁止解释模型/算法结果。

### Stage 0A — Offline validation support gate

不跑 Waymax。

K 只在 validation sidecar 上从：

```text
K = {2, 4, 8, 16}
```

选择，规则：取达到 observed max `FullHypothesisRootCoverage` 的 95% plateau 的最小 K，然后冻结到 selected checkpoint。禁止用 lost7/CF48 选 K。

进入 closed loop 的 GO 条件：

```text
FullHypothesisRootCoverage(selected RCRSO)
  - max(frozen fixed baseline, historical V44 analytic baseline)
>= 3 percentage points

AND VerifiedRootRecall > 0
```

且所有 hard verifier semantics 0 regression。

失败：当前 RCRSO architecture **STOP before Waymax**。

### Gate E1 — Common-path equivalence

Stage-0 PASS 后才跑：

```text
base_equivalence16_parallel2
```

目标 contract：16 scenes / 1120 fields / 0 mismatch，analyzer `passed=true`。

失败：engineering regression，**REPAIR ONLY**。

### Stage 0B — frozen lost7

原冻结规则不变：

```text
new rescues >= 2/7 -> GO to retained3
new rescues <  2/7 -> current RCRSO architecture STOP
```

建议使用固定顺序 `2+2+3` progressive batches；只能根据“剩余 scene 数在数学上是否还能达到 2 rescues”提前停止，禁止 outcome-adaptive 重排。

### Stage 0C — retained3 / rescue10

lost7 PASS 后运行 retained3。

```text
historical RVR rescues retained >= 5/10 -> GO induced9
otherwise -> STOP/archive current architecture
```

### Stage 0D — induced9

```text
historical RVR induced collisions avoided >= 7/9 -> GO remaining29
otherwise -> STOP/archive
```

### Stage 1 — original frozen counterfactual48 conjunction

只有前面都 PASS，才跑 remaining29 并 stitch 原冻结 48 scenes。

六项必须同时满足：

1. old RVR rescues retained >= **5/10**；
2. old RVR induced avoided >= **7/9**；
3. relative to COWP net collisions removed >= **3 scenes**；
4. Kinematics net regression <= **1 scene**；
5. paired mean EP delta >= **-0.05**；
6. action-changing intervention > **0**。

任一失败：archive，不允许改阈值救结果。

### Stage 2 — fresh37 development generalization

CF48 PASS 后才运行。

必须同时：

1. no net collision harm；
2. no net CR harm；
3. offroad regression <=1 scene；
4. kinematics regression <=1 scene；
5. paired mean EP delta >= -0.03；
6. nonzero intervention。

失败：不进 exact200。

### Stage 3 — historical exact200

只作为 development confirmation，不是 publication holdout。不能再用于调 K/threshold/architecture。

### Publication gate

算法 freeze 后重新冻结从未参与 V25→V45 机制选择的 final set；至少 3 independent seeds + paired scene CI。strong causal-burden claim 仍需要 reactive-agent evaluation + human-audited false-safe stress set。

---

## 6. Learned-recourse family 的内部收敛 / 自动关闭规则

当前 V45R1 是该 family 的第一个可靠 implementation，**现在不能提前设计 V46**。

只有新的 preregistered architecture 相对上一版至少满足之一，才值得继续：

```text
FullHypothesisRootCoverage absolute gain >= 3 pp
OR
lost7 rescue count +1
```

同时必须：

```text
hard verifier semantics: 0 regression
common path: 0 mismatch
no beta/root/CSP/horizon relaxation
```

自动关闭：

- 连续两次 architecture 都既没有 >=3pp FHR coverage gain，也没有 +1 lost7 rescue -> close learned-recourse proposal family；
- 总共 3 个 preregistered architectures 都没有达到 lost7 >=2/7 -> close family；
- 任何版本只有通过放宽 hard verifier 才能提高 recall -> immediate STOP。

关闭后下一科学分支锁定为：

> **natural-root validity / interaction-model uncertainty decomposition**

而不是继续加 K、control knot、loss weight、beta、root mass 或 verifier tolerance。

---

## 7. 当前下一轮唯一允许的科学分支

### 现在锁定：V16.8.45R1 RCRSO fidelity rerun

顺序：

```text
sidecar_smoke
-> full train/val sidecar
-> RCRSO training
-> Stage-0 support
```

**Stage-0 之前不跑 Waymax。**

Stage-0 PASS 后：

```text
base_equivalence16
-> profile4 (engineering only)
-> progressive lost7 2+2+3
```

只有 lost7 >=2/7 才按 frozen retained3 / induced9 / remaining29 继续。

因此当前不允许：

- 设计 V46；
- 调 K 于 lost7；
- 改 p_min/floor/root count/mass/beta；
- 重建 compact-5k；
- 缩短 80-step；
- 加 learned hard shield；
- 放宽 environment/CSP；
- 把 accepted-path kinematics 与本轮混改。

---

## 8. 性能判断与下一次 profiling

已有可靠 V44R1 profiler：lost7 18,740 s，selection 约 99.33% policy runtime；candidate build ~0.6%，base model forward ~0.05%。因此性能优化重点不是 base COWP Torch forward。

V45R1 保留：

- 一次 RCRSO forward 批量输出 K proposals，替代 V44 1.35M online bisection profile evaluations 的算法形态；
- static burden/roadgraph/kinematics 前置；
- ego/environment expensive checks 后置；
- semantic cache identity；
- root scarcity / minimum-domain fail-fast；
- sidecar/snapshot offline architecture gate。

注意：R1 修复后 RCRSO 会真正进入之前被提前挡住的 proposal holes，并可补充 nonempty domain 的 CSP diversity，因此它可能比原“under-implemented V45”做更多有效 verification。不能预先承诺服务器 end-to-end speedup。Stage-0 PASS 后用 `profile4_parallel2` 测真实 wall-clock；它不参与 scientific promotion。

不要把 80 steps 改成 40/60。历史 closed-loop failure 可以到接近 step 78，缩 horizon 会系统性制造 false rescue。

---

## 9. 验证结果

最终工作树：

- V45R1 dedicated tests：22/22 PASS；
- V16.8.25→V45R1 focused semantic/integrity launcher sanity：136/136 PASS；
- Python compileall：PASS；
- launcher `bash -n`：PASS；
- `conventional_check=False` bypass scan：无新增 bypass；
- V45 proposer/sidecar/Stage-0：无 logged-future / online-GT access；
- flat raw WOMD roadgraph sidecar smoke：PASS；
- synthetic Stage-0 no-lift case：正确 fail-closed，exit code 4。

Full repository `pytest -q` 仍在三个历史 collection problems 上停止：缺少 external `ocrap` package 的两个 tests，以及已归档 `_recovery_bridge_viability_mask` historical API test。相同 collection failures 已在用户上传的 pristine V45 原包中复现，因此不是 R1 引入。

---

## 10. 推荐运行命令

```bash
cd COWP_V16_8_45R1_RCRSO_SEMANTIC_FIDELITY_REPAIR

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh make_ids

# only if TFExample index is missing
# bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh build_tfindex

# First reproduce the repaired schema path.
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh sidecar_smoke

# Full offline learned-recourse preparation.
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh build_sidecar_train_parallel4
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh build_sidecar_val_parallel2
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh train_rcrso
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh stage0_support
```

到这里先看 `stage0_support_gate.pass`。

只有 Stage-0 PASS：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh base_equivalence16_parallel2

# runtime-only
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh profile4_parallel2

# 推荐 fixed-order progressive lost7
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh lost7_batch1_parallel2
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh analyze_lost7_progressive
```

若 `continue_progressive=true`：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh lost7_batch2_parallel2
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh analyze_lost7_progressive
```

仍需继续：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh lost7_batch3_parallel2
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh analyze_lost7_progressive
```

或者一次跑完整 frozen lost7：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh lost7_parallel2
bash NEXT_RUN_COMMANDS_V16_8_45R1_RCRSO_CN.sh analyze_lost7
```

后续 launcher 已按 prereg gate fail-closed。

---

## 11. 下一轮最值得回传的文件

第一阶段只需要：

```text
outputs/v16_8_45r1_rcrso_operator/stage0_val_support_audit.json
outputs/v16_8_45r1_rcrso_operator/training_history.json
outputs/v16_8_45r1_rcrso_operator/rcrso_stage0_selected.pt  # 若需要复核 checkpoint provenance，可给 hash/metadata，不必优先传整个大文件
```

Stage-0 PASS 后再回传：

```text
equivalence16_cowp_vs_v16_8_29.json
profile4_* runtime/profile outputs
lost7_v45_progressive_gate.json 或 lost7_v45_failfast_gate.json
对应 lost7 merged/shards
```

最关键的下一判断不是先看 headline collision，而是：**RCRSO 是否先在 outcome-blind validation sidecar 上真实提升 universal retained-root verified coverage，并且这种 offline completeness 提升能否转化为 frozen lost7 的至少 2/7 new rescue。**
