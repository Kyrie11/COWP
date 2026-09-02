# V48.71 工程审计、科学归因与 V48.72 OC-IORW 设计落地报告

日期：2026-08-31  
目标：按 CCF-A 级别的可证伪、可归因与可复现标准，审计 V48.71 OC-BORW，收紧论文主线和 dominant bottleneck，并交付下一轮代码、预注册分支和无算法污染的加速。

## 0. 最终判决

**工程判决：V48.71 结果正确、完整、可靠，足够进行算法归因；不存在需要先修复并重跑 V48.71 的工程阻断。**

**科学判决：V48.71 OC-BORW = `STOP`。** 它没有达到 promotion 条件，也不能进入当前 Main 机制组合。失败不是 permissive relapse；harmful/teacher-infeasible pass 仍约为 `0.011–0.023`，说明模型仍然擅长拒绝，真正缺失的是跨 split 稳定、独立于 teacher floor 的 true-witness credibility 与 safe-positive admission。

**本轮最关键机制结论：** observation history / set-valued external-dynamics information 有真实、可复现的排序信号；但 V48.71 的 isotropic circumball 与 boundary-deficit localization 不是可 promotion 的实现。dominant bottleneck 已收紧为：

> **candidate/recovery-conditioned interaction geometry for observation-only external-dynamics reachability**

下一版已落地为 **V48.72 OC-IORW — Observation-Consistent Interaction-Oriented Reachability Witness**。它只改变外部可达集的交互几何，不改变 Stage-I、证书 sign、projection、top-K、threshold、teacher、dataset 或 boundary transport。

## 1. 论文 idea、motivation 与目标结果

论文的核心不是训练一个普通 feasibility classifier，而是解决 **oracle-to-deployable recoverability gap**：隐藏未来分支上分别存在 recovery，并不意味着部署时仅凭共享后缀观测可以选择一个合法 recovery。OC-MERO 要求 observation-compatible roots 共享可执行 recovery option；RIFA 再把 Stage-I candidate-vs-nominal 的相对证据与 Stage-II absolute deployable feasibility 做角色隔离和字典序组合。Safe/Near/Contact 是 supervision/evaluation strata，而不是运行时 regime id、router、expert、阈值或 proposal budget。

CCF-A 主线应表述为：在部分可观测动态下，如何构造 observation-consistent、actuator-realizable、set-valued 且具有物理边界语义的 recovery certificate，并在三个 regime 上用同一 policy 证明安全非干扰、近接触可恢复性和接触后恢复。

当前 TeX 仍有四项投稿前债务：

1. 文本停留在 V48.58 语义，尚未吸收 V48.64–V48.72 的 active-set、projection、projection fidelity 与 external-dynamics trust 证据链。
2. main closed-loop table 尚空，absolute source 未 GO，不能提前做三 regime SOTA/deployment 宣称。
3. teacher margin 的论文叙述过度简化为 active normalized slack 的纯 minimum，而真实 teacher 含 structural floors/overrides；必须如实统一。
4. 数据报告能支持 Waymax runtime、DRS/FRA/ODG、observation consistency 等性质，但 `paper_support` 对 WOMD-primary provenance 并非全真；投稿前需补 provenance 证据或收紧措辞。

## 2. 数据集性质（理解，不重构）

12 个正式 train/val/calibration/test × Safe/Near/Contact 报告均 `failures=[]`，shape/finite/schema 检查通过，OC-MERO 重算误差为约 `1e-7` 或更小，各报告 `leakage_scenes=[]`。

| Split | Regime | Samples | Scenes | Neg. deployable | Oracle artifact | Oracle recoverable | Alias incompat. |
|---|---|---:|---:|---:|---:|---:|---:|
| train | safe | 20,000 | 1,171 | 0.099 | 0.000 | 0.901 | 0.000 |
| train | near_contact | 13,324 | 600 | 0.553 | 0.189 | 0.636 | 0.161 |
| train | contact | 16,790 | 500 | 0.543 | 0.166 | 0.623 | 0.095 |
| val | safe | 2,328 | 132 | 0.073 | 0.000 | 0.927 | 0.000 |
| val | near_contact | 3,445 | 176 | 0.504 | 0.246 | 0.742 | 0.204 |
| val | contact | 6,477 | 211 | 0.461 | 0.219 | 0.757 | 0.135 |
| calibration | safe | 2,544 | 135 | 0.053 | 0.000 | 0.947 | 0.000 |
| calibration | near_contact | 6,039 | 316 | 0.448 | 0.240 | 0.792 | 0.196 |
| calibration | contact | 16,843 | 543 | 0.417 | 0.212 | 0.795 | 0.142 |
| test | safe | 3,216 | 175 | 0.069 | 0.000 | 0.931 | 0.000 |
| test | near_contact | 4,723 | 250 | 0.488 | 0.244 | 0.756 | 0.209 |
| test | contact | 6,687 | 209 | 0.444 | 0.218 | 0.774 | 0.141 |

Near/Contact 具有约 `0.42–0.55` 的 negative-deployable 比例、约 `0.17–0.25` 的 oracle artifact、非零 alias incompatibility 和多 option diversity，足够测 absolute feasibility、oracle-to-deployable gap 与 witness selectivity。Safe 是高可恢复、零 oracle artifact/zero oracle gap 的 non-interference bucket，但仍有约 `0.05–0.10` 的 deployable negatives；不应拆成单独 Safe expert。Safe 中大量 targeted-future warning 是 replay/reactive nominal bucket 的结构属性，不是 V48.71 工程错误。

## 3. V48.71 工程门

顶层契约：

```text
valid                  = true
attribution_ready      = true
engineering_version    = v48.71.0-OC-BORW
errors                 = []
test_roots_read        = false
dataset_reconstruction = false
```

独立复核：H/J/K × balanced/precision 共 6 个训练均完成 21 epochs；6 个 state-isolation audit 全部有效；170 个共享 Stage-I tensors bitwise identical；唯一新增 trainable state 是 `direct_absolute_semantic_witness_gain[2]`（2 parameters）；schema=7；factor/variant isolation、runtime import provenance、source/reference reuse、protocol 与 scene-disjoint contract 均通过。RC20 是预注册 scientific gate failure，不是 RC30 engineering failure。

上传结果包未包含 `best.pt` binaries，因此当前离线环境不能再次加载二进制 checkpoint 重算 SHA。这是残余审计限制；训练完成标记、记录 checkpoint SHA、状态隔离、runtime provenance、artifact hashes 与 fail-closed sentinel 相互一致，不构成归因阻断。

## 4. 按 V48.71 预注册顺序裁决

| Gate | Result | Verdict |
|---|---:|:---:|
| H/J/K vs E70 positive-certificate sign/set | 8/8 exact | PASS |
| K−E source AUC > 0 | 2/8 | FAIL |
| K−E positive-cert probability AUC > 0 | 2/8 | FAIL |
| teacher-feasible retention > teacher-infeasible | 5/8 | FAIL |
| harmful/TI non-relapse | 8/8 | PASS |
| non-floor K−E AUC > 0 | 4/8 | FAIL |
| K−B full-source AUC > 0 | 2/8 | FAIL |
| K−P66 safe-positive pass | 8/8 Δ=0 | FAIL |
| Formal status | `STOP` | **STOP** |

| Variant | Split | K−E AUC | K−E cert-prob AUC | TF>TI | non-floor K−E | K−B AUC |
|---|---|---:|---:|:---:|---:|---:|
| balanced | dev_near | +0.009472 | +0.032124 | ✗ | +0.000000 | -0.000466 |
| balanced | dev_contact | -0.001184 | -0.005472 | ✗ | -0.005421 | +0.001350 |
| balanced | certificate_near | -0.008737 | -0.025349 | ✓ | +0.002862 | -0.009194 |
| balanced | certificate_contact | -0.000666 | -0.001511 | ✓ | +0.002201 | -0.002060 |
| precision | dev_near | +0.009528 | +0.033537 | ✗ | +0.000000 | -0.000428 |
| precision | dev_contact | -0.000831 | -0.004188 | ✓ | -0.004993 | +0.001694 |
| precision | certificate_near | -0.008675 | -0.025349 | ✓ | +0.002848 | -0.008624 |
| precision | certificate_contact | -0.000519 | -0.001091 | ✓ | +0.002169 | -0.001692 |

四个事实同时成立：certificate sign/set 没有被隐藏删除；K 只在 dev-Near 有主排序正信号；non-floor 结果在 dev-Contact 为负、certificate 只有约 `+0.0022–0.0029`；safe-positive admission 对 P66 完全没有提升。因此不能从 aggregate AUC 或高 abstention precision 越级宣称机制成功。

## 5. 2×2 factorial 机制归因

| Variant | Split | H−E | J−E | K−H | K−J | Interaction |
|---|---|---:|---:|---:|---:|---:|
| balanced | dev_near | -0.006522 | +0.008657 | +0.015994 | +0.000815 | +0.007337 |
| balanced | dev_contact | -0.001525 | -0.003033 | +0.000341 | +0.001849 | +0.003374 |
| balanced | certificate_near | -0.019204 | +0.001529 | +0.010467 | -0.010267 | +0.008937 |
| balanced | certificate_contact | -0.002110 | +0.000577 | +0.001444 | -0.001243 | +0.000866 |
| precision | dev_near | -0.006455 | +0.008672 | +0.015983 | +0.000856 | +0.007311 |
| precision | dev_contact | -0.001769 | -0.002803 | +0.000939 | +0.001973 | +0.003742 |
| precision | certificate_near | -0.019061 | +0.001192 | +0.010386 | -0.009867 | +0.009194 |
| precision | certificate_contact | -0.001983 | +0.000462 | +0.001464 | -0.000981 | +0.001002 |

计数：H−E 正 0/8；J−E 正 6/8；K−H 正 8/8；K−J 正 4/8；factorial interaction 正 8/8。

**H71 boundary localization 明确失败。** H−E 8/8 为负，certificate-Near 约 `−0.019`。当前 `max_t[d_safe-clearance_alt(t)]_+` 没有形成 selective credibility，而是在错误 strata 提升 support；不允许继续做权重/阈值 sweep。

**J71 history tube 的 premise 有效、geometry 失败。** J−E 6/8 为正，证明 history-derived set-valued information 有稳定信号；但 component-wise extrema 的 L2 circumball 会收费切向 acceleration spread、组合未共同出现的 Cartesian corners，并让任何旧 acceleration 在未来第一时刻立即生效。因此可 promotion 的是历史集合数据通路与假设，而不是 J 的圆球实现。

**K71 interaction 为正不等于 Main 成功。** K−H 8/8 为正，但 K−J 仅 4/8 且两个 certificate splits 都为负。它说明 history set 相对 current point 有信息，不足以通过 trust/non-floor/full-source gates。

## 6. 模型各层状态

| Layer | Maturity | Current conclusion |
|---|---|---|
| Problem formulation / OC-MERO | mature | 核心论文主线，保留 |
| One regime-agnostic policy | mature | Safe/Near/Contact 不做 router |
| Dataset protocol | attribution-ready | provenance 文本仍需补齐 |
| Stage-I roots/proposal/relative evidence | stable/frozen | 不是当前 bottleneck |
| Common witness / non-compensatory semantics | validated | 保留 |
| Active-set alignment | validated repair | 保留 |
| Route / persistent re-entry | useful semantics | 保留诊断，不单独当 hard solution |
| Actuator projection | validated primitive | 必须保留 |
| Projection severity | validated soft trust | 保留 |
| Demand-only forgiveness | falsified | 禁止 |
| Raw CV/CA disagreement | ordering signal only | 非 selective physical trust |
| V48.71 boundary deficit | falsified | 移出 Main |
| V48.71 isotropic history tube | premise signal / geometry fail | 方向化集合替换 |
| Absolute boundary transport | real downstream debt | trust+non-floor GO 前冻结 |
| Relative-ranker/deployment | not yet authorized | full-source GO 后再开 |
| Safe non-interference/final SOTA | not yet authorized | full-source GO 后评测 |

模型已经学到“怎样构造 observation-consistent、actuator-realizable recovery，并强力拒绝 false witnesses”；尚未学到“针对某条具体 recovery，外部 uncertainty 中哪一部分真正能够侵蚀 signed reserve”。这就是当前 dominant bottleneck。

## 7. V48.72 OC-IORW 算法

对 projected ego recovery `x_e(t)` 和 agent CV center `x_j^CV(t)`：

```text
r_j(t) = x_e(t) − x_j^CV(t)
n_j(t) = r_j(t) / ||r_j(t)||
h_A(n) = max_{a∈A} nᵀa
d_lower(t,j) = ||r_j(t)|| − g(t) h_Aj(n_j(t)) − r_ego − r_agent,j
```

其中 `g(t)` 沿用既有 acceleration-hold displacement coefficient。由 `||r−ga|| ≥ nᵀ(r−ga)`，这是 recovery-conditioned conservative lower clearance。与各向同性半径不同，它只惩罚能够沿具体 line-of-sight 把外部 agent 推向 recovery trajectory 的 acceleration component。

| Arm | Acceleration set | Causal question |
|---|---|---|
| J71 historical | component box 的 isotropic circumball | frozen baseline |
| L72_BOX_SUPPORT | observed component box + zero；`h=nᵀc+|n|ᵀw` | 是否主要失败于 direction-blind/tangential overcharge |
| M72/Main OC-IORW | empirical joint hull `conv({0,aτ})`；`h=maxτ nᵀaτ` | 是否还受未观测 Cartesian corners 影响 |

L/M 使用同一 schema-8 20-D materialized features，只有模型消费 coordinate 不同。historical signed CV certificate 保持不变；trust multiplier 严格为正，不能改变 positive-certificate sign/set。仍只训练两个 gains。

### 预注册顺序

1. L/J、M/J、M/L certificate sign/set 必须 8/8 exact。
2. 先判 L−J（direction blindness），再判 M−L（unobserved joint corners），最后判 M−J Main。
3. Trust GO：M−J source AUC 与 positive-cert probability AUC 各至少 6/8 为正；TF retention>TI 至少 6/8；harmful/TI 8/8 不 relapse。
4. Non-floor physical GO：排除 exact-0.5 feasible rows 后 M−J AUC 至少 6/8 为正。
5. Full-source GO：M−B 8/8 为正且至少 6/8 ≥+0.01；safe-positive 对 P66 8/8 提升且至少 6/8 ≥+0.05；历史 selectivity 保持。
6. trust+non-floor GO 但 source STOP，才允许下一轮 trust-conditioned boundary transport；label GO/non-floor STOP 则先做 teacher truth-contract adjudication；box/hull STOP 则进入 observation-only interaction-response dynamics，禁止调 scale/threshold/horizon。

## 8. Changelog 禁区复核

V48.72 没有重开任何历史无效方向。继续禁止：threshold/LR/horizon/feature-weight grids；无 support 证据的 proposal/top-K/option expansion；generic AFE/MLP；candidate-only/compensatory replacement；per-option negative veto；quantifier sweep；regime router/expert/threshold/budget；broad encoder/root/margin retraining；privileged future distillation；class-local/path-stop Main；post-hoc hard control veto；hard CV/CA min；demand-only forgiveness；raw disagreement 直接当 physical trust；trust/non-floor GO 前 boundary transport；source GO 前 relative ranker/deployment。

本轮新增禁止：不再 sweep isotropic tube radius/scale；不再 sweep H71 boundary deficit；不因 J 的 6/8 正 AUC 直接把 J 加入最终组合；不把 geometry 与 temporal response 放进同一 Main。

## 9. 运行性能与等价优化

V48.71 的主要开销在 dataset materialization，而不是 2-parameter head：H71 约 `390.26–390.40 s`，J71 约 `468.63–468.68 s`，K71 约 `486.97–487.06 s`。旧 telemetry 中所谓 cache hit 进程可能大部分时间在等待同 key 文件锁。

已落地：schema-8 L/M 共用 tensor/cache key；box/hull 共用一次 CV geometry；empirical support vectorized；cold-cache 支持保序并行构建（launcher 默认 8 workers，历史默认仍为 1）；telemetry 分离 lock wait 与 build time；active shell 统一 LF。

64-sample × 2 arms old/new equivalence：bitwise mismatches=0，max abs diff=0.0；L/M materialized tensor mismatch=0。80-call synthetic microbenchmark 为约 `1.024×`，只说明单样本 kernel 没有退化，不是 end-to-end speed claim。结构上会消除第二个 L/M cold rebuild；在 V48.71 同规模日志中相当于避免一次约 469–487 秒的重建，目标机并行 speedup 由新 telemetry 实测。

## 10. 工程验证

```text
V48.72 focused tests                  11 / 11 PASS
V48.46 → V48.72 relevant regression 212 / 212 PASS
compileall src + tools + tests               PASS
recursive shell bash -n              150 / 150 PASS
runtime-code preflight                       valid=true
schema-8 L/M contract                        PASS
L/M tensor/cache-key identity                PASS
serial / 8-thread ordered exactness          PASS
```

这里不把未执行的 GPU 训练包装成 V48.72 科学成功。当前验证只证明：代码实现了预注册干预、保持状态/证书隔离，并且性能修改不改变 features/order。

## 11. 下一步指令

```bash
cd /home/senzeyu2/code
unzip -o /path/to/OC-RAP-v48.72-OC-IORW-drop-in.zip
cd /home/senzeyu2/code/OC-RAP

export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

python tools/check_v48_72_runtime_code_contract.py \
  --repo "$PWD" \
  --output /home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.72-runtime-code-contract.manual.json

GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_v48_72_dcp_drfc_bcde_rifa_iorw_two_gpu.sh
```

运行后必须上传 Main 与 audits；为了独立判定 `M−L`，同时上传 L/box arm：

```text
/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_72_dcp_drfc_bcde_rifa_iorw_main.zip
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.72-OC-IORW-audits.zip
/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_72_dcp_drfc_bcde_rifa_iorw_box.zip
```

## 12. 一句话论文结论

> V48.71 证明 observation history 中存在 external-dynamics credibility signal，同时反证“各向同性历史圆球 + 当前 boundary deficit”作为 selective physical trust；V48.72 因而只验证一个更窄、更因果的问题：不确定性是否必须沿具体 recovery interaction geometry 被度量。
