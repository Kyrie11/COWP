# COWP v16.8.1 全量算法审计、代码修复与下一阶段方案

## 0. 审计范围与结论边界

本次审计覆盖：

- 论文 `interactive_planning_v16_7_revised.tex`；
- 完整代码包 `COWP.zip`；
- `大模型建议.md` 与 `ALGORITHM_CHANGELOG.md`；
- `cowp_v16_8_pipeline_v9labels_seed2026.zip`；
- `cowp_v16_8_rcot_v9base_seed2026.zip`。

结论分为三类：

1. **已由上传结果确认**：数据覆盖、诊断统计、停止位置和错误日志；
2. **已由本地代码验证**：公式修复、张量形状、checkpoint 兼容、shell 执行链、单元测试与 CPU forward/loss smoke；
3. **必须回服务器验证**：GPU 收敛、learned-offline mechanism gate、真实 Waymax probe/full、最终论文主表与 SOTA 声明。

当前版本不能诚实地宣称已达到 SOTA。它完成的是在继续耗费 GPU 之前必须完成的“定义—标签—模型—损失—证书—执行链”一致性修复，并将论文最有价值的 novelty 收敛到可检验的 RCOT 机制。

---

## 1. 论文核心 idea 的复原

论文要解决的不是普通碰撞风险分类，而是如下 false-safe 现象：ego 轨迹表面无碰撞，但仅因为具有优先权或平等协商地位的其他交通参与者被迫急刹、让出原有间隙或放弃其自然交互意图。

核心对象为：

- **protected relation**：根据路权/优先级确定必须保护的交互对象；
- **natural roots**：非受迫条件下稳定的行为根及其概率质量；
- **same-root transport**：ego 候选与自然根冲突后，是否还能在不改变空间/拓扑 maneuver 的前提下，通过有限纵向时序调整恢复；
- **option-preservation ratio, OPR**：受保护 agent 的自然选择概率质量被保留或恢复的比例；
- **burden tail**：冲突根上最低安全响应的超预算负担尾部；
- **hard non-coercive feasibility**：对所有 protected agents 同时满足 same-root recovery、OPR 和 burden certificate。

论文中的关键传输公式为：

```text
s_ikm = (1 - c_ikm) r_ikm + c_ikm q_ikm
O_i   = sum_m p_tilde_im s_ikm
```

其中 `q_ikm` 是冲突根的低负担 same-root recovery 概率。它不是普通“候选是否安全”的分类概率。

---

## 2. 问题 1：当前数据集是否足以支撑训练和测试

### 2.1 可以支撑的部分

当前 `v9 raw base + v16.8 RCOT overlay` 对本轮机制开发是足够的：

- train raw/overlay：`20,440 / 20,440`；
- val raw/overlay：`5,013 / 5,013`；
- overlay `error_count=0`；
- train/val 文件名交集为 0；
- sampled scenes 中 natural、response、witness、planner core、Waymax outcome 必要字段覆盖率均为 100%；
- critical mapping 未映射数为 0；
- response root 越界数为 0；
- selected Waymax rollout 成功率为 1.0；
- train/val transport label 的 aggregate conflict 与 OPR 自一致性通过。

候选分布也足以训练当前机制：每场景平均约 50.59 个有效候选，固定 12 个候选具有已附着的 Waymax rollout，约 81.1% train 和 79.3% val 场景同时包含 physical safe/unsafe 样本。

因此当前数据可以用于：

- natural basis 的继续复用与诊断；
- RCOT/root transport 监督；
- witness、BCOT、selector 的 learned-offline 机制验证；
- 训练完成后的真实 Waymax online probe/full。

### 2.2 不能支撑的主张

当前数据**不能**单独支撑以下结论：

1. **不能作为 fresh v15/v16 causal-label dataset。** sampled cache 中缺少 `obs_contamination`、`map_compliant`、`map_distance_max`、`map_verified` 等新协议物化字段，因此只能作为 v9 base 上的 RCOT/certificate isolation。
2. **不能训练或评价 log-divergence。** finite logdiv coverage 为 0；将缺失值当真零会制造错误监督。当前配置已保持 `outcome_logdiv=0`。
3. **不能把 cached Waymax outcome 当作无偏全候选闭环证据。** 只有约 23.7% 的有效候选有 rollout outcome，而且这些候选由旧选择流程决定；它们适合辅助训练/诊断，不适合证明完整候选空间上的闭环优势。
4. **不能替代 reactive non-ego closed loop。** 上传结果显示实际 non-ego protocol 仍为 logged replay；最终论文的交互安全主张必须由真实 Waymax/响应式仿真或更强交互协议支撑。

### 2.3 最终判断

- **本轮不重建数据是合理的。** 现有 overlay 足够验证 RCOT 定义和证书是否学得起来。
- **论文主表前仍需要新数据或新在线评测。** 若 mechanism gate 通过，再只针对 validation/困难子集构建 fresh causal labels，避免现在就重做全部训练集。

---

## 3. 问题 2：算法层面的主要问题与修复

### 3.1 最严重问题：模型前向遗漏冲突恢复项

原标签/论文采用：

```text
s = (1-c)r + cq
```

但原模型前向的 OPR 实际只用了：

```text
s_model = (1-c)r
```

这会产生三个连锁后果：

- 冲突根即使能 same-root recovery，模型 OPR 仍把它算作丢失；
- retention head 被迫代偿本应由 recovery head 表达的概率；
- RootTransport AUPRC、OPR calibration 和 hard certificate 同时被污染。

**已修复：** `cowp/models/set_transport_head.py` 现在显式计算：

```python
transported_root_prob = retain_prob + conflict_prob * mode_recovery_prob
opr = (canonical_root_weight * transported_root_prob).sum(-1)
```

并加入公式级回归测试。

### 3.2 `q` 与最低安全负担 `b*` 的语义混淆

`q=0` 只表示不存在 `b* <= beta` 的低负担恢复，不表示完全没有安全响应。例如某根可能存在 `b*=1.2` 的高负担安全响应；把它直接替换为“无响应 sentinel=2.0”会夸大 burden tail。

**已修复：** 对每个 candidate–agent–root，使用共享 root-conditioned latent 的两个独立输出：

- `q_ikm`：低负担 same-root recovery 概率；
- `b*_ikm`：最低 same-root safe burden，范围 `[0,2]`。

两者各自直接监督，并只在高置信冲突根上加入：

```text
q ≈ sigmoid((beta - b*) / T)
```

的双向停止梯度一致性正则。这样保持语义相关，但不错误地令二者代数等价。

### 3.3 全局 response bank 不应定义 root recoverability

旧路径是“全局通用 response bank → top-R 截断 → 事后 root assignment”。这会把响应槽覆盖不足误判为根不可恢复，尤其伤害转弯、合流及不同速度时序的根。

**已修复：** primary certificate 使用 candidate–root 直接预测的 `q` 与 `b*`；通用 response bank 仅保留为：

- 轨迹可视化；
- 辅助重构；
- root assignment 辅助消融。

它不再定义 OPR 或 root-CVaR。

### 3.4 概率质量测度在标签、训练适配器和推理间不一致

论文包含 `p_min` 截断和概率 floor，但模型推理此前直接 softmax；同一根在标签中可能被过滤/重归一化，而推理仍保留不同质量，造成证书阈值不可校准。

**已修复：** 三处统一使用同一 canonical measure：

1. `p < p_min` 的根先从支持集移除；
2. surviving mass 重新归一化；
3. 只在 active support 上施加 `epsilon_p` floor smoothing；
4. 若全部低于阈值，则保留全部根作为防空集 fallback。

当前默认：`p_min=0.03`，`epsilon_p=0.02`。

### 3.5 recovery loss 被大量无关非冲突根稀释

全根 BCE 中非冲突根数量大，容易得到表面低 loss、实际冲突根 recall 很差的模型。

**已修复：** 主 recovery loss 改为 conflict-only BCE + conflict-conditioned ranking；全根 recovery BCE 权重置 0。直接 root burden 与 q–b* consistency 同步加入。

### 3.6 理论 SOTA 路线

当前 v16.8.1 完成“定义正确性”，但理论领先性还需要第二层：**calibrated non-coercion certificate**。

建议下一版形成以下一侧置信证书：

- `LCB(q_ikm)`：恢复概率下界；
- `UCB(b*_ikm)`：最低安全负担上界；
- `LCB(OPR_i)`：OPR 下界；
- `UCB(CVaR_i)`：burden tail 上界；
- 对同一候选全部 protected pairs 做 simultaneous calibration；
- 仅当所有下/上界同时通过时接受，否则 abstain/fallback。

在理想 calibration 假设下，可以把“错误接受 coercive 候选”的概率控制在预设 `delta`；在交互环境中还必须处理 ego policy 更新引起的 distribution shift，不能直接套用 IID conformal。该理论层应在本轮 mechanism gate 通过后实现，否则会把标签定义错误和不确定性校准问题混在一起。

---

## 4. 问题 3：可能误判算法优劣的工程问题

### 4.1 本轮训练/测试完全未开始的直接原因

两个上传结果包都停止在同一命令行参数错误：

```text
argument --data-protocol: invalid choice:
'v16_8_root_conditioned_overlay'
(choose from 'v15', 'v9_reuse')
```

launcher 已传入新协议名，但 `36_audit_causal_protocol.py` 未注册该枚举，因而在 causal audit 阶段退出。checkpoint 目录为空、learned-offline 目录为空，与此完全一致。

**已修复：** audit 支持 `v16_8_root_conditioned_overlay`，并区分：

- `engineering_pass`；
- `mechanism_overlay_protocol_pass`；
- `full_v15_label_protocol_pass`。

当前上传数据的预期结果是前两项 true、最后一项 false；这是诚实的机制开发协议，而非伪装成 fresh v15 数据。

### 4.2 `strict=False` 不能处理 head shape mismatch

自然 checkpoint 的 transport output 为 4 行，新模型为 5 行。PyTorch `load_state_dict(strict=False)` 仍会对同名张量形状不一致报错，导致恢复训练、learned-natural 诊断或 Waymax wrapper 在进入真正训练前失败。

**已修复：** 新增 `cowp/utils/checkpoint_compat.py`，只允许经过审计的 `mode_out` 4→5 行扩展：复制前四行，保留新 `b*` 行初始化；其他 shape mismatch 仍硬失败。

### 4.3 后台 launcher 父进程可能继续执行

父 shell 启动 nohup child 后若不 `exit 0`，父/子可能同时运行同一个 pipeline，产生重复训练、文件竞争和 provenance 混乱。

**已修复并测试锁定。**

### 4.4 分阶段重启丢失 external natural checkpoint

mechanism 运行时通过环境变量传入 v16.6 natural checkpoint，但新 shell 启动 probe/full 时这些变量会消失，而 OUT_ROOT 内没有 natural checkpoint，从而再次在 natural hard gate 前失败。

**已修复：** transfer manifest 现在记录 natural checkpoint 和 history 的路径及 SHA-256；新增 probe/full launcher 会从 manifest 恢复它们，并检查 transport/planner checkpoint 存在。

### 4.5 旧 OUT_ROOT 的严格 provenance 会拒绝新代码

这是正确行为，不应关闭。修复后必须使用新的：

```text
outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026
```

避免把旧错误代码生成的中间文件与新代码混用。

### 4.6 cached outcome 的选择偏差

每场景只有 12/约 50.6 个有效候选有 rollout outcome。若离线 selector 选中了未 replay 的候选，使用“已 replay 子集”评价会错判算法。当前方案把这些 outcome 限定为辅助训练/诊断，论文闭环指标必须来自 online Waymax。

---

## 5. 问题 4：保留、深化、删除或修改

### 5.1 应保留

- protected-priority semantics；
- typed natural decoder 与 natural source separation；
- OBS capacity 与自然选择质量的显式表示；
- stable root + probability mass；
- mass-aware root envelope；
- pair witness；
- source-aware multi-horizon alignment；
- monotone BCOT aggregator；
- PBTR、protected OPR、BTE-CVaR25；
- mechanism gate 与 planner freeze；
- learned-offline calibration/held-out 划分。

### 5.2 应继续深化

- RCOT timing family 的表达能力和覆盖率；
- conflict-conditioned root ranking；
- `q` 与 `b*` 的 aleatoric/epistemic uncertainty；
- simultaneous one-sided calibration；
- PBTR–coverage–efficiency frontier；
- protected relation 错误对最终证书的敏感性；
- mechanism 通过后的 proposal refinement；
- 交互 distribution shift 下的保证。

### 5.3 应从主算法删除或降级为辅助

- 全局 response bank 的 nearest-root `q`；
- 不含 `cq` 的 OPR；
- `safe == low-burden recovery`；
- 无条件 all-critical aggregate severe hard veto；
- planner 阶段继续更新 certificate；
- legacy flat candidate certificate；
- 仅靠放大 `BCOT_RISK_BUDGET` 解决低接受率；
- 用稀疏 cached Waymax outcome 证明闭环 SOTA；
- 在 mechanism 未通过前继续堆叠 planner/world-model 结构。

---

## 6. 已落地的代码修改

| 文件 | 修改 |
|---|---|
| `cowp/models/set_transport_head.py` | 5-channel root head；完整 `s=(1-c)r+cq`；direct `b*`；canonical root mass；generic bank 降为 auxiliary |
| `cowp/models/losses.py` | 统一 target mass；conflict-only recovery；direct root burden；q–b* consistency |
| `cowp/models/cowp_model.py` | 将 p_min、floor、CVaR 配置传入 certificate head |
| `cowp/scripts/26_augment_transport_labels.py` | future overlay 支持写入 canonical root weight |
| `cowp/scripts/27_diagnose_transport_labels.py` | 兼容 canonical weight 并维持旧 overlay 自一致性 |
| `cowp/data/cache_schema.py` | canonical root weight 可选 schema |
| `cowp/scripts/36_audit_causal_protocol.py` | 注册 overlay 协议并分离机制协议与 fresh-v15 协议 |
| `cowp/utils/checkpoint_compat.py` | audited 4→5 transport head checkpoint migration |
| `03_train.py`、`39_diagnose_learned_natural.py`、Waymax wrapper | 接入兼容加载器 |
| `NEXT_RUN_COMMANDS_V16_8_CN.sh` | 修复 background parent 重复执行；使用新 OUT_ROOT |
| `NEXT_RUN_COMMANDS_V16_8_MECHANISM_CN.sh` | manifest 记录 checkpoint + history |
| `NEXT_RUN_COMMANDS_V16_8_PROBE_CN.sh` | 新增独立 probe 阶段与自包含 checkpoint 恢复 |
| `NEXT_RUN_COMMANDS_V16_8_FULL_CN.sh` | full 与 probe 分离；强制先完成 probe |
| `interactive_planning_v16_7_revised.tex` | 同步 p_min/floor、q/b* 双头、RCOT primary/response bank auxiliary 定义 |

现有 v16.8 overlay 不需要因本次模型修复重建；训练适配器会现场重建 canonical target。只有未来重新生成 sidecar 时，才会额外物化 canonical root weight。

---

## 7. 本地验证结果

- `pytest`: **144 passed, 5 skipped**；
- Python `compileall`: PASS；
- 所有顶层 `.sh` 的 `bash -n`: PASS；
- TeX 未转义花括号计数平衡：PASS；
- CPU realistic forward/loss smoke：PASS；
- patched causal audit smoke：
  - `pass=true`；
  - `engineering_pass=true`；
  - `mechanism_overlay_protocol_pass=true`；
  - `full_v15_label_protocol_pass=false`。

当前环境无 CUDA、Waymax 和服务器数据，因此没有声称新模型已经收敛或 mechanism gate 已通过。

---

## 8. 下一轮必须观察的判定指标

只有以下条件同时成立，才进入在线 probe：

- `calibration_feasible=true`；
- `mechanism_verification.pass=true`；
- priority NCF recall ≥ 0.30；
- priority NCF precision ≥ 0.50；
- priority RootTransport AUPRC ≥ 0.50；
- witness AUPRC ≥ 0.60；
- priority BCOT AUPRC ≥ 0.50；
- accepted candidate rate ≥ 0.10；
- fallback ≤ 0.25；
- 相对 conventional 的 priority burden-transfer 改善 ≥ 0.03；
- global false-safe 改善 ≥ 0.03。

若 ranking 仍强而 RootTransport 不达标，应继续检查 direct q/b* representation、root feature 与 target confidence；不要先增大风险预算。

若 mechanism 通过但 Waymax 性能差，瓶颈才真正转移到 proposal generator / selector utility，此时再引入 proposal refinement、world model 或强 planner backbone。

---

## 9. CCF-A 投稿定位建议

不要把贡献表述为“又一个 end-to-end planner”。近期强方法主要在历史动量、world model、VLM reasoning、proposal generation/selection 与闭环分数上竞争，单纯堆网络结构很难形成清晰可守的理论 novelty。

更有辨识度的主线是：

> Collision-free is not sufficient. COWP explicitly protects the probability mass of legally/socially protected agents' natural interaction options through root-conditioned counterfactual transport, and rejects ego plans whose apparent safety is purchased by transferring tail burden to others.

建议论文贡献顺序：

1. coercive false-safe 的形式化与 protected relation；
2. natural-option probability transport；
3. RCOT 的 same-root recoverability 与 burden representation；
4. calibrated hard certificate；
5. PBTR/OPR/BTE-CVaR 及闭环验证。

最终 SOTA 需要同时满足：定义新、理论保证清楚、机制消融成立、强 baseline 公平、reactive closed loop 多 seed 有显著性。当前 v16.8.1 只完成前两项的关键正确性基础和第三项的大部分实现。
