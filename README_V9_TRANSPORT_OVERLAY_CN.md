# COWP 论文、数据、v8 结果与 v9 实现诊断报告

## 1. 结论先行

1. **论文的核心问题定义是成立的，而且具有形成 CCF-A 论文贡献的潜力。** COWP 关注的不是普通“礼让成本”，而是一个更强的安全缺陷：ego 轨迹虽然没有碰撞，但其安全依赖其他交通参与者急刹、强制让行、放弃合法间隙或丧失自然选择空间。论文将此定义为 false-safe planning，并把 non-coercive feasibility 提升为硬可行性约束。
2. **v8 的主要错误不是原始 tensor cache 缺文件、字段损坏或 train/val 分布错误。** 原始 natural、response、witness 和 planner 标签基本完整。真正的问题是 v8 对“逐自然模式的 option transport”只有聚合监督，模型可以用很多不真实的 mode 分解得到同一个 OPR/witness，因此核心中间机制不可辨识。
3. **对当前 v9 算法而言，transport_v9 监督是必要的；但不必再复制一份完整数据集。** 它本质上是由现有 cache 派生出的少量 `cowp/transport/*` 标签，而不是重新构建 WOMD，也不是重新执行 Waymax replay。可以使用小 sidecar 叠加到原始 cache 上。
4. **原第一阶段并非只包含数据构建。** 脚本先并行构建 train/val transport 标签，然后显式 `wait`；只有两者完成，才会依次执行双卡 transport DDP、双卡 planner DDP、offline gate 和 Waymax probe。你目前停留在 613/20000+，所以训练尚未开始是正常的依赖关系，而不是训练指令丢失。
5. **我已经重写增广与调度代码。** 新版本默认使用 overlay sidecar：输出目录中的基础 NPZ 是指向原 cache 的符号链接，新标签单独写入 `.transport_v9/`，dataset loader 自动合并。它还能继续复用当前已生成的 613 个完整 materialized NPZ，不需要删除重来。
6. **代码层面不能保证 SOTA。** 当前只能确认监督链路和执行逻辑已修复；是否达到 SOTA 必须由 gate、100/1000/5000 场景、3 个随机种子、置信区间及强基线共同证明。

---

## 2. 论文核心 idea、目标和算法 pipeline

### 2.1 核心 idea

论文将互动式规划的安全问题从：

> ego 能否避免碰撞？

改写为：

> ego 能否在不迫使其他交通参与者承担冲突解决负担的情况下保持安全？

其关键区分是：

- **conventional safe**：ego 本身无碰撞、可行；
- **non-coercive feasible**：每个关键交通参与者在 ego 候选执行后，仍保留足够的低负担安全响应；
- **false-safe**：conventional safe，但其他参与者的自然低负担选择被破坏，只能通过高负担 ceding response 保证 ego 安全。

这使 COWP 不只是“增加一个 social cost”，而是用可审计的 coercion witness 否决一类表面安全候选。这是论文最值得保留的贡献定位。

### 2.2 论文 pipeline

根据论文与代码，完整逻辑为：

1. 从 WOMD scene/proto 与 tf.Example 构建统一场景状态、地图和交互关系；
2. 构建 burden-oriented interaction graph，确定关键参与者及候选条件下的冲突关系；
3. 生成 ego 候选轨迹；
4. 对每个关键参与者生成 counterfactual natural alternatives，包括 observed、ego-neutral、priority-preserving 等来源；
5. 对每个 ego candidate 生成/评价 ego-conditioned response set；
6. 计算安全、低负担、自然支持、option retention、OPR、最小负担和 witness；
7. 先做 conventional safety 过滤，再做 non-coercive hard certificate；
8. 在通过证书的候选中按 planner utility 选择；若没有候选通过，则进入保守 fallback；
9. 训练时联合 natural、response、transport/witness、planner 和稀疏 Waymax outcome 辅助目标；
10. 在真实 Waymax online rollout 中评价 conventional 指标与 FSR/OPR/HBCR 等机制指标。

### 2.3 论文最终应证明的命题

论文不是只要“FSR 更低”就成立，而必须同时证明：

- COWP 真正学习了 same-root option transport，而不是依赖 candidate-level false-safe classifier；
- semantic-only 设置下，关闭 online physical outcome penalty 后，FSR/OPR/HBCR 至少有稳定改善；
- full COWP 在降低 FSR/HBCR 时，collision、offroad、wrong-way、route、kinematics 不恶化；
- fallback 不应通过牺牲大量 EP 来制造“安全改善”；
- 优势对不同 seed、不同交互类别和不同阈值稳定；
- witness 有足够精度、召回和校准，解释 token 与几何事件相符。

---

## 3. 数据集性质与现有 cache 是否有问题

现有数据流分为三层：

1. WOMD scenario proto：用于场景索引和 COWP counterfactual 标签构建；
2. WOMD tf.Example：用于 tensor cache 与 Waymax 环境；
3. Waymax candidate replay outcome：只覆盖平衡采样的约 12 个候选/scene，作为稀疏 collision/offroad 辅助标签。

`cache_sufficiency_full.json` 表明，诊断时：

- train 有 14,640 个可读 scene，val 有 5,013 个；
- natural/response/witness/planner 核心字段几乎 100% 完整；
- 每个 scene 平均约 50.6 个 valid candidates；
- 每个 scene 约 12 个候选有 Waymax replay；
- Waymax outcome 对所有 valid candidates 的覆盖率只有约 23.4%–23.7%；
- finite log-divergence 覆盖为 0；
- train/val 的候选数量、unsafe 比例和 replay coverage 相近，且 scenario ID 无重叠。

因此：

- 你现在看到 train 大于 14,640、val 大于 5,013，并不自动表示错误；旧 JSON 是当时 cache 的快照，你已经说明当前正式构建规模更大；
- 原 cache 足以继续 natural/response/witness/planner 的基础训练；
- collision/offroad outcome 只能作稀疏辅助，不可作为唯一 planner 目标；
- logdiv 必须继续禁用，不能把缺失值当成真实 0；
- 真实 online Waymax evaluation 不依赖 train candidate replay 完整覆盖。

---

## 4. v8 结果说明了什么

v8 的关键离线结果为：

| 指标 | Conventional | COWP v8 |
|---|---:|---:|
| EP ↑ | 0.3891 | 0.2218 |
| Fallback ↓ | 0.1043 | 0.3585 |
| FSR ↓ | 0.6624 | 0.5899 |
| OPR ↑ | 0.7413 | 0.7905 |
| HBCR ↓ | 0.3980 | 0.2320 |
| Selected False-Safe ↓ | 0.5933 | 0.3784 |

同时：

- Witness AUPRC 约 0.4300；
- Candidate False-Safe AUPRC 约 0.8243；
- accepted NCF recall 约 0.0727；
- fallback 约 0.3585；
- mechanism gate 为 `pass=false`；
- 因 gate 失败，完整 100 场景 probe 没有继续。

正确解读是：

- 证书已经连接到 selector，且确实降低了 selected false-safe 和 HBCR；
- 但 witness/option-transport 远弱于 candidate classifier；
- 模型不能准确保留真正 non-coercive 的候选，导致过度拒绝和高 fallback；
- EP 大幅下降，说明当前收益不能被视为有效闭环优势；
- v8 只证明“证书会影响选择”，没有证明“核心 transport mechanism 被正确学习”。

---

## 5. transport_v9 到底是什么，是否必须生成

### 5.1 它不是修复“标签文件不完整”

原始 cache 已经有：

- natural alternatives；
- response trajectories、safe/low-burden/burden；
- aggregate witness、OPR；
- candidate false-safe、NCF 等标签。

问题在于这些主要是 aggregate labels。假设一个 agent 有多个 natural modes，模型只被要求输出最终 OPR/witness，那么不同的逐 mode conflict/retain 分解可能得到相同 aggregate 值。模型没有充分理由学习真实对应关系：

`固定 natural primitive -> candidate 是否破坏该 primitive -> 是否通过同 root 的低负担 response 恢复`。

### 5.2 v9 新增的监督

transport_v9 派生以下关键标签：

- `cowp/transport/mode_valid`
- `cowp/transport/mode_conflict`
- `cowp/transport/mode_retained_low_safe`
- `cowp/transport/response_root_index`
- `cowp/transport/response_is_min_burden`
- `cowp/transport/root_recovery_mass`

它们让 direct mode conflict、mode retention、same-root response recovery 和 minimum-burden response loss 可以真正训练。

### 5.3 是否必须

结论分两层：

- **如果继续使用 v9 的 Primitive-Indexed Same-Root Option Transport 作为主算法：必须有这些派生监督。** 否则新增 loss 会缺失或恒为 0，v9 退回 v8 的不可辨识状态。
- **不必生成一份完整复制的数据集。** 这些字段可以作为很小的 sidecar 附着在原 NPZ 上。也不需要重新跑 WOMD proto label build、tf.Example tensor cache 或 Waymax candidate replay。

所以，必须的是“新监督”，不是“完整新数据副本”。

---

## 6. 为什么原构建特别慢

原 `26_augment_transport_labels.py` 有四个主要瓶颈：

1. 每个 scene 加载 NPZ 中所有字段，包括大体积 state、roadgraph 和与 transport 无关的张量；
2. 每个 scene 将所有原始字段连同 6 个新标签重新写成完整 NPZ，产生大量读写和磁盘空间开销；
3. 对 candidate × agent × natural mode 重复调用几何和 burden 计算；
4. 一次性向 `ProcessPoolExecutor` 提交所有 future，且 20 workers 与 BLAS/OpenMP 多线程叠加，容易造成 CPU 过度订阅和随机 I/O 拥塞。

因此，613/20000+ 的慢速并不意味着标签算法本身完全不可运行，更多是数据组织和并行策略不合理。

---

## 7. 已完成的代码优化

### 7.1 Overlay sidecar

新脚本默认：

- `--storage-mode overlay`；
- 在 `TRAIN_CACHE/.transport_v9/` 中只写 6 个新标签；
- 在 `TRAIN_CACHE/` 中为每个基础 NPZ 创建指向 `RAW_TRAIN_CACHE` 的符号链接；
- `COWPNpzDataset` 自动读取基础 NPZ，再覆盖/补充 sidecar 字段；
- 对 train、offline eval 和 Waymax evaluator 保持原有 `--cache-dir` 接口，不需要改训练命令。

### 7.2 选择性读取

只读取 transport 推导所需字段，不再加载 roadgraph 等无关数组。

### 7.3 等价计算优化

- 对 candidate/root 组合先做向量化、严格负判定 broad phase；只有可能接近的 pair 才调用原始 `unsafe_between`；
- natural root 的 intrinsic burden 只计算一次；若 pair 不触发 unsafe，其 candidate-conditioned risk 项严格为 0；
- response 到 natural root 的距离分配改为批量 NumPy 运算；
- min-burden response 按 root 聚合。

### 7.4 可恢复执行

- 当前已经生成的 613 个完整 transport NPZ 会被识别为 `skipped_materialized` 并继续复用；
- 新生成场景使用 sidecar；完整文件与 overlay 文件可以混合加载；
- summary 记录 total/completed/error，只有全部成功才标记 `complete=true`；
- 中断后重启会跳过已完成 sidecar。

### 7.5 双 GPU 调度

- transport DDP 使用 GPU 0+1；
- planner DDP 依赖 transport checkpoint，因此不能与 transport 同时训练；
- 第一轮 Waymax：COWP 与 conventional 分占 GPU 0/1；
- 第二轮 Waymax：semantic-only COWP 与 Pareto ablation 分占 GPU 0/1；
- full evaluation：conventional 与 planner-score-only 并行，随后 COWP 两卡分 shard；
- CPU augmentation 阶段 GPU 空闲是合理的，因为这部分没有可直接复用的 GPU kernel，强行同时训练会使用不完整标签。

### 7.6 已完成验证

- Python `compileall` 通过；
- shell `bash -n` 通过；
- 18 个相关测试全部通过；
- 新旧 `_derive` 在 10 个随机合成 scene 上的 6 类 transport 标签逐元素完全一致；
- overlay loader 测试确认基础字段和 sidecar 字段可透明合并。

本环境没有你的完整 WOMD、服务器磁盘和 GPU runtime，因此无法给出真实硬件上的加速倍数，也没有替你完成实际训练或 Waymax 闭环。

---

## 8. 下一步正确执行顺序

### 8.1 先停止旧增广进程

部署新代码前，先查看：

```bash
pgrep -af 'cowp.scripts.26_augment_transport_labels|run_cowp_v9_dual_gpu.sh'
```

只终止你当前旧版 v9 driver/augmentation 对应 PID。不要删除现有 `tensor_cache_*_transport_v9`，其中 613 个 train 文件与完整 val 文件均可复用。

### 8.2 部署并测试

使用随附压缩包覆盖代码后，执行静态测试。详细命令见 `双GPU运行指令_v9_加速版.txt`。

### 8.3 推荐直接执行整阶段

推荐保留 `RUN_AUGMENT=1`。脚本会：

1. 检测 val 是否已完整；完整则跳过；
2. 复用 train 的 613 个旧文件；
3. 为剩余 scene 写 overlay sidecar；
4. cache 完整后自动开始双卡 transport；
5. transport 完成后自动双卡 planner；
6. offline gate 通过后才进行 online probe。

初始建议 worker：train=12、val=6，且 augmentation 子进程强制每 worker 使用单线程 BLAS。若 `iostat -xz 1` 显示磁盘长期 100% util 而 CPU 很低，降到 8；若 NVMe 很快且至少 32 个物理核心，可试 16。不要直接恢复 20 workers × 4 OpenMP threads。

---

## 9. 训练和 gate 必须观察的信号

训练日志中应存在并有学习趋势：

- `val/set_transport/mode_conflict`
- `val/set_transport/mode_retain`
- `val/set_transport/root_recovery`
- `val/set_transport/response`
- `val/response_aux/root`
- `val/response_aux/min_burden`

若缺失、NaN 或长期恒为 0，不应继续跑 1000/5000 Waymax。

Offline gate 至少要求：

- threshold selection points >= 3；
- Witness AUPRC >= 0.50；
- LearnedAcceptNCFRecall >= 0.25；
- LearnedAcceptedCandidateRate >= 0.08；
- FallbackRate <= 0.30；
- SelectedFalseSafeRate 相对 conventional 至少下降 0.03。

还需要重点看：

- conflict/retain 正负样本比例是否极端；
- per-root recovery 是否只集中在少数 mode；
- threshold sweep 是否形成真正的 Pareto 变化，而非 selection 不变；
- semantic-only probe 是否仍改善至少一个 FSR/OPR/HBCR；
- EP 下降是否由 fallback 或错误的 hard veto 主导。

---

## 10. 投稿 CCF-A 前必须修正的论文—代码不一致

### 10.1 Non-ego simulator protocol

论文写的是 logged replay、learned reactive、rule-reactive mixture；当前实际 evaluator 只控制 SDC，non-ego 是 logged/background evolution，配置中也明确 `reactive_mixture_implemented=false`。

投稿前必须二选一：

- 真正实现并验证 reactive non-ego mixture；
- 或把论文改成诚实的 log-replay closed-loop，并将 counterfactual stress set/human audit 作为独立证据。

否则会成为严重的实验协议问题。

### 10.2 Ego-neutral diffusion branch

论文附录声称 conditional denoising diffusion decoder；当前 `NaturalDecoder` 是 MLP trajectory head + mixture/source/priority heads，并非 diffusion。必须实现 diffusion，或修改论文表述。

### 10.3 Candidate generator

论文中的“graph-conditioned lattice-MPC”表述强于当前代码。当前主要是规则/运动学 primitive 与 terminal lattice variants。应准确描述，或实现真实优化式 MPC refinement。

### 10.4 外部基线

仓库中的 GameFormer/DTPP 是对当前 cache、候选生成器和预算的公平重实现，不是官方 checkpoint。论文必须明确：

- 实现来源；
- 输入和候选集；
- 训练预算；
- 是否复用 COWP candidate generator；
- 与官方论文设置的差异。

### 10.5 结果协议

达到有说服力的 CCF-A 水平，至少需要：

- 3 个训练 seed；
- 100 场景机制 probe；
- 1000 场景开发；
- 5000 场景最终一次性评估；
- paired bootstrap 95% CI；
- 按 merge/lane-change/crossing/unprotected-turn/car-following 分层；
- conventional、planner-score-only、soft burden、universal NCF、COWP；
- option transport、witness rejection、same-root recovery、semantic-only、outcome penalty 等消融；
- witness calibration 与错误案例人工审计；
- 不报告当前缺失的 logdiv 证据。

---

## 11. 最终判断

**新的 transport 监督必须保留，但“重新生成完整 transport_v9 数据集”并不必要。** 最合理的工程方案是复用原始 cache + 小型 overlay sidecar。这样既修复了 v8 的核心监督不可辨识问题，也避免为每个 scene 重写整份 NPZ。

你当前第一阶段尚未训练不是脚本遗漏，而是旧脚本在等待 train augmentation 完成。部署加速版后，可以无损续用当前 613 个 train transport 文件与已完成的 val cache，并在剩余标签完成后自动进入双卡训练、offline gate 和并行 probe。

从研究角度看，当前最重要的不是先调低 witness threshold，而是先确认 v9 direct transport losses 真正学习、semantic-only 机制成立，并解决论文与 evaluator/decoder/candidate generator 的协议不一致。只有这些通过，才值得扩大到 1000/5000 场景并讨论 SOTA。
