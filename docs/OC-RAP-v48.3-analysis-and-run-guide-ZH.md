# OC-RAP v48.1 结果审计、Calibration 判断与 v48.3 优化说明

## 0. 结论先行

1. **当前 dedicated calibration 构建流程本身是按 Safe → Near-contact → Contact 串行分阶段执行的。** Near-contact 尚未完成时没有 Contact 日志是正常现象，不是漏启动。脚本甚至在状态信息中明确写了 `contact logs are not expected yet`。Contact 的 worker 4/5 只有在 Near worker 2/3 均正常结束后才会启动。
2. `ocrap_v48_1_existing_data_screening/dataset_splits/calibration_safe`、`calibration_near_contact`、`calibration_contact` 是本轮 **proxy calibration datasets**。它们由旧 `val_*` 按 scene 以 50%/50% 拆成 calibration 与 early-stopping dev，样本文件通过硬链接优先生成。它们不是后台从 WOMD validation 新建的 dedicated calibration。
3. 本轮输出目录虽然叫 `v48_1`，但日志和 checkpoint 配置表明：**训练实际已包含 v48.2 的工程修复、SRC 风险-覆盖正则、exact teacher-PCD sampler、encoder anchor、robust experts。** 因此不能再说这轮“没有进入训练”。Balanced 训练到第 4 epoch early-stop，Precision 训练到第 5 epoch early-stop。
4. v48 的工程修改大多生效；算法修改只在候选级产生部分信号，**没有解决策略级组内选择**。Near/Contact candidate AUC 可达约 0.70–0.82，但组内 top-1 correlation 约为 −0.11～+0.03，所有校准规则都选择 0 个动作，Natural gate 正确拒绝了全部候选。
5. 现在不应继续放宽阈值。主问题是模型仍按候选独立打分，训练损失虽是 setwise，但网络缺少候选间显式交互。为此代码升级为 **v48.3 OC-TRAC-NASC/RCD**：
   - NASC：Nominal-Anchored Set Context，显式使用 nominal 与 recovery set 的交换不变上下文；
   - RCD：Regret-Consistent Distillation，用完整 teacher advantage 分布与期望 top-1 regret 训练最终组合策略；
   - 保留 v48.2 SRC 的 harmful mass budget 与 positive coverage floor。
6. 在 dedicated Near/Contact 尚未完成前，**继续使用 proxy val split 做快速算法筛选是合理的**，但只能作为开发结果。最终论文表格必须切换到独立 WOMD validation 区间构建的 dedicated calibration，并在 test 上只做一次最终确认。

---

## 1. 论文与实现的统一理解

论文核心不是一般意义上的碰撞风险预测，而是区分：

- oracle recoverability：隐藏未来身份已知时，每个未来分别存在某个恢复动作；
- deployable recoverability：执行候选前缀后，仅根据当时可观测信息，观测不可区分的未来必须共享兼容恢复动作；
- oracle-to-deployable gap：oracle 分支可恢复但部署端无法选择正确恢复动作的差距。

OC-RAP 的主链条是：

1. recovery-sufficient latent roots；
2. post-prefix observation equivalence kernel；
3. affordance-conditioned recovery margins；
4. OC-MERO lower-tail aggregation；
5. calibrated action admission/CRISP selection。

当前代码对该链条做了大量实现，但 v40–v48 的训练重点已经逐渐转向一个直接 candidate recovery-value/opportunity/harm 分支。v48 的核心失败不在 OC-MERO teacher 标签完全无信号，而在“单个候选的可学习性”无法转换成“同组候选的正确相对选择”。

论文 Contact 数据当前标签为 `post_contact_counterfactual`，并禁止 `post_contact_observed`。所以论文应使用：

> contact-conditioned counterfactual recovery

而不应把当前实验直接描述为真实碰撞动力学下的 post-impact control。真实撞后控制结论还需要 observed-contact 数据或更高可信度的碰撞/轮胎/车身动力学。

---

## 2. WOMD v1.3.1 tf.Example 与 Waymax 的作用

### 2.1 WOMD Motion v1.3.1

- Motion Dataset 提供大规模对象轨迹、地图和交互场景。
- v1.3.1 新增公开 `sdc_paths`，包含候选未来路线的位置、弧长、road-part IDs 与 `on_route` 元数据。
- tf.Example 形式中包含过去、当前和未来状态张量以及 roadgraph/map 特征；你的数据构建器利用这些内容形成场景前缀、候选动作、未来 rollout 与 teacher 标签。
- `sdc_paths` 对 route progression、wrong-way/off-route 等指标特别重要。

### 2.2 Waymax

- Waymax 是基于 JAX 的轻量多智能体闭环仿真器，原生面向 WOMD。
- 它用 bounding-box 级状态表达行为仿真，适合规划/行为研究，不等价于高保真碰撞动力学。
- 官方基础指标包括 overlap、offroad、wrong-way、route-following、kinematic infeasibility 与 log divergence。
- 你的数据构建中使用 Waymax closed-loop rollout、replay/reactive/targeted futures 与 teacher margins，因此结果适合讨论行为级 recoverability 与 secondary-overlap 风险，不宜夸大为真实车辆碰撞后动力学验证。

---

## 3. Dedicated calibration 构建状态判断

### 3.1 为什么没有 Contact 日志不是问题

`build_v48_calibration_regimes.sh` 的执行顺序为：

1. Safe：worker 0、1；
2. Near-contact：worker 2、3；
3. Contact：worker 4、5；
4. merge/filter/overlap audit/diagnose；
5. 检查三个最终根目录的最小 scene 数。

脚本在 Near 阶段写入：

```text
workers 2,3; contact logs are not expected yet
```

所以，只要 Near 尚在运行，以下文件不存在是预期行为：

```text
$OUTPUT_ROOT/logs/calibration_contact_w4.log
$OUTPUT_ROOT/logs/calibration_contact_w5.log
```

另一个路径概念：

- 日志在 `$OUTPUT_ROOT/logs`；
- 数据 shard 在 `$OUTPUT_ROOT/shards/calibration_*_w*`。

如果你说“shard 中只有日志”，应再核对实际查看的是 `logs/` 还是 `shards/`。

### 3.2 Safe “构建完”可能有两种含义

- 若只有 `shards/calibration_safe_w0/manifest.csv` 与 `w1/manifest.csv`，表示 Safe 两个 worker shard 完成；
- 最终可直接使用的 `$OUTPUT_ROOT/calibration_safe` 要等 Safe/Near/Contact 六个 shard 都完成后，在 merge/filter 阶段生成。

因此，Near 未完成时通常还不能把 dedicated Safe 当成完整最终 calibration root 使用，除非你手工合并并完成排重审计。

### 3.3 目前能否判断 dedicated Safe 的性质正确

上传内容中没有 `/data0/.../OCRAP/calibration` 的实际 shard manifest、diagnose JSON 和 overlap audit，因此只能判断 **构建合同正确**，不能对已生成 Safe 样本的实际分布作最终确认。

Safe 最终至少要满足：

- 两个 shard manifest 正常且无损坏；
- merge 后 scene 数 ≥ 80；
- nominal regime 为 normal；
- 不含 near/contact/prefix collision/prefix contact；
- 与旧 val/test scene overlap 为 0；
- Waymax runtime coverage 正常；
- Safe 用于 nominal non-inferiority，不要求 targeted futures。

建议先执行：

```bash
cd /path/to/OC-RAP
python tools/inspect_calibration_build_v48.py \
  --root /data0/senzeyu2/dataset/OCRAP/calibration
```

重点查看：

- `status_file.state/stage`；
- `stages.near.complete`；
- `recommended_start_stage`；
- `contact_logs_expected_now`。

不要在原 controller 仍运行时启动第二个 controller；脚本有 flock 锁，第二个会被拒绝。

若 controller 已退出：

- Near 两个 manifest 未完成：`START_STAGE=near RESUME=1`；
- Near 两个 manifest 完成而 Contact 未开始：`START_STAGE=contact RESUME=1`；
- 六个 shard 都完成但最终 root 未生成：`START_STAGE=merge RESUME=1`。

---

## 4. `dataset_splits/calibration_*` 是否是 calibration dataset

是，但它们是 **proxy calibration**。

本轮 manifest 明确记录：

```json
"calibration_mode": "proxy_val_split",
"proxy_calibration_fraction": 0.50,
"proxy_calibration_seed": 4801,
"test_roots_used_during_screening": false
```

实际拆分：

| Proxy calibration | scenes | samples | 来源 |
|---|---:|---:|---|
| calibration_safe | 71 | 1,224 | val_safe |
| calibration_near_contact | 89 | 1,715 | val_near_contact |
| calibration_contact | 107 | 3,351 | val_contact |

Dev 与 calibration 之间 scene overlap 为 0：

- proxy dev：252 scenes；
- proxy calibration：267 scenes；
- overlap：0。

`samples/*.npz` 是实际样本文件，脚本优先 `os.link` 硬链接，失败后依次尝试 symlink、copy。它们共享原样本内容，但目录角色和 manifest 中的 `split_id` 已改成 `calibration`。上传的结果压缩包没有带入这些大体积/链接文件，只保留 manifest 和 provenance；这不影响判断服务器原目录中的 samples 身份。

它们与 dedicated calibration 的区别：

| 项目 | Proxy | Dedicated |
|---|---|---|
| 原始来源 | 现有 val_* | WOMD standard validation 的保留区间 |
| 用途 | 快速算法开发 | 最终论文校准 |
| 是否减少 dev | 是，val 被拆半 | 否，可保留完整 dev |
| 分布独立性 | 与旧 val 同源 | 更强 |
| 当前可用性 | 三个 regime 均可用 | Safe shard 完成，Near 未完，Contact 未启动 |

---

## 5. v48 修改是否生效

### 5.1 已确认生效的工程修改

1. **训练真正运行**
   - Balanced：4 epochs，best epoch 1；
   - Precision：5 epochs，best epoch 2；
   - 均正常 early-stop。

2. **两 GPU 并行与快速路径**
   - 日志显示 A30；
   - direct-only fast path、AMP/BF16、worker/prefetch 配置进入实际训练。

3. **exact teacher-PCD index/sampler**
   - 30,114 samples；
   - 3,800 scene-time groups；
   - teacher index 中正优势 group：442；
   - Near：210 positive groups / 84 positive scenes；
   - Contact：232 / 76；
   - sampler 实际识别 362 个受 macro allowlist 约束的 positive groups，并启用 boost=5.0；
   - 不存在“sampler 完全没采到正机会”的问题。

4. **encoder 解冻与 L2-SP anchor**
   - encoder 未冻结；
   - anchor weight=0.02；
   - 50 tensors、970,944 params 被约束。

5. **proxy calibration 隔离**
   - dev/calibration scene-disjoint；
   - screening 没有读取 test root。

6. **Natural gate**
   - 所有 candidate calibration 失败后，没有继续进入 test；
   - 这是正确保护，而不是脚本失败。

### 5.2 部分生效但没有解决核心问题的算法修改

| 修改 | 观察结果 | 判断 |
|---|---|---|
| nominal 显式 abstention 类 | 最终全部 abstain | 保护有效，但策略塌缩 |
| 三状态监督 | candidate AUC 有信号 | 候选分类部分有效 |
| harm head | AUC 约 0.50–0.55 | 基本接近随机，未学到可迁移 harm 排序 |
| robust experts | 输出/分歧均存在 | 工程生效，未改善 top-1 |
| SRC harmful budget + coverage | checkpoint 中已启用 | 未找到同时满足风险与覆盖的规则 |
| setwise CE/pairwise/top-rank | top-1 corr 仍接近 0 | 损失层面的 setwise 不足 |

### 5.3 一个重要的初始化损失

v47 checkpoint 加载时，direct heads 第一层从 1152 维变成 1308 维，相关权重 shape mismatch，因此 direct heads 被重新初始化。虽然共享 encoder 等 100 个 key 成功加载，但 v47 已学到的 candidate head 没有完整继承。

下一轮 v48.3 推荐从本轮 **Precision best checkpoint** 初始化：

```text
runs/ocrap_v48_1_existing_data_screening/candidates/precision/model_v48_trac_sr/best.pt
```

这样旧 direct heads 可以完整加载，仅 NASC 新模块重新初始化。

---

## 6. 当前结果与主指标

### 6.1 策略级筛选结果

| Variant | Regime | positive AUC | harm AUC | pred-teacher corr | group top-1 corr | verify selected |
|---|---|---:|---:|---:|---:|---:|
| Balanced | Near | 0.6957 | 0.5382 | 0.0368 | -0.0189 | 0 |
| Balanced | Contact | 0.7855 | 0.5034 | -0.0545 | -0.1097 | 0 |
| Precision | Near | 0.7289 | 0.5483 | 0.0836 | 0.0337 | 0 |
| Precision | Contact | 0.8216 | 0.5336 | 0.0447 | -0.0785 | 0 |

关键解释：

- AUC 是跨候选的二分类可分性；
- 部署选择要求在每个 scene-time 组内，把正确候选排第一；
- 当前两者明显脱节；
- Contact 的 AUC 最高，但 top-1 反而为负，说明模型更可能学习了跨场景/macro 的 shortcut，而不是组内相对恢复优势。

### 6.2 为什么当前无法评价闭环主指标

Natural gate 未通过，候选策略没有进入 test/closed-loop，所以当前并没有可信的：

- Safe collision/offroad/route progression non-inferiority；
- Near minimum clearance/TTC/DRS/FRA/ODG 改善；
- Contact secondary overlap/stable-stop/recontact 改善。

当前结果只能说明训练/校准前置条件与离线可学习性，不能声称闭环性能提升。

### 6.3 数据分布对泛化的影响

Near：

| split | r_dep mean | hard violation mean | harm_proxy mean |
|---|---:|---:|---:|
| train | -1.794 | 0.0894 | 0.0289 |
| val | -0.801 | 0.0087 | 0 |
| test | -0.690 | 0.0161 | 0 |

Contact：

| split | r_dep mean | hard violation mean | harm_proxy mean |
|---|---:|---:|---:|
| train | -1.792 | 0.0936 | 0.0283 |
| val | -0.561 | 0.0148 | 0 |
| test | -0.572 | 0.0212 | 0 |

训练集显著更恶劣，且 `harm_proxy` 在 val/test 完全退化为 0。这会使模型容易学习“训练分布严重性”而非组内恢复优势。用户明确暂不重构数据，因此 v48.3 只从算法上降低该问题，但最终上限仍可能受数据合同漂移限制。

---

## 7. CCF-A 稿件的内部 readiness 门槛

不存在通用的“CCF-A 数值录取线”。以下是建议作为内部 go/no-go 标准的门槛，而非官方标准。

### 7.1 Natural gate / 离线选择门槛

建议三个 seed 均满足：

- Near candidate positive AUC ≥ 0.78；Contact ≥ 0.82；
- group top-1 Spearman/Pearson correlation ≥ 0.20，理想 ≥ 0.30；
- verify precision Wilson LCB90 ≥ 0.60；
- verify harmful-group exposure UCB90 ≤ 0.10，主结果争取 ≤ 0.05；
- positive opportunity recall ≥ 0.35，主结果争取 ≥ 0.50；
- selected count 不少于 15–20/group fold，避免依赖极少样本；
- selected macro max share ≤ 0.70，避免单一 macro shortcut；
- fit→verify precision drop ≤ 15 percentage points。

当前最大差距不是 AUC，而是：

- top-1 corr 差约 0.17–0.31；
- recall 从目标 0.35–0.50 到当前 0；
- 没有任何可验证 precision。

### 7.2 Safe 主指标

内部建议：

- collision/offroad 不增加，paired 95% CI 上界不超过 +0.1～0.2 percentage point；
- route progression 相对下降 ≤ 0.5%；
- NUP 相对下降 ≤ 1%；
- jerk/yaw-rate p95 增幅 ≤ 5%；
- intervention episode rate ≤ 2%～3%；
- 至少 3 seeds、scene-paired bootstrap CI。

### 7.3 Near-contact 主指标

内部建议：

- collision rate 相对下降 ≥ 15%～25%，或有统计显著的绝对下降；
- min-clearance p05 提升 ≥ 0.20 m；
- min-TTC p05 提升 ≥ 0.20 s；
- near-contact exposure 降低 ≥ 15%；
- DRS 提升 ≥ 8 percentage points；
- PCD 提升 ≥ 0.03；
- FRA 降低 ≥ 30%；
- ODG 降低 ≥ 25%；
- harmful switch rate ≤ 5%～10%。

### 7.4 Contact-conditioned counterfactual recovery

内部建议：

- secondary overlap event rate 降低 ≥ 20%；
- re-contact count / overlap duration 降低 ≥ 20%；
- stable-stop rate 提升 ≥ 10 percentage points；
- time-to-stable-stop 降低 ≥ 10%；
- post-contact clearance 提升 ≥ 0.20 m；
- uncontrolled displacement 降低 ≥ 15%；
- route-rejoin rate 提升 ≥ 5 percentage points；
- harmful selection UCB90 ≤ 0.10。

CCF-A 竞争力还依赖：强外部 baseline、完整 ablation、统计显著性、算力/延迟、失败案例与限制，而不是只超过上述数值。

---

## 8. v48.3 新算法：OC-TRAC-NASC/RCD

### 8.1 NASC：Nominal-Anchored Set Context

v48.2 的 direct branch 对每个候选独立编码和打分。即使 loss 在 group 内计算，head 本身看不到其他候选。

NASC 对每个 scene-time group 构造：

- 当前候选 embedding；
- candidate − nominal 的相对 embedding；
- recovery candidates 相对 nominal 的 mean summary；
- recovery candidates 相对 nominal 的 max summary。

四者拼接后进入共享 adapter，再通过一个初始较小的可学习残差 gate 加回原 candidate feature。

性质：

- 对 recovery candidates 排列交换不变/等变；
- 显式 nominal anchoring；
- 无完整 group 或 singleton 时回退到旧 pointwise path；
- 不输入 hidden regime label；
- 与论文“candidate 的恢复价值是相对 nominal 与候选集合定义的”更一致。

### 8.2 RCD：Regret-Consistent Distillation

旧 setwise CE 只监督一个硬 argmax；多个相近正候选时，argmax 标签不稳定，而且 CE 不直接度量选错候选的 teacher regret。

RCD：

1. 由 teacher PCD(candidate) − PCD(nominal) 构造完整 teacher policy distribution；
2. 对模型部署时实际使用的 score + opportunity − harm 组合 logits 做 KL distillation；
3. 计算模型 policy 下的期望 teacher advantage；
4. 对 oracle best advantage − expected advantage 的超额 regret 做平方惩罚。

它与 SRC 互补：

- SRC 控制 harmful probability mass 与正机会 coverage；
- RCD 决定 coverage 应落在哪个候选上，并直接减少组内 top-1 regret。

### 8.3 Novelty 边界

单独使用 DeepSets/mean pooling 不构成足够 novelty。论文贡献应表述为：

> an observation-consistent, nominal-anchored candidate-set recovery policy whose admission distribution is calibrated by harmful-mass risk control and trained with teacher recoverability regret.

需要用以下 ablation 证明贡献：

1. v48.2 SRC；
2. NASC only；
3. RCD only；
4. NASC + RCD；
5. 去掉 harm/SRC；
6. pointwise head + 同样参数量对照。

论文还可增加一个简单命题：NASC 对同组 recovery candidate 排列是 permutation equivariant，并在缺少有效 recovery set 时退化为 pointwise nominal-preserving scorer。

---

## 9. 下一步实验

### 9.1 当前优先：Proxy calibration 筛选 v48.3

推荐从 v48.1 Precision best 初始化，以保留已获得的 candidate AUC：

```bash
cd /path/to/OC-RAP-v48.3

OUTPUTDIR=runs/ocrap_v48_3_nasc_rcd_proxy \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
INIT_CKPT=runs/ocrap_v48_1_existing_data_screening/candidates/precision/model_v48_trac_sr/best.pt \
CALIBRATION_MODE=proxy_val_split \
CALIBRATION_FRACTION=0.50 \
CALIBRATION_SEED=4801 \
BUILD_TRAIN=0 BUILD_CALIBRATION=0 STRICT_TRAIN_DATA_GATE=0 REUSE_TEACHER_INDEX=0 \
GPU0=0 GPU1=1 BATCH_SIZE=72 NUM_WORKERS=6 PREFETCH_FACTOR=2 \
EPOCHS=8 PATIENCE=2 \
SET_CONTEXT_ENABLED=true \
POLICY_DISTILL_WEIGHT=1.0 POLICY_REGRET_WEIGHT=1.0 \
bash run_v48_two_gpu_fast_commands.txt
```

为何 `REUSE_TEACHER_INDEX=0`：新 output directory 应生成与当前路径一致的 exact index，避免复用旧 output 中的绝对路径。

为什么 batch 先降到 72：NASC 增加 set-context 激活与约 1M 级 adapter 参数，先保证 A30 稳定；显存足够后再提高到 96。

### 9.2 第一轮 go/no-go

先看：

- Near top-1 corr 是否 ≥ 0.10；
- Contact 是否从负值转正；
- 至少一个 variant 在 fit 与 verify 均有 selected；
- harmful exposure UCB 没有明显恶化。

若仍全部 0 selection，不要降 gate；先跑 NASC-only/RCD-only ablation，定位是 context 还是 regret objective 未起作用。

### 9.3 Proxy 多 seed

主方向通过后，再用相同 checkpoint/配置至少测试：

```text
CALIBRATION_SEED=4801
CALIBRATION_SEED=4802
CALIBRATION_SEED=4803
```

若仅 4801 通过，说明规则仍过拟合 proxy split，不应进入 test。

### 9.4 Dedicated calibration 完成后

不必立刻重训。先用通过 proxy gate 的 best checkpoint 在 dedicated calibration 上重新校准：

```bash
CALIBRATION_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP/calibration \
CHECKPOINT=runs/ocrap_v48_3_nasc_rcd_proxy/candidates/<winner>/model_v48_trac_sr/best.pt \
bash scripts/recalibrate_v48_on_dedicated_set.sh
```

只有 dedicated fit/verify 也通过，才运行 Safe/Near/Contact test closed-loop。

---

## 10. Calibration 后台任务的建议操作

当前 Near 仍运行时只监控：

```bash
tail -f /data0/senzeyu2/dataset/OCRAP/calibration/logs/calibration_near_w2.log
tail -f /data0/senzeyu2/dataset/OCRAP/calibration/logs/calibration_near_w3.log
cat /data0/senzeyu2/dataset/OCRAP/calibration/calibration_build_status.json
```

controller 退出后：

```bash
python tools/inspect_calibration_build_v48.py \
  --root /data0/senzeyu2/dataset/OCRAP/calibration
```

按工具建议的 `recommended_start_stage` 恢复。例：

```bash
nohup env \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
OUTPUT_ROOT=/data0/senzeyu2/dataset/OCRAP/calibration \
GPU0=0 GPU1=1 \
CALIBRATION_START_INDEX=11000 PARTITION_STRIDE=6 \
SAFE_RAW_PER_WORKER=600 NEAR_RAW_PER_WORKER=700 CONTACT_RAW_PER_WORKER=700 \
MIN_CAL_SAFE_SCENES=80 MIN_CAL_NEAR_SCENES=120 MIN_CAL_CONTACT_SCENES=120 \
START_STAGE=near RESUME=1 RUN_DIAGNOSTICS=1 \
bash scripts/build_v48_calibration_regimes.sh \
>/data0/senzeyu2/dataset/OCRAP/calibration/logs/calibration_controller.log 2>&1 </dev/null &
```

将 `START_STAGE=near` 替换成 inspect 工具实际给出的阶段。

---

## 11. 本次代码验证

已完成：

- Python compileall：通过；
- 主训练 shell syntax：通过；
- 模型 set-context forward/backward smoke：通过；
- RCD loss forward/backward smoke：通过；
- 全部测试：**119 passed**；
- 新增测试：
  - NASC permutation equivariance；
  - singleton fallback；
  - RCD 对 teacher-best 候选的偏好。

未完成：

- 当前环境没有真实 WOMD/JAX/A30 运行；
- v48.3 是否改善 top-1、通过 Natural gate、改善闭环指标必须由下一轮服务器实验验证。

---

## 12. 论文投稿前必须补齐

1. 论文当前主结果表仍为占位符，尚不能投稿。
2. 明确三个 regime，而不是正文/附录中混杂五个 regime 的叙述。
3. Contact 改称 contact-conditioned counterfactual recovery。
4. Dedicated calibration 与 test 完全隔离。
5. 三个以上随机 seed、paired scene bootstrap CI。
6. 与 log replay、IDM/MPC proxy、risk-aware/backup/contingency 等强基线比较。
7. NASC/RCD/SRC 完整 ablation。
8. 运行时延迟、候选数/K/L 的复杂度分析。
9. 报告失败模式：无正机会、always-nominal、macro concentration、fit→verify collapse。
10. 不以放宽 Natural gate 获得表面闭环结果。
