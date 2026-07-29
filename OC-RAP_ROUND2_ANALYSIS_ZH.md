# OC-RAP 第二轮代码审查：混合训练、连续专家、双 A30 构建与安全断点续增

## 1. 结论摘要

1. **现有 `train_near_contact` 可以暂时不重建。** 约 1.88% 的 post-contact 混入本身很小；更重要的是旧实现把数据集目录名当作 hard expert id，导致所有样本被强制送入 near head。该 hard routing 已改为 observation-conditioned soft mixture-of-experts（MoE），因此这些边界/过渡样本可以作为连续性正则，而不是错误的 policy 标签。
2. **原实现不是三套完整 policy。** 它是一个共享 OC-RAP encoder/root/OC-MERO 主干，加两个 near/contact direct-value 辅助 head；Safe 主要走共享 OC-MERO。旧版在辅助 head 上做 hard bucket routing，现已改成连续软融合。
3. **不需要再新增一个显式 regime classifier。** 显式分类会引入阈值误差和离散跳变，也会削弱“recoverability 本身驱动动作准入”的论文叙事。修改后 router 只读取可观测的 scene-prefix representation，输出连续专家权重；不输入 dataset path、regime label 或 teacher future identity。
4. **near/contact 构建最慢的部分是 Waymax recovery teacher rollout。** 主要开销是候选 prefix × futures × recovery options × 40-step recovery rollout；不是 NPZ 写盘。原脚本还同时启动 6 个未绑定 GPU 的进程，造成两张 A30 争抢和诊断提前开始。现已改成每次两个任务、每张卡一个进程，并等待两者完成。
5. **数据集现在支持强语义安全的断点续增。** 新增 dataset-level semantic contract、原子 NPZ、scene-time completion marker、manifest 重建、损坏文件隔离、输出目录互斥锁。改变 teacher、future、regime threshold 或 quality contract 会直接拒绝续写。
6. **输出目录使用原名。** `train_safe`、`val_safe`、`test_safe`、`val_near_contact` 等不再加 `_v2`。train-safe 的双卡 shard 放在隐藏目录 `.train_safe_shards`，最终原子替换 `$OCRAP_ROOT/train_safe`。
7. **`run_v45_two_gpu_fast_commands.txt` 的 checker 已按正确口径工作。** Safe 合同检查 nominal candidates 中的 `nominal_normal_fraction`，而不是所有替代候选中的 `normal_fraction`。旧 diagnostics 需要用新代码重新生成，才能包含 nominal-only counts。

---

## 2. 是否保留现有 `train_near_contact`

### 可以保留，但要明确它的角色

旧 `train_near_contact` 更准确地说是 **near-contact-centered training mixture**，而不是纯净的 near-only evaluation set：

- 约 78.52% candidate 样本带 near-contact 标签；
- 约 1.88% 带 post-contact；
- 其余主要是同一 near-centered 构建流程中被保留的边界、low-headroom 或非 nominal 候选。

对测试集而言，regime purity 非常重要；对训练集而言，适量边界样本有助于学习连续决策面。真正不合理的是把这些样本全部通过目录名硬路由到 near policy。

### 保留它需要满足的条件

- val/test 仍应保持严格 regime contract，不用训练集的宽松口径替代测试纯度；
- 论文中不要把旧 train set 描述为 100% pure near-contact；可称为 near-contact-centered mixture；
- 训练与评测应按独立 scene 划分；
- v45 必须重训，因为模型结构从 hard expert routing 改为 soft MoE；
- 最终可做一个低成本 ablation：不重建 NPZ，只从 manifest 创建一个 near-label-only 视图，与完整 mixture 对比。这不是当前主流程的必要条件。

### 为什么 1.88% post-contact 不值得单独重建

这个比例不足以抵消完整重建的成本，而且 contact train set 本身提供了更强的 contact supervision。soft MoE 下，梯度由可观测特征与 recoverability targets 驱动，不再把这 1.88% 当成“near policy 身份标签”。

---

## 3. 原算法到底有几套 policy

### 修改前

- 一个共享 scene/prefix encoder；
- 一个共享 recovery-sufficient root / observation kernel / margin / OC-MERO 主干；
- 两个轻量 direct recovery value heads；
- near bucket 选择 head 0，contact bucket 选择 head 1；
- safe 不依赖 direct head 做主要动作准入。

因此它不是三套独立 policy，而是共享主干 + 两个辅助专家。问题在于 head 选择是离散的、由 bucket id 控制。

### 修改后

- 保留共享 OC-RAP 主干；
- 同时计算两个辅助专家输出；
- 用 observable scene-prefix feature 生成 softmax 权重；
- 对两个专家输出做连续加权；
- `bucket_id` 在 soft 模式中不会选择神经网络专家；
- 增加很小的 unsupervised load-balance loss，防止早期所有样本塌缩到一个专家；它不监督 regime label。

形式上可写为：

\[
\hat v(x)=\sum_{e=1}^{2}w_e(x)\hat v_e(x),\qquad
w(x)=\mathrm{softmax}(r_\phi(x)/T).
\]

这里的 \(x\) 是执行时可观测的 scene-prefix 表征，而不是 near/contact 标签。

### calibration bucket 仍然存在，但不是 policy switch

运行脚本仍按 Safe/Near/Contact 分层保存 gamma、macro allowlist 和风险上界。这些是 **安全校准 strata**，用于控制不同压力区域的 false admission，不是让模型识别一个类别后切换整套 policy。论文中应这样表述，避免把 novelty 变成 regime classification。

---

## 4. 构建为什么慢

### 主要复杂度

near-contact 的典型设置：

- 24 个 candidate prefixes；
- replay + 2 reactive + 8 targeted，约 11 个 futures；
- 12 个 recovery options；
- 4 秒 recovery horizon × 10 Hz = 40 个 Waymax steps；
- balanced two-pass 还可能多次尝试 artifact/non-artifact prefixes。

contact 使用约 13 个 futures，并且 biased times、candidate attempts 更多。

未经缓存的量级接近：

\[
N_{prefix}\times N_{future}\times N_{option}\times 40.
\]

代码已有 post-prefix state、teacher metric 和 JIT scan 缓存，因此实际小于该上界，但 teacher margins 仍是绝对主开销。

### 已做的精确加速

这些修改不改变 teacher label 语义，可安全用于续跑：

- 两张 A30 每次只运行两个构建进程，每卡一个；
- `CUDA_VISIBLE_DEVICES` 正确绑定；
- 启动前验证 CUDA-enabled JAX 能看到 GPU；
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免每个进程抢占整张卡；
- recovery horizon 的 40-step loop 使用 `jax.jit + lax.scan`；
- 显式开启 env cache、post-prefix rollout cache、teacher metric cache、identical rollout cache；
- 新增 future materialization cache：共享同一 Waymax rollout state 的 latent metadata branches 不再重复 trajectory extraction 和 metric summary；
- profiling 不再触发每个 scene-time 重写整个 manifest，避免 O(N²) I/O；
- 双卡脚本会等待当前 pair 完成后再启动下一 pair，诊断只在所有构建结束后运行。

### 可选的 screened-hybrid 加速

`teacher_rollout_top_k_options > 0` 会先用 structural teacher 筛选，再只对 top-k 和强制 mode 做 Waymax rollout。它能显著减少 recovery scans，但属于 teacher contract 变化：

- **不能直接追加到已经用 `top_k=0` 构建的 partial 目录；** 新 resume contract 会拒绝；
- 仅适用于删除相应输出后的 clean rebuild；
- 论文中要报告这一设置，或在小规模 full-teacher audit set 上验证标签一致率。

建议 clean-build 起点：top-k=6，并强制关键 modes：

- near：`stop,brake_lane,lateral_escape,yield_rejoin`；
- contact：`brake_lane,mitigate_contact,post_contact_stabilize,avoid_secondary`。

默认脚本仍为 `top_k=0`，可无语义变化地续跑当前 partial 数据。

---

## 5. 安全断点续增机制

### 保证

1. 每个 NPZ 先写同目录临时文件，flush 后 `os.replace`；不会留下半个正式 NPZ。
2. resume 启动时验证已有 NPZ schema；损坏文件移动到 quarantine，不会当作已完成样本。
3. manifest 缺行时从 NPZ 元数据重建；重复 manifest row 按文件名去重。
4. 每个 scene-time 完整处理后写 completion marker，空组/不足组也能避免反复计算。
5. `resume_contract.json` 保存 generator version 和 semantic config fingerprint。
6. `max_scenarios`、worker index、stride、start offset 只属于 scan scope；扩大扫描预算不会破坏 fingerprint。
7. teacher backend、top-k、future kinds、thresholds、quality gates 等发生变化会拒绝续写。
8. `.build.lock` 禁止两个进程并发写同一个目录。双卡必须写不同 output/shard 目录。
9. shard merge 先生成完整临时目录，再原子替换最终输出，避免旧 samples 残留。

### 旧 partial 数据的首次接管

旧目录没有 `resume_contract.json`，默认会拒绝。首次必须显式：

```bash
--resume --adopt-resume-contract
```

这表示你确认当前命令和原构建语义一致。接管后，后续只用 `--resume`。

---

## 6. 推荐命令

### 6.1 直接重建 `train_safe`，输出原名

你会删除旧 `train_safe`，执行：

```bash
cd /path/to/OC-RAP_round2
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example

rm -rf "$OCRAP_ROOT/train_safe" "$OCRAP_ROOT/.train_safe_shards"

RESUME=1 \
GPU0=0 GPU1=1 \
RAW_PER_WORKER=6000 \
MIN_SAMPLES=15000 MAX_SAMPLES=20000 \
REQUIRE_JAX_GPU=1 \
bash scripts/rebuild_ocrap_train_safe_two_gpu.sh
```

若不足 15k，只提高 `RAW_PER_WORKER` 后重跑，不要删除 shard：

```bash
RESUME=1 RAW_PER_WORKER=9000 GPU0=0 GPU1=1 \
bash scripts/rebuild_ocrap_train_safe_two_gpu.sh
```

由于 scan-budget 不进入 semantic fingerprint，旧 shard 会安全跳过已完成 scene-times，只扫描更远的数据。

### 6.2 删除 `val_safe`，同时续跑旧 near/contact/test partial 数据

第一次使用新版 resume contract：

```bash
rm -rf "$OCRAP_ROOT/val_safe"

RESUME=1 \
ADOPT_LEGACY_RESUME=1 \
GPU0=0 GPU1=1 \
REQUIRE_JAX_GPU=1 \
NEAR_TEACHER_TOP_K=0 \
CONTACT_TEACHER_TOP_K=0 \
STRESS_COMPUTE_FUTURE_METRICS=true \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

之后再次扩大 RAW budget 或中断后恢复：

```bash
RESUME=1 \
ADOPT_LEGACY_RESUME=0 \
GPU0=0 GPU1=1 \
VAL_NEAR_RAW=1200 TEST_NEAR_RAW=1600 \
VAL_CONTACT_RAW=1200 TEST_CONTACT_RAW=1600 \
NEAR_TEACHER_TOP_K=0 CONTACT_TEACHER_TOP_K=0 \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

不要在同一 partial 目录中把 top-k 从 0 改为 6；contract 会阻止这种混写。

### 6.3 仅在 clean rebuild 时启用 screened-hybrid

```bash
rm -rf "$OCRAP_ROOT/val_near_contact" "$OCRAP_ROOT/test_near_contact" \
       "$OCRAP_ROOT/val_contact" "$OCRAP_ROOT/test_contact"

RESUME=1 ADOPT_LEGACY_RESUME=0 \
GPU0=0 GPU1=1 \
NEAR_TEACHER_TOP_K=6 \
NEAR_TEACHER_ROLLOUT_MODES=stop,brake_lane,lateral_escape,yield_rejoin \
CONTACT_TEACHER_TOP_K=6 \
CONTACT_TEACHER_ROLLOUT_MODES=brake_lane,mitigate_contact,post_contact_stabilize,avoid_secondary \
STRESS_COMPUTE_FUTURE_METRICS=false \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

最终论文优先使用 exact continuation，除非 screened-hybrid 已做 full-teacher audit。

### 6.4 查看实际瓶颈

构建时设置：

```bash
PROFILE_BUILD=1 RESUME=1 ... bash scripts/rebuild_ocrap_val_test_regimes.sh
```

汇总：

```bash
python tools/summarize_dataset_build_profile.py \
  "$OCRAP_ROOT/val_near_contact" \
  "$OCRAP_ROOT/test_near_contact" \
  "$OCRAP_ROOT/val_contact" \
  "$OCRAP_ROOT/test_contact" \
  --output "$OCRAP_ROOT/reports/build_speed_summary.json"
```

### 6.5 重新生成 diagnostics 并运行 v45 checker

```bash
mkdir -p "$OCRAP_ROOT/reports"
for d in train_safe val_safe test_safe val_near_contact test_near_contact val_contact test_contact; do
  python -m ocrap.cli diagnose \
    --dataset "$OCRAP_ROOT/$d" \
    --output "$OCRAP_ROOT/reports/diagnose_${d}.json"
done

export DATASET_DIAGNOSTICS_DIR="$OCRAP_ROOT/reports"
FINAL_RUN=0 bash run_v45_two_gpu_fast_commands.txt
```

Safe diagnostics 也可以显式传入 nominal contract：

```bash
python -m ocrap.cli diagnose \
  --dataset "$OCRAP_ROOT/val_safe" \
  --set dataset_quality.nominal_regime_dataset=true \
  --set 'dataset_quality.require_nominal_regimes=[normal]' \
  --output "$OCRAP_ROOT/reports/diagnose_val_safe.json"
```

---

## 7. 是否重新训练

- 重建 `train_safe` 后，共享 base 必须 refresh；旧 v30/v39 可以作为初始化，但不能作为最终冻结 backbone。
- soft MoE 修改了 v45 head 结构与 checkpoint state，因此 v45 必须重新训练。
- `train_near_contact` 可继续使用，无需等它重建。
- 最终流程：clean safe base refresh → soft-MoE v45 training → scene-disjoint calibration → val selection → test once。

---

## 8. 数据评判标准确认

`tools/check_regime_dataset_contract.py` 对 Safe 使用：

- `nominal_normal_fraction`；
- nominal count / scene-time group 完整性；
- forbidden near/post/artifact/prefix-collision contamination；
- all-candidate `candidate_normal_fraction` 仅做信息提示，不再要求大于 95%。

`run_v45_two_gpu_fast_commands.txt`：

- development：`--scope v45dev --mode development`；
- final：`--scope all --mode paper`。

因此 checker 口径已经正确。必须重新运行新版 `diagnose`，旧报告没有 nominal-only regime counts 时只能走 legacy fallback。

---

## 9. 验证结果

- Python compile：通过；
- Shell syntax：通过；
- 全部 pytest：94 passed；
- soft routing 测试：相同 observation 在不同 bucket id 下输出相同连续融合结果；
- resume fingerprint：扩大 max-scenarios / 改 worker scan scope 不变，改变 targeted futures 会变化；
- semantic mismatch：正确拒绝；
- legacy partial：无 adopt 时拒绝，显式 adopt 后可续跑；
- synthetic resume：无重复样本、completion markers 生效、manifest 与 samples 数量一致。
