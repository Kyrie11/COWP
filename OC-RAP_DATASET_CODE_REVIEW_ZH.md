# OC-RAP 数据集、训练与评测代码审查报告

## 1. 结论摘要

这次失败不是单一原因，而是两个问题叠加：

1. **旧 `train_safe` 数据集确实不合格，必须重建。**
   - 5072 个样本、634 个 scene-time group、261 个场景；每组恰好 8 个候选，说明现有文件更像是完整构建出的一个小数据集，而不是简单的“中途停跑残片”。
   - `normal=0`；约 48.74% 被标为 `low_headroom`；还含 51 个 `prefix_collision` 和 17 个 `prefix_contact`。
   - 因此 `train_safe: normal_fraction=0.000` 不是诊断器误报，旧数据本身不满足 clean Safe 合同。

2. **`val_safe: normal_fraction=0.606` 主要是诊断口径错误，不等价于 val_safe 被污染。**
   - Safe 合同应是“每个 scene-time group 恰有一个 nominal，且 nominal 为 normal，并且全组无 near/post/artifact/prefix collision/contact 污染”。
   - 当前 builder 有意保留安全场景中的困难非 nominal 候选作为负样本；这些替代候选可不被标成 normal。
   - 旧 checker 用“所有候选中 normal 占比 ≥95%”检查 Safe，错误地把候选级比例当成场景级纯度。
   - 重构后的 `val_safe/test_safe` 的 forbidden contamination 为 0；60.6% 只是所有候选的 normal 比例。仍应使用补丁后的 diagnose 重新生成报告，获得 `nominal_counts` 后做最终确认。

最终建议：

- **现在可以用当前 near/contact 的 partial val/test 做开发性、机制性和 smoke test。**
- **最终论文结果前必须重建 train_safe、刷新共享 base、重新训练 v45 heads、重新校准，并扩大 stress test 独立场景数量。**
- 旧 v30/v39 checkpoint 可作为初始化，不能继续冻结为最终 backbone。

---

## 2. 论文方法与代码 pipeline 对齐

论文的核心问题是 oracle-to-deployable recoverability gap：不同隐藏未来可以分别存在恢复动作，但部署车辆若无法从动作执行后的观测区分这些未来，就不能使用隐藏分支身份选择不同动作。

实现链路应为：

1. 生成 nominal 与可执行替代 prefix；
2. 预测 recovery-sufficient latent roots；
3. 预测 post-prefix observation embedding，并形成 observation-equivalence kernel；
4. 对 root × recovery option 预测 signed margin；
5. OC-MERO 先在不可区分 roots 内要求共享恢复选项，再做 lower-tail 聚合，得到 `R_dep`、`R_orc` 和 gap；
6. CRISP 使用校准阈值做 action admission，在 nominal 合格时保持 nominal；
7. 训练损失包括 root、observation、margin、anti-oracle 和 utility preservation；
8. 评测以 FRA、ODG、DRS、NUP 为主，并在 contact 条件下补充 secondary collision、stable stop、yaw、route rejoin 与 harm 指标。

整体代码框架与论文想法基本一致，但当前数据分桶、专家路由和评测口径中有几处会影响论文结论，见第 7 节。

---

## 3. 当前数据集性质

| 数据集 | 样本 | 独立场景 | scene-time groups | normal | near | post | artifact | 主要判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train_safe | 5,072 | 261 | 634 | 0.0% | 0.0% | 0.0% | 0.0% | 真实不合格；含 prefix collision/contact |
| val_safe | 2,328 | 132 | 291 | 60.57% | 0.0% | 0.0% | 0.0% | 候选级 normal 60.6%，无 forbidden 污染；需 nominal 级复诊 |
| test_safe | 3,216 | 175 | 402 | 60.60% | 0.0% | 0.0% | 0.0% | 同上；safe 测试规模可用 |
| train_near_contact | 13,324 | 600 | 1,800 | 0.0% | 78.52% | 1.88% | 18.94% | 与新 val/test near 分布不完全对齐 |
| val_near_contact | 826 | 45 | 104 | 0.0% | 100% | 0.0% | 24.46% | 可做开发评测，独立场景仍偏少 |
| test_near_contact | 614 | 32 | 77 | 0.0% | 100% | 0.0% | 24.76% | 可做一次性 preliminary test，不足以支持精确论文结论 |
| train_contact | 16,432 | 490 | 1,957 | 0.0% | 78.51% | 100% | 16.63% | 量足；旧报告未区分 observed/counterfactual post-contact |
| val_contact | 754 | 25 | 84 | 0.0% | 52.52% | 100% | 22.02% | 可做 smoke/mechanism；独立场景少 |
| test_contact | 837 | 25 | 93 | 0.0% | 58.06% | 100% | 22.22% | 可做 preliminary；不能把 837 candidates 当 837 个独立样本 |

### 3.1 为什么 `train_safe` 只有 5072

原命令的理论上限是：

`800 scenarios × 3 times/scenario × 8 accepted prefixes = 19,200 samples`

实际只有 634 个被接受的 scene-time group，而且每组完整保留 8 个候选：

`634 × 8 = 5,072`

这说明主要瓶颈是：

- `max_scenarios=800` 的原始扫描预算偏小；
- uniform time、nominal/quality gate、WOMD 有效历史等筛掉了大量 scene-time；
- 旧命令没有显式要求 nominal normal，却又受到其他质量条件影响，最终得到的是小而不纯的集合；
- 仅从 report 不能 100% 排除进程提前结束，但数据恰好由完整 8-candidate groups 构成，不支持“只是写了一半文件”的主要解释。

要获得 15k–20k 样本，应增加**原始场景扫描预算**，而不是降低 normal 安全阈值。补丁脚本默认两 worker 各扫描 6000 个 raw scenarios，再按完整 scene-time group 截断至 15k–20k。

---

## 4. Safe 合同应如何定义

### 错误合同

```text
normal candidate samples / all candidate samples >= 0.95
```

该口径会把安全 scene-time 中刻意生成的困难替代动作当成数据污染。

### 正确合同

对每个 scene-time group：

1. `nominal_sample_count == 1`；
2. nominal 的 `regime_label.normal == true`；
3. nominal 不属于 near-contact、post-contact、oracle-artifact、prefix-collision、prefix-contact；
4. 全组不存在上述 forbidden 标签；
5. 至少保留 2 个候选，保留困难但合法的非 nominal 负样本；
6. 诊断中同时报告：
   - `nominal_normal_fraction`：合同指标；
   - `candidate_normal_fraction`：描述性指标，不作 95% 硬门槛。

因此，新的 Safe 目标应为：

- `nominal_normal_fraction >= 0.95`，最好 1.0；
- `forbidden_fraction == 0`；
- `nominal_sample_count == scene_time_groups`。

---

## 5. 是否需要重新训练

### 最终答案：需要。

原因：

- v30/v39 的共享表示、root/margin heads 与 safe nominal preservation 都看过旧 `train_safe`；
- 旧 safe 中 `normal=0` 且存在 low-headroom、prefix collision/contact；
- v45 当前主要训练 near/contact 轻量专家头，并冻结共享 base；如果继续冻结旧 base，重建后的 Safe 分布不会真正进入共享表示；
- 阈值、calibration 和 selector 也依赖模型分数分布，数据分布改变后必须重新校准。

推荐顺序：

1. 重建 `train_safe_v2`；
2. 用补丁 checker + manifest audit 验证；
3. 以旧 v30/v39 checkpoint 作为 initialization；
4. 至少解冻共享 encoder、root、observation 和 margin 相关模块，执行 clean-base refresh；
5. 基于 refresh 后 checkpoint 重训 v45 near/contact heads；
6. 按 scene-disjoint calibration 重新拟合阈值；
7. 在 val 上选模型；
8. 最后只运行一次 test。

### 可以先跑结果吗

可以，但必须标成 development-only：

- 使用旧 base；
- 训练 v45 near/contact heads；
- 在 partial val 上筛选；
- closed-loop 用可观测路由跑短程 probe；
- 不据此写最终论文主表。

补丁后的 `run_v45_two_gpu_fast_commands.txt` 已提供：

- `FINAL_RUN=0`：开发模式，允许 partial stress data，UCB 上限临时为 0.12；
- `FINAL_RUN=1`：最终模式，强制 clean-base marker、完整数据合同、论文级校准门槛；
- `RETRAIN_CLEAN_BASE=1`：先刷新共享 base；
- `RUN_HELDOUT_TEST=1`：模型选择完成后，一次性运行 test。

---

## 6. 当前 val/test 能否使用

### Safe

- val: 132 scenes；test: 175 scenes。
- 已足够做开发，并达到当前代码中建议的 100-scene paper 下限。
- 仍需用 patched diagnose 证明 nominal normal 纯度。

### Near-contact

- val: 45 scenes；test: 32 scenes。
- 可用于方向判断、零使用检查、机制性 gate 和 preliminary comparison。
- 不适合对 1%–5% 的小差异给出强结论。

### Contact

- val/test 各 25 个独立 scenes。
- 可做 smoke test、case study 和检测大回归；不适合作为最终主表唯一证据。

按最保守的二项比例近似，独立场景数对应的 95% 最大误差约为：

- n=25：±19.6%；
- n=32：±17.3%；
- n=45：±14.6%；
- n=132：±8.5%；
- n=175：±7.4%。

必须以 **scene 为 bootstrap 单元**，不能把同一 scene 的多个 time/candidate 当独立样本。

### Partial contact 对校准门槛的影响

原先 `harmful_group_exposure_ucb90 <= 0.06` 在小验证集上可能数学上不可达：即使 0 次 harmful，至少约 43 个被验证 group 才能使一侧 Wilson 上界低于 0.06。当前 contact val 只有 84 groups，scene-disjoint 对半后常常不足。

因此：

- 开发阶段临时使用 0.12；0 harmful 时约 20 groups 即可达到；
- 最终论文恢复 0.06，并扩充独立 contact scenes。

---

## 7. 额外算法与评测问题

### 7.1 离线 evaluator 存在 oracle regime routing

离线 evaluator 当前从数据集目录名设置 `active_bucket_name`，即：

- `val_near_contact` 直接调用 near expert；
- `val_contact` 直接调用 contact expert。

这给模型提供了评测分桶身份。它可作为“oracle router / expert upper bound”，但不能作为 deployable 主结果。

闭环 runner 默认设置 `selection.auto_regime_from_observation=true`，根据可观测 clearance/TTC 路由，这是正确的 deployable 路径。

建议报告两组结果：

1. `offline oracle-bucket diagnostic`：只看专家上限与候选排序；
2. `closed-loop observable-router result`：论文主结论，并报告 router confusion、regime transitions、head usage。

### 7.2 当前 contact 数据是 counterfactual contact surrogate

重构脚本要求 nominal 含 `post_contact_counterfactual`，同时禁止 `post_contact_observed`。因此当前 contact 集更准确的名称是：

- counterfactual contact；
- incipient-contact/contact-surrogate stress set。

它不是严格意义上“历史已经发生碰撞后的 post-contact dataset”。论文如果声称真实 post-contact stabilization，应另外构造：

- 在 Waymax 中执行至 first contact；
- 从 contact 后状态重新切 history；
- 令 `post_contact_observed=true`；
- 再评估 stabilize / avoid-secondary / route-rejoin。

当前集合仍可支持“动作会进入接触分支时的恢复可部署性”分析，但不要把它与真实 post-impact control 混称。

### 7.3 `train_near_contact` 与新 val/test 不对齐

旧 train near 中只有 78.52% candidates 带 near-contact 标签，并混入 1.88% post-contact；新 val/test near 为 100% near、0% post。

训练代码按数据集根目录推断 bucket，而不是按每个样本的真实 regime label，因此旧 train near 中所有样本都会进入 near expert。这会造成 expert supervision 噪声。

开发阶段可暂时使用；最终建议：

- 重构 train near，要求 nominal near；
- 禁止 post-contact 与 prefix collision/contact；
- 或训练加载时按 nominal scene-time regime 过滤/加权；
- 报告 train/val/test 的 nominal regime fractions，而非只看 all-candidate fractions。

### 7.4 Calibration 原先存在同 scene 泄漏

原脚本按 `scene|time` 哈希分 fold，同一 WOMD scene 的不同时间可能落入 fit 与 verify 两边。补丁默认改为 scene-disjoint，并加入最小独立 scene 数检查。

### 7.5 测试集不能参与模型选择

原快速脚本容易让 test 介入多阶段筛选。补丁将 Stage 2 明确放在 val，并提供独立的可选 one-shot heldout test。任何依据 test 调参数后重新测试的结果都不再是严格 held-out。

---

## 8. 已完成的代码修改

1. `src/ocrap/data/build/diagnose.py`
   - 新增 nominal candidate regime 统计；
   - 输出 `nominal_sample_count`、`nominal_counts`、`nominal_fractions`；
   - Safe 数据不再因两 root 的 `y_obs=1` 被通用极端值规则误判。

2. `src/ocrap/data/build/builder.py`
   - 在 dataset summary 中记录 regime threshold 与生成合同信息，便于复现和审计。

3. `tools/check_regime_dataset_contract.py`
   - Safe 改为 nominal-level purity + forbidden contamination；
   - 增加 `all / trainval / v45dev` scope；
   - 兼容旧 diagnostics，并明确旧报告只能做必要条件判断。

4. `scripts/rebuild_ocrap_train_safe_two_gpu.sh`
   - 新增两 GPU、可续跑、严格 Safe 重建脚本；
   - 默认目标 15k–20k；
   - 完整 scene-time group 截断；
   - 最终 manifest 硬审计。

5. `tools/cap_dataset_scene_time_groups.py`
   - 按完整 group 截断，避免随机截 sample 导致候选组残缺。

6. `scripts/rebuild_ocrap_val_test_regimes.sh`
   - Safe 使用 strict nominal regime contract；
   - `keep_nominal_even_if_quality_fails=false`；
   - under-min-quality 时丢弃整组。

7. `tools/calibrate_direct_value_risk_v45.py`
   - 默认 scene-disjoint fold；
   - 检查独立 scenes 与 fold overlap。

8. `scripts/calibrate_ocrap_v45_rave.sh`
   - 透传 scene-level 校准参数。

9. `scripts/train_ocrap_v39_ocrac.sh`
   - 支持通过环境变量清空 freeze prefixes，执行 clean-base refresh。

10. `run_v45_two_gpu_fast_commands.txt`
    - 分离 development 与 final；
    - 可选 clean-base retraining；
    - partial-data calibration gate；
    - val-only selection + one-shot heldout test。

---

## 9. 验证结果

已完成：

- 修改后的 Python 文件全部通过 `py_compile`；
- 修改后的 Shell 脚本全部通过 `bash -n`；
- `tests/test_v45_rave.py` 与 `tests/data/test_dataset_rebuild_integrity.py`：6 passed；
- group-preserving cap 合成 smoke test 通过；
- 旧 reports 上：
  - `--scope trainval` 正确失败，且只把旧 train_safe 的真实问题列为 failure；
  - `--scope v45dev` 通过并给出 development-only warnings。

旧 reports 无 `nominal_counts`，所以 `val_safe` 最终 nominal purity 仍需在真实数据目录上重跑 patched diagnose 后确认。

---

## 10. 推荐执行顺序

### A. 立即开发性结果

```bash
cd /path/to/OC-RAP-patched
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export DATASET_DIAGNOSTICS_DIR=$OCRAP_ROOT/reports

FINAL_RUN=0 \
RETRAIN_CLEAN_BASE=0 \
RUN_HELDOUT_TEST=0 \
bash run_v45_two_gpu_fast_commands.txt
```

用途：发现模型/selector 的大问题，不作论文最终结论。

### B. 重建 strict train_safe

```bash
cd /path/to/OC-RAP-patched
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP

RAW_PER_WORKER=6000 \
MIN_SAMPLES=15000 \
MAX_SAMPLES=20000 \
GPU0=0 GPU1=1 \
bash scripts/rebuild_ocrap_train_safe_two_gpu.sh
```

若最终少于 15k，不降低 `tau_normal_dep/tau_normal_occ`，优先增加 `RAW_PER_WORKER` 至 8000 或 10000。

### C. 重新诊断

```bash
python -m ocrap.cli diagnose \
  --dataset "$OCRAP_ROOT/train_safe_v2" \
  --set dataset_quality.nominal_regime_dataset=true \
  --set 'dataset_quality.require_nominal_regimes=[normal]' \
  --output "$OCRAP_ROOT/reports/train_safe.json"

python tools/check_regime_dataset_contract.py \
  "$OCRAP_ROOT/reports" --scope trainval --mode development
```

同时用 patched diagnose 重新生成 val/test safe 报告，确保有 nominal counts。

### D. 最终 clean-base + v45 重训

```bash
FINAL_RUN=1 \
RETRAIN_CLEAN_BASE=1 \
TRAIN_SAFE_DATA="$OCRAP_ROOT/train_safe_v2" \
CLEAN_BASE_RUN=runs/ocrap_v39_ocrac_clean_safe \
RUN_HELDOUT_TEST=1 \
bash run_v45_two_gpu_fast_commands.txt
```

建议最终跑 3 个 seeds，并对 scene 做 bootstrap confidence intervals。

