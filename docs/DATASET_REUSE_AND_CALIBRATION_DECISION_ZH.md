# v48.1 数据集复用与 Calibration 决策报告

## 结论先行

本轮不重建 `train_contact`、`train_near_contact`、`val_contact`、`val_near_contact`，也不改写现有 `test_*`。

先使用现有数据完成以下可证伪实验：

1. exact teacher-PCD 正恢复机会覆盖诊断；
2. v48 setwise nominal-vs-recovery 学习；
3. 独立 harm head 与 joint policy-risk gate；
4. scene-disjoint proxy calibration；
5. 未接触 test 的 development offline/小规模 closed-loop probe。

只有在以下情况之一出现时，才值得支付两三天以上的训练集重建成本：

- Contact exact-PCD 正机会接近不存在或集中到极少 scene；
- 训练集上的 group top-1 也无法由随机附近提升为正相关；
- 训练可学、proxy calibration可学，但独立calibration持续失效，且可归因于构建合同漂移；
- 论文最终实验需要统一的新数据合同与更窄置信区间。

## 现有数据是否足够验证算法方向

| 数据集 | 样本 | scene | scene-time group | 候选/group | oracle recoverable | artifact | Waymax runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_near_contact | 13,324 | 600 | 1,800 | 7.40 | 0.636 | 0.189 | 1.0 |
| train_contact | 16,790 | 500 | 2,000 | 8.40 | 0.623 | 0.166 | 1.0 |
| val_near_contact | 3,445 | 176 | 433 | 7.96 | 0.742 | 0.246 | 1.0 |
| val_contact | 5,723 | 187 | 639 | 8.96 | 0.758 | 0.218 | 1.0 |
| test_near_contact | 4,723 | 250 | 595 | 7.94 | 0.756 | 0.244 | 1.0 |
| test_contact | 5,514 | 171 | 616 | 8.95 | 0.773 | 0.218 | 1.0 |

这些规模足以回答“模型能否从候选级信号学成策略级正确选择”，尤其是训练集有500–600个独立scene、1800–2000个group、完整8-root结构和100% Waymax runtime字段。现在最缺的不是总样本数，而是exact teacher-PCD下的正机会分布统计；v48.1会在训练前生成该索引。

## train_contact的瑕疵是否致命

### 1. 训练集比val/test更难

`train_contact`的平均 `r_dep_star=-1.792`，而`val_contact=-0.563`；hard violation均值分别约0.0936和0.0105。候选数也从8.40升到8.96。

这是中等难度/候选前沿偏移，会影响绝对校准与阈值迁移，但不会阻止以下学习目标：

- 同一个scene-time内候选相对nominal的PCD差值；
- 是否存在正恢复机会；
- 某个候选是否比nominal有害；
- nominal与恢复动作之间的setwise选择。

这些标签在组内构造，较少依赖全局均值一致性。因此它是“需要通过校准处理的covariate shift”，不是“无法训练的label contradiction”。

### 2. val/test的harm_proxy为0

这会使依赖旧 `harm_proxy` 字段的legacy诊断或baseline不完整，但不会直接破坏v48主风险头：

- v48 harm标签由exact teacher-PCD相对nominal下降量生成；
- `teacher_adv <= -negative_gain`即为harmful candidate；
- joint calibration也重新计算teacher-PCD，而不是读取原始harm_proxy。

因此不应仅因为harm_proxy退化而重建。

### 3. train_contact使用旧`post_contact`标签，val/test显式为`post_contact_counterfactual`

现有`train_contact`中所有样本都属于旧的`post_contact`总类；新val/test进一步区分为counterfactual。v48训练的bucket来自数据根路径，主损失来自teacher-PCD，不使用`post_contact_observed`和`post_contact_counterfactual`作为隐藏router标签，因此当前实验仍可进行。

但论文必须把现阶段Contact称为contact-conditioned/counterfactual recovery，不应写成真实碰撞动力学恢复。最终若要研究observed post-collision，需要新数据，而不是简单重建同一surrogate数据。

### 4. 外部baseline公平性

所有模型使用同一`train_contact`，可以保证数据资源公平；但它不能自动保证科学结论完整。若正机会过少或集中，所有模型都可能学不到，结果只能说明“在当前数据合同下没有可学习证据”，不能说明提出的方法无效。

因此本轮增加exact-PCD覆盖报告，而不是先重建：

- `positive_groups`、`positive_scenes`；
- `max_positive_macro_share`；
- `top10_positive_scene_share`。

建议解释标准：

| 状态 | Near | Contact | 处理 |
|---|---|---|---|
| 足以做方向验证 | ≥80正group且≥40 scene | ≥60正group且≥30 scene | 直接训练 |
| 边缘可调试 | 20–79 group或10–39 scene | 15–59 group或8–29 scene | 可训练，但不作强结论 |
| 数据阻断 | 接近0或集中于1–3 scene | 接近0或集中于1–3 scene | 才考虑增补/重建 |

v48.1保留论文级目标门槛（Near 200/80，Contact 120/60），但当前screening只告警，不再自动中止；只有某个regime完全没有正group/scene才硬停止。

## 临时calibration协议

### 为什么不用test

测试集必须在模型结构、损失、超参数、阈值搜索和候选选择全部冻结后才能打开。拿`test_*`临时代替calibration，会把test信息写入：

- opportunity/gain/harm thresholds；
- supported macro set；
- balanced/precision候选选择；
- Natural gate通过与否。

这会使后续test结果不再held-out。

### 当前做法

把每个现有val根按scene做50/50确定性拆分：

- `proxy calibration`：只拟合阈值和证书；
- `development val`：只做early stopping、offline evaluation和小规模closed-loop probe；
- 原始`val_*`完全不修改；
- `test_*`完全不读取。

拆分覆盖Safe、Near和Contact，避免Safe仍在完整val上校准、又在同一val上报告的问题。

预计每个分区scene数量约为：

- Safe：约66/66；
- Near：约88/88；
- Contact：约94/93。

calibrator内部还会再按scene划分fit/verify，因此这是开发筛查协议，置信区间会宽。Natural gate失败可能来自模型，也可能来自proxy校准样本量；报告会保留完整警告，不能直接宣称算法理论失败。

## 最终dedicated calibration来源

使用：

```text
waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/
```

不使用：

- `testing`：未来ground truth隐藏，不能生成teacher-PCD标签；
- `testing_interactive`：同样不适合本地带标签校准；
- `validation_interactive`：可作为交互压力/OOD评估，但不作为主IID calibration，以免与当前标准validation构建的val/test形成新的来源偏移。

最终脚本采用：

- 标准validation 150 shards；
- `scenario_start_index=11000`，避开现有同步val/test构建默认预算所扫描的最大原始索引；
- `scenario_stride=6`；
- Safe worker 0/1，Near 2/3，Contact 4/5；
- Safe每worker 600 raw，Near/Contact每worker 700 raw；
- 仍然过滤所有现有val/test scene，并做硬重叠审计；
- 最低独立scene目标：Safe 80、Near 120、Contact 120。

Near/Contact参数与同步`rebuild_ocrap_val_test_regimes.sh`保持一致，包括targeted futures、候选数、root数、artifact配额、future metrics和regime purity约束。

## 不重训切换到dedicated calibration

专用calibration完成后，不需要重训模型。运行：

```bash
OUTPUTDIR=<已完成的proxy screening目录> \
CALIBRATION_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
GPU0=0 GPU1=1 \
bash scripts/recalibrate_v48_on_dedicated_set.sh
```

脚本会：

- 复用balanced/precision两个checkpoint；
- 重新计算OC-MERO gamma和Near/Contact joint risk rules；
- 将结果写入`calibration_dedicated`；
- 创建`dedicated_candidates/*`视图；
- 不读取test，不改变模型权重。

## A30 + Xeon 5220R运行配置

两张A30分别训练balanced与precision；默认每个训练进程：

- batch size 96；
- BF16 AMP；
- TF32开启；
- dataloader workers 6；
- prefetch factor 2；
- OMP/MKL/OpenBLAS/NumExpr threads 4。

相比每进程8 workers、prefetch 3，这一配置更适合两个训练进程共享Xeon Gold 5220R，降低CPU线程和页锁定内存争用。若GPU利用率长期低于约70%，先把`NUM_WORKERS`提高到8；若内存/IO抖动或load average过高，则降到4。

## 本轮代码改变

1. `run_v48_two_gpu_fast_commands.txt`
   - 默认复用现有训练和val数据；
   -  expensive rebuild双确认；
   - Safe/Near/Contact全部scene-disjoint proxy拆分；
   - test不参与screening；
   - 数据覆盖门从默认硬失败改为告警+零机会硬底线；
   - 写出dataset role manifest和screening status。

2. `tools/build_teacher_pcd_index_v48.py`
   - 新增`--quality-mode strict|warn|off`；
   - summary记录`quality_passed`与`quality_failures`。

3. `scripts/build_v48_calibration_regimes.sh`
   - 标准validation专用calibration；
   - 新增Safe calibration；
   - reserved tail、scene filtering、overlap audit、最低scene门槛。

4. `scripts/recalibrate_v48_on_dedicated_set.sh`
   - 专用calibration完成后只重校准，不重训。

5. `scripts/train_ocrap_v48_trac_sr.sh`
   - A30/5220R双任务CPU线程和dataloader默认优化。
