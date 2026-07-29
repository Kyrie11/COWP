# OC-RAP v48.6 完整实验审计与 v48.7 OC-TRAC-SPIRE 设计

## 1. 本轮实验是否完整

上传的三个结果包可以支持本轮归因：

- 主实验两个 variant 均完成，完成性审计 `comparable=true`；
- balanced 与 precision 均完成 12 个 epoch，best epoch 均为 8；
- 4801/4802/4803 三个 proxy calibration seed 均完成；
- A/B/C/D 四组核心消融均有完整训练摘要、校准诊断和完成性审计；
- 所有 Natural gate 结果均为拒绝，因此没有合规的 Near/Contact development closed-loop 结果。

## 2. 人工检查步骤 5 的结论

### 2.1 三 seed 聚合

| Variant | Regime | Candidate AUC | Rank corr | Group top-1 corr | Rank-margin correctness AUC | Verify selections |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.7062 | 0.0612 | -0.0024 | 0.4223 | 0 |
| Balanced | Contact | 0.8238 | 0.0306 | -0.0536 | 0.6512 | 0 |
| Precision | Near | 0.7028 | 0.0390 | -0.0390 | 0.4349 | 0 |
| Precision | Contact | 0.8178 | -0.0091 | -0.0874 | 0.6253 | 0 |

三 seed 的 top-1 相关性：

- Balanced Near：0.0239 / 0.0022 / -0.0335；
- Balanced Contact：-0.0998 / 0.0225 / -0.0835；
- Precision Near：-0.0282 / -0.0270 / -0.0619；
- Precision Contact：-0.1435 / 0.0069 / -0.1256。

因此步骤 5 的检查结果是：

| 检查项 | 结果 |
|---|---|
| 三个 seed 的 Near/Contact top-1 全部为正 | 失败 |
| top-1 均值达到 0.10 | 失败 |
| Contact 不再系统性反向 | 失败，只有 seed 4802 局部转正 |
| rank-margin correctness AUC ≥0.65 | 仅 Balanced Contact 勉强达到；Near 明显不达标 |
| 每个 seed 产生非零 verify selection | 全部失败 |
| precision LCB90、positive recall 达到最低门槛 | 全部失败 |
| Natural gate 通过 | 全部失败 |

### 2.2 不能把“低 harmful exposure UCB”解释为低 harmful switch

旧 multi-seed 汇总记录的是：

`harmful_group_exposure_ucb90 = harmful actions / all groups`。

在零选择或极低覆盖时，这个值会很低；但真正对应投稿目标“harmful switch ≤5%–10%”的是：

`harmful_selected_ucb90 = harmful actions / selected actions`。

零选择时后者应为 1.0，而不是 0。v48.7 已将两种风险分别报告并分别约束，避免再次把空覆盖解释成安全选择。

## 3. v48.6 三个辨识对象是否学会

### 3.1 Preference：同组候选应该选谁

没有稳定学会。

- Near 三 seed 均值接近零，precision 三个 seed 全为负；
- Contact 大多数 seed 为负；
- 正恢复 top-1 accuracy 约 0.40–0.50，与可靠策略仍有明显距离；
- v48.5 曾得到 Contact top-1 约 0.14–0.15，v48.6 反而退化。

四组消融证明，`B_preference_context_only` 是唯一出现正向迹象的模块：

| 消融 | Balanced Near top-1 | Balanced Contact top-1 | Precision Near top-1 | Precision Contact top-1 |
|---|---:|---:|---:|---:|
| A reference | 0.0085 | -0.0375 | -0.0383 | -0.0566 |
| B preference context only | 0.0191 | 0.0225 | 0.0233 | -0.0040 |
| C direct delta only | -0.0086 | -0.0907 | 0.0093 | -0.0524 |
| D full RPGC | 0.0239 | -0.0998 | -0.0282 | -0.1435 |

相对上下文的出发点成立，但它被联合训练中的 delta/certificate 梯度抵消。

### 3.2 Relative gain：候选是否优于 nominal

只学到候选级的粗粒度信号，没有形成可靠的策略证据。

- Near candidate AUC 约 0.70；
- Contact candidate AUC 约 0.82；
- risk-harm AUC 只有约 0.55–0.58；
- 同一模型在高 candidate AUC 下仍可能选择 teacher 更差的组内候选；
- direct-delta-only 消融没有改善 top-1，完整模型还进一步破坏排序。

所以 Relative gain 是“部分可辨识”，但其不确定度和相对 nominal 的证书没有校准到可执行水平。

### 3.3 Certificate：模型是否足够确定、允许执行

没有学会可用证书。

- 所有 seed、variant、regime 的 verify selection 都为 0；
- Contact 最接近通过的 fit near-miss 通常只有 9–12 个选择；
- precision 约 0.56–0.60，但 LCB90 只有约 0.30–0.36；
- recall 约 0.15–0.26；
- conditional harmful rate 约 0.08–0.20，样本小导致 UCB 仍高；
- 最大 macro 占比约 0.80–0.92；
- seed 4803 明显退化。

Near 的 near-miss 更差，常见 precision 只有 0–0.25，且 macro share 接近 1.0。

## 4. v48.6 中有效与无效的设计

### 有效

1. **Exact teacher-PCD 合同**：训练与校准的 teacher 语义保持一致，是可信归因的必要条件。
2. **Preference-only relative context**：B 消融提高 rank correlation，并局部将 Contact top-1 转正，值得保留。
3. **场景级 multi-seed calibration**：准确暴露了模型不是单个 seed 偶然失败，而是存在系统性排序和证书问题。
4. **Natural gate**：在 conditional harmful rate 高、排序不稳时拒绝执行是正确保护。
5. **Candidate encoder/value 表示**：Contact AUC 仍较强，说明无需推翻整个基础表示。

### 无效或负作用

1. **Preference 与 direct-delta 联合反向传播**：C、D 消融证明 delta 目标会破坏偏好排序，尤其是 Contact。
2. **严格单一 winner 监督**：Near 中 teacher-best 与 runner-up 经常接近，单 winner 标签随 scene split 反转。
3. **Gaussian delta 自报方差直接作为证书**：风险区分能力低，opportunity 与 harm 分布重叠。
4. **旧 checkpoint 指标与部署证书尺度不一致**：验证使用 raw delta，而 calibration 使用 Gaussian CDF；best epoch 未必是部署最优 epoch。
5. **旧风险汇总混淆 exposure 与 conditional harmful switch**。
6. **Macro balance 仍不足**：near-miss 规则高度集中于 macro 5。

## 5. 三个 regime 投稿目标完成情况

### Safe

当前没有 paired Safe closed-loop 结果，以下目标均未验证：

- collision/offroad 非增；
- paired-scene 95% CI 上界；
- route progression；
- NUP；
- jerk/yaw-rate p95；
- intervention episode。

Safe 在策略中是 nominal-locked，因此可以单独运行 Safe 非劣探针，不需要 Near/Contact Natural gate。它只能验证 Safe，不得用于授权 stress-regime 恢复动作。

### Near-contact

由于实际策略始终 abstain，图片中的所有改善目标都没有实现证据：

- collision 相对下降 15%–25%；
- clearance p05 +0.20 m；
- TTC p05 +0.20 s；
- exposure -15%；
- DRS +8 个百分点；
- PCD +0.03；
- FRA -30%；
- ODG -25%。

零选择不能把 harmful switch 记为 0 并作为贡献，因为覆盖率同样为 0。

### Contact

Contact 比 Near 更接近形成证书，但仍没有进入闭环，因此 secondary overlap、recontact、stable-stop、time-to-stop、clearance、uncontrolled displacement 和 route-rejoin 均未验证。

当前最主要差距是：

- top-1 仍为负；
- seed 稳定性不足；
- conditional harmful-switch 证书不足；
- 正机会 recall 低；
- macro 集中严重。

## 6. `missing checkpoint runs/ocrap_v48_trac_sr_regime_balanced/...` 的原因

这不是因为 v48.6 没有训练模型。

实际训练 checkpoint 应位于：

- `runs/ocrap_v48_6_rpgc_proxy_4801/candidates/balanced/model_v48_trac_sr/best.pt`
- `runs/ocrap_v48_6_rpgc_proxy_4801/candidates/precision/model_v48_trac_sr/best.pt`

报错链路是：

1. 两个 candidate 都未通过 Near+Contact Natural gate；
2. controller 没有写出 `chosen_base_run.txt`；
3. 手工执行 closed-loop 时没有得到有效 `BASE_RUN`；
4. 旧评估脚本静默回退到历史默认目录 `runs/ocrap_v48_trac_sr_regime_balanced`；
5. 该旧目录不是本轮训练输出，因此报 missing checkpoint。

不建议为了满足这个旧路径而额外训练一个 legacy 模型。v48.7 已取消该默认回退：必须明确提供 `BASE_RUN` 或 checkpoint；Natural gate 失败时写出 `GATE_FAILED.json`，并禁止生成 stress closed-loop 指令。

## 7. 新算法 v48.7：OC-TRAC-SPIRE

SPIRE = **Set-valued Preference with Isolated Relative-gain Evidence**。

### Stage P：Preference

- 冻结 encoder、value、opportunity、harm 和 delta；
- 只训练独立 pointwise preference residual 和 relative-context preference residual；
- Near 使用 0.025 exact-PCD acceptable-set epsilon，Contact 使用 0.010；
- 对 acceptable set 使用集合 KL，而不是强迫某一个近似并列候选成为唯一正确答案；
- 对 acceptable set 与 nominal/明显较差候选施加 margin；
- 保留 confidence-paced best-vs-rest 和 expected regret；
- early stopping 使用三折 worst-fold tie-aware preference risk。

### Stage C：Relative gain / Certificate

- 冻结 Stage P 的全部排序路径和共享表示；
- 只训练 candidate-minus-nominal direct-delta adapter；
- 这样 Relative gain 不再通过共享 encoder 破坏 Preference；
- checkpoint 使用与 calibration 相同的 Gaussian CDF 语义；
- risk 包含 admitted harmful、false intervention 和 missed opportunity；
- always-abstain 会因 missed-opportunity penalty 被惩罚。

### Calibration

- 分开报告 strict top-1 与 acceptable-set top-1；
- 分开报告 harmful exposure UCB 与 harmful-selected conditional UCB；
- proxy calibration 使用较宽松的开发证书；
- 论文 promotion 必须使用更大的 dedicated calibration，并收紧 conditional harmful UCB。

## 8. Dedicated calibration 合并

新增脚本：

`scripts/merge_v48_calibration_regimes_to_eval_root.sh`

它允许 Safe/Near 先合并，不需要等待 Contact。最终目录为：

- `/data0/senzeyu2/dataset/OCRAP/calibration_safe`
- `/data0/senzeyu2/dataset/OCRAP/calibration_near_contact`
- `/data0/senzeyu2/dataset/OCRAP/calibration_contact`

脚本会：

- 检查两个 worker manifest；
- 合并分片；
- 排除已有 val/test scene；
- 原子替换目标目录；
- 写入 `MERGE_COMPLETE.json`；
- 执行 scene overlap audit。

## 9. 下一轮最先观察的指标

第一轮不是直接宣布达到 CCF-A 门槛，而是验证 SPIRE 是否修复因果链：

1. 三个 seed 的 Near/Contact top-1 全部转正；
2. top-1 均值至少 0.10；
3. Contact 不再被 direct-delta stage 拉回负相关；
4. acceptable-set top-1 明显高于 strict top-1，特别是 Near；
5. 至少一个 variant 在两个 regime 都产生非零 verify selection；
6. conditional harmful-selected UCB 不失控；
7. macro share 不再接近 1.0。

达到这些条件后，才值得运行 Near/Contact development closed loop；最终投稿仍需 top-1 ≥0.20、precision LCB90 ≥0.60、recall ≥0.35，以及图片中的 scene-paired closed-loop 改善目标。
