# OC-RAP v48.5完整结果审计与v48.6 OC-TRAC-RPGC设计

## 1. 结论先行

v48.5没有完全解决六个问题，但已经出现一个具有明确消融证据的实质进展：**独立ECPR偏好头使Contact的组内top-1排序从v48.4三个seed持续为负，变为v48.5两个variant在4801/4802/4803上全部为正。** 当前瓶颈不再是“Contact完全不会排序”，而是：

1. Near排序仍受scene split影响，跨seed可转负；
2. Contact排序虽转正，但尚未达到投稿级强度；
3. 相对nominal的准入收益与风险分布仍重叠严重；
4. 近似可用的规则被单一macro支配；
5. Natural gate仍找不到同时满足支持度、精度、harm和macro多样性的规则，因此全部 abstain。

因此v48.6不再继续强化共享NASC，也不放宽Natural gate。新版本保留有效的独立Preference路径，并将“同组选谁”和“是否值得离开nominal”进一步拆成两个直接监督、可独立校准的对象。

---

## 2. v48.5三seed结果

### 2.1 主要离线指标

| Variant | Regime | Candidate AUC均值 | Harm-risk AUC均值 | Candidate rank corr均值 | Group top-1 corr均值 | Positive top-1 accuracy | Positive regret |
|---|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.6816 | 0.5948 | 0.1900 | 0.0042 | 0.4507 | 0.1746 |
| Balanced | Contact | 0.7809 | 0.5906 | 0.2322 | 0.1508 | 0.5776 | 0.1871 |
| Precision | Near | 0.7279 | 0.5942 | 0.2001 | -0.0076 | 0.5175 | 0.1132 |
| Precision | Contact | 0.7957 | 0.5657 | 0.2142 | 0.1372 | 0.5883 | 0.1804 |

Contact三个seed的top-1相关性均为正：

- Balanced：0.1737 / 0.1551 / 0.1237；
- Precision：0.1551 / 0.1268 / 0.1297。

Near仍不稳定：

- Balanced：0.0619 / 0.0430 / -0.0922；
- Precision：0.0285 / 0.0192 / -0.0704。

这说明独立Preference head已经学到Contact中一部分“同组候选谁更优”的结构，但Near的teacher gap更小、机会更稀疏，排序对scene划分和噪声仍敏感。

### 2.2 Natural gate near-miss

三个seed中最接近通过的规则仍有明显缺陷：

| Regime | near-miss选择数 | Precision范围 | Precision LCB90范围 | Harm rate范围 | Positive recall范围 | 最大macro占比 |
|---|---:|---:|---:|---:|---:|---:|
| Near | 8–18 | 0.154–0.556 | 0.052–0.303 | 0–0.375 | 0.105–0.316 | 1.000 |
| Contact | 10–49 | 0.265–0.400 | 0.127–0.242 | 0.242–0.510 | 0.111–0.520 | 0.898–1.000 |

Near规则几乎100%集中于一个macro；Contact平均macro占比约0.954，同时harm rate过高。因此零选择不是单纯由于阈值过严：即使取最接近的规则，也没有达到可信部署质量。

---

## 3. 六个问题是否被解决

| 原问题 | v48.5状态 | 判断 |
|---|---|---|
| 候选级恢复信号存在 | 仍存在，但AUC较v48.4有所下降 | 保留；并非主要瓶颈 |
| Near排序不稳、Contact持续反向 | Contact已在全部seed转正；Near仍不稳 | **部分解决** |
| Harm head接近随机 | 当前`candidate_harm_auc`主要评价delta-distribution风险，并非原始Harm head | 未证明原Harm head改善；仍只应作辅助 |
| 所有规则选择0动作 | 仍然如此 | 未解决 |
| Candidate AUC与policy top-1脱节 | Contact脱节明显缓解；Near仍存在 | 部分解决 |
| 未学会同组谁优于nominal/其他候选 | Contact候选间排序已学到一部分；是否优于nominal仍未校准成功 | 部分解决，但准入失败 |

关键转变是：**排序问题与准入问题已经可以分开定位。** Contact排序开始有效，但相对nominal的gain/risk certificate仍不能支撑执行。

---

## 4. v48.5消融归因

| Ablation | Balanced Near top-1 | Balanced Contact top-1 | Precision Near top-1 | Precision Contact top-1 | 结论 |
|---|---:|---:|---:|---:|---|
| A Exact pointwise | -0.0302 | -0.0991 | -0.0292 | -0.0102 | Exact target本身不够 |
| B Exact + ZI-NASC | -0.0119 | -0.0926 | -0.0021 | -0.0931 | NASC未解决排序 |
| C Exact + ECPR | **0.0382** | **0.0785** | -0.0337 | **0.0257** | 唯一能明确使Contact转正的模块 |
| D Full ECPR + NASC | -0.0663 | -0.0047 | -0.0414 | -0.0139 | NASC与ECPR发生干扰 |

### 有效设计

1. **Exact teacher-PCD合同统一**：排除了训练和校准teacher定义不同的干扰，是可信归因的基础。
2. **独立Preference head**：消融C是唯一明确改善Contact top-1的结构，出发点成立。
3. **Confidence-paced preference/regret**：在teacher差距明确的Contact group上有效，值得深化。
4. **风险聚焦checkpoint与多seed固定checkpoint流程**：能够稳定暴露Near split sensitivity和Contact一致改善。
5. **Natural gate**：在规则精度不足时拒绝执行，安全逻辑正确。

### 无效或作用较小的设计

1. **共享NASC**：修改了value和ranking共同依赖的表示，提升部分candidate AUC，却破坏策略top-1；不应作为主路径继续增强。
2. **用两个absolute value分布相减构造delta**：默认把candidate和nominal误差视为独立，实际上二者共享scene encoder，方差相加可能过度保守，导致opportunity/harm概率重叠和零coverage。
3. **Harm/opportunity head参与admission composition**：其标签跨split不稳定，继续让它影响主准入会放大shortcut；应降为辅助诊断。
4. **仅用分数top-1而无rank confidence certificate**：模型即使选对概率稍高，也不知道何时自己“足够确定”。
5. **fit阶段未把macro集中度作为硬约束**：旧版本只在verify后报警，容易搜索出macro捷径规则。

---

## 5. 工程审计

### 已排除的疑似问题

完整检查确认，执行的v48.5配置使用`direct_value_output_mode=score`，训练validation和calibration均使用score语义。此前怀疑的“raw logit与sigmoid概率尺度不一致”不是本轮失败根因。v48.6仍将该分支显式化，以保护未来probability-mode消融。

### 确认需要修复的问题

1. **相对gain不应由两个absolute预测相减并假设独立方差**；已改为直接预测candidate-minus-nominal gain及其log-variance。
2. **共享NASC干扰Preference**；已改为只给Preference head增加零初始化relative context，不重写value特征。
3. **macro捷径**；新增基于训练teacher index的positive teacher-best-macro逆频率采样，并在calibration fit阶段强制macro share约束。
4. **rank置信度缺失**；新增best-vs-runner-up margin监督、校准和runtime abstention。
5. **checkpoint对proxy split敏感**；early stopping改为Near/Contact三个scene-hash folds的worst risk。
6. **风险诊断混在一个AUC中**；新增direct-risk harm AUC、legacy Harm-head AUC和rank-margin correctness AUC分别报告。

---

## 6. 三个regime投稿目标完成情况

### Safe

当前包没有paired closed-loop结果，因此以下目标均**尚未验证**：collision/offroad非增、paired 95% CI、route progression、NUP、jerk/yaw-rate和intervention episode。Safe calibration文件仅说明标量阈值可生成，不等于闭环非劣成立。

### Near-contact

Natural gate没有有效规则，模型没有进入恢复执行，因此：

- collision相对下降15%–25%：未实现/未验证；
- minimum clearance p05 +0.20 m：未验证；
- minimum TTC p05 +0.20 s：未验证；
- exposure下降15%：未验证；
- DRS +8个百分点、PCD +0.03：未实现；
- FRA下降30%、ODG下降25%：未实现；
- harmful switch≤5%–10%：零执行时形式上为0，但这是空覆盖，不构成算法优势。

差距来源主要是Near top-1跨seed不稳、precision LCB太低、positive recall低以及macro 100%集中。

### Contact

Contact top-1已经取得方向性进步，但规则harm rate约0.24–0.51，仍不能部署。secondary overlap、recontact、overlap duration、stable stop、time-to-stop、post-contact clearance、uncontrolled displacement和route-rejoin全部没有闭环证据。

差距来源主要是：排序相关性只有约0.14–0.15，尚未达到>=0.20；直接准入风险不够区分；规则高度集中于单一macro；Contact near-miss precision仅约0.27–0.40。

---

## 7. v48.6：OC-TRAC-RPGC

### 7.1 Preference-only relative context

候选、candidate-minus-nominal、recovery mean和recovery max只进入独立Preference残差，不再修改value专家特征。输出层零初始化，因此加载v48.5 checkpoint时初始策略严格不变。

### 7.2 Direct relative-gain distribution

新增头直接预测：

```text
Delta_PCD = PCD(candidate) - PCD(nominal)
```

以及log-variance。该输出统一用于：delta regression/NLL、validation admission、calibration opportunity/harm和closed-loop selector，避免两个absolute分布的独立性假设。

### 7.3 Confidence-paced listwise + rank-gap

在best-vs-rest之外增加完整recovery候选分布KL，并监督teacher-best与runner-up的gap。teacher近似并列时降低权重；明确机会组中增强排序。校准联合搜索rank-margin阈值，让模型在排序置信度不足时abstain。

### 7.4 Macro-balanced exact-opportunity sampler

只基于training teacher index，对positive group按teacher-best macro逆频率加权，降低macro-5 shortcut。它不读取calibration/test，比minibatch GroupDRO稳定且可解释。

### 7.5 Fold-robust checkpoint

每个scene按hash分成3 folds，early stopping使用Near/Contact所有fold中的worst policy risk，减少某一个proxy split主导best epoch。

---

## 8. v48.6首轮目标

首轮不直接声称达到CCF-A门槛，应按以下顺序判断：

1. Near和Contact在4801/4802/4803均为正，且top-1 corr初步>0.10；
2. Contact均值向>=0.20靠近，Near不再出现负seed；
3. rank-margin correctness AUC>=0.65；
4. 至少一个variant在Near和Contact都有非零verify selections；
5. development阶段precision LCB90>=0.40、positive recall>=0.20，harm UCB不恶化；
6. selected macro share<=0.85；
7. 达到上述条件后才运行development closed-loop；论文门槛仍为precision LCB90>=0.60、recall>=0.35、harm UCB<=0.10和top-1 corr>=0.20。

这些修改提高了理论可辨识性和实验归因质量，但真实训练前不能保证所有闭环指标必然达到目标。
