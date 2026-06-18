# COWP 论文/数据/代码审查摘要

## 数据诊断结论

当前 train/val labels 可以支撑 COWP 的核心监督信号和初步模型训练，但还不足以单独支撑最终论文级闭环结论。

### 支撑点

- train/val schema 均无 validation error。
- train 共约 20k scenes，val 共约 3k scenes，规模适合先做模型训练和消融验证。
- train/val 分布一致性较好：positive pair ratio、false-safe candidate ratio、NCF candidate ratio、OPR/CBS 等指标接近。
- false-safe 与 witness 信号充足：train false_safe_candidate_ratio≈0.365，val≈0.379；positive_pair_ratio≈0.287/0.294。
- response safe coverage 较高：train≈0.874，val≈0.869。

### 风险点

- 候选终点多样性偏弱：train/val mean_candidate_endpoint_spread_m 约 5.46m，低于配置阈值 6.0m。
- val 的 scenes_with_ncf_candidate_ratio≈0.3467，略低于 0.35 目标线。
- mechanism token 只覆盖 HB/AY/SR/OR，缺 PA/GS，说明 priority/gap-space burden 没有被充分传递或触发。
- waymax_enabled_scene_ratio=0，当前 diagnostics 不能证明闭环实验已经可运行或已完成。

## 代码复现判断

代码已覆盖论文 pipeline 的工程骨架：WOMD index、label construction、critical agent、ego candidate、natural alternatives、safe response、witness certification、tensor cache、diagnostics、training/eval/Waymax 接口。但它不是完整论文级神经算法复现：graph decoder、natural alternative generator、learned burden/response prediction 等均为规则/搜索式近似或接口化实现。

## 本次修改

- 将 critical-agent priority relation (`rho`) 传入 safe response burden 和 typed safe-budget search。
- 在 burden 中根据 natural reference 推断 progress loss、delay loss 和 gap loss，使 PA/GS token 可被触发。
- 修正 OPR：由 mass 改为低负担 natural alternatives 中被保留的比例。
- 新增 `cowp/witness/natural_mass_by_source`，修复 natural branch ablation 中 OPR 没有 source-normalized 的问题。
- 修正 offline planner fallback，避免在没有可接受 NCF candidate 时把 coercive conventional candidate 当成已选择 COWP 计划。
- 修正 label-only EP：输出归一化 EP，同时保留 `EP_m` 和 `FallbackRate`。
- 扩大 candidate lattice，并降低 endpoint dedup tolerance，以提升 endpoint diversity。

## 验证

- `python -m compileall -q cowp` 通过。
- `python -m pytest -q` 通过：13 passed。

## 下一步建议

1. 重新构建 labels 和 diagnostics，重点检查：endpoint spread 是否超过 6m，PA/GS token 是否出现，val NCF scene ratio 是否稳定高于 0.35。
2. 重新生成 tensor cache 并训练模型。
3. 用 Waymax rollout dataset 与 closed-loop standard metrics 验证 CR/Offroad/EP 等闭环指标；label diagnostics 只能证明监督数据质量，不能替代闭环实验。
