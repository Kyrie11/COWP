# EICDRR H2-V3 修订说明

原始 COWP/H2 Gate 源码完整保留，重点修复跨时域动作不一致，增加交互局部的观测拟合审计、不同规则结构的反证性测试、Waymax map metric 初步检验、H1 正向 witness 验证器与强规划基线实际选择验证器。

入口：`bash NEXT_EICDRR_H2_V3_CN.sh`。完整版科学结论与命令位于另附的 `H2_V3_SCIENTIFIC_RESEARCH_DECISION_CN.md`。

修订后新增/修改：

- `research_probe/matched_response_test.py`：`pulse_steps` 参数。原 `steps=40` 默认保持原有行为以不破坏历史实验，**必须明确传入 `--pulse-steps 20` 才是真正 4s→8s fixed-prefix extension**。
- `research_probe/h2_fixed_prefix_audit.py`：分别重测 4 秒和 8 秒，保持前 20 步动作完全一致；报告共享车辆对和同 bad/good 动作。
- `research_probe/h2_observational_stratified.py`：全体 vs 近 ego vs 纵向 corridor 等，只按当前状态选子组，报告逐场景配对 bootstrap。
- `research_probe/reactive_stress.py`：新增 `ttc_gap_rule` 响应规则，不进入老 `REGIMES` 和已有结果重计算。
- `research_probe/h2_ttc_rule_probe.py`：单独运行 TTC 几何规则与三种 IDM 配置，仍是 shared-simulator screening。
- `research_probe/waymax_paired_idm.py`：`--steps`、`--validate-map` 以及地图 metric fail-closed；目标速度干预非 cosine 相同 ego 动作，不可作为 Q1 严格配对证明。
- `research_probe/h1_joint_witness_gate.py`：输入显式 JSONL joint 轨迹，验证模型条件正向 SAT；失败只能 UNKNOWN。
- `research_probe/h4_planner_choice_gate.py`：输入真实执行输出，检查共同候选库和已选择动作；无原始输出时不能执行。
- `research_probe/h2_three_question_audit.py`：三问证据闭合状态机；不以缺失实验判定 H2 已证伪。
- `tests/test_eicdrr_h2_v3.py`：控制前缀一致性、反证规则、H1/H4 保守资格测试。

**限制**：本地运行 78 项测试成功，未在服务器原生 NPZ、原生 WOMD Waymax 环境执行新任务。正式论文结果需以服务端运行输出为准。
