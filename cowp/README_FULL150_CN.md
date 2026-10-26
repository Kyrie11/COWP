# WOMD validation 150 分片全量扫描与 H1–H4 证据门禁

从 `NEXT_WOMD_FULL150_EICDRR_CN.sh` 开始。该脚本提供 `inventory`、`tests`、`cohort_full`、`identity`、`matched_dev`、`matched_dev_all`、`raw_all`、`waymax_pilot`、`holdout_once`、`finite_h1_dev`、`finite_h3_dev`、`audit_uploaded`。

**关键要求**：使用新的 `EICDRR_FULL_ROOT`，不能沿用只包含四个分片的旧 cohort。先读取原始 WOMD validation 的所有 150 个分片，每 shard 不限记录；扫描会保存 `coverage`、`per_shard_counts` 和 `source_record_index`。如果 `identity_raw32.json` 不通过，停止科学解释，并检查 `errors`。在最终确认之前，执行 `raw_all`；局部 32 个 raw identity 通过不等于全部被验证。

**新特性**：SHA256 hash 保留集；SDC 在 local NPZ 的索引 0 且真实 source SDC 可反查；20 邻车/75 m；历史/当前观测 CPA 抽样；最小非 ego pair center-distance 是初筛 proxy；未来反应闭环仍需独立模型和 Waymax 真实服务器验证。

**禁止过度解读**：当前 3 个 IDM 配置不属于 3 个独立反应模型；test split 无 WOMD 8 秒真值未来；有限 CSP UNSAT 不等于全物理不可行；未运行服务器上的 TFExamples、Waymax、GameFormer 或 RA-CBF。

审计结果与论文实验协议分别在 ZIP 同级的 `EICDRR_FULL150_SCIENTIFIC_AUDIT_20261014_CN.md` 与 `EICDRR_FULL150_PAPER_PROTOCOL_APPENDIX_20261014.tex`。
