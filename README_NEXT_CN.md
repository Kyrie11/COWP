# RIDE v4 — 实验审计与下一轮指令

工作标题：RIDE: Response-Model-Induced Decision Errors in Interactive Driving。

**结论：RQ1、RQ2 未通过证据关口；RQ3 尚不应正式启动。**

详细实验分析、场景诊断、数据需求、全部服务器命令请见：

📄 `docs/RQ12_FORENSIC_AND_NEXT_GATE_CN.md`

证据摘要：`docs/RQ12_EVIDENCE_SUMMARY_20261027.json`

在服务器：`conda activate mfrp; source scripts/00_env.sh; python -m pytest -q`。
所有新实验必须使用**新**的 RIDE_OUTPUT_ROOT；旧 v3 frozen anchors 不可混用。
