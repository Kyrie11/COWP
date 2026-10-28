# RIDE RQ1/RQ2 v5

该仓库继承用户上传的 RIDE v4，修复真实 CARLA 证据揭示的并线间距、目标车道跨 OpenDRIVE 边界识别、model-dependent 预热初态、道路中心线有效性诊断以及场景执行门禁，并提供 ReactSim 现有 nuPlan DB/地图软链接安装器和严格的标准化两模型配对审计接口。

从 `docs/V5_SCIENCE_AND_EXECUTION_CN.md` 开始。重要：v5 与 v4 anchors **不能混用**；本地仅做离线测试，CARLA 物理 smoke 和 ReactSim 模型实验必须在拥有原始数据与 GPU 的用户服务器上执行。任何仿真正例不等于现实人类反事实因果真值。旧 RIDE/COWP 的研究假设不会由工具自动宣称通过。

一键入口：`scripts/15_reactsim_nuplan_stage.sh`（只读检查/软链接），`scripts/16_reactsim_run.sh`（官方预处理与推理封装），`python -m ride.reactsim_coverage`（不反序列化的场景文件名覆盖审计），`python -m ride.reactsim_paired_audit`（严格 2×2 配对）。
