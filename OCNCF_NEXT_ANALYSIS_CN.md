# OC-NCF 下一轮补丁说明

核心目标：让 COWP 不再退化为 conventional_safety，而是让“非胁迫可行性证书”在 planner-only retrain 后真正参与闭环选择。

主要改动：
1. candidate_certificate_loss 新增 outcome-calibrated risk BCE、risk ranking、scene spread 正则，缓解 0.5/0.5/0.5 证书塌缩。
2. 在线 frontier 从弱 score hint 改为 COWP 的实际 feasibility layer：一旦存在 least-coercive frontier，就只在 frontier 内选。
3. action risk 从单步检查升级为短视窗 jerk/yaw/accel 检查，用于继续压低 Waymax kinematic infeasibility。
4. run_mpncf_next.sh 新增 CKPT_SELECTION=latest，避免 val loss 被饱和 witness 项主导时继续评估旧 best checkpoint。
5. run_ocncf_next.sh 封装双卡 planner-only retrain + offline/online eval 默认参数。

建议先跑 120 或 300 场景 probe，确认：
- COWP 的 accepted_candidates 明显低于 conventional_candidates；
- CandidateCertificate/SelectedNcfProbMean、FalseSafeProbMean、QualityProbMean 不再全是 0.5；
- COWP 与 conventional_safety 的 closed-loop 指标不再完全相同；
- KinematicsInfeasibilityRate 低于 0.128，最好继续降到 0.08–0.10。
