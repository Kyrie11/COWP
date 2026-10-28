# 外部基线接入矩阵（RIDE v6）

本文件把“仓库已下载”“模型能运行”“模型能够回答 RIDE 的 RQ3”分开。源码压缩包的存在不能替代 commit、权重、数据预处理和一次独立成功推理的证据。

## 当前上传分支的角色

| 上传源码 | 当前分支能证明什么 | 不能直接证明什么 | 接入建议 |
|---|---|---|---|
| `ReactSim-Bench-main`（archive `71e54fda...`） | nuPlan 上 MTR、CTG、VBD、SMART、CATK、TrajTok 的响应世界模型闭环评测脚本 | 不是 ego planner；作者 benchmark 的平均指标不能直接变成 RIDE 的 ex-ante 规划选择 | 先作为独立 reactive evaluator，完成官方 2,636 validation 场景覆盖审计；再设计固定初态、固定 A/B 身份和候选动作注入接口 |
| `interactive-closed-loop-main` | Bosch 的 nuPlan + SMART reactive agents 复现框架，提供 `planning/checkpoint` | 不是 CARLA 适配器，也不是 RIDE ego planner；AGPL-3.0 需要单独处理再发布边界 | 作为第二个响应环境或交叉验证；保留独立环境，不能把其输出和 CARLA TM/BehaviorAgent 混成同一个模型 |
| `GameFormer-main`（archive `fcb0d4a...`） | WOMD interaction prediction 与 open-loop planning；README 明确不含 WOMD closed-loop | 当前分支不能作为 nuPlan 闭环强规划器；也没有可假设存在的通用 checkpoint | 若 RQ3 必须有 learned planner，另行获取作者 `GameFormer-Planner` 仓库并按其官方数据/权重流程复现；不能修改当前 WOMD 分支硬接 CARLA |
| `learning_responsibility_allocation-main`（archive `107c7b...`） | nuScenes/trajdata 上责任分配模型训练脚本与静态场景资源说明 | README 明确 closed-loop simulation 未发布；没有 nuPlan/CARLA 适配器 | 仅作为责任分配/安全过滤近邻的概念对照；若要放入 RQ3，需新建数据适配、动力学闭环和 ex-ante 选择接口，工作量相当于新项目 |
| `r3r_decentralized_gatekeeper-main` | Julia 的 2D 去中心化 Gatekeeper、DoubleIntegrator/Dubins 仿真与安全机制 | 非 learned planner；无 nuPlan/CARLA、地图、感知和 checkpoint 接口 | 仅在 toy/Waymax 作为安全过滤或理论近邻；不要放入 CARLA planner 主表，除非完成独立动力学/地图适配并锁定公平预算 |

## ReactSim 权重和环境

`zzzy0729/ReactSim-Bench-Models` 是 Hugging Face **model** repository。`smart.ckpt` 和 `mtr.pth` 是作者发布的响应世界模型 checkpoint，不需要先从源码训练才能复现作者 benchmark。RIDE 的 `scripts/14_download_reactsim_weights.sh` 使用 `--repo-type model`、固定 revision，并检查 Git-LFS pointer 与 SHA256。训练只在以下情况启动：需要重训作者模型、换数据集、或做明确的 domain-adaptation ablation；不能用自训 checkpoint 替代作者 checkpoint 后仍声称“官方复现”。

ReactSim 评测至少要同时锁定：

1. ReactSim commit、模型 revision 和 checkpoint SHA256；
2. nuPlan devkit 版本、地图版本、原始 DB 文件集合与 SHA256；
3. `custom_behaviors` 文件集合、官方 `scenario_ids_{train,val}.json` SHA256；
4. `nuplan_processed/{train,val}/processed_scenarios` 的生成命令和覆盖率；
5. 独立 Python/CUDA 环境、GPU、batch、replan interval 和 rollout seed。

`scripts/17_reactsim_preprocess_checked.sh` 在预处理前做只读 token 审计。`ELIGIBLE_BY_SQL_JOINS` 只表示最小 SQLite 关联成立，仍需作者 `build_cache.py` 成功生成完整 validation cache；`RAW_TOKEN_ABSENT`、`FILTERED_SCENE_BOUNDARY`、`MISSING_LIDAR_LOG_JOIN` 和 `SQLITE_INTEGRITY_FAIL` 必须分别记录，不能删除失败 ID 后把剩余样本称作完整 benchmark。

## RIDE RQ3 的最小接入契约

外部 planner 只有同时满足以下条件才进入 RQ3 主表：

- 在开发集锁定 commit、checkpoint、配置、预处理和候选预算；
- 在独立 holdout 上输出 `method_id`、`scene_id`、`initial_state_digest`、候选动作/轨迹摘要、`selected_action`、选择分数和 `choice_locked_before_rollout=true`；
- 由同一个独立响应评测器执行候选动作，禁止用 rollout 后的碰撞/制动结果反过来决定 planner 选择；
- 对失败、无候选、无 anchor、动作无效、模型崩溃和输入缺失逐项计数，统一输出 `UNKNOWN` 而不是删除；
- 在相同初态、相同 A/B 身份、相同地图和相同计算预算下，与 RIDE 的 proxy selector、硬安全过滤器和至少一个强 learned planner 做 paired per-scene 比较。

因此，下一阶段的顺序是：先修复 ReactSim token/DB 覆盖并完成 SMART/MTR 官方 validation 复现，再在 RIDE holdout 上实现一个 planner adapter 的 ex-ante 选择记录，最后才扩展到 GameFormer-Planner 或其他强基线。将源码复制到 CARLA、改动模型预测或只运行单个成功场景都不足以构成 RQ3 证据。
