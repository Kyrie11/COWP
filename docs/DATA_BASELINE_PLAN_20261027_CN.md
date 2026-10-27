# 数据集与强基线后续计划（基于 RIDE.zip 的静态审计）

## 可确认的输入与限制

当前 RIDE 工程把三个东西明确区分开：

1. **WOMD v1.3.1 原始 TFExample**：用于从日志盲抽样、身份/几何抽取、观察响应校准；`ride.womd.observational` 只能做 logged observational fit，不能当作 `do(ego action)` 的反事实真值。
2. **ReactSim-Bench 的 custom_behaviors**：`07_download_reactsim_data.sh` 只下载约 28 MB 的作者 AV 反事实行为文件；脚本已经明确写出，完整评测还需要独立获取 nuPlan state/map 数据。因此这批文件单独不能支撑跨模型闭环评测。
3. **research_baselines**：`08_clone_external_baselines.sh` 只克隆作者源码，不等于下载 checkpoint、成功复现或已接入 RIDE。任何论文表格必须记录 commit、checkpoint、预处理版本、输入可见性、推理硬件/预算和失败原因。

当前容器看不到用户服务器的 `/data0/senzeyu2/dataset/...` 挂载，所以下述是基于代码接口和官方仓库说明的执行计划，需在服务器上先运行 inventory/doctor 验证实际文件。

## WOMD：先做什么，暂时不做什么

### 立即做

- 在 `/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1` 下确认 `uncompressed/tf_example` 及 `validation` 的 shard 数量、大小、命名和可解码性；不要只按目录存在判断。
- 按 RIDE 既有脚本先跑 `scripts/09_native_womd_cohort.sh dev`（4 shards、有上限的盲抽样），再跑 `scripts/11_observational_response_calibration.sh`。先查看 cohort manifest、保留/剔除原因、heading-velocity 异常比例、constant-velocity 对照和 IDM 家族的拟合残差。
- Gate 0 通过后，再建立冻结的 validation cohort。代码中的 `full150` 只有在 validation 150 shards 全部完整且 schema 通过时才运行；它使用 `holdout_fraction=0.20`，应把 holdout 用于最终检查，不能反复调参。
- 每个 cohort 固定 scene ID、raw shard/record 索引、manifest 哈希和历史排除表；按 scene 去重，禁止同一 segment 同时进校准和测试。

### 暂时不要下载/扫描

- 不需要先下载 WOMD training 或 testing 来证明 RQ1/RQ2；当前问题是验证机制是否存在与跨模型是否稳定，validation + 盲抽样已经足够作为第一阶段。
- 若后续要训练 GameFormer/交互预测模型，才单独申请与官方版本匹配的 training（用于训练）和 `validation_interactive`（用于交互预测测试），并保留完全独立的 RIDE holdout。不能把训练集或 validation_interactive 的调参结果当作 RIDE test。

## ReactSim-Bench / nuPlan：需要的最小闭环组合

ReactSim-Bench 本身是响应世界模型评测，不是一个已接入 RIDE 的 planner。为了把它用于 RQ1/RQ2，最小闭环材料应包括：

- ReactSim-Bench 的完整场景索引、AV counterfactual 行为、作者提供的评测脚本/协议和与其版本匹配的模型 checkpoint（若要比较作者模型）；
- 对应版本的 **nuPlan state/database 与 map**（至少 mini/dev 规模先打通，之后固定 test/holdout；不要只下载 custom_behaviors）；
- 与 RIDE 同一批 ex-ante ego candidate/action IDs，或明确的映射层；
- 独立的响应模型输出（ReactSim、SMART/interactive-closed-loop、IDM 仅作规则基线），以及 collision、map-rule、kinematic feasibility、第三方 pair risk、ego progress 的统一指标。

推荐顺序：先在 nuPlan mini/dev 上只做 schema、坐标系、候选动作回放和响应模型接口 smoke；再扩大到作者 benchmark 的三类场景。ReactSim 结果只能作为**独立响应环境/评测器**，不能直接证明 RIDE 算法更强，也不能把作者 AV 行为文件当作真实人类反事实。

如果缺少 nuPlan 授权/磁盘空间，ReactSim 只能写成外部响应有效性 pilot，RIDE 主结论仍需在 WOMD logged calibration + CARLA 物理闭环中限定为“仿真/证据支持”。

## `research_baselines` 的角色与优先级

### P0：先跑、成本最低且与 RIDE 命题直接相关

1. **SMART / interactive-closed-loop（Bosch）**：优先作为 CARLA/nuPlan 外部 reactive evaluator，而不是把它当作一个 planner baseline。先确认仓库中是否包含可下载权重、nuPlan 版本、输入特征和 rollout 接口；用同一冻结 action set 回放，至少比较 IDM、SMART 和另一结构独立响应模型的第三方 pair risk/ego progress。
2. **GameFormer-Planner（与 `GameFormer` 区分）**：如果要作为强 planner，优先在官方支持的 nuPlan mini/dev 上复现一个端到端闭环 checkpoint，然后导出每个场景的 ex-ante action choice 和 candidate scores。原始 GameFormer 仓库明确包含 WOMD 交互预测和 WOMD open-loop planning，但不包含 WOMD closed-loop；nuPlan planning 应使用作者的 GameFormer-Planner 仓库及其数据预处理。若只能得到预测轨迹而不能导出 action choice，应标为 prediction/evaluator baseline，不得放进 RQ3 的 planner regret 表。
3. **硬 courtesy / robust safety filter**：先在 RIDE 内实现透明、可调参数的 hard/soft courtesy 参照，并以同一候选、同一计算预算调参；这不是论文作者代码，但能验证 RQ2 正例是否仅由显然的保守规则即可消除。

### P1：作为机制近邻或安全过滤器，不直接宣称“强规划器”

- **RA-CBF**：官方工作是责任分配与控制 barrier filter。它更适合作为“责任感知安全过滤器”近邻：在相同候选轨迹上后处理，报告是否消除 third-party collision、代价多少 ego progress。必须保留其训练/责任假设和控制约束；不能把运行 RA-CBF 当作对 RIDE 选动作机制的完整复现。
- **R3R_Julia gatekeeper**：如果能编译，则作为形式/去中心化安全过滤器敏感性检查；没有候选排序或学习 planner 接口时不纳入主 RQ3 排名。

### P2：验证外部响应有效性，不是 planner baseline

- **ReactSim-Bench**：用作者场景和独立模型评估响应/世界模型，报告其和 CARLA TM、BehaviorAgent、SMART 的结构差异；不能把其 benchmark score 直接当 RIDE planner score。

## 公平比较协议（RQ3 前必须冻结）

每个 planner/selector 必须接收相同的：

- 初始观测历史、地图、route goal、candidate action IDs 和候选时域；
- 可用的响应证据（不能让某个方法偷看 CARLA outcome、真实未来或评测模型 rollout）；
- rollout 时间步、GPU/CPU、候选数量和后验评估预算；
- 碰撞、规则、舒适、ego progress 和第三方 pair burden 指标。

执行顺序：先用训练/开发数据调参；锁定 checkpoint、配置、commit、候选选择 manifest 与 SHA；再在未看过的 RIDE holdout 上运行一次；之后把锁定选择交给独立响应评测器。报告 paired per-scene regret、collision regret、unsafe-selection precision/recall、progress difference、coverage/UNKNOWN，而不是只报平均碰撞率。所有 no-anchor、invalid-action、模型崩溃和无匹配动作都要进入审计计数。

## 建议的下载/运行顺序

1. 在服务器上做 WOMD raw inventory + TFExample decode smoke；完成 `dev` cohort 和 observational calibration。
2. 跑四类 CARLA 场景的全量冻结审计（所有 anchor/repeat，含删失原因）；确认 RQ1/RQ2 是否有跨响应模型的有效正例。
3. 只在上述 Gate 通过时，获取 nuPlan mini/dev state+map，并打通 ReactSim-Bench / SMART 的响应接口；先不下载全量 nuPlan。
4. 复现 GameFormer-Planner 一个官方 checkpoint；若依赖/权重不可得，明确记录为 unavailable，先用硬 courtesy + RA-CBF 过滤器完成机制消融，不伪造强基线。
5. 在独立 holdout 上锁定所有 planner choices，再运行 CARLA、WOMD/Waymax 和 ReactSim/SMART 响应环境，做 paired statistics 和 cluster bootstrap。
6. 只有当 RQ1/RQ2 在至少一个真实记录校准的外部响应模型和一个结构独立的响应环境中成立，才开始 RQ3 大规模强基线比较；否则把 RQ3 限制为失败/消融分析，避免先做昂贵基线。

## 实施上的硬检查

- `RIDE_OUTPUT_ROOT` 每一阶段新建，禁止覆盖冻结 manifest；所有输出带 run_end 完成标记。
- 记录 `git rev-parse HEAD`、checkpoint SHA/URL、环境 lockfile、GPU、seed、scene manifest SHA。
- 不把 TM 与 BehaviorAgent 直接称为两个人类真值；两者是 CARLA 内不同控制器。
- 不把 full WOMD logged fit 写成反事实因果证明；不把 ReactSim AV-only 文件写成完整 benchmark；不把源代码 clone 写成 baseline 已运行。
