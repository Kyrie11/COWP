# RIDE 工程清洁、依赖闭包和科学逻辑审计

**审计对象**：用户提供的 RIDE ZIP 及 `ride升级.md` 两次模型回答；以第二次回答指定的 RIDE 决策错误研究为最终研究定位。以下所有「修复」指代码与静态/合成测试中的实现修复，**不等于已在用户的 CARLA 服务器、原生 WOMD 或 Waymax 上验证物理结论**。

## 一、为什么出现 `No module named research_probe.carla_h2.run_carla`

原 ZIP 中 `ride/` 是薄层研究分析包；真实 CARLA 执行器、WOMD 提取与观察拟合、Waymax IDM 运行器分散在旧工程 `research_probe/` 代码下。原 scripts 通过把旧工程目录追加到 `PYTHONPATH` 继续调用 `python -m research_probe.carla_h2.run_carla`。一旦旧目录没有被正确纳入 Python 搜索路径，或只有 RIDE 的源码被拷贝，就会报上述 ModuleNotFoundError。**改环境变量只是在掩盖包依赖，不是彻底工程清洁。**

现在 `scripts/03_precommit_scenes.sh` 调用 `python -m ride.carla.run --manifest-only`，`scripts/03_run_carla_case.sh` 调用 `python -m ride.carla.run`。所有 Python 入口均以 `ride` 为根包，直接解压后在项目根目录可以执行 `--help`；真实 CARLA 物理执行时仍需安装外部仿真环境。

### 保留且已迁移的真实依赖

| 科学功能 / 被哪个脚本调用 | 独立 RIDE 代码 | 外部依赖 |
|---|---|---|
| `03_precommit_scenes`, `03_run_carla_case` | `ride/carla/run.py`、`ride/carla_scenarios.py`、`ride/carla/protocol.py`、`ride/geometry.py` | CARLA 0.9.16，`agents.navigation` 的同版本源码 |
| `03_run_carla_case`, `04_analyze_carla` | `ride/evidence.py`、`ride/carla/analyze.py`、`ride/science.py`、`ride/ranking.py`、`ride/gates.py`、`ride/combine.py` | 无新的重型仿真依赖 |
| `03_support_ablation` | `ride/support.py`、`ride/carla/run.py` | CARLA 0.9.16 与 agents |
| `05_waymax`, `05b_waymax_native` | `ride/waymax/paired_idm.py`、`ride/womd/reactive.py` | Waymax / JAX / TensorFlow / 原始 WOMD TFRecord |
| `09_native_womd_cohort` | `ride/womd/native.py`、`ride/datasets.py` | TensorFlow、NumPy、原始 WOMD 数据 |
| `11_observational_response_calibration` | `ride/womd/observational.py`、`ride/womd/reactive.py`、`ride/womd/footprints.py` | NumPy + 已生成的 native cohort |
| `06a_lock_proxy_selectors`, `06_compare_locked_planners` | `ride/selectors.py`、`ride/commit.py`、`ride/decision.py` | **必须外部产生**独立、事先冻结的预测和实际选择 |
| `06b_burden_proxy_ablation` | `ride/burden.py`、`ride/selectors.py` | 同一外部预测 CSV |
| `01_doctor`, `10_bundle_run_outputs` | `ride/doctor.py`、`ride/datasets.py`；打包脚本自带代码 | 可访问的数据/服务环境 |
| `07_download_reactsim_data`, `08_clone_external_baselines` | shell 下载工具，**不作为强基线复现** | 网络、Hugging Face Hub / Git |

### 删除的遗留及理由

- 删除旧工程的**整个**源码、训练管线、旧测试、配置、缓存、历史 shell 操作脚本、已供应第三方项目副本与旧名称入口。没有继续依赖旧项目的 Python 包。
- 保留之前使用过的场景 ID **去重排除表**（`references/prior_development_exclusion_ids.txt`，共 7,200 个 ID）：它是研究划分防泄漏所需输入信息，不是旧工程代码，也**不能**被用来训练或挑选测试时的结果。
- 外部 GameFormer / RA-CBF / 反应式模拟器不再供应成大体积仓库副本；研究者可自行通过 `scripts/08_clone_external_baselines.sh` 下载官方代码、锁定 commit、独立安装和复现。

## 二、实际发现并修复的逻辑问题

| 层次 | 原有潜在不合理行为 | 当前处理 |
|---|---|---|
| 路径 | scripts 对额外的旧目录和不属于 RIDE 的入口模块存在硬依赖 | 整体迁入 `ride.carla`、`ride.womd`、`ride.waymax`，脚本根目录自动发现，`pip install -e .` 可独立安装 |
| CARLA 地图 | 并线相邻车道筛选以 5 m 作为最小距离，且比较不同纵向位置的路点，容易过滤真实 3.5 m 车道 | 改为**同纵向 road waypoint** 的 2.4–6.5 m 距离过滤；用伪 waypoint 回归测试 |
| CARLA 版本/依赖 | 对 CARLA 0.9.15 与用户 0.9.16 的要求不一致，BehaviorAgent 路径隐藏 | 显式限定 0.9.16 server/client；只在匹配源码文件存在时加入 agents PythonAPI 路径；doctor 诊断缺依赖 |
| 场景一致性 | 运行时临时选择不同的地图位置或 lane 可能使分支不配对 | 先冻结 anchor/map/seed；保存 road ID、lane ID、yaw；恢复时校验身份，错配直接拒绝 |
| 不真实执行 | 只按 profile 名称比较即使 ego 未实际变道/动作不明显，也可产生「干预效果」统计 | 必须校验轨迹时间、ego 实际位置/速度差异；并线必须进入目标车道，路口必须完成交互接近条件 |
| 纵向进展/TTC | 用轨迹路程充当任务目标进度；用标称跟车距离加累计里程计算 TTC，特别在转弯、变道时错误 | 记录几何投影/目标距离减少与独立路程；只对仍为同路同车道 follower pair 的实际速度、间隙计算 TTC；0 间隙 TTC 为 0 而非无穷大 |
| 碰撞语义 | 将几何重叠直接当成物理碰撞，或忽略传感器不一致 | CARLA 物理传感器报告 collision，OBB 仅作重叠/有效性诊断；明显不一致试验标记 invalid |
| 路况/场景真实性 | 离开车道或在红灯中移动仍可能被当作低负担安全响应 | 记录 road/lane/red 状态，不合格作为排除理由；不伪装成安全见证 |
| 研究证据 | 事后从所有动作组合自由挑最大效果，存在选择偏差 | 配对动作方向预注册为 `early_slow→late_slow`、`mild_early→mild_late`，输出包括所有无效及缺失原因 |
| RQ1 定义 | 一个模型的预测/风险变化可能被误述为真实规划错误 | 严格区分后验模拟排序变化、真实 ex-ante 选择和真实模型校准；无强基线时 RQ3 不标通过 |
| 决策 regret | 如果「安全但急刹」与「碰撞但不急刹」比较，可能产生虚假的刹车后悔值 | 采用**字典序**第三方碰撞优先；只有碰撞等级相等时才计算刹车代理 regret |
| 有限证据 | 小响应库找不到 witness 被误当不可行；共享候选轨迹若不同只对比二值状态 | NARROW/WIDE 强制集合嵌套；重跑共同候选还要核对轨迹和进展；未找到只给 UNKNOWN，不宣称全局 UNSAT |
| Waymax | 原始 SDC 的本地索引被假设为 0、老 roadgraph 规格被套用、错误被静默算成测试通过 | 从 raw TFExample 校验真实 SDC ID、原始位置与 shard/record，要求 WOMD v1.3.1 规格；执行异常留在 errors 并返回非零状态（待实机 API 验证） |
| WOMD 响应校准 | 纵向响应模拟对速度方向与车辆朝向矛盾的异常状态盲目模拟 | 排除明显 heading–velocity 不一致；保留 constant-velocity 对照，不把 ADE 当真实反事实真实性保证 |
| 结果完整性 | 异常中断的 JSONL 可能被读取并作为无事件样本；重复覆盖破坏冻结 | 所有 CARLA outcome 要求 run_end 与 trial_count 一致；明确拒绝旧文件覆盖；Waymax 新样本 panel 绑定 manifest+ID 哈希 |

## 三、当前 RQ / 场景 / 实验状态矩阵

**「可执行入口」仅意味着代码与参数已提供；不是「完整实验已跑通」。**

| 实验 | 研究任务 | 对应命令 | 当前实现状态及局限 |
|---|---|---|---|
| CARLA `convoy` | RQ2 连锁制动/固定第三方 pair | `03_precommit_scenes.sh convoy` → `03_run_carla_case.sh convoy` | 物理执行器已迁移，当前环境没有 CARLA；需真实 smoke，严重性不得从脚本逻辑推断 |
| CARLA `cut_in` | RQ2 并线强制退让及 hard courtesy 价值 | 同样脚本换 `cut_in` | 地图车道和变道完成性已增强；变道控制器仍是轻量 waypoint PID，不是完整导航算法 |
| CARLA `intersection` | RQ1/RQ2 联合可行与独立响应差距 | 换 `intersection`，可在新冻结目录尝试 Town03/Town05 | 有有限 joint-speed positive witness；没有通信/合作策略的全空间搜索及交叉道路规则证书 |
| CARLA `support_gap` | 有限 recourse 库不完备与 UNKNOWN | `03_support_ablation.sh` | NARROW/WIDE 相同锚点、种子、候选一致性审计；不是认证不可行性 |
| CARLA 响应模型变化 | RQ1 同候选后验排序翻转 | `04_analyze_carla.sh` | TM/BehaviorAgent 都属于 CARLA、尚未独立观测校准 |
| 预先冻结 planner | RQ3 真实在线选择错误 | `06a_lock_proxy_selectors.sh`/`06_compare_locked_planners.sh` | 只有 proxy selectors、外部选择审计接口；RACP/GameFormer/HYPE 官方 checkpoint **未接入** |
| burden on/off | RQ3/增量负担 | `06b_burden_proxy_ablation.sh` | 相同预测下的**代理筛选**消融，原始神经规划/teacher 机制的完整对照**尚未实现** |
| 原生 WOMD | RQ0 数据身份、盲抽样/避免历史泄漏 | `09_native_womd_cohort.sh dev/full150` | 代码完成，原始 150 shard 完整性/读 schema 必须在用户服务器上验 |
| WOMD logged fit | RQ0 IDM 家族 vs CV | `11_observational_response_calibration.sh` | 只拟合观察日志，不识别干预下的真实响应 |
| Waymax 对照 | RQ0/WOMD 第三方交互 pilot | `05_waymax.sh`、`05b_waymax_native.sh` | 版本敏感接口，缺真实 Waymax 环境；只有 IDMRoutePolicy，不等于 ReactSim 或 SMART-reactive |
| 独立 reactive benchmark | RQ1/RQ2 跨世界模型一致性 | `07_download_reactsim_data.sh` | 只下载 ego 行为材料；nuPlan 地图、反应模型和评测集成尚需实现 |
| 统计/报告 | RQ1–RQ3 自证风险管控 | `ride.evidence`、`ride.gates` | 报告尝试/成功/删失、簇 bootstrap；没有真正临床式因果识别或代表性人类交通比率 |

### 建议的后续研究闸门

1. **Gate 0（优先于任何新算法）**：服务器验证 CARLA 0.9.16 server/client/BehaviorAgent 与至少一组 `convoy` 物理轨迹；人工检查地图冻结、初始 overlap、日志完成、动作实际发生、TTC 与 ego progress。Waymax 修复 API、原始 SDC 及 map 验证，WOMD logged fit 至少不应无条件劣于 CV。
2. **Gate 1（RQ1/RQ2）**：分别对 `convoy/cut_in/intersection/support_gap` 收集不筛选的全部锚点、重复种子与删失原因。在独立校准过的结构不同的 reactive 模型上检查相同动作/固定车辆对/任务进展以及非 ego–非 ego pair 的风险趋势。用 ReactSim-Bench、SMART-reactive 或 nuPlan-R 等作外部评测；不能将 TM + BehaviorAgent 当作两项独立的人类真值。
3. **Gate 2（RQ3）**：在冻结同一候选集合与可用信息后，接入作者实现或严格标记的复现版 hard courtesy、RACP、HYPE/GameFormer/INTERACT（以及适当 safety filter），锁定动作再运行独立评测，报告第三方风险对 ego progress 的 Pareto 曲线、配对置信区间与强基线调参预算。
4. **停止条件**：若进展匹配且跨模型成立的可避免第三方风险不能稳定复现，就不应宣称 RIDE 主线获得支持；转而检验有限响应证据、三值 UNKNOWN 和决策错误的更窄问题。**不得**因为测试代码可运行就写论文结果。

## 四、已验证事项 vs 必须在真实服务器验证的事项

- 代码/打包可验证：Python 入口导入、`--help`、bash 语法、模拟场景回归测试、同一 RIDE 压缩包在**新目录**独立解压、文件名和路径不含旧工程依赖。
- 当前沙箱无法验证：CARLA 0.9.16 服务端、`agents`、地图 lane 安全、BehaviorAgent 闭环控制、Waymax/JAX API、原始 1.3.1 WOMD 文件、nuPlan / ReactSim 数据、任何真实物理或统计性论文结果。
- 因而「**清洁且相对路径完全自洽**」和「**用户服务器所有外部依赖均已安装、全部实验实机跑通**」是两个不同的工程声明。本 ZIP 可交付前者，后者必须先满足上述外部环境与实验条件。
