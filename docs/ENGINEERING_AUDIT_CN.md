# RIDE 代码包相对于用户 COWP(1).zip 的修改与实际验证

## 设计原则

保留 `COWP/` 的原始主要代码及训练/Waymax/数据接口，在项目顶层增加独立的 `ride/` 研究问题审计层。**不再叠加未经证实的新的规划神经网络**。原工程仍是历史研究实现，不应宣称最终论文算法已完成。

## 对原有 COWP 修改的范围

1. `COWP/research_probe/carla_h2/run_carla.py`：CARLA 客户端与服务端双 0.9.16 检查；地图依赖型四类场景、frozen anchor、同步物理配对重置、实际路径进展、非 ego 碰撞以及联合候选/备份诊断。
2. `COWP/research_probe/carla_h2/protocol.py`：物理实验明确受限的 legality、pair risk、hard brake、TTC、独立角色归因等证据指标。
3. `COWP/NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh`：CARLA 版本门槛从 0.9.15 调整到 0.9.16；其独立 `waymax_*` 分支保持原有语义。
4. `COWP/README_EICDRR_V4_CARLA_WAYMAX_CN.md`：标记为历史说明并指向新顶层运行手册。

## 新增模块

- `ride/carla_scenarios.py`：场景生成器，不是已核实的真实问题实例。
- `ride/geometry.py`：朝向矩形重叠；避免圆盘/中心距离导致误检。
- `ride/science.py` / `ride/evidence.py`：初始状态一致性、任务进展匹配、真实 CARLA 执行对的双边干预差异，原始失败 trial 不丢弃。
- `ride/ranking.py`：不同应答家族对**同一** ego 动作对在后验风险排序上是否严格翻转。
- `ride/support.py`：严格嵌套的 narrow/wide response bank；无正向 witness 仅是 UNKNOWN，不是 UNSAT。
- `ride/selectors.py` / `ride/commit.py`：由外部、事前预测分数形成的可重复基线选择与文件哈希。仅为简化参考评分器，不冒充 GameFormer/RACP 的完整方法。
- `ride/decision.py`：独立执行结果中的选择后悔值，统计 ego unsafe 选择，不在分母中悄悄丢弃。
- `ride/freeze_ids.py` / `ride/datasets.py`：原始 WOMD 盲 ID 选择和文件盘点。
- `ride/gates.py` / `ride/doctor.py`：严格阻断未经验证的研究声明与环境前置检查。
- `scripts/00_env.sh` 至 `11_observational_response_calibration.sh`：CARLA、Waymax、WOMD、ReactSim 数据及作者基线源码下载/运行入口。

## 已完成的测试与未测试的部分

- 新增 RIDE 13 项 `pytest` 单元测试通过。
- 保留的 COWP 84 项选择性 `pytest` 单元测试通过；两条 PyTorch Transformer 运行警告，未阻断。
- 脚本 `bash -n` 静态语法与 Python `compileall` 通过。
- 合成数据的 `ride.evidence` → `ride.ranking` → `ride.gates` 全流程命令通过，且明确标为 `SYNTHETIC_ONLY`。
- **没有在用户机器上运行 CARLA 0.9.16、Waymax、原始 WOMD 全量抽样、ReactSim-Bench、真实论文 baselines**；因此严格 RQ1~RQ3 的真实系统实验目前均未达到证据门槛。

## 高优先级运行风险

- CARLA 0.9.16 Linux Python wheel、CARLA root PythonAPI 下 `agents/` 与真实服务器版本必须匹配。`pip install carla` 不能代替导航代理目录。
- CARLA 原始地图中的 junction 连接和 traffic-light 可能使某场景无合格 anchor；不要在缺失时伪造失败样本。
- CARLA 几何/责任/行为模型仅是人工实验环境，不能声称与 WOMD 中某个城市路口完全同一场景。
- IDM/TM/BehaviorAgent 没有完成独立人类校准。先用原始 WOMD 观测拟合否定明显不合理模型，再用不同结构反应环境验证外推。
- 若 22-ID 旧版 Waymax cohort 不存在，使用 `09_native_womd_cohort.sh` 与 `05b_waymax_native.sh`，后者明确是新的盲抽样 panel，**不是原 22 优先案例的复现**。
- 只有 RQ1、RQ2 的高置信正例及强基线真实复现（RQ3）都支持同一具体机制时，才考虑升级主算法并锁定正式 CCF-A 投稿标题。
