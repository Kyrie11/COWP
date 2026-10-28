# RIDE v10：CARLA 科学机制验证工程

本工程建立在用户上传的 `RIDE_V9_CARLA_STABILITY_20261028_MAPFIX.zip` 上，保留显式地图切换机制。目标是使 RQ1/RQ2 能够被严谨地证伪或支持，而不是人为调参制造碰撞。**尚未在真实 CARLA 服务器上执行新四车实验；不存在已得到的四车第三方风险正例。**

- **旧实验再审计**： `python -m ride.v10_mechanism_report --input /path/cut_in.jsonl --output /path/report.json`。用户最新上传 M0 数据经回放为 `20 valid, 3 matched cross-family comparisons, 0 strict reversal`，RQ1/RQ2 未建立。
- **三车 M0**：保留 `scripts/23_v8_smoke.sh`，更新 BehaviorAgent 名义速度封顶，需使用全新 frozen 输出。
- **四车链式风险实验**： `scripts/31_v10_chain_m0.sh`，固定 B–C 为第三方受影响车辆对，记录 Ego/A/B/C，保留所有 invalid 与 cruise 控制。
- **四车确认**： `scripts/33_v10_chain_confirm.sh`，要求开发 M0 有有效跨模型动作对，参数冻结与物理走廊排重后运行。
- **地图准备**：原用户代码 `RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04`；仅在本人的空闲 CARLA 服务器上显式执行，不能在 rollouts 内偷偷加载地图。

详细科学边界、旧实验测量、速度混杂、命令和可能转向：[`docs/V10_RQ12_SCIENCE_AND_EXECUTION_CN.md`](docs/V10_RQ12_SCIENCE_AND_EXECUTION_CN.md)。

代码审计通过本地 `pytest` 126 项及编译、Shell 语法；不代表 CARLA 路径、Vulkan、真实交互或第三方风险已实测通过。
