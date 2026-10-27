> **2026-10-27 actor lifecycle hotfix**：上传 smoke A 的全零初始位置造成无效的“初始重叠”，随后 smoke B 检查到 3 辆残留车。请先读 [中文修复、清理和实验有效性说明](docs/CARLA_ACTOR_LIFECYCLE_FIX_CN.md)。新增 `python -m ride.carla.actor_lifecycle`（默认只读）、首次同步 tick 与服务端验证的清理。`RIDE_SMOKE_PASS` 只是物理试验检查，不是 RQ1/RQ2 的研究结论。

> **2026-10-27 CARLA reload timeout fix:** If `convoy` fails at `client.reload_world()` after `RIDE_TRIAL_RESET_BEGIN`, see [docs/CARLA_RELOAD_TIMEOUT_20261027_CN.md](docs/CARLA_RELOAD_TIMEOUT_20261027_CN.md). The default trial reset is now `actors` (reuse a dedicated current Town, respawn only RIDE vehicles) rather than `reload_world` per trial. Scientific cross-reset caveats and isolated tests are documented there.

# RIDE — Response-Model-Induced Decision Errors in Interactive Driving

> **2026-10-27 环境/端口修复说明：** 见 [`docs/CARLA_PORT_ENV_FIX_CN.md`](docs/CARLA_PORT_ENV_FIX_CN.md)。
> 默认 TM 改用 **8100**，CARLA RPC 仍默认 **2000**；若需要在 GPU 1 开独立 CARLA，可配置 **CARLA_RPC_PORT=2002 / CARLA_TM_PORT=8102 / CARLA_GPU_ADAPTER=1**。在启动前运行 `bash scripts/00_diagnose_carla.sh`。


> **研究验证工程（research gate），不是已完成的 RIDE 规划器，也不是已证明的论文结论。**
> 本工程只使用 `ride.*` 包名；CARLA / Waymax / WOMD 的运行代码均在本仓库内，不需要任何旧项目源码目录。

## 1. 研究问题与验证顺序

本工程以 `ride升级.md` **第二次**「#大模型回答」中确定的 RIDE 方向为准。研究对象是：在**任务进展近似相同**的 ego 候选之间，有限或失准的他车响应证据是否会导致候选风险排序或实际决策错误，并带来本可避免的、固定第三方车辆对的碰撞风险或负担。

| 研究问题 | 必须获得的证据 | 本工程实际可提供的证据等级 |
|---|---|---|
| **RQ1** 响应模型是否改变规划相关排序？ | 同场景、同候选、不同独立响应模型下的**严格**风险排序翻转 | `ride.ranking`：CARLA TM / BehaviorAgent **后验排序敏感性**；未等同实际在线 planner 决策 |
| **RQ2** 风险是否由 ego 动作诱发、任务匹配且可避免？ | 同初始状态、同固定车辆对、合法且进展匹配的动作对，独立反应模型重复支持 | `ride.evidence` + `ride.carla.analyze`：有明确删失条件的 **CARLA 物理模拟对照**；尚非人类驾驶因果证明 |
| **RQ3** 强基线经过公平调优仍会选错吗？ | hard courtesy、RACP、HYPE/GameFormer 等冻结选择、同候选/预算、公平闭环评估 | `ride.selectors` 为**代理评分器**；`ride.decision` 可审核外部冻结选择，但**没有**官方强基线模型接入或结果 |

只有 RQ1、RQ2 获得跨模型且经过观测校准的稳健支持，才值得继续 RQ3 和新算法设计。**不能**把 RACP/GameFormer 的源码下载、CARLA 假设模型分歧或合成单测算作强基线实证。

## 2. 清洁目录（从 RIDE.zip 解压后直接使用）

```text
RIDE/
├── README.md
├── pyproject.toml
├── ride/
│   ├── carla/             # CARLA 0.9.16 物理场景执行、结果协议、配对分析
│   ├── carla_scenarios.py # 基于道路地图生成四类场景、冻结及验证 waypoint
│   ├── womd/              # WOMD 原生读取、观察拟合、简化反应模型、几何诊断
│   ├── waymax/            # Waymax 版本敏感的 WOMD/SDC 配对 IDM 探针
│   ├── science.py         # 统一配对合法性、执行差异和进展匹配协议
│   ├── evidence.py        # RQ2 配对第三方风险/删失/统计
│   ├── ranking.py         # RQ1 不同响应策略下排序变化
│   ├── support.py         # S4 窄/宽有限响应库、UNKNOWN 语义
│   ├── selectors.py       # 外部预测下的简化 hard/soft courtesy 评分器
│   ├── decision.py        # 预先冻结的选动作与事后风险后悔值审核
│   ├── burden.py          # 固定预测下 burden on/off **代理**消融
│   ├── gates.py           # 保守的 RQ1–RQ3 状态，不自动宣称通过
│   └── ...                # doctor、datasets、combine、freeze_ids 等实用模块
├── scripts/               # 00–11 执行脚本，全部基于当前 RIDE 根目录
├── tests/                 # 无仿真器依赖的合成/回归测试
├── inputs/                # 预测输入 CSV **表头模式**，不含伪造结果
├── references/            # 历史数据 ID 排除表（不是旧项目代码）
└── docs/                  # 研究定位与工程审计说明
```

所有脚本自动通过自身目录定位 `RIDE_HOME`。模型导入路径为：

- CARLA：`python -m ride.carla.run`
- CARLA 配对：`python -m ride.carla.analyze`
- 原生 WOMD：`python -m ride.womd.native`
- 观测校准：`python -m ride.womd.observational`
- Waymax：`python -m ride.waymax.paired_idm`

**不需要**手工设置指向旧项目的 `PYTHONPATH`。`pip install carla` 不会自动提供 `agents.navigation.behavior_agent`，它应来自匹配版本 CARLA 源码安装目录下的 `PythonAPI/carla/agents/`。

## 3. 在有 CARLA 0.9.16 的服务器上启动最小实验

```bash
unzip RIDE.zip
cd RIDE

conda create -n ride-carla python=3.10 -y
conda activate ride-carla
python -m pip install -e .
python -m pip install carla==0.9.16 pytest

export CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
# 建议每次实验独立一个输出目录；不能覆盖冻结的场景或评测结果
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/run_001
source scripts/00_env.sh
bash scripts/01_doctor.sh
python -c "from agents.navigation.behavior_agent import BehaviorAgent; print('CARLA agents import OK')"
python -m ride.carla.run --help
```

在**终端 A**（服务器需要显示/GPU/依赖配置时按实际环境调整）：

```bash
conda activate ride-carla
cd /path/to/RIDE
source scripts/00_env.sh
bash scripts/02_start_carla_headless.sh
```

在**终端 B**：

```bash
conda activate ride-carla
cd /path/to/RIDE
source scripts/00_env.sh
python -m ride.doctor --server             # 必须核对 server/client 均为 0.9.16

export CARLA_TOWN=Town04
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
bash scripts/01b_probe_anchors.sh convoy Town04  # 先确认候选数 > 0
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

`03_precommit_scenes.sh` **只冻结**道路场景和车辆身份；`03_run_carla_case.sh` 在相同冻结地图下运行两类响应控制器及事先指定的 ego 动作对。`Town04` 未找到路口时，可对 `intersection` **另建新输出目录**并改用 Town03/Town05，不能混用地图/anchors 统计。

### 当 `NO_ELIGIBLE_MAP_ANCHOR` 出现时：先诊断，不直接更换阈值

原版的 `convoy` 搜索只用 `get_spawn_points()`，并要求 100m 的终点与
原点具有完全相同的 `road_id`、`lane_id`，道路总偏航变化小于 12°。
这不能证明 Town04 本身没有合适道路，只说明原先的**有限采样与过严限制**
没有找到场景。CARLA 的 `Waypoint.next(distance)` 允许返回多条后续分支，
大跨度 `next(100)` 也无法证明中间没有穿过路口。

修订版先检索推荐出生点，不足时对 `generate_waypoints(10)` 的驾驶车道中心线做
确定性扫描；对 B(-18m)、A(0m)、ego(+20m)、目的点(+100m) 之间的路线
采用 3–5m 逐段检测，拒绝路口、路径分叉、车道 ID 跳变与过大转向。
允许远处连续道路跨越 OpenDRIVE `road_id` 边界，但初始跟车对保持同一
`road_id`，总偏航变化限制为 25°。**车辆间距、目标位置和实验动作协议保持不变。**

在已启动 CARLA 0.9.16 的终端 B 中，先执行**只检查地图、不生成结果文件**：

```bash
source scripts/00_env.sh
export RIDE_ANCHORS=1
bash scripts/01b_probe_anchors.sh convoy Town04
```

成功时 `eligible_anchors` 大于零，并输出候选的物理坐标、road/lane 身份；
`CARLA_MAP_ANCHOR_DIAGNOSTICS` 包含 `spawn_points_scanned`、
`map_waypoints_scanned` 和 `rejections`，可辨认地图不足、路口、分叉与弯曲限制。
如果 Town04 仍然返回零，分别**单独运行**：

```bash
bash scripts/01b_probe_anchors.sh convoy Town05
bash scripts/01b_probe_anchors.sh convoy Town03
# 若你的 CARLA 安装了 Town10HD，也可以尝试：
# bash scripts/01b_probe_anchors.sh convoy Town10HD
```

选一个实际返回非零的 Town，且**正式冻结前建立新输出目录**：

```bash
export CARLA_TOWN=Town04  # 改成你刚刚验证有合格锚点的 Town
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_town04_fix_v2
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

如果全部 Town 都为零，保留诊断输出，不应删除道路和几何验证或随机制造
“正例”；应根据 `rejections` 检查该场景模板是否适合已安装地图。
不要在同一个服务端执行中的 CARLA 实验上切换地图，也不要把不同 Town 的
场景 ID 混到同一冻结数据中。**旧版冻结 manifest 不能复用**：本版会验证
`anchor_protocol_version`，要求从头冻结新的地图锚点。

### 全部四类机制筛查

在**全新**的 `RIDE_OUTPUT_ROOT` 下执行以下四场景流程（若之前已在同一目录运行 `convoy`，请先换新输出目录）：

```bash
for case in cut_in convoy intersection support_gap; do
    bash scripts/03_precommit_scenes.sh "$case"
    bash scripts/03_run_carla_case.sh "$case"
done
bash scripts/03_support_ablation.sh
bash scripts/04_analyze_carla.sh
```

若已执行某个场景，第二次执行会拒绝覆盖：请换用新的 `RIDE_OUTPUT_ROOT`。当前 v3 场景协议把 `support_gap` 定义为真实的相邻车道并线间隙，和直道 `convoy` 不再共享锚点；旧版 evidence 中把二者坐标复用的结果必须作废并重新冻结。`support_gap` 关注有限响应库的证据缺口，`convoy` 关注第三方连锁风险。`intersection` 的联合响应只是有限动作库中的**正向存在性见证**；搜索不到不等于不可行。`cut_in` 或 `support_gap` 若实际没有变道，`mechanism_executed=false`，不纳入有效配对。

### 主要 CARLA 输出

```text
$RIDE_OUTPUT_ROOT/frozen/{case}_anchors.jsonl
$RIDE_OUTPUT_ROOT/carla/{case}.jsonl
$RIDE_OUTPUT_ROOT/carla/{case}_audit.json
$RIDE_OUTPUT_ROOT/carla/all_scenes_audit.json
$RIDE_OUTPUT_ROOT/carla/risk_ranking_disagreement.json
$RIDE_OUTPUT_ROOT/carla/science_gates.json
$RIDE_OUTPUT_ROOT/carla/support_gap_bank_ablation.json
```

分析只接受带 `run_end` 完成标记的物理实验日志；异常中断不计作无碰撞负样本。CARLA 碰撞采用物理传感器事件，定向包围盒用于审查初始交叠和传感器一致性。场景可能**没有合法锚点、没有合法动作对或没有任何碰撞**，这并不是代码必然出错，也不能被宣传为科学否证。

## 4. WOMD / Waymax（建议独立于 CARLA 的 Python 环境）

先完成单场景 WOMD 身份和响应模型**观测**拟合，不要把日志复现误当成 `do(ego action)` 人类因果效应。原始数据根目录应指向 WOMD v1.3.1 的 `tf_example` TFRecord；需按你服务器实际结构设置。

```bash
conda create -n ride-womd python=3.10 -y
conda activate ride-womd
python -m pip install -e .
python -m pip install 'numpy<2' 'tensorflow-cpu==2.16.2'
cd /path/to/RIDE
export RAW_WOMD_ROOT=/path/to/WOMD/v1.3.1/tf_example
export RIDE_OUTPUT_ROOT=/path/to/RIDE/evidence/run_001
source scripts/00_env.sh
bash scripts/09_native_womd_cohort.sh dev
bash scripts/11_observational_response_calibration.sh
```

从 **开发样本**到严格完整扫描：

```bash
# 只有原始 validation 全部 150 shards 均完整时才执行 full150
bash scripts/09_native_womd_cohort.sh full150
```

Waymax 需要独立安装**与 WOD v1.3.1 配套**的 Waymax + JAX + TensorFlow，并实机核对版本敏感 API；不能承诺当前沙箱已经执行。先：

```bash
bash scripts/05_waymax.sh waymax_inspect
bash scripts/05b_waymax_native.sh smoke
bash scripts/05b_waymax_native.sh panel
```

`05b` 从 cohort manifest 按哈希锁定 22 个场景 ID，并校验 ID/manifest 哈希；不会自动将旧 pilot 的 22 个场景当成同一批。若你有原来的 22-ID 和包含 raw-shard 记录索引的 cohort，可显式指定 `RIDE_PREVIOUS_COHORT` 与 `RIDE_PRIORITY_IDS` 后运行 `scripts/05_waymax.sh waymax_priority22`。Waymax 当前只使用 IDMRoutePolicy，不是独立学习式 reactive evaluation。

## 5. 事先锁定选择、Burden 消融及强基线

`inputs/predictions_SCHEMA_ONLY.csv` 仅有**字段表头**。必须由真实预测器使用**历史及当前可用信息**生成数值预测，不能将 CARLA outcome/TTC/真实未来填写进预测输入。

```bash
# 必须在打开独立响应结果之前冻结预测/动作选择；以下仅是代理方法
bash scripts/06a_lock_proxy_selectors.sh /path/to/ex_ante_predictions.csv hard_courtesy
bash scripts/06b_burden_proxy_ablation.sh /path/to/ex_ante_predictions.csv

# 有合法、预先冻结的选择 manifest 且已得到全场景闭环 outcome 后再审核：
bash scripts/06_compare_locked_planners.sh \
    "$RIDE_OUTPUT_ROOT/frozen/hard_courtesy_choices.csv" \
    "$RIDE_OUTPUT_ROOT/frozen/hard_courtesy_choices.sha.json"
```

`burden.py` 只是在**同一预测 CSV、同一风险和规则阈值**下开/关 predicted burden 上限，是**proxy incremental-value audit**；不是原始论文模型的完整 burden-on/off checkpoint 级消融。完整算法消融、官方 RACP/GameFormer/HYPE 等的作者实现接入、独立 learned reactive evaluator、第三方风险-ego progress 曲线和跨数据集显著性统计仍需要后续真实实验。

## 6. 本地代码验证与重要限制

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests
python -m ride.carla.run --help
python -m ride.waymax.paired_idm --help
```

当前交付的测试以合成数据、纯函数和假的 CARLA waypoint 为主，**并不等于在你的服务器上完成了 CARLA 0.9.16 物理执行、Waymax/TensorFlow 集成或真实 WOMD 标定**。当场景证据不足、任务/道路/车辆身份不匹配、有限响应库未找到 witness 时，必须报告 UNKNOWN、INVALID 或 NOT_TESTED；不得填写不经证实的正例率。

具体文件迁移、实验逻辑修复、检查项与尚未实现的研究工作见 [docs/CLEANUP_AND_SCIENCE_AUDIT_CN.md](docs/CLEANUP_AND_SCIENCE_AUDIT_CN.md)，科研定位见 [docs/RESEARCH_GAP_CN.md](docs/RESEARCH_GAP_CN.md)。

更多关于本次 `NO_ELIGIBLE_MAP_ANCHOR` 修复的诊断理由与服务器命令见 [docs/CONVOY_ANCHOR_FIX_CN.md](docs/CONVOY_ANCHOR_FIX_CN.md)。
