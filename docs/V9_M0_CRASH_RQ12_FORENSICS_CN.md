# RIDE v9 取证报告：Town04 M0 配速阻断、Town05 Signal 11、RQ1/RQ2 证据不足

日期：2026-10-28。依据：本次上传的 `RIDE_V8_RESEARCH_GATE_20261028(1).zip`、`rq12_v8_m0_smoke.zip`、`rq12_v8_confirm.zip`，以及 v8 源码内的工程协议。此报告不是 CARLA GPU 执行证明。

## A. 真实上传数据的审计结果（不是估计）

### M0 smoke
- 原始文件有 1 个 manifest、20 个 trial 和 1 个 `run_end(completed=true)`。
- TM：10 次试验、全部在产生 trace 前抛出 `ValueError: TM_DESIRED_SPEED_EXCEEDS_LIMIT: 9.00 m/s > 30.00 km/h`。
- BehaviorAgent：10 次有完整的 8 秒轨迹，10/10 `valid=true`、8/10 `mechanism_executed=true`、0 ego 物理碰撞。剩余 2 个是各 anchor 的 `cruise` 不并线负对照，不能以机制未执行判它们的实验程序错误。
- BehaviorAgent 2 个 anchor 各有 early/late 及 mild-early/mild-late 两组有效的**同响应模型、任务匹配**动作对，合计 4 对；TM 有效对为零，因而跨响应模型可比较对也为零。
- BehaviorAgent Anchor 0：两对的碰撞/持续急刹/短 TTC 主端点均无差异。Anchor 1：`cruise` 自身已经产生后车持续急刹，early/late 仍没有产生严格的主端点差异。
- **这个结果并非 RQ1/RQ2 失败**：需要的跨 TM/BehaviorAgent 有效反事实配对为零。只具有单响应模型的局部阴性信号和 M0 的执行性正面信号。

### Town05 confirm
- 上传的 ZIP 只有目录和 `seed_23/frozen/`，没有 JSONL、冻结 manifest、引擎日志和 completed run_end。
- 所以既不能知道是否实际进入第一条 trial，也不能依据包内证据精确定位崩溃发生在 map load、actor spawn、sensor、TM 还是 Unreal RenderThread。
- 用户终端显示 Signal 11，是引擎段错误，不是 Python 的规划代码普通异常，也不能从 `Malloc Size` 一行推断系统内存不足。

## B. 崩溃风险排名与证据边界

**优先假设**：原 `scripts/02_start_carla_headless.sh` 固定 `-quality-level=Low`；原 `ride/carla/run.py` 在期望 Town05 但当前运行 Town04 时会 `client.load_world('Town05')`。CARLA 官方 issues #4940 记录过 `-quality-level=Low` 配合 `load_world('Town05')` 导致 Signal 11，去掉 Low 后恢复的同类现象；#9061 也记录了 Unreal map load / RenderThread timeout 伴随 Signal 11。这与用户结果高度一致，但不构成该服务器的堆栈证据。

次级排查：GPU Vulkan 驱动及资源压力、同时启动多个 CARLA 服务器导致显存不足、Unreal 保存配置、地图资源损坏或加载阻塞、其他进程对 CARLA server 发出 kill/中断、其他调用误进入 `reload_world`。不要通过无限次自动重新加载 Town05 来“修复”段错误。

**排查路径**：先只用 Town04 进行 v9 M0，分别验证 server `get_world().get_map().name` 和 2 个 anchor 的正常物理轨迹；只有确认没有 server crash 后，再用一个**专用独立 CARLA 进程**尝试 Town05，首先不用 Low 质量。若 Town05 仍发生崩溃，执行 `bash scripts/30_v9_crash_snapshot.sh` 保存显卡、内存、进程、Unreal 日志与可能的内核/Xid 记录；依据最新日志含 `LowLevelFatalError/RenderThread/Out of memory/Vulkan` 的具体内容再判断。无日志时不推荐重装 CARLA，也不能断言崩溃是 nuPlan DB 引起。

**重要区分**：`-RenderOffScreen` 仍进行渲染并可能初始化 Vulkan；`world_settings.no_rendering_mode=True` 则是 CARLA Python 在世界加载后设置。两个概念不等价。后者无法修复世界加载前的所有 GPU 问题。

## C. v8 科学逻辑尚未完成的环节

1. **RQ1**：所需为同物理场景、同初始状态、同一组两个真正不同的 ego 候选动作，两个反应模型给出严格相反的**风险优劣次序**；不是模型轨迹不同本身。v8 没有完整跨模型对。
2. **RQ2**：场景 B 的急刹必须相对同上下文 `cruise` 负对照检查“ego 干预前已存在的背景制动”，但 `cruise` 没有并线，不能当成任务匹配替代动作。真正 RQ2 正例需要两个都执行并线且进展相近的动作 A/B，固定第三方受害对象，A 导致较大风险而 B 避免，排除 ego 碰撞和道路违规。
3. **三车模型的结构限制**：ego 插入前车 A 与后车 B 之间时，当前非 ego 车辆对只有 A–B，中间常有 ego，容易首先发生 ego 碰撞。若论文重点是“ego 安全而第三方连锁碰撞”，下一轮应实现四车链：ego 合流在 lead L 与 follower B 之间，另设后车 C，使被测对象为 B–C，且证实 B 的减速发生于 ego 干预后。v9 没有声称新增此四车物理闭环实现；不能将旧 A–B 统计改名为 B–C。
4. **动作多样性**：原 cut-in 中 `early_slow` 等名称容易误导；其 ego 纵向控制在此场景固定为名义巡航速度，差异主要是目标车道切换时间（1.5、2.5、3.5 秒）。许多二元风险端点都为平局。应先记录完整 lane-entry、持续制动和增量负担，不可事后替换主端点宣称成功。
5. **反应模型公平性**：TM 与 BehaviorAgent 名义速度差异是 v7/v8 的混杂，v9 先设所有车辆共同巡航 8m/s，并记录各角色的最高速度、时间和跟车行为。如果 BehaviorAgent 和 TM 的实际速度仍持续严重分离，应另做 calibration 对照，不许直接声称响应“结构性失败”。
6. **已执行轨迹比较的后验筛选**：`same_ego_execution` 对观察到的 ego 轨迹施加相似性约束，有利于排除明显不匹配，但不能代替事前冻结的相同 ego 轨迹候选。下一研究层需要独立的 open-loop ego candidate replay + reactive other-agent evaluation，再单独评测 closed-loop planner control；不能把后验风险排序写成真实在线规划器事前选错。
7. **TTC 和责任语义**：只对同 lane / 同 road 适用的 TTC 子样本报告分母；交叉口需完整红灯和路权审计，不能拿 trigger-zone proxy 当违法证明。跨 city/scene 不能仅对照 CARLA 的 TM/BehaviorAgent 两策略就推断真实人类异质反应。

## D. v9 已落实的工程改动

- `ride/carla/run.py`：默认 `--cruise-mps=8.0`；预期 Town 与当前地图不一致时拒绝调用 `client.load_world`；保留明确但不推荐的 `--allow-world-switch` 作为有意识实验设置；名义速度/道路限速不匹配立即中断整个 batch。v9 使用新的 manifest 协议。
- `ride/carla/tm_policy.py`：保留严格合法性约束与 km/h→m/s 换算；**不允许只裁剪 TM 速度而让 ego/BehaviorAgent 仍跑 9m/s**。
- `scripts/02_start_carla_headless.sh`：默认 Epic 而非 Low，支持 `RIDE_CARLA_START_MAP` 的明确 UE map asset 参数。该选项是否由你安装的 CARLA 0.9.16 接受必须以 server preflight 为准；不能默认为已启动到指定地图。
- `scripts/29_v9_carla_preflight.sh`：只读检查版本/端口/地图与外来车辆；不加载世界。
- `scripts/24_v8_confirm.sh`：沿用原文件名但在冻结及运行任何种子前验证开发 M0 必须已通过、地图必须一致，保留冻结参数锁与开发 corridor 排斥。
- `scripts/23_v8_smoke.sh`：失败仍输出结构化 M0 诊断；若主运行异常不会继续按科学通过处理。
- `ride/carla/m0_gate.py`：给出有效单模型动作对、跨模型任务匹配对、无效原因、每个 family 的目标机制试验数。
- `scripts/30_v9_crash_snapshot.sh`：只读取服务器日志、GPU、内核和进程状态，不修改服务器。
- `ride/ranking.py`：新协议也需要执行轨迹相似性检查。
- 新增回归测试和新的 `ride_lane_corridor_v9_20261028` 冻结协议；v8 frozen anchors 不能复用。

## E. 下一步运行方式

### E1. 先在 Town04 执行 M0

```bash
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V9_CARLA_STABILITY_20261028
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export DATASET_ROOT=/data0/senzeyu2/dataset
source scripts/00_env.sh
python -m pip install -e '.[test]'
python -m pytest -q

# 先检查当前 server 处于 Town04 且无其他车辆
export CARLA_TOWN=Town04
bash scripts/29_v9_carla_preflight.sh Town04

export RIDE_CRUISE_MPS=8
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_WORLD_RESET=actors
export RIDE_WARMUP_TICKS=20
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export RIDE_SEED=17
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v9_town04_m0_new
bash scripts/23_v8_smoke.sh
cat "$RIDE_OUTPUT_ROOT/carla/m0_physics_gate.json"
```

若新 M0 仍 FAIL，先看 `status`。`ENGINEERING_EXCEPTION_BLOCKER` 表明还有非科学问题；`M0_MECHANISM_OR_VALIDITY_BLOCKER` 表示动作没有有效执行；`M0_PHYSICS_EXECUTES_BUT_NO_CROSS_FAMILY_TASK_MATCH` 表示仍没有有效 cross-family 分母；最后状态 `M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF` 才允许进入发现性科学筛查（它仍不代表 RQ1/RQ2 已通过）。

### E2. 需要重启服务时，明确服务器配置并记录日志

终端 A，CARLA 进程**未运行**时：

```bash
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V9_CARLA_STABILITY_20261028
conda activate mfrp
source scripts/00_env.sh
export RIDE_CARLA_QUALITY=Epic
export RIDE_CARLA_START_MAP=Town04
bash scripts/02_start_carla_headless.sh 2>&1 | tee /data0/senzeyu2/dataset/RIDE/carla_town04_v9_server.log
```

若启动后 `29_v9_carla_preflight.sh Town04` 仍显示别的地图，不要反复自动 `load_world`。请先检查上述启动参数是否被该 CARLA 包接受。

### E3. 保持 M0 通过后，先在相同 Town04 找到未使用的物理 corridor

```bash
# 使用独立新输出目录；不要把此结果掺入 M0
export RIDE_DEV_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v9_town04_m0_new/frozen/cut_in_anchors.jsonl
export RIDE_CONFIRM_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v9_town04_holdout_new
export RIDE_CONFIRM_SEEDS='23 41 73'
export RIDE_ANCHORS=4
export RIDE_EXCLUDE_RADIUS_M=90
export CARLA_TOWN=Town04
bash scripts/24_v8_confirm.sh
```

这一步先验证有无独立 corridor，再从相同可控环境中观察筛查结果。**不能**把三个 seed 当成三个独立道路；应审计重复 corridor 和每个 anchor 的配对。若排斥后找到的 corridor 数不足，报告 `HOLDOUT_INSUFFICIENT_DISJOINT_CORRIDORS`，而不是放宽 90m 或删除失败案例。

### E4. 只在需要跨 Town 验证时测试 Town05

不要在正在跑 Town04 trial 的 Unreal 实例里切地图。先结束**你自己持有**的 Town04 实例，确认 RPC 端口及 GPU 资源已释放，再在一个专用会话里尝试：

```bash
export RIDE_CARLA_QUALITY=Epic
export RIDE_CARLA_START_MAP=Town05
# 同一套 CARLA 0.9.16 程序；仅一个 Unreal 实例占用该 GPU
bash scripts/02_start_carla_headless.sh 2>&1 | tee /data0/senzeyu2/dataset/RIDE/carla_town05_v9_server.log
```

在另外终端：

```bash
export CARLA_TOWN=Town05
bash scripts/29_v9_carla_preflight.sh Town05
```

若仍 Signal 11：

```bash
bash scripts/30_v9_crash_snapshot.sh
```

根据 `carla_town05_v9_server.log`、`carla_crash_*.log` 中 `LowLevelFatalError` / `Vulkan` / `RenderThread` / Xid / OOM 证据再确定驱动或地图修复。不要把崩溃记作 RQ1/RQ2 的负例，也不要在崩溃循环中不断重复 `load_world`。若 Town05 不稳定，可明确报告环境限制，先完成 Town04 的控制实验，再选择稳定的不同地图进行独立验证。

## F. 正式科学推进门槛与方向分叉

**M0**：固定 Town 与地图，两个响应 family 均能无异常执行，至少一个物理 corridor 存在两条有效且真实不同的早/晚并线、ego 无碰撞、任务进展匹配；初态与执行轨迹按冻结容差检查。若只有 BehaviorAgent，最多说明单模型场景执行，不支持跨模型 RQ1。

**RQ1**：在预设风险键上有跨响应模型严格反向动作排序，统计应以独立 corridor 为 cluster；相同动作必须有相同前提和可比控制。描述性 rollout 反转不能替代预执行 planner 的错误选择记录。

**RQ2**：同一个反应环境中 A/B 两个合法且任务匹配的 ego 动作对固定非 ego 车辆造成不同的额外风险；排除 background brake、预热碰撞、不同路段速度差与隐藏 traffic-controller 状态；对第三方连锁碰撞需明确四车被试对象 B–C。首先寻找速度/急刹负担方向，再检查是否有独立响应策略复核，不强行制造 collision。

**RQ3**：尚不应做正式主表。可提前准备 learned reactive simulator 和强 planner，但必须先获得 RQ1/RQ2 的有效分母及独立模型复核，并事前冻结 planner choice。同源 CARLA 两控制器并非两个独立世界模型。

**原命题负面趋势的应对**：如果在新的合法、完整、不同 corridor 上连续无排序翻转，不能靠提高后车速度、事后改变阈值或筛掉失败 seed 得出“强规划失败”。可转向 `Decision-Critical Behavioral Recourse Evidence`，探索有限响应库、UNKNOWN/feasible/infeasible 的证明语义及有交通责任/负担约束的预算分配；但该方向仍需比 hard courtesy、risk-aware contingency 和 decision-focused prediction 有可信的剩余 gap。当前没有充分证据应立即改题。

## G. 需保存给下一轮分析的文件

`$RIDE_OUTPUT_ROOT/frozen/cut_in_anchors.jsonl`、`carla/cut_in.jsonl`、`carla/cut_in.log`、`carla/m0_physics_gate.json`、`carla/risk_ranking_disagreement.json`、`carla/cut_in_mechanism_diagnostics.json`、原始 CARLA console+UE logs、`scripts/30_v9_crash_snapshot.sh` 输出。不要只传 `science_gates.json`，否则无法定位有效性和因果混杂。
