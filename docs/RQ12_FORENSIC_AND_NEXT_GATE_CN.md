# RIDE 2026-10-27｜RQ1/RQ2 独立审计、v4 修订与下一轮实验

> **本文件可直接在服务器查看。** 实验输入：`自动驾驶论文定位分析.md`、`RIDE.zip`、`rq12_discovery_20261027.zip`、`rq12_fix_smoke_20261027.zip`、`womd_dev_20261027.zip`。在本地沙箱完成代码审计、数据解析、60 项单测，**没有 CARLA 服务器、WOMD 原始数据或外部基线 checkpoint，不能声称实际闭环 v4 已通过。**

## 1. 工作论文题目与可证伪命题

工作标题：**RIDE: Response-Model-Induced Decision Errors in Interactive Driving**。不要声称首次提出交互预测、风险排序偏差、他车负担或第三方碰撞。PDT（2026）已系统研究预测指标向决策转移，INTERACT（2026）已有 ego-intent 条件化交互规划；RACP 与 GameFormer 已处理多模态交互。

- **RQ1**：预先固定同状态、同动作候选 A/B、同信息集与 progress/rule 门槛。在两个响应模型中，第三方风险评分对 A/B 的符号**严格反转**（不含平局），且实际在线规划的 ex-ante 选择会因此不同吗？CARLA 的事后 `ride.ranking` 只验证仿真输出排序敏感性，**不能自动宣称 planner 错选**。
- **RQ2**：在无初始碰撞、真实执行动作且同任务的前提下，同一组第三方固定车辆对的碰撞、短 TTC、持续急刹、延误是否因 ego 干预而变化？是否存在合法、进展匹配、可避免的替代动作？效应仅对指定 reactive simulator 有意义，WOMD 单条日志不识别 `do(a)`。
- **RQ3**：如果 RQ1/RQ2 成立，给 hard courtesy、风险规划（RACP）、ego-conditioned planner（GameFormer-Planner）**真正公平**的 ex-ante 选择对照，考虑部署可获得观测、算力、候选池与参数调优；比较第三方风险/进展 Pareto 曲线，而不是强行令纯 CBF 和 R3R 完全接收同样输入。

## 2. 从三个新实验归档看到的结果

### CARLA smoke

`all_scenes_audit.json`: 65 trial；完整预注册对 8，缺失对 8；有效 task-matched 对 2；`risk_ranking_disagreement.json`: 4 个跨模型待检查对，仅 1 个满足完整效度条件，**strict_reversals=0**。`support_gap_bank_ablation.json`: 有效嵌套比较 0，finite miss=0，global UNSAT=false。唯一可比较的有效对来自同一个 convoy anchor；无碰撞、急刹、TTC 风险差别。此结果不足以接受或拒绝 RQ1/RQ2。

cut_in 的 4/4 普通 trial 标 `valid=true`，但是 `mechanism_executed=false`；旧冻结场景的 ego 位于目标车道第一辆 follower **前方**，并非处于该车与后车之间的插入间隙。支持缺口中的 44/44 trial 不构成可验证 merge 机制；`intersection` 只有 3/13 条为普通 validity 合格，且红灯停止线、优先权和 route arc 都没有验证，不能提出守法的可避免风险结论。convoy 4/4 trial 物理运行成功、配对中 2 组有效，但第三方风险相同。

### CARLA discovery（不完整，不能与完整 smoke 混合统计）

原始 `support_gap.jsonl` 直接在 ZIP 中流式解析：3233 trials、缺 `run_end`、manifest 计划 10 anchors、执行只触及 5 anchors / 75 anchor×seed contexts。family=`tm` 450、`behavior` 450、`joint_witness` 2333。请勿对 invalid 总数进行选择性报告：`valid=2346`、`invalid=887`、`offroad_or_lane_departure=783`、`simulation_exception=15`；`mechanism_executed=30`，且全部 30 出现在无效配对链路中。300 组有 A/B 完整记录的 family-context，对其中 157 组至少一条 invalid，剩余 143 组未执行目标动作，因此 **0 组走到真实物理差异/progress 配对条件**。

全部 **843 条 non-ego pair 碰撞**与 **684 条 ego 碰撞**出现在 `joint_witness` 的外部纵向控制搜索；TM 和 BehaviorAgent 自主响应分支第三方碰撞均为 0。因此它们不是 RQ2 的证据。原始 hard-brake 的 3233/3233 由首秒模拟器瞬态严重污染；事后检查 1 秒以后的瞬时触发 1918/3233，而连续至少 0.20s 的触发 700/3233。后两个统计是探索性敏感性，不是正式 v3 结果。2513/3233 条试验进入了 junction，与预期“非路口并线”的场景语义不符。

### WOMD

`native_full150_predefined` 的 scan **完成了 150/150 validation shards**：43,695 records examined，40,912 usable，排除 2200 overlapping IDs，冻结选择 1,758 development + 600 sealed_holdout。`native_dev_partial` 是独立的有限 shard 开发样本。注意 cohort 是**按机制富集抽样**，不代表自然数据分布；不要在 600 个 holdout 上不停开发。

Logged observational calibration: attempted 320, eligible 310；constant-velocity 的 ADE(1s/2s)=**0.154/0.451m**，cautious=0.294/0.868，assertive=0.336/1.053，heterogeneous=0.302/0.917，TTC-gap=0.163/0.460。CV 在 1s 比三种 IDM-like 假设分别优于 282/310、290/310、288/310；这是观测拟合信号，不等于 CV 在所有反事实人类响应中最佳。此新 ZIP **没有 native Waymax 闭环 panel 的输出**，不能把历史讨论中的 17/22 直接当成本轮实验证据。

## 3. 四个场景究竟是不是四个独立、合格的研究场景？

| 场景 | 原设计问题 | v4 修订/边界 | 当前可声称的结论 |
|---|---|---|---|
| S1 cut_in | ego 位置不在真实 gap，路口/road_id 干扰，车道编号变化≠实际并线 | 重排同车道 front A、ego 邻道的纵向位置、rear B；目标车道前方验证 90m 无 junction；真正进入并连续驻留至少 0.5s | 待服务器验证，不能预设通过 |
| S2 convoy | 急刹启动瞬态，单 anchor，正常/干预风险均零 | 保留干净 100m 连续道；warm-up + 原始加速度和持续急刹双通道；增 seed | 可作为控制/负例，暂无正例 |
| S3 intersection | 交通灯触发区移动不等于越红灯停止线；goal-distance proxy 不等于路由进展 | v4 科学配对对该场景 **fail-closed**，直到 route arc 和优先权/合法性得到明确验证 | 只用于探索，**不允许 RQ2 正例** |
| S4 support_gap | 原版与 convoy 重合；v3 merge 失败，narrow/wide 跨运行不一致 | 新版复用 S1 空间几何但测试不同命题：同一 WIDE 运行精确子集导出 NARROW，以消除随机轨迹漂移 | **不是第四种独立空间场景**；只能声称有限库中正向可行 witness，不能从 no-witness 推全空间不可行 |

S1/S4 仍可能在 Town04 找不到足够 90m 可用目标车道；这种诚实拒绝应报告 `NO_ELIGIBLE_MAP_ANCHOR`，并换 Town05/Town03 的新输出根。只搜索直到碰撞出现会导致采样偏差；保持独立盲抽样和 holdout。

## 4. 本次代码修复，何处是真正的科学修复

代码在 `ride/carla_scenarios.py`、`ride/carla/run.py`、`ride/carla/protocol.py`、`ride/science.py`、`ride/support.py` 及实验脚本中修订：

1. 删除历史 support-gap/convoy 几何别名：support-gap 现在是 *真实 merge geometry + 有限响应库的证据协议*；修复 cut-in 原始 ego 不在车间隙的问题。冻结 `anchor_protocol_version=ride_lane_corridor_v4_20261027`，旧 v3 anchors **不可继续使用**。
2. 排除交叉路口或分叉的 90m 目标车道段，避免一部分 `BehaviorAgent NoneType.is_junction` 与场景无关的路口干扰。保留初始化朝向、waypoint 身份和初始 OBB/碰撞核验。
3. 用 CARLA 传感器事件的 frame ID 显式区分 prime/warm-up 碰撞与 ego 干预后碰撞；若干预前已有接触，整条 trial 判为无效，绝不计入因果风险差。
4. 在原有速度初始化后增加 20 ticks 统一 warm-up（含 controller），记录 `warmup_ticks` 和每步控制输入、目标路段 ID；场景初始条件以 warm-up **结束后**实测为准。warm-up 不能保证 CARLA 隐状态与随机反应完全相同。
5. `hard_brake` 采用 action t>=1s，持续低于 -4m/s² 至少 0.20s 的预注册 v4 指标；同时输出 `startup_brake_proxy_observed`、`hard_brake_instantaneous_post_startup_proxy`、两名 follower 最长持续制动、全部加速度原始极值，防止屏蔽真实风险。碰撞从不因 warm-up 执行被结果处理屏蔽。
6. 非 convoy 场景，如果没有真实执行 lane merge 或 junction/legality/route 证据，严格拒绝 RQ 配对；仅 lane_id 变化、末端三帧驻留都不够。
7. `scripts/03_support_ablation.sh` 新版只执行 **一次 WIDE**，NARROW 是相同 joint-target 已记录轨迹的**精确子集**；消除两个独立运行的随机漂移，但仍不等于外部独立重复。缺少 `run_end`、任一候选执行未完成、任一共用候选不一致，都不得报告 finite-bank miss。
8. 新增 `python -m ride.forensics` 可在不解压大量轨迹的情况下流式诊断不完整实验；`python -m ride.baseline_inventory` 只读审计已有 5 个外部仓库、git revision、真实 ckpt 文件与 LFS 指针；`scripts/14_download_reactsim_weights.sh` 按名称单独下载官方模型。

修订版沙箱检查：`python -m pytest -q` **60 passed**；`python -m compileall -q ride` 和 `bash -n scripts/*.sh`；没有运行 CARLA server、JAX Waymax 或官方 RQ3 planning checkpoints，物理成功率仍须服务器检查。

## 5. 首先运行 v4 最小闭环，不要直接执行 10×15×36 大搜

安装/运行目录：

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip RIDE_RQ12_AUDITED_FIX_20261027.zip -d /data0/senzeyu2/dataset/RIDE/code/
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_AUDITED_FIX_20261027
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export DATASET_ROOT=/data0/senzeyu2/dataset
source scripts/00_env.sh
python -m pytest -q
python -m ride.doctor --server
bash scripts/13_baseline_inventory.sh
```

单独终端（仅在尚未启动 CARLA 时）：

```bash
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_AUDITED_FIX_20261027
conda activate mfrp
source scripts/00_env.sh
bash scripts/02_start_carla_headless.sh
```

新冻结 **smoke**，每组输出根不能复用，否则会拒绝覆盖：

```bash
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_AUDITED_FIX_20261027
conda activate mfrp
source scripts/00_env.sh
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v4_smoke_20261027
export CARLA_TOWN=Town04
export RIDE_WARMUP_TICKS=20
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_ALL=0
for case in convoy cut_in support_gap; do
  bash scripts/03_precommit_scenes.sh "$case"
  bash scripts/03_run_carla_case.sh "$case"
done
bash scripts/04_analyze_carla.sh
bash scripts/12_research_status.sh
```

**注意** 在 cut-in/support-gap 中，`early_slow`、`late_slow` 等遗留 profile 名称实际控制的是**并线启动时间**，而不是不同纵向目标速度；必须通过连续的轨迹横向位移证明动作确实不同，不能凭 profile 名称判为行动差异。上述 `03_run_carla_case.sh support_gap` 会额外执行 36 个 joint trials；如果只做几何 smoke、希望节省运行，先只运行 `convoy` 与 `cut_in`，待实际车道占据通过后再执行 support_gap。某个 shell 因 `NO_ELIGIBLE_MAP_ANCHOR` 终止时，不要用 `|| true` 隐藏失败；换 Town05/Town03 与独立输出路径，并保留 rejection 统计。`intersection` 暂不纳入科学 gate；它可以单独冻结/执行做探索，但必须补合法性/路由验证才能使用。

核查新日志（仅判执行，不假装科学证明）：

```bash
python -m ride.forensics \
  --input "$RIDE_OUTPUT_ROOT/carla/cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/cut_in_forensics.json"
python - "$RIDE_OUTPUT_ROOT/carla/cut_in.jsonl" <<'PY'
import json,sys
trials=[json.loads(x) for x in open(sys.argv[1]) if '"type": "trial"' in x]
for t in trials:
 m=t.get('metrics',{}); trace=t.get('trace',[])
 print('family',t.get('family'),'action',t.get('profile'),
       'valid',m.get('valid'),'mechanism',m.get('mechanism_executed'),
       'lane_dwell_s',m.get('target_lane_dwell_max_s'),
       'end_progress',m.get('ego_progress_m'),'reason',m.get('reasons'),
       'error',t.get('simulation_exception'),
       'start_lane',trace[0].get('ego_lane_id') if trace else None,
       'target_lane',trace[0].get('target_lane_id') if trace else None)
PY
```

只有 S1 证实**真实物理并线**、各 family 两动作初始对齐且 A/B 实际轨迹不同、progress 差<=3m 后，才扩大 `RIDE_ANCHORS=5 RIDE_REPEATS=3`，最后进入 ≥20 个独立 anchor 的发现性评估。第一次扩大采用新输出路径 `rq12_v4_discovery_*`。每次完整运行保存 `run_end`，不允许提前中断后汇报成 zero-effect。

对于 support-gap，只有在切入机制 smoke 稳定后运行：

```bash
# 在一个全新 RIDE_OUTPUT_ROOT 内先冻结 support_gap anchors
bash scripts/03_precommit_scenes.sh support_gap
bash scripts/03_support_ablation.sh
# 这会一次性生成 support_gap_WIDE.jsonl、精确子集 NARROW.jsonl 和 ablation.json
```

## 6. WOMD 后续如何用，哪些数据暂时不需要下载

已扫描全 validation。开发阶段重用**现有 1758 development**，避免重复生成可能不同的全量富集 cohort。独立 `600 sealed_holdout` 只用于冻结最终方案后的验证。论文报告须分别展示 (i) 盲抽样自然分布、(ii) 机制富集的 stress 分布，并明确富集率不能估算真实道路事件发生率。

WOMD v1.3.1 的 `sdc_paths` 包含示例路线及 arc length、road-part metadata，可用于替代目前 Waymax 初始朝向投影的 progress proxy。需要在本地确认 TFExample 中 `sdc_paths` 是否存在、有效路线覆盖率；不存在时明确标为未知，不要偷偷用 logged ego future 当预测时输入。官方 WOMD 轨迹只提供自然观测的响应拟合，不能真实判定干预后的他车行为。当前**不需要下载 WOMD train 或 testing**；有规划网络训练需求再下载 train 的必要 shards。可以检查你已下载的 `validation_interactive` 是否齐全，优先用这些文件发现困难交通场景，但样本富集结构会引入 selection bias。

```bash
export RAW_WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
find "$RAW_WOMD_ROOT" -type f -iname '*validation*' | wc -l
# 只列目录，不假设你服务器的文件名；不再重跑全150扫描覆盖 600 holdout
python - <<'PY'
import json
p='/mnt/data/ride_work/womd/womd_dev_20261027/womd/native_full150_predefined/cohort_manifest.json'
print('Use the copied cohort_manifest.json at your own RIDE output path; sandbox path is not a server path.')
PY
```

如需在新路径单独构建、正式锁定一个新的 WOMD full cohort，执行 `bash scripts/09_native_womd_cohort.sh full150` 之前要确保新路径、排除清单和随机盐一致，避免把“同名队列”误当成原先 600 holdout。Waymax `bash scripts/05b_waymax_native.sh smoke` 与 `panel` 仅验证接口/模型，不是 RQ1/RQ2 因果结论；训练或真实校准需额外数据与 protocol。

## 7. ReactSim、nuPlan 与五个已有仓库

`bash scripts/07_download_reactsim_data.sh` **仅**下载 `ReactSim-Bench-Data` 的 custom ego behavior 文件，不包含 nuPlan 原始状态 DB/map；使用 ReactSim 评测必须还具备对应 nuPlan `nuplan-v1.1/splits/trainval` 数据库、maps、作者预处理/索引和模型 checkpoint。官方安装文档：https://github.com/Thinklab-SJTU/ReactSim-Bench/blob/main/docs/install_and_preprocess.md。`nuPlan mini` 可用于安装 smoke，但未必包含 ReactSim 的 2,636 个 benchmark scene，因此不能以 mini 的成功运行宣称 benchmark 完成。

**现有仓库职责**：

| 仓库路径后缀 | 对 RIDE 的合理角色 | 官方权重情况 |
|---|---|---|
| `Bosch_SMART_reactive` | nuPlan+SMART 反应式独立评测器，不是所提出的 ego planner | 官方说明 `planning/checkpoint` 已提供模型；注意需配套 SMART、nuPlan devkit 和地图 |
| `GameFormer` | WOMD 交互预测/开环模块；不等于 nuPlan 闭环 GameFormer-Planner | 不假设存在通用 `.pth`；closed-loop 应克隆独立 `MCZhi/GameFormer-Planner` 并按该仓库训练或核验作者权重 |
| `R3R_Julia` | Julia 的去中心化安全控制/理论假设对照 | 无 PyTorch `.pth` |
| `RA_CBF` | 责任分配/安全过滤近邻；不能冒充完整 planner | 官方提供指向预训练权重的链接，但明确**完整闭环模拟未发布** |
| `ReactSim_Bench` | 各类反应模型的外部评测环境 | 作者 HF `zzzy0729/ReactSim-Bench-Models` 有 `smart.ckpt`, `mtr.pth`, `ctg.ckpt`, `vbd.ckpt`, `clsft.ckpt`, `trajtok.ckpt` |

### 真实 checkpoint 下载命令（先 SMART 和 MTR，再选择性扩展）

```bash
conda create -n reactsim python=3.9 -y
conda activate reactsim
python -m pip install -U huggingface_hub
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_AUDITED_FIX_20261027
source scripts/00_env.sh
bash scripts/14_download_reactsim_weights.sh smart.ckpt
bash scripts/14_download_reactsim_weights.sh mtr.pth
# 按需：bash scripts/14_download_reactsim_weights.sh ctg.ckpt
# 按需：bash scripts/14_download_reactsim_weights.sh vbd.ckpt
```

这些 checkpoint 是**反应世界模型**，不是 RACP/GameFormer 规划器的共享 weights；不能把一个 `.pth` 随意加载到另一个架构。不要在 `mfrp` 的 CARLA/JAX 环境里同时 pip 安装所有项目依赖。

接入已下载 ReactSim ego behavior 的路径可以以符号链接实现（先确认目录存在）：

```bash
export BASE=/data0/senzeyu2/dataset/RIDE/research_baselines
export CUSTOM=/data0/senzeyu2/dataset/ReactSim-Bench/custom_behaviors
mkdir -p "$BASE/ReactSim_Bench/data"
test -d "$CUSTOM" && test ! -e "$BASE/ReactSim_Bench/data/custom_behaviors" && \
  ln -s "$CUSTOM" "$BASE/ReactSim_Bench/data/custom_behaviors"
```

要使用 `ReactSim-Bench` 仍需将**官方 nuPlan 原始** `maps/` 和 `nuplan-v1.1/splits/trainval/` 放到作者预期目录（可符号链接）。先查看 S3 文件树，不要猜 URL 下载整套：

```bash
python -m pip install awscli
aws s3 ls --no-sign-request --region ap-northeast-1 s3://motional-nuplan/
# 从官方 S3 列表或已注册 nuPlan 官网下载需要的 DB/maps 后再运行作者流程
cd "$BASE/ReactSim_Bench/reactsim_data_pipeline"
# 请按官方 requirements，在独立 conda 环境中安装：
# python -m pip install -r requirements.txt
# bash preprocess.sh
```

此外后续比较完整 planner 时补充：

```bash
cd /data0/senzeyu2/dataset/RIDE/research_baselines
[ -d GameFormer_Planner ] || git clone https://github.com/MCZhi/GameFormer-Planner.git GameFormer_Planner
[ -d RACP ] || git clone https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions.git RACP
# RACP 原生在 CommonRoad 场景上运行；不是直接可调用的 CARLA .pth。
# GameFormer-Planner 需要按官方 nuPlan + 训练模型运行，不能直接把 WOMD v1.3.1 TFExample 送入。
```

官方作者的测试入口，**只在所需环境与数据已经齐备后使用**：

```bash
# RACP 官方 CommonRoad 本地 demo
cd /data0/senzeyu2/dataset/RIDE/research_baselines/RACP
# 使用独立 Python 3.8 环境，pip install -r requirements.txt
python planner/Frenet/frenet_planner.py

# GameFormer-Planner nuPlan 闭环 (按作者 README 确认 checkpoint 和数据布局)
cd /data0/senzeyu2/dataset/RIDE/research_baselines/GameFormer_Planner
python run_nuplan_test.py \
  --experiment_name closed_loop_reactive_agents \
  --data_path /data0/senzeyu2/dataset/nuplan/nuplan-v1.1/splits/mini \
  --map_path /data0/senzeyu2/dataset/nuplan/maps \
  --model_path /PATH/TO/YOUR/VERIFIED_GAMEFORMER_MODEL

# R3R Julia 自身测试，非 CARLA planner score
cd /data0/senzeyu2/dataset/RIDE/research_baselines/R3R_Julia
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Bosch SMART-reactive nuPlan 的作者示例，要求单独、未污染的 nuPlan-devkit 及 SMART 代码环境（其 README 规定拷贝/替换部分 devkit 目录，必须在**专用副本**做）：

```bash
# 在 Bosch 官方配置完环境之后，按其 README 中的 nuPlan devkit 路径执行
python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  experiment_name=simulation_smart_agents \
  +simulation=closed_loop_smart_reactive_agents \
  planner=log_future_planner \
  scenario_builder=nuplan \
  scenario_filter=test_one_scenario
```

**警告**：这些命令均来自官方发布的原生使用流程，本沙箱没有你的 DB、服务器、模型权重；没有把它们移植到共享 RIDE 候选池，更没有“成功运行五个强基线”。

## 8. RQ3 进入标准、提前锁定的扩展计划

正式 RQ3 的开关须同时满足：至少 20 个**独立物理 anchor** 完成 RQ1 有效比较；严格排序翻转或经过等价的决策失配证据；至少约 10 个独立 task-matched 可避免第三方风险正例（这个数是研究预设的筛查门槛，不是统计学显著性的替代）；既有 TM/BehaviorAgent 以外，**至少一个不同结构的反应评测环境** 支持；对模型/预算/TTС/-4m/s² 及进展阈值变化进行 cluster-level CI 检验；候选选择来自实际在线 planner 在真实 rollout 前锁定，而不是同一 evaluator 事后知道风险。审稿会要求独立数据来源上复现，而不是只在人工 stress corridor 上 20 个相似 seed 复现。

建议下次新增的场景或机制是：**高密度相邻车道并线 + 后车队列的固定 pair 二次风险**（连接 S1+S2）；作为独立 stress class 的**无保护左转/无信号让行交叉**（须修复路权和 stop-line 证明）；以及**真实 WOMD 的多样化任务匹配候选**。这比增加一批名字不同但物理动作一样的场景更有说服力。

如果 v4 有效进度提高但仍 0 个 RQ1/RQ2 正例：**不能无限调参直至正例出现**。预注册广范围负例和最小检测效应/上界，然后转向研究不同反应模型支持域、有限 safe response evidence 的三态 `verified feasible / certified infeasible under bounded model / unknown`，并验证 `unknown-aware` 是否真正改善错误候选选择；同样不能拿有限搜索找不到解冒充全局不可恢复。若 RQ1/RQ2 成立但 hard courtesy 或 RACP/交互预测全部能避免，不能继续声称强基线系统性失败，应进一步缩小贡献为模型可信度诊断或放弃 RIDE planner 主线。

**最终建议**：本轮不要开始规模化 RQ3，也不要下载大规模 WOMD train/测试集。先复核 CARLA 的真实并线、控制与急刹测量并形成至少 3–5 个可理解的有效执行 case；随后独立 WOMD/ReactSim 反应复核。新规划算法的设计应该由哪一个稳定、独立的决策失败模式决定。
