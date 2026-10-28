# RIDE RQ1/RQ2 v6：新上传 V5 证据复核、ReactSim Token 诊断、下一步命题验证

**审计边界**：本版根据 `/mnt/data/RIDE_RQ12_V5_20261028(1).zip` 与 `/mnt/data/rq12_v5_smoke_20261028.zip` 静态及逐轨迹复核构建。本地无用户的 CARLA/nuPlan DB/GPU；未验证作者 checkpoint 推理、未运行新参数场景。新增科学协议 `ride_lane_corridor_v6_20261028`，V5 anchor 不得偷偷复用作 V6 预注册样本。

## 1. 主要证据与研究判断

V5 `convoy.jsonl` 和 `cut_in.jsonl` 都有 `run_end completed=true`，分别 20 个 trial（2 anchor × 1 seed × 2 family × 5 actions）。

| 指标 | convoy | cut_in |
|---|---:|---:|
| 完整 trial | 20 | 20 |
| 合格有效 trial | 20 | 18 |
| 机制执行 `mechanism_executed` | 20 | 16 |
| Ego 碰撞 | 0 | 0 |
| 固定第三方对 A–B 碰撞 | 0 | 0 |
| 后车 B 持续急刹（所有 trial） | 0 | 17 |
| cruise 未并线对照仍持续急刹 | 0/4 | **4/4** |
| 预注册动作对（双 family 完整） | 4 | 4 |
| 任务匹配跨 family 有效动作对 | **2** | **2** |
| 严格第三方风险排序翻转 | 0 | 0 |

**正面信号**：V4 的必然碰撞 / 不成功并线问题已被 V5 明显改善：V5 cut-in 有 16/20 次符合目标车道连续驻留的物理执行，ego 零碰撞，部分动作及进度配对有效。**负面信号**：四个严格有效的跨 family 配对全部为相同的原有序数风险键，既无排序翻转，也无可避免的 A–B 碰撞；仅 2 个独立物理 anchor 和 1 seed，不能统计排除普遍存在的机制。

`cut_in` 的后车 B 急刹高度敏感于初始设计。在 anchor 0 中，TM/Behavior 的 `cruise`（**不进入目标车道**）同样出现严重制动：第一个持续急刹约 1.75 s / 1.65 s；早并线的目标车道完整占据约在 3.05 s。**这不表示早于完整并线的急刹一定不是 ego 诱发**，因为横向控制指令可能早于完全占据目标车道。真正的反证是：未并线 `cruise` 同样使后车 B 急刹；所以不能把全部 B 制动计为“并线新增负担”。

连续制动超限积分（`∫ max(0, -a_b-3)dt`，m/s）仍显示对响应模型敏感：anchor0 的 TM `early_slow` 与 `late_slow` 分别约 4.10 / 12.04，Behavior 对应约 1.70 / 1.70。它们是**事后发现性**的连续指标差异，不是预注册的 RQ1 排名翻转，更不代表执行规划器产生了错误选择。详见 `docs/RIDE_V5_REAUDIT_V6_DIAGNOSTICS.json`，完整逐 trial `ride.carla.mechanism_audit` 输出包含精确时序。

## 2. ReactSim 错误性质：先诊断，不要重下所有 DB

`preflight` 8732 DB / 2636 behaviors / 地图齐备，只验证文件链接和数量，不验证每个 `scenario_token` 的对应 `lidar_pc` 数据。`15... check` 是 dry-run，因此 `ready_for_preprocess=false` 并非数据损坏。

官方 `reactsim_data_pipeline/preprocess.sh` **先处理 train，再处理 val**；报错可能来自 `.scenario_ids_train.json`，不能凭 `processed val: MISSING` 假设是 validation 出错。ReactSim 官方 `scenario_cache.py` 调用 nuPlan `get_scenarios_from_db`，而 nuPlan 查询会排除日志边界若干 scene，并需要 `lidar_pc`→`lidar`→`log` join；即使 token 在原始 DB 中存在，也可能不能成为一个合格场景。**缺 token、被 scene 过滤、数据库损坏是三种不同的问题。**

进一步发现当前官方 `preprocess.sh` 的默认输出目录是 `../data/test/nuplan_processed/train|val`，而 README 中供 SMART/MTR 读取的目录是 `data/nuplan_processed/val`。本版改为直接调用官方 `build_cache.py` 指向经软链接的数据工作区，保留官方 ID 列表、不跳过任何 token。若原始 token 或查询关系真的不匹配，将明确拒绝 full benchmark，而不是“补齐”或删掉失败场景。

### 2.1 首先跑单 token 只读诊断

```bash
conda activate ride_reactsim_dp
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V6_20261028
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_REACTSIM_REPO=/data0/senzeyu2/dataset/RIDE/research_baselines/ReactSim_Bench
export RIDE_REACTSIM_WORK=/data0/senzeyu2/dataset/RIDE/reactsim_workspace
source scripts/00_env.sh

bash scripts/17_reactsim_preprocess_checked.sh token \
  2021.08.27.03.47.52_veh-53_01182_01302.db 2cdb26865dfa5261

cat "$RIDE_REACTSIM_WORK/diagnostics/token_2cdb26865dfa5261.json"
```

结果解释：`RAW_TOKEN_ABSENT`：**确切的这个 DB** 没有所需原始 token，优先核对官方 DB 版本、文件大小、schema、scenario ID 文件对应 commit，必要时只获取这个 DB 的正确发布版本，不直接全量重下。`FILTERED_SCENE_BOUNDARY`：token 在原始 DB 内，但落在 nuPlan 排除的日志边缘；先查该样本是否具备 1s 历史+8s 未来，不能自动绕过筛选。`MISSING_LIDAR_LOG_JOIN`：DB 内部元数据缺失或不匹配。`ELIGIBLE_BY_SQL_JOINS`：SQLite 基础连接成立，若 ReactSim 原函数仍失败，则在 `ride_reactsim_dp` 环境直接调官方 `get_scenarios_from_db` 定位版本/代码差异。`SQLITE_INTEGRITY_FAIL`：仅此时直接考虑文件损坏/定向重新下载。

**本地没有真实 DB，以上是分支诊断，不是断言这次必然属于某个分支。**

### 2.2 比较 train 与 val ID 列表

```bash
# 先只读抽查 train 前 200 个；不完整抽样不能当全量认证。
RIDE_REACTSIM_AUDIT_LIMIT=200 \
  bash scripts/17_reactsim_preprocess_checked.sh inspect-train

# val 只有 ~2636 ID，可完整检查
bash scripts/17_reactsim_preprocess_checked.sh inspect-val
cat "$RIDE_REACTSIM_WORK/diagnostics/val_token_audit.json"
```

如果 val 的全量 SQL audit 检查结果全部 `ELIGIBLE_BY_SQL_JOINS`，继续下面的 validation-only 预处理。Train 若存在几条失败，**不妨碍先尝试 val 的官方预训练模型评测**，但 val 必须自身完整且后续 `build_cache.py` 无错误。

### 2.3 Validation-only 预处理（不碰 train/test）

```bash
export RIDE_REACTSIM_WORKERS=4
bash scripts/16_reactsim_run.sh preprocess
# 等价于：
# bash scripts/17_reactsim_preprocess_checked.sh val
bash scripts/16_reactsim_run.sh preflight
```

`preprocess` v6 默认改为验证完整 ID 之后只处理 `val`；**不再默认触发官方 train 的 642k+ 场景预处理**，也不再误将输出写到 `data/test/` 下。只有明确计划重新训练 ReactSim 模型时才用：

```bash
RIDE_ALLOW_FULL_REACTSIM_TRAIN=1 bash scripts/17_reactsim_preprocess_checked.sh train
```

若 validation IDs 全部通过 SQL 审计但 `build_cache.py` 仍报同样 `Failed to resolve`：保存 `val_token_audit.json`、ReactSim Git commit、nuplan-devkit 版本、完整 traceback，先核对作者过滤/加载逻辑。若是其他运行期地图/轨迹异常，保留 `process_scenarios_errors.json`，不要将错误样本删除后把剩余结果当完整 benchmark。

## 3. 新版 CARLA 预注册因素筛查：先提高“信息性”，不要求碰撞

V5 的 `cut_in` B 从静态几何上距离 ego 约 25m，且 B 的初始速度比 ego/A 高 1.5m/s。该条件下 `cruise` 已触发强烈制动，导致主要指标饱和。下一轮采用**预先固定**的 2×2 因素网格：

- `RIDE_CUTIN_EGO_GAP_M=20`（A→ego）；
- A→B rear gap ∈ `{35,45}` m；
- B 额外初速 ∈ `{0,1.0}` m/s；
- 每格 2 anchor × 1 seed × 2 response family × 5 ego profiles，共 20 trials，四格合计 80 trials；
- 每格包含未并线 `cruise` 负对照，及两个 **任务目标一致的** early/late 并线对。`cruise` 不能计入 RQ2 的 task-matched 替代动作。

上述参数探索只决定哪些响应条件值得进一步验证，不能从 4 个格子挑一个“最好”的结果在同一个开发集上当论文统计结论。进入后续正式确认时，必须冻结因素、候选选择规则、独立 anchor 划分、seed、主端点和 cluster-robust 不确定性。

### 3.1 CARLA v6 部署/冒烟

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip RIDE_RQ12_V6_20261028.zip -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V6_20261028
conda activate mfrp
python -m pip install -e '.[test]'
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pytest -q
python -m ride.doctor --server
```

无头 CARLA 未启动时，在单独终端执行 `source scripts/00_env.sh; bash scripts/02_start_carla_headless.sh`。为避免混合旧输出，使用**新的**证据目录。

```bash
export CARLA_TOWN=Town04
export RIDE_SEED=17
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export RIDE_WARMUP_TICKS=20
export RIDE_WORLD_RESET=actors
export RIDE_FACTORIAL_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v6_factorial_20261028
export RIDE_REAR_GAP_GRID='35 45'
export RIDE_REAR_SPEED_GRID='0 1.0'

bash scripts/19_cut_in_factorial_pilot.sh
bash scripts/20_factorial_report.sh
cat "$RIDE_FACTORIAL_OUTPUT_ROOT/all_cells_audit.json"
```

如果只想先测 1 格 1 anchor（更适合排错），另设从未用过的根目录，执行：

```bash
export RIDE_FACTORIAL_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v6_one_cell_smoke
export RIDE_ANCHORS=1
export RIDE_REAR_GAP_GRID='35'
export RIDE_REAR_SPEED_GRID='0'
bash scripts/19_cut_in_factorial_pilot.sh
bash scripts/20_factorial_report.sh
```

**不要先运行一格再在同目录重复所有格**：脚本显式拒绝覆盖冻结的 anchors 和 trial JSONL。这是为防止根据实验结果反复改变场景。不同配对参数应分别命名、全量保留失败格子。完成第一格物理 smoke 后用全新根目录重跑 2×2 网格。

### 3.2 继续审计现有 V5 轨迹（无需重新跑 CARLA）

```bash
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V6_20261028
source scripts/00_env.sh
bash scripts/18_reaudit_mechanism.sh cut_in \
  /path/to/rq12_v5_smoke_20261028/carla/cut_in.jsonl \
  /path/to/cut_in_v6_mechanism_diagnostics.json
```

报告记录 `first_lateral_command_s`、`first_ego_target_lane_s`、A/B 持续急刹首次时间、连续制动超限积分、相同 family/anchor/seed 下相对于 `cruise` 的差值。此为**辅助发现指标**，不能事后替代原始二元风险通过条件。

## 4. RQ1、RQ2 接下来的明确命题与停止条件

**RQ1（决策相关的响应模型差异）**：同 scene-anchor、同初态、相同 ego 控制目标且**实际 ego 轨迹也近似一致**、两个 task-matched merge 候选，经过 TM/Behavior 响应后第三方风险排序非平局且严格翻转。v6 新增跨 family ego 执行轨迹门禁，不能把只有 profile 名字相同但实际 ego 行为不同的 trial 直接比较。当前 V5 0 个严格翻转。连续风险积分只能作为**预注册 V6 新主端点的候选**；需要独立 heldout 和敏感性验证，而不能篡改 V5 主端点。

**RQ2（可避免的 ego 诱发第三方负担）**：要求同初态、同模型/固定第三方 A–B、同路权、两个合法 ego 并线都完成相近的纵向进展；候选之间存在显著的 A/B 风险差；不能是 B 的高速追前车导致的独立基线制动。保留不并线 `cruise` 只用于判断是否存在**原有背景风险**；它不是与并线同任务的替代动作。

如果 2×2 屏检表明（i） `cruise` 本身已经大量刹车，则优先剔除这种饱和机制条件（作为预先制定的科学纳入条件并同时公开其排除率），选择更合理的 B 初始速度/空间，不要人为挑碰撞；（ii）低背景制动条件下，早/晚并线仍无第三方差异，在新增的**独立** 20+ anchor、多个 seed、外部 SMART/MTR 评测下仍无严格翻转，则必须缩小 RIDE 主张或转向“有限/失准响应证据的决策可信性”，并正面比较 PDT、RACP、HYPE/SIGMA 等已覆盖的近邻概念。（iii）存在连续负担差异但没有二元风险翻转，必须**新预注册**鲁棒负担效应、动作选择后悔值、由正确/错误模型选出的 ego 动作的损失，不能直接宣布原 RQ1 通过。

**RQ3 暂缓正式强基线。** 可并行安装 ReactSim 官方 SMART 和 MTR 作为独立反应环境，但其原生 off-log ego stress 数据不天然有同初态双动作、进展配对，也不是强规划器的选择失误证明。先做作者自身 benchmark 复现、记录 scene-token 覆盖和模型版本，再设计 RIDE pair injection、固定 A/B 身份、闭环第三方风险和 candidate-selection 接口。所有 planning baselines 必须锁定 ex-ante 决策，不能使用后验 rollout risk 当作选择证据。

## 5. 执行验证 / 交付边界

本地运行单元测试、纯 Python 编译、Shell `bash -n`、旧 V5 数据复读、ZIP 验证。本地不存在 `/data0/senzeyu2` 下真实 nuPlan 数据和 CARLA 0.9.16 Server，**没有运行 V6 物理实验或完整 ReactSim checkpoint**。只有服务器上执行上述 token 诊断后才能断定该 token 缺失的确切原因。
