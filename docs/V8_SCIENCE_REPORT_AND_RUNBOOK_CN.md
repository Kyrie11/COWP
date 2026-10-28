# RIDE v7 数据审计、v8 修订与下一步完整执行指南

> 审计时间：2026-10-28。研究工作标题：RIDE — Response-Model-Induced Decision Errors in Interactive Driving。研究证据只支持当前的**模拟器内诊断**；尚无校准的人类反事实真值，也没有任何已复现强规划器的 ex-ante 选择错误。**所有旧 v7 结果均只读，未用 v8 指标重新标签后冒充旧实验。**

## A. 科学命题与研究边界

- RQ1：对于同一冻结场景 `s` 和两条任务进展相近、均合法可执行的 ego 候选 `(a,b)`，两个独立响应模型 `P,Q` 对第三方风险的严格顺序是否相反？主端点为预先给定的 `(A–B pair collision, sustained hard brake, short TTC)` 字典序，连续制动积分为**单独预注册于 v8 的次端点**。事后 rollout 风险排序不是规划器的提前选择。
- RQ2：固定初始状态和 A/B 身份后，某个 ego 动作对该第三方车辆对的风险是否具有可复现的正向干预效果？是否存在既任务相近、ego 自身安全、又能避免这一额外风险的替代动作？`cruise` 不并线是干预负对照，而**不是**任务匹配替代动作。提前于并线指令的刹车须单独排除或标记。
- RQ3：经过充分训练调优的 courtesy、risk-aware 和交互规划器，在事先冻结的信息、候选池、预算及 ex-ante 评分的公平条件下，是否仍选择独立环境中更危险的动作？**本阶段不把 Bosch/ReactSim 的世界模型误叫作规划器。**
- 主要研究风险：RACP、HYPE、GameFormer、FeAR 与已有决定价值/预测差异研究覆盖了大部分宽泛问题；即使证明存在一条响应模型敏感性曲线，也不等于已有论文未研究过。必须最后落实为“交通语义具体、任务匹配、第三方响应依赖、证据可靠性导致可避免的规划决策错误”。

## B. 对四份 v7 证据包重新计算的结论

所有统计来自真实 JSONL 中 `type=trial` 且属于 TM / BehaviorAgent 的记录；joint-witness 没有混进分母。

| 上传实验 | 全部 trial | valid | 机制执行 raw | valid 且执行机制 | A–B 碰撞 raw | A–B 碰撞 valid | 跨模型任务匹配候选对 | 主端点严格翻转 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v7-dev seed17 | 100 | 87 | 80 | 68 | 0 | 0 | 9 | 0 |
| v7-confirm seed17 | 80 | 38 | 64 | 29 | 9 | 0 | 0 | 0 |
| v7-confirm seed23 | 80 | 34 | 64 | 25 | 10 | 0 | 0 | 0 |
| v7-confirm seed41 | 80 | 40 | 60 | 32 | 7 | 0 | 0 | 0 |
| **合计（不是独立样本）** | **340** | **199** | **268** | **154** | **26** | **0** | **9** | **0** |

特别地，确认 seed17、23、41 中 TM 分支各为 **6/40、4/40、0/40 valid**，而 BehaviorAgent 各为 **32/40、30/40、40/40**。这不是已确定的 TM 模型科学失败：错误包括 `uncontrolled_actor`、`offroad_or_lane_departure`，还存在物理撞车/场景初始化/控制器离开车道的混淆。26 条第三方碰撞全部在无效 trial 内，且 8 条发生于没有并线的 `cruise` 中（3、3、2），不能认定是 ego 并线引起的 A–B 次生风险。

开发集固定主端点中有 9 组跨模型有效配对，0 次翻转；连续制动 B 的次端点超过 0.25 m/s 预定阈值的配对有 **2/9**，但连续负担跨 TM/BehaviorAgent **严格排序翻转依然为 0/9**。这最多是弱模型敏感性线索，而非 RQ1 通过。旧硬刹二元指标在不同动作间广泛饱和，不能把“硬刹全是 1”理解为没有行为差异。

**致命的确认协议偏移**：开发集 `cutin_late_switch_s=3.5`、`cutin_lateral_lookahead_m=20`；三组 confirm 使用 `4.5` 与 `15`，即发生了控制协议回退。四包均为 gap35/delta0 一个 cell，**并没有跑出完整因素网格**。4 包一共 25 个唯一近似物理走廊签名，8 个签名跨实验重复，因此“3 个 seed = 3 倍独立场景”的理解错误；统计须按真实物理走廊聚类。

实验原始控制代码对 cut-in 各动作**维持 ego 目标速度不变**，`early_slow/late_slow` 和 `mild_early/mild_late` 实际代表**不同横向并线启动时间**，不是它们名字暗示的纵向减速。v8 在 manifest 中明确记录这一语义。旧 `action_protocol_v5` 静态描述不再作为 v8 可信来源。

**RQ1 = NOT SUPPORTED**；**RQ2 = NOT SUPPORTED**；**RQ3 = NOT TESTED**。当前样本既不能声称强基线系统性失败，也不能证明研究命题普遍不存在。相比以前，CARLA 确实出现过有效、安全的真实并线（dev 68/100 valid且机制执行），但这些实验没有识别出独立可复现的第三方可避免风险。

## C. nuPlan / ReactSim 数据：为什么缺 1036 个日志

来自 `diagnostics.zip` 官方作者 validation 场景列表的审计：

- ReactSim frozen validation ID **2636 个 scenario tokens**，分组后关联 **1984 个不同 DB log names**。
- 你的现有本地划分里（`val/train_boston/train_pittsburgh/train_singapore/train_vegas/test`）可以访问 8732 个 train/val 不同 DB 文件，另有 test 1310 个，合计 **10042**。
- 在 1984 个作者要求的 DB log names 中，**948** 个存在；另外 **1036** 个在这六个本地 split 中同名文件均缺失。具体影响 **1501 个 validation scenario tokens**；余 **1135** 个通过初步 SQL 表关联检查。
- 之前的 `DB_MISSING` 不能再归因于“已在 train_* 但未 stage”：这次 `MISSING_IN_ALL_LOCAL_SPLITS` 的含义是六个实际 split 中**文件名均不存在**。但仍可能存在于未纳入扫描的其他数据根目录/原始 archive ZIP，也可能是 ReactSim 的 frozen ID 采用了另一版 nuPlan 发布清单。
- 单 token `2cdb26865dfa5261` 对应的 DB 原始 SQLite quick_check 正常，token 存在，却位于 scene-0002/7；被 nuPlan 场景边界/历史窗口筛除，属于**解析/过滤条件问题**，不能靠重下相同 DB 修好。

**结论：不要重下 8732 个已存在 DB。** 如果要严谨复现完整 2636-scene ReactSim，需要先用原始完整官方 nuPlan v1.1 release manifest 和作者 `.scenario_ids_val.json` SHA256/仓库 commit 验证同一数据版本，再针对真正缺失的 1036 个日志增量补齐。填完后需另处理边界过滤、scene 映射和 91 帧未来/历史完整性。绝不能 `DELETE missing IDs` 后依然称为官方 2636 benchmark。

nuPlan AWS 官方公开资源：https://registry.opendata.aws/motional-nuplan ，区域 `ap-northeast-1`，匿名 S3 bucket。实际压缩 archive 文件名由 `aws s3 ls` 列出，不得按猜测的 S3 key 下载。nuPlan 数据遵循其原始许可与发布版本。建议先找本地其它复制及已经下载的 ZIP，最后才下载缺失的原始发行包。

### 只读查找、缺失名单、已下载 ZIP 索引

```bash
conda activate ride_reactsim_dp
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V8_EVIDENCE_GATE_20261028
source scripts/00_env.sh
export RIDE_REACTSIM_REPO=/data0/senzeyu2/dataset/RIDE/research_baselines/ReactSim_Bench
export RIDE_REACTSIM_WORK=/data0/senzeyu2/dataset/RIDE/reactsim_workspace
export RIDE_NUPLAN_SOURCE_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan

bash scripts/22_reactsim_db_coverage.sh
python -m ride.reactsim_recovery \
  --coverage "$RIDE_REACTSIM_WORK/diagnostics/val_db_coverage.json" \
  --extra-root /data0/senzeyu2/dataset/RIDE \
  --output "$RIDE_REACTSIM_WORK/diagnostics/v8_missing_search.json" \
  --missing-list "$RIDE_REACTSIM_WORK/diagnostics/v8_missing_db.txt"
# 注意 --extra-root 只用于扫描真实存在的别处数据目录。避免把整个 /data0 根扫描多次。
```

如果你已有一个官方下载的 `nuplan-v1.1_*.zip`，可通过 `--archive /真实路径/原始.zip` 复查本地压缩包有无这批文件。确认该 ZIP 确实属于 `val`（而不是 test）后，再逐个原始 split 增量提取：

```bash
python -m ride.nuplan_extract_missing \
  --archive /真实路径/已验证来源于val的nuplan压缩包.zip \
  --missing-list "$RIDE_REACTSIM_WORK/diagnostics/v8_missing_db.txt" \
  --splits-root "$RIDE_NUPLAN_SOURCE_ROOT/nuplan-v1.1/splits" \
  --split val \
  --report "$RIDE_REACTSIM_WORK/diagnostics/val_archive_dryrun.json"
# 先查看 dry-run 的 matched_in_archive 和文件名；只有发布清单及 split 核对无误时追加 --apply。
```

如本地归档没有：

```bash
conda activate ride_reactsim_dp
python -m pip install awscli
aws s3 ls --no-sign-request --region ap-northeast-1 s3://motional-nuplan/
# 按上步列出的真实前缀再查看归档目录；不要盲目 s3 sync 整个 bucket。
aws s3 ls --no-sign-request --region ap-northeast-1 s3://motional-nuplan/public/
```

从授权官方发布源下载对应缺失日志所在原始训练/验证档案，然后解压或使用选择性工具。对于 `FILTERED_SCENE_BOUNDARY`，即使 DB 存在也要修复或解释作者 devkit 场景筛选，不得擅自把过滤掉的 token 当作完整场景。

### 在缺 DB 时依然可以做 ReactSim 小子集**安装冒烟**（不能用于论文主表）

```bash
conda activate ride_reactsim_dp
source scripts/00_env.sh
bash scripts/15_reactsim_nuplan_stage.sh apply \
  --splits val train_boston train_pittsburgh train_singapore train_vegas
export RIDE_REACTSIM_SMOKE_N=12
bash scripts/25_reactsim_subset_smoke.sh
```

输出在 `$RIDE_REACTSIM_WORK/subset_smoke_v8/`，标注 `NON_BENCHMARK_EXPLORATORY_SUBSET` 且保留原始作者 ID 列表 SHA256。能否完成预处理仍取决于 token 的完整场景窗口和 devkit 版本；失败必须保留日志，**不是数据或研究命题失败**。官方 full benchmark 仍需原始 ID 全覆盖。

## D. v8 源码的修改和局限

已保留 RIDE 原有实验工程和脚本，增量新增/修复：

1. 将冻结协议升级为 `ride_lane_corridor_v8_20261028`；`--anchor-selection-seed` 与 `--seed` 解耦；固定 dev 与 confirm 的动作时间、车辆间隙、响应模型、道路、常速行为及 warmup，拒绝静默覆盖已有输出。
2. 增强初态检查，连同位置和速度比较航向（环绕 360°）；跨模型实际 ego 轨迹等价性门禁扩展到 v8。
3. 使用冻结目标车道中心线**弧长站点 + 横向投影**计算并线进展、A/B 实际同车道跟驰 TTC，避免跨 CARLA road_id 分段时 TTC 总是缺失和 goal-heading 进展失真。
4. 物理碰撞事件保存外部物体 `type_id`/actor ID，后续可区别静态障碍和普通第三方车辆；试验摘要分开报告 raw / valid / valid且机制执行。
5. 对当前四份 v7 JSONL 的离线审计（`ride.v7_forensic_panel`）；另提供 `ride.secondary_effect_gate` 输出连续负担第二主张，但严格不重定义旧主端点。
6. `ride.reactsim_recovery` 盘点所有本地 split 以外的额外文件/ZIP；`ride.nuplan_extract_missing` 先 dry-run 并逐 DB 校验 SQLite、禁止覆盖和 `test` 混入。
7. `ride.reactsim_subset` / `scripts/25_...` 保留官方 ID 列表不动，在独立目录生成可追溯的非官方小规模安装测试。
8. `ride.anchor_disjoint` 与 `scripts/27_probe_disjoint_v8.sh`：冻结后、执行前检查 dev/confirm 走廊近邻重叠；只使用地图几何进行 seed 候选筛选，不读取风险 outcome。
9. `ride.baseline_preflight` 静态检查 GameFormer-Planner、ReactSim、Bosch 源码及 checkpoint；`ride.choice_contract` 验证 ex-ante baseline selection manifest，**并未伪造官方 planner**。

**没有在这台审计容器里启动你服务器的 CARLA、nuPlan 和 GPU 模型。** 以上静态/单元测试不能取代 v8 物理复现，也无法保证所选 Town04 新 corridor 持有足够独立空间。

## E. CARLA 0.9.16 无头服务器操作：先新协议的单次 dev，再确认

### 1) 环境及 tests

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip -q RIDE_V8_EVIDENCE_GATE_20261028.zip \
  -d /data0/senzeyu2/dataset/RIDE/code
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V8_EVIDENCE_GATE_20261028
conda activate mfrp
python -m pip install -e '.[test]'
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pytest -q
bash scripts/01_doctor.sh
python -m ride.doctor --server
```

若 CARLA 无头服务还没启动，请在独立终端先启动 `bash scripts/02_start_carla_headless.sh`；不要在多个终端同时独占同一个 CARLA sync world。

### 2) Dev，先看 M0：禁止旧 env 变量悄悄覆盖

```bash
# 建议在新 shell 执行；过去用过的默认设置必须清除
unset RIDE_ANCHOR_SELECTION_SEED RIDE_V8_ANCHOR_SEED_EXPECTED \
  RIDE_CUTIN_LATE_SWITCH_S RIDE_CUTIN_LATERAL_LOOKAHEAD_M \
  RIDE_REAR_GAP_GRID RIDE_REAR_SPEED_GRID RIDE_FAMILIES RIDE_PROFILES
export RIDE_V8_EVIDENCE_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_locked
bash scripts/24_run_v8_panel.sh 17 dev
```

dev 输出在：`$RIDE_V8_EVIDENCE_ROOT/dev/seed_17/`，包含 frozen manifest、原始轨迹、risk ranking、mechanism diag、forensic。首先要求至少一条 `valid=true AND mechanism_executed=true AND ego_collision=false` 的早/晚任务匹配动作对；不足则先**修复场景控制**，不要直接开 10×多 seed 大规模运行。

对应原始 JSONL 的次端点：

```bash
python -m ride.secondary_effect_gate \
  --input "$RIDE_V8_EVIDENCE_ROOT/dev/seed_17/rear_gap_35/rear_delta_0/carla/cut_in.jsonl" \
  --output "$RIDE_V8_EVIDENCE_ROOT/dev/seed_17/secondary_burden.json"
```

### 3) 仅在 dev 的 M0 稳定后进行三组确认

新的 v8 还增加了**事前位置互斥门禁**：在任何 confirm outcome rollout 前，与 dev 冻结 anchor 在 25 m 半径内重合的场景将使执行命令拒绝继续。若默认 anchor selection seed 9901 重合，可以根据**地图几何而非 outcome**选择另一个 seed，先执行：

```bash
# 只生成 frozen anchors，不跑任何行动，不观测碰撞/风险/规划分数
bash scripts/27_probe_disjoint_v8.sh 9901 9920
# 按脚本输出记录真实通过的 seed，例如（仅举例，不能事前保证9907通过）
export RIDE_V8_CONFIRM_ANCHOR_SEED=实际通过互斥检查的数字
```

再运行确认批次。程序会在**冻结后、rollout 前**再次检查 dev/confirm 25m 近邻重合；这一检查不是地图拓扑级独立性的数学证明，仍要人工核对道路 ID、走廊路线、物理位置及交互身份：

```bash
# 环境变量仍应清除 stale 参数；下面 anchor selection seed=9901 固定，
# 响应 seed 分别为17/23/41；所有 confirm 复用同一个冻结 corridor 集。
for seed in 17 23 41; do
  unset RIDE_ANCHOR_SELECTION_SEED RIDE_V8_ANCHOR_SEED_EXPECTED
  bash scripts/24_run_v8_panel.sh "$seed" confirm
done
```

**注意：不同 anchor selection seed 本身不能保证 dev 与 confirm corridor 不重合。** v8 新增 rollout 前 25m 近邻冻结门禁；运行后仍须使用下面的聚合程序检查 `physical_corridors_reused_across_runs` 并查看 manifest；如果有重复走廊，这些不能当作完全独立的 held-out scene。真正最终 confirmation 应固定一份事前核验过、与 dev 完全不同的 scene-token/road-corridor 集合。按 corridor 聚类而不是只按 seed 或 trial 计数。

```bash
python -m ride.v7_forensic_panel \
  --panel "$RIDE_V8_EVIDENCE_ROOT/dev/seed_17/rear_gap_35/rear_delta_0/carla/cut_in.jsonl" \
  --panel "$RIDE_V8_EVIDENCE_ROOT/confirm/seed_17/rear_gap_35/rear_delta_0/carla/cut_in.jsonl" \
  --panel "$RIDE_V8_EVIDENCE_ROOT/confirm/seed_23/rear_gap_35/rear_delta_0/carla/cut_in.jsonl" \
  --panel "$RIDE_V8_EVIDENCE_ROOT/confirm/seed_41/rear_gap_35/rear_delta_0/carla/cut_in.jsonl" \
  --output "$RIDE_V8_EVIDENCE_ROOT/v8_pooled_forensic.json"
```

### 4) 单独的补充**发现性**因素实验（不得混进 v8 confirmation）

- 原 v7 仅 gap35/delta0。对真正存在的背景风险饱和，建议开发实验另行尝试 `rear_gap=35,45 m` × `rear_speed_delta=0,1.0 m/s` 并保留 `cruise` 负对照。此前 v6 gap45 在 `cruise` 已高度饱和，因此不承诺 gap45 更好。
- 同时记录不同切入时刻而不是把 `early_slow` 当成速度策略；关心 A/B 的 decel onset 是否先于 ego 横向指令，哪怕发生在入道前，也不自动否认受并线**意图**影响，必须区分指令开始和完成并线两个时间。
- 你也可扩展第二个机制：cut-in + convoy 的固定 follower_a→follower_b 连锁响应。以每个 corridor 的第三方 A-B TTC、碰撞、持续减速与车辆进展为报告端点，切勿通过预设碰撞的 joint_witness 人为驱动车辆并声称独立响应 RQ2 成立。
- `intersection` 暂不进入主证据表：当前没有严格的交通灯停止线、路权、转向合法性与精确路线弧长；`support_gap` 是同一道路几何上的有限证据消融，不是独立第四类地图拓扑。

## F. RQ3 的正确预备：Bosch vs GameFormer vs ReactSim

| 项目 | 用户上传及源码核对 | 实际身份 | 现在可做 | 不能直接声称 |
|---|---|---|---|---|
| GameFormer-Planner-main | 有 `Planner/planner.py` / `run_nuplan_test.py`，确实是 nuPlan planner，不是原 GameFormer open-loop 仓库 | 强规划器（闭环支持 IDM） | 适配 nuPlan env、模型训练/权重检查、原生 nuPlan IDM 5 个场景 smoke | 跑完 5 个场景就等于公平 RQ3 |
| Bosch interactive-closed-loop | 本次未上传该源码，仅服务器有已 clone 的 Bosch 仓库 | SMART reactive nuPlan 改造版，response evaluator | 独立 nuPlan-devkit 工作树安装、复现实验里 log_future_planner + SMART | 它本身是完整的 RQ3 新 planner |
| ReactSim-Bench | 已上传源码，官方 `scenario_cache.py`、SMART、MTR 等均存在 | off-log ego 干预后的 reactive world model benchmark | 现有 1135 条初步 SQL eligible 的 subset smoke，补齐 DB 后官方 SMART/MTR | ReactSim 的世界模型分数等于 planning regret |
| RA-CBF / R3R | 先前已检查 | 安全责任过滤/Julia toy 安全机制 | 机理近邻与假设消融 | 与 GameFormer 同条件直接放入 planner 主表 |

### 外部仓库离线清单

```bash
python -m ride.baseline_preflight \
  --gameformer /data0/senzeyu2/dataset/RIDE/research_baselines/GameFormer-Planner \
  --reactsim /data0/senzeyu2/dataset/RIDE/research_baselines/ReactSim_Bench \
  --bosch /data0/senzeyu2/dataset/RIDE/research_baselines/Bosch_SMART_reactive \
  --nuplan /data0/senzeyu2/dataset/CapPlan/data/nuplan \
  --output /data0/senzeyu2/dataset/RIDE/evidence/baseline_preflight_v8.json
```

本次上传的 `GameFormer-Planner-main` 只有源码，未发现 `.pth`；`ReactSim-Bench-main` 只有源码，没有嵌入大模型权重；不能把源码 clone 等于安装完成。GameFormer 需与网络结构匹配的 nuPlan 训练 checkpoint，**不是 ReactSim 的 `mtr.pth`**。

如果服务器的 GameFormer 原仓库位于 `.../research_baselines/GameFormer-Planner`（请以真实目录为准），并且已经有模型权重：

```bash
# 使用另一个与作者 nuPlan-devkit v1.2.2 兼容的专用环境，
# 不要和替换 planning/ 的 Bosch fork 共用 site-packages/PYTHONPATH。
conda activate ride_gameformer
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V8_EVIDENCE_GATE_20261028
source scripts/00_env.sh
export RIDE_GAMEFORMER_REPO=/data0/senzeyu2/dataset/RIDE/research_baselines/GameFormer-Planner
export RIDE_NUPLAN_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan
export GAMEFORMER_MODEL_PTH=/真实路径/与GameFormer网络配置匹配的model.pth
export RIDE_GF_TOTAL=5
bash scripts/26_gameformer_native_smoke.sh
```

如果确实没有可用权重，先找作者提供的正式 checkpoint；找不到则按其 README 执行 `python data_process.py --data_path ... --map_path ... --save_path ...`，再 `python train_predictor.py --train_set ... --valid_set ...`，需要保证 training/validation 不混用与计算资源足够。**不要拿任意 PyTorch .pth 充当 GameFormer**。

Bosch 官方复现从独立 `nuplan-devkit/nuplan/planning` 中替换为 Bosch 的 `interactive-closed-loop`，作者建议参考：

```bash
# 单独新工作目录，不要对现有 GameFormer 依赖的 nuPlan 安装执行 rm -rf planning。
mkdir -p /data0/senzeyu2/dataset/RIDE/baseline_envs/bosch
cd /data0/senzeyu2/dataset/RIDE/baseline_envs/bosch
git clone https://github.com/motional/nuplan-devkit.git
git clone https://github.com/rainmaker22/SMART.git
# 若已 clone 的 Bosch 源码可用，以其 commit 为准，复制/链接到全新的
# nuplan-devkit/nuplan/planning 工作树；记录 git commit 与 license。
# 请先阅读官方 README 的安装和 checkpoint 位置，不要在旧目录原地覆盖。
```

Bosch 官方原生 smoke：

```bash
export NUPLAN_DEVKIT_ROOT=/data0/senzeyu2/dataset/RIDE/baseline_envs/bosch/nuplan-devkit
export NUPLAN_DATA_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan
export NUPLAN_MAPS_ROOT="$NUPLAN_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/data0/senzeyu2/dataset/RIDE/bosch_native_exp
export NUPLAN_CACHE_ROOT="$NUPLAN_EXP_ROOT/cache"
export SMART_ROOT=/data0/senzeyu2/dataset/RIDE/baseline_envs/bosch/SMART
export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$SMART_ROOT:$PYTHONPATH"
# 在作者提供的 nuplan_smart 专用环境中、确认 planning/checkpoint 可读后：
python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  experiment_name=simulation_smart_agents \
  +simulation=closed_loop_smart_reactive_agents \
  planner=log_future_planner scenario_builder=nuplan \
  scenario_filter=test_one_scenario
```

重要：Bosch 与 ReactSim 的 SMART 可能共享模型家族，**不算两个结构独立的响应假设**；ReactSim MTR 与 SMART 跨架构对照更有独立性，但也需要在相同 ID、输入和候选动作下配对。

GameFormer 原生闭环运行成功后，还需适配 **ex-ante 选择导出接口**，即在 rollout 前保存候选动作、每条动作基于本 planner 假设得到的风险/代价、时刻、初态哈希、模型 checkpoint SHA256、所选动作等。RIDE v8 `ride.choice_contract` 只做此 schema 审计，**没有替用户训练或替换作者的 Planner 模型**。运行：

```bash
python -m ride.choice_contract \
  --choices-jsonl /真实路径/gameformer_prelayout_choices.jsonl \
  --output /真实路径/gameformer_choice_contract_audit.json
```

## G. 下一轮科学 Gate 和研究故事转向的客观条件

### M0：先证明物理试验有资格作为分母

1. 实际初态的 ego/A/B 位置、速度、航向跨模型成对一致，并排除初始重叠、预热碰撞与未清理 actor。
2. 每条被对比的 ego 动作满足 `mechanism_executed`、同一路线弧长进展容差、道路和法律有效性、ego 自身安全。**为了完全避免只选择成功 sample 的选择偏差，同时报告所有 attempted/invalid 原因与比例。**
3. `cruise` 作为本底对照，观察后车减速是否先于横向指令；跨模型记录同一物理走廊和同一响应种子的 reset 模式。
4. 至少有足够可比较的跨模型 pair 才进入 RQ1 统计；以物理 corridor cluster bootstrap、paired sign/permutation 和区间估计，不把多候选×重复 seed 当作完全独立 N。

### RQ1 的支持条件

- 至少在冻结开发协议之外可复现严格非平局翻转；在相同动作池、相同初态、相似实际 ego 轨迹和相同风险主端点上成立。
- 原主端点和新连续 burden 次端点**分别报告**。若只有连续 burden 差异但没有翻转，不得回填为旧 RQ1 阳性。
- 由可独立校准的 SMART/MTR（或其他新响应环境）验证至少一种有交通语义的可重复模型差异。单纯 CARLA TM 与 BehaviorAgent 的差异仍缺真实分布外推。

### RQ2 的支持条件

- 同一 scene A/B 两个合法、ego 安全、进展匹配的动作；固定第三方 A/B 身份；固定另一模型的状态和评测预算。
- 报告第三方碰撞、短 TTC、持续减速、相对 `cruise` 背景增量的完整时序，且风险先后与 ego 指令合理，不能把任意背景刹车都归因于切入。
- 逐 corridor 配对效果及不确定性；反事实结论仅限于明确指定的仿真响应模型，不声称人类真实因果效应。

### 进入正式 RQ3 之前

- RQ1 和 RQ2 在**预冻结确认集**具有可重复正例；至少一组 learned reactive 环境不与 CARLA 控制器共用相同机制；排除设置错误。
- GameFormer 学习 checkpoint 和至少一个风险/courtesy 强对照在真实候选和闭环预算下经官方复现，ex-ante 锁选择日志经 `choice_contract` 校验；统计进展—第三方风险 Pareto 前沿，避免永远停车的伪优。
- 若强基线可以稳定解决你找到的所有失败，不应继续声称他们系统性失败。

### 需要调整论文故事的情况

**结果 A：valid pair 充分但严格排序几乎都不翻转。** 不无限调参碰运气；转向“模型不确定性何时真的影响规划决策？”的负结果分析，或用以下方向做更具原创性的假设。

**结果 B：连续负担/有限响应可行证据有决策差异，但碰撞端点差异弱。** 论文可收敛为 *Decision-Critical Behavioral Recourse Evidence*：区分 `VERIFIED_FEASIBLE`、`CERTIFIED_INFEASIBLE`（需显式完备搜索域与证明）和 `UNKNOWN`，并研究在受道路优先权、自然行为与负担约束下，如何为减少错误决策分配有限验证预算。必须和决策感知预测 / RACP / HYPE / FeAR 等已有方法逐项对比。不能把普通有限搜索 miss 称为全球不可行性。

**结果 C：只有 CARLA TM / BehaviorAgent 出现差异，SMART/MTR 不复现。** 论文的 scope 应降为某些仿真器和控制器的评价偏差，不宜声称普遍交通失效。

**结果 D：强 planner 都能充分解决。** 不能强行包装 gap；可选择定义更窄的有证据诊断问题，或放弃 RIDE 这一主张。

当前最有价值的研究路径：**锁定 v8 物理/策略协议 → 获得可信 M0 分母 → 检验 RQ1/RQ2 → 在 nuPlan 子集安装并补足原始数据 → Bosch/GameFormer 原生 smoke → 真正公平 RQ3 → 最后设计新算法。**

## H. 已完成的本地工程验证及不可替代的服务器工作

审计容器：`python -m pytest -q` **93 项通过**，`compileall -q ride` 与所有 Shell `bash -n` 通过；对四份 v7 原始 JSONL 的法证摘要及连续 burden 次端点已重算，源码包含单元回归。**没有**对用户服务器 CARLA、nuPlan、模型 GPU 做任何动态成功宣称。
