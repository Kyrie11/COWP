# RIDE v5：RQ1/RQ2 实验审计、预注册修订与下一步运行指南

**代码日期：2026-10-28。所有 v5 代码均只在本地离线经过 70 项 pytest、compileall 和 shell 语法验证；没有在用户的 CARLA/nuPlan/ReactSim 服务器上实际完成 v5 物理闭环。**

## 1. 证据级别与最新结果

对应上传的 `rq12_v4_smoke_20261027.zip` 和 `rq12_v4_support_bank_20261027.zip`，以完整 JSONL 和原审计文件重算：

| 指标 | 实际数字 | 可得结论 |
|---|---:|---|
| v4 smoke: convoy | 8/8 完整、8/8 valid | 有一类可执行的三车跟车机制 |
| v4 smoke: cut-in | 8/8 完整、2/8 valid、0/8 mechanism | 旧并线场景无法建立科学正例 |
| v4 cut-in ego–前车物理碰撞 | 8/8 | 目标第三方风险不符合 ego 安全前提 |
| v4 cut-in 道路/车道报警 | 6/8，合计 13 帧 | 可能包括地图 lane-boundary 投影伪影；不能一概当作违规或忽略 |
| v4 smoke 预注册动作对 | 8 对；0 对 eligible | 完全没有 RQ1/RQ2 的分母 |
| v4 RQ1 严格排序翻转 | 0，eligible 也是 0 | **不可把 0 解释为真实否定 RQ1** |
| v4 Support WIDE | 40 trial，其中 36 为联合控制候选 | 所有 40 条均无法通过原 valid predicate |
| v4 Support NARROW | 从 WIDE 原始结果提取 13 trial | 比较方式一致，但 valid nested comparisons=0 |
| v4 有效有限库 miss | 0 | 无 positive witness；不构成全空间 UNSAT |

`support_gap_WIDE` 中可看到 28 条 ego 碰撞、4 条非 ego 碰撞，但这 4 条来自人工执行的 joint-witness 搜索，不是可解释的独立驾驶者导致第三方碰撞。旧碰撞传感器每帧可能重复触发，原始 collision event 数量不是事件数，不得计成多个独立事故。

不同模型（CARLA TM、BehaviorAgent）在 v4 的部分预热起点存在米级差异；convoy 各行动的末端任务 progress 也存在 4.6 米以上偏差，因此没有通过 3 米的 action matching gate。

**整体判断**：RQ1=NOT_YET_IDENTIFIABLE；RQ2=NOT_YET_IDENTIFIABLE；RQ3=NOT_RUN。比“没有发现现象”更准确的说法是：这批数据没有给出可用于检验假设的合格样本。

## 2. 找到的实质性代码缺陷

1. `cut_in_candidate` 的 front ego 纵向间距只有 9m；v4 中 ego 与 follower_a 八次全部物理碰撞。v5 把相对中心线位置从 `A=0, ego=-9, B=-22` 调整为 `A=0, ego=-20, B=-45`，仍是场景*筛查*而不是实证正例。不同 road/lane 的完整地图走廊已冻结。
2. v4 的 `ego_in_target_lane` 需要 ego 当前 `road_id` 恰好等于冻结的 `follower_a.road_id`，因此穿越 OpenDRIVE 道路分段（例如 Road 45→6）时严重漏检。v5 使用目标 lane 的冻结中心线和横向误差，要求连续 0.5s 驻留。
3. v4 使用 `get_waypoint(project_to_road=False)` 的单帧未命中判定整个 trial invalid，导致已执行的合法车道线横跨可能被误当 offroad。v5 同时记录 `road_strict_valid_by_role` 和最近 drivable lane 的中心线误差，主控以车身中心是否在路面 ribbon 内判定；**车辆整个 bbox 和路权仍需单独复核**，不应把 center-valid 称为完整合法性证明。
4. v4 在 warm-up 期间已经分别使用 TM 和 BehaviorAgent 控制他车，违反跨响应模型共初始状态；v5 统一以 PID 对 ego/A/B 预热，记录初始状态后才交给各 responder。`Protocol` 比较门槛收紧到位置 0.5m、速度 0.5m/s。Actor-only reset 仍不消除全部 Unreal 隐状态，应在稳定环境复现 strict reload。
5. v4 convoy `early_slow/late_slow` 最终 progress 经常差大于 3m，未形成任务匹配；v5 在实验前预注册减小速度扰动幅度（`mild: 1.8m/s`, primary: `3.0m/s`），**仍然要用实际执行的轨迹判断是否 match**。
6. v5 添加 `--require-mechanism` 门禁：必须存在至少 1 条 valid + 真实机动完成 + ego 无碰撞的 trial。它只是 M0 工程门槛，不代表 RQ1/RQ2 pass。
7. 支持库从一次完整 WIDE rollout 提取精确 NARROW 子集，这一实现已在 v4 保留；但所有候选均 invalid 的话，结论必须是 UNKNOWN，不是 infeasible。

旧 `ride_lane_corridor_v4_20261027` anchors 不能复用；v5 为 `ride_lane_corridor_v5_20261028`，不同 rear speed delta 也被纳入 manifest 与恢复时一致性校验。

## 3. 必须分开论证的科学命题

**P0/M0 场景物理可达性**：真实道路、真实交通角色、无预热碰撞，A/B 同一初始状态（<=0.5m/0.5m/s）、ego 动作真的不同；cut-in 至少一条真实完成并线且 ego 无碰撞。此关失败不得进入规模化敏感性搜索。

**P1/RQ1 决策相关的模型分歧**：针对同一冻结场景、同一 ego 候选对和同一第三方车辆身份，至少两个响应模型的第三方风险给出*严格非平局排序翻转*；每一动作在评测前由相应模型产生 ex-ante 风险估计/选择。CARLA 仿真后的 `risk_ranking_disagreement.json` 仅说明 post-hoc 不同响应环境的排序敏感性；没有 ex-ante 选择，仍不能叫真实 planner 的 decision error。

**P2/RQ2 自主响应环境中可避免的额外代价**：固定同一场景和 actor pair、progress 差 <=3m、ego 两个动作合法且无碰撞，A 有比 B 更严重第三方风险，B 能避免，而且结果不来自初始重叠、预热冲击、道路投影错误或人为控制 joint witness。单一急刹阈值不能独立成为正例，需给出持续制动/短 TTC/非 ego 物理碰撞分解和负向控制。至少一套外部 ReactSim SMART/MTR 等 reactive world model 复核；**这些模型也不是现实人类反事实真值**。

**P3/RQ3 强规划器的剩余缺口**：在上述 P1/P2 成立后，要求提供公平的 hard courtesy、risk-aware contingency、ego-conditioned plan selector 预承诺选择，并统一 candidate pool、progress、预算、合法性和闭环评价。不能把 ReactSim 的 SMART/MTR 反应环境直接称为规划器基线；R3R 和 RA-CBF 的通讯/安全假设须单列。

建议正例数：先完成 M0 的 2–3 anchors × 3 seeds 工程稳定性，再在独立预注册场景 >=20 anchors 中找 RQ1 严格翻转，并在 >=10 anchors 形成可避免第三方风险方向一致的 RQ2 正例（这些是建议的研究 go/no-go 检查，不是统计功效保证）。正式论文仍须配对场景/anchor 聚类置信区间、机制分层和盲选 heldout。

## 4. CARLA v5 使用说明

### 4.1 解压、安装、代码和服务端校验

```bash
mkdir -p /data0/senzeyu2/dataset/RIDE/code
unzip RIDE_RQ12_V5_20261028.zip -d /data0/senzeyu2/dataset/RIDE/code/
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V5_20261028
conda activate mfrp
python -m pip install -e '.[test]'
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export DATASET_ROOT=/data0/senzeyu2/dataset
source scripts/00_env.sh
python -m pytest -q
bash scripts/01_doctor.sh
python -m ride.doctor --server
```

如未启动 CARLA，请在**独立终端**用 `conda activate mfrp; cd ...; source scripts/00_env.sh; bash scripts/02_start_carla_headless.sh` 启动无头服务；不要并发执行两个使用同一 CARLA 世界的实验。

### 4.2 优先跑两条物理 smoke；先不要跑 support-gap 的 36 witness 大搜索

```bash
export CARLA_TOWN=Town04
export RIDE_SEED=17
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export RIDE_WARMUP_TICKS=20
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v5_smoke_20261028
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_ALL=0
export RIDE_SMOKE_REQUIRE_VALID=1
source scripts/00_env.sh

export RIDE_REAR_SPEED_DELTA_MPS=0
bash scripts/01b_probe_anchors.sh convoy Town04
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy

export RIDE_REAR_SPEED_DELTA_MPS=1.5
bash scripts/01b_probe_anchors.sh cut_in Town04
bash scripts/03_precommit_scenes.sh cut_in
export RIDE_SMOKE_REQUIRE_MECHANISM=1
bash scripts/03_run_carla_case.sh cut_in
```

`--require-mechanism` 可能以非零退出，这是实验不满足 M0 的正确报警；日志和 JSONL 仍保存。若中途失败，**先看日志和有效轨迹**，不要在同目录直接重跑：结果文件默认禁止覆盖。可在独立执行新的 `RIDE_OUTPUT_ROOT` 下使用 Town03/Town05 进行预注册探索，记录地图选择，不能在正式 holdout 中挑选更好 seed。

若两类 smoke 都运行完整，可以：

```bash
bash scripts/04_analyze_carla.sh
bash scripts/12_research_status.sh
python -m ride.forensics --input "$RIDE_OUTPUT_ROOT/carla/cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/cut_in_forensics.json"
```

需要人工检查的物理诊断包括：`ego_target_centerline_error_m`、`target_lane_dwell_max_s`、`ego_collision`、`road_center_valid_by_role`、`road_strict_valid_by_role`、`pre_action_collision_events`、`initial_states`、`hard_brake_roles`、`min_rear_pair_ttc_s` 和 `ego_progress_m`。真实踏线行为需要用 CARLA 可视化回放核验 bbox/道路边界，不要仅凭中心点判定法规合法。

### 4.3 仅在 cut-in 的 M0 通过后运行 support-gap

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v5_support_bank_20261028
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
export RIDE_SEED=17
export RIDE_REAR_SPEED_DELTA_MPS=1.5
unset RIDE_SMOKE_REQUIRE_MECHANISM
source scripts/00_env.sh
bash scripts/03_precommit_scenes.sh support_gap
bash scripts/03_support_ablation.sh
cat "$RIDE_OUTPUT_ROOT/carla/support_gap_bank_ablation.json"
```

WIDE/NARROW 是同一次真实 CARLA joint-control 实验的 nested 子集。全部候选 valid+完整且独立存在正向 witness 时才有局部 finite-bank miss；无 witness 一律 UNKNOWN。

**Intersection**：原构造未验证信号/路权/冲突区域/路由弧长，保留 exploratory；当前不得进入 RQ2 合法性主表。优先把 cut-in→后车刹车→第三车追尾这一具体第三方链条做深，而不是无控制地增加新场景数量。

## 5. ReactSim + 用户已有 nuPlan DB/地图（不用重复下载 nuPlan）

官方 `ReactSim-Bench` 需要 `data/nuplan/maps`、`data/nuplan/nuplan-v1.1/splits/trainval`、`data/custom_behaviors`、`data/nuplan_processed` 和 checkpoints；你已有的原始 DB 路径 `/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/{val,train_boston,train_pittsburgh,train_singapore,train_vegas,test}` 可以*按需软链接*构建官方期望的 `trainval` 扁平视图。严禁把 `test` 混入 training。Stage 会自动拒绝同名不同路径 DB 和已存在冲突文件，绝不会删除或复制原 DB。是否涵盖 ReactSim 2636 个场景需要后续作者预处理进行验证。

### 5.1 检查/建立 repo（不要重新 clone 已有仓库）

```bash
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_REACTSIM_REPO=$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench
export RIDE_NUPLAN_ROOT=$DATASET_ROOT/CapPlan/data/nuplan
export RIDE_REACTSIM_WORK=$DATASET_ROOT/RIDE/reactsim_workspace
# 如果历史 07_download... 已将数据下载在旧根目录，请使用如下位置：
export RIDE_REACTSIM_BEHAVIORS=$DATASET_ROOT/ReactSim-Bench/custom_behaviors
# 如果实际上已移动到 RIDE/ReactSim-Bench/custom_behaviors，改为：
# export RIDE_REACTSIM_BEHAVIORS=$DATASET_ROOT/RIDE/ReactSim-Bench/custom_behaviors

ls -l "$RIDE_REACTSIM_REPO/reactsim_data_pipeline/preprocess.sh"
ls -l "$RIDE_NUPLAN_ROOT/maps/nuplan-maps-v1.0.json"
find "$RIDE_REACTSIM_BEHAVIORS" -maxdepth 1 -name '*.pkl' | wc -l

# 在 RIDE v5 工程目录执行
conda activate mfrp
source scripts/00_env.sh
bash scripts/15_reactsim_nuplan_stage.sh check --splits val
bash scripts/15_reactsim_nuplan_stage.sh apply --splits val
bash scripts/16_reactsim_run.sh preflight
```

`val` 是优先用于测试适配器的最小原生数据阶段，而**不是完整 ReactSim 基准**；作者官方 preprocessing 可能要求 train/val 共同数据，必须先查实际脚本。要按用户现有完整训练/验证 DB 整理时，请使用新的 workspace（不能覆盖旧 staging）：

```bash
# 继续使用第一次 stage 的相同 workspace。不要更换 RIDE_REACTSIM_WORK：
# repo/data 已指向原 workspace，改变目标会触发安全冲突检查。
export RIDE_REACTSIM_WORK=$DATASET_ROOT/RIDE/reactsim_workspace
bash scripts/15_reactsim_nuplan_stage.sh check \
  --splits val train_boston train_pittsburgh train_singapore train_vegas
bash scripts/15_reactsim_nuplan_stage.sh apply \
  --splits val train_boston train_pittsburgh train_singapore train_vegas
```

**切记**用户的现有 DB 和地图只是文件存在性描述，我没有远程访问服务器验证它们真的可读或能匹配 2636 个 ReactSim 场景。Stage 输出 `ready_for_preprocess=true` 只表示路径链接结构可供预处理尝试，不表示数据科学评测通过。

### 5.2 建立 author preprocessing 环境

```bash
conda create -n ride_reactsim_dp python=3.9 pip -y
conda activate ride_reactsim_dp
cd "$RIDE_REACTSIM_REPO/reactsim_data_pipeline"
python -m pip install -r requirements.txt
# 先审查 preprocess.sh 的 GPU、线程、输入范围、硬编码路径与输出规模
sed -n '1,240p' preprocess.sh
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V5_20261028
source scripts/00_env.sh
bash scripts/16_reactsim_run.sh preflight
# 仅在看过官方脚本并确认剩余磁盘/CPU/GPU后，显式允许耗时预处理
export RIDE_ALLOW_FULL_REACTSIM_PREPROCESS=1
bash scripts/16_reactsim_run.sh preprocess
```

**不要假定这个脚本对 val 仅处理少数日志，官方完整处理规模可能很大。** 保持独立环境，避免 CARLA Python 版本与 ReactSim Torch、nuPlan-devkit 冲突。

预处理完成之后，先检查自定义动作场景与作者预处理场景的**文件名覆盖**（不反序列化 pickle，不推断行为是否正确）：

```bash
python -m ride.reactsim_coverage \
  --behaviors "$RIDE_REACTSIM_BEHAVIORS" \
  --processed "$RIDE_REACTSIM_REPO/data/nuplan_processed/val/processed_scenarios" \
  --output "$RIDE_REACTSIM_WORK/reactsim_filename_coverage.json"
```

若出现大批 `missing_processed_count`，说明现有 DB/拆分或预处理策略并未覆盖所需 benchmark 场景；不要直接跳过缺失场景称作完整 ReactSim 评估。文件名相同也**不保证** DB token 与模型标签匹配，仍须作者原生样本级身份验证。


### 5.3 官方 SMART 或 MTR 推理（响应环境，不是 planner RQ3）

先确认 GPU 驱动 `nvidia-smi` 以及 PyTorch CUDA 12.1 wheel 的支持性。官方 SMART/CATK/TrajTok 的典型环境为 Python 3.11、PyTorch 2.5.1 CUDA 12.1：

```bash
conda create -n ride_reactsim_smart python=3.11 pip -y
conda activate ride_reactsim_smart
python -m pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' \
  'torchaudio==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-geometric==2.6.1
python -m pip install torch-scatter torch-cluster \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
cd "$RIDE_REACTSIM_REPO/SMART_CATK_TrajTok"
python -m pip install -r install/requirements.txt
python -m pip install --no-deps nuplan_devkit==2.0.0
python -m pip install -U huggingface_hub

cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V5_20261028
source scripts/00_env.sh
bash scripts/14_download_reactsim_weights.sh smart.ckpt
# 权重在 HF datasets 仓库；脚本已显式使用 --repo-type dataset。
# 若本地此前已有同名权重，下载脚本默认拒绝覆盖：先检查文件是否真实完整。
bash scripts/16_reactsim_run.sh preflight
bash scripts/16_reactsim_run.sh smart
```

MTR 使用独立 Python 3.9 / Torch 2.5.1 CUDA12.1 环境，运行其仓库 `MTR/requirements.txt`、`pip install --no-deps nuplan_devkit==2.0.0` 与 `python setup.py build_ext --inplace`，然后：

```bash
# 激活你安装好的 MTR 环境
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_RQ12_V5_20261028
source scripts/00_env.sh
bash scripts/14_download_reactsim_weights.sh mtr.pth
bash scripts/16_reactsim_run.sh mtr
```

上述评估命令按 ReactSim 作者官方 `docs/SMART_CATK_TrajTok.md` 和 `docs/MTR.md` 编写。**本地没有这些仓库及 CUDA/GPU，故尚未对用户本机模型做实测，也未能确认 checkpoint 与所有预处理输出完全匹配**。失败时先检查作者对应 commit 的依赖和接口而非修改模型预测以适配论文结论。

### 5.4 ReactSim 何时真正能支持 RQ1/RQ2

ReactSim 官方主要评估每个场景下一个 off-log AV 行为，且 2,636 个干预按纵向、方向、横向分组；**这些不天然构成同一初始状态、任务匹配的动作 A/B 对**。须先在同一原生场景构造或确认至少两个合法 AV 动作，确保相同角色 ID、相同历史、相同地图、相同动作执行、相同采样方案。SMART 和 MTR 作为两套响应仿真器，不能把它们各自的后验第三方碰撞次数直接当作真实决策错误。

v5 提供了 `python -m ride.reactsim_paired_audit` 对人工核验并标准化的 **4 组结果（2 actions × 2 models）** 做严格验收：拒绝缺失 arm、初始状态/固定第三方 ID 不一致、动作轨迹跨模型不同、任务进度不匹配和 ego 自身不合法。它**不自动解析作者原始 pickle**（没有仓库实物及 schema，不能编造兼容性）。见 `docs/REACTSIM_EXPORT_SCHEMA_CN.md`。

## 6. 下一步研究停止/转向规则

- **M0 仍不能得到合法、ego 安全并线**：这是场景/控制工程问题，不应据此否定 RIDE；先试不同 Town 和少量预先固定的 gap/speed 配置，不要重复扩充 response-bank。
- **M0 通过但 RQ1 0/足够样本，外部模型也无排序翻转**：RIDE 作为 *response-model-induced ranking failure* 的中心主张会受损；应报告无翻转的范围和不确定性，转向有更明确可证伪性的 *decision-critical recourse evidence coverage*。
- **只有 narrow/wide 的 UNKNOWN→witness 转变**：将重点切向部分可见反事实响应证据的可信性、顺序验证预算与三值决策，不把有限候选库未找到响应等同不可行性。
- **CARLA 有、ReactSim/WOMD 独立模型没有**：限定为模拟器反应假设敏感性，不上升到“现实人类交通系统性失败”。
- **RQ1/RQ2 成立但 hard courtesy/GameFormer-Plan 等强方法消除风险**：研究现象有现实意义，不能把 RIDE 误包装成这些方法未解决的 algorithmic gap，须进一步缩窄问题或停题。

真正新增算法之前要锁定：场景生成及排除表、初始状态归一化、执行与地图合法性、任务进展配对、明确受影响第三方 ID、行动前锁定的响应估计/选择、独立响应模型复核、负向控制、按 anchor 聚类的区间估计。
