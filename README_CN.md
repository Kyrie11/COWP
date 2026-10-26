# RIDE 研究问题证据审计工程（2026-10-26）

**暂定论文标题**：**RIDE: Response-Model-Induced Decision Errors in Interactive Driving**

**中文研究问题**：其他交通参与者的响应模型、联合可行性或有限 recourse 证据存在偏差时，ego 规划是否会在一组任务表现相近的候选动作中选错，从而对特定第三方增加可避免的碰撞风险或不合理负担？如何将这种 *decision-relevant* 响应依赖与普通预测误差、合法协作、保守停车区分开？

## 重要边界与执行状态

- 本包不是最终投稿算法；**未在作者 CARLA 0.9.16 服务器、原生 WOMD、ReactSim-Bench、nuPlan 上运行过端到端物理实验**。
- 包中 `ride/` 是全新、独立的证据审计层。`COWP/` 是所上传 COWP 代码的保留副本，CARLA 研究入口做了 0.9.16 适配和四类场景实验性扩展，保留 COWP 的训练、自然 roots、Waymax、旧审计工具。旧脚本不得因其文件名而被认为已验证。
- 四类 CARLA 代码是 **candidate scenario constructors**，不是四个已发生的真实交通失败。无法在某地图找到符合条件的 anchor 时明确失败；仿真未完成、动作进展不匹配和初始碰撞不会计作负例。
- Traffic Manager 与 BehaviorAgent 是两套软件策略，但使用相同 CARLA 物理与地图，既不等同于两种经真实人类观测校准的独立模型，也不提供人类 do-effect 证据。
- `ride.decision` 只接受在实验之前提交的 planner choices，严格禁止使用事后观测 outcome 来冒充 GameFormer/RACP/COWP 正式基线。
- 后续算法设计前必须先有真实实验正例；如果 RQ3 只使用简单 constant-velocity 或原生 COWP 内部阈值，不得声称强算法系统失败。

## 0. 解压/准备

```bash
mkdir -p /home/senzeyu2/code
unzip RIDE_RESEARCH_GATE_20261026.zip -d /home/senzeyu2/code/
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
```

以下命令均在此根目录运行，不能从任意目录直接执行 `python -m research_probe.*` 而未设置 PYTHONPATH。

## 1. 创建单独 CARLA Python 环境

CARLA Python 0.9.16 官方轮子支持 CPython 3.10/3.11/3.12（请根据系统 glibc 选择）；不要直接安装进正在使用的 Waymax JAX 环境。以下用 Python 3.10：

```bash
conda create -n ride-carla python=3.10 -y
conda activate ride-carla
python -m pip install --upgrade pip
python -m pip install carla==0.9.16 pytest numpy
export CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
bash scripts/01_doctor.sh
python -c 'from agents.navigation.behavior_agent import BehaviorAgent; print("agents OK")'
```

**如果提示 `No module named agents`**：检查 `$CARLA_ROOT/PythonAPI/carla/agents/navigation/behavior_agent.py` 是否真的存在。如果文件存在，`source scripts/00_env.sh` 后 `python -m ride.doctor` 会列出 Python 路径及导入位置。如果文件缺失，安装 `carla` wheel 本身不会补上 BehaviorAgent；从对应 0.9.16 完整发布/源码获取 `PythonAPI/carla/agents/`，不要用不同版本混装。**如果服务器显示 0.9.15，不要绕过检查。**

## 2. CARLA 0.9.16 无头运行

在独立的 Terminal/TMUX 窗口启动服务端，必须让它在客户端实验期间保持运行：

```bash
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
source scripts/00_env.sh
bash scripts/02_start_carla_headless.sh
```

上述服务端使用 `-RenderOffScreen`。它与 Python 环境设置的 `--no-rendering` 不同：前者禁用窗口、仍允许 GPU 渲染；后者是物理场景测试专用的无渲染模式。当前实验只用 CARLA 位置与碰撞传感器，不依赖 RGB 相机。服务端需具备足够的 GPU/内存与仿真资源。

在另一个终端：

```bash
conda activate ride-carla
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
source scripts/00_env.sh
python -m ride.doctor --server
```

确认服务端与客户端均以 `0.9.16` 开头后，先进行最简单的因果配对链路验证：

```bash
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
export CARLA_TOWN=Town04
# 首先冻结地图场景 anchor，在任何 response rollout 之前执行
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
```

注意：脚本默认存放到 `/data0/senzeyu2/dataset/COWP/ride_evidence_20261026/`。要改变输出位置，设置 `export RIDE_OUTPUT_ROOT=...`。

## 3. 运行四类实验性失效机制筛查

```bash
# S1: 并线/抢占间隙；相同目标速度，不同横向插入时间
bash scripts/03_run_carla_case.sh cut_in

# S2: 三车跟驰制动传播；同积分目标速度、实际路程仍需审核
bash scripts/03_run_carla_case.sh convoy

# S3: 路口多车不同决策；joint witness (有限可控响应) vs 独立响应
# 若 Town04 找不到有效垂直路口，先检查输出，再尝试 Town03/Town05。
bash scripts/03_run_carla_case.sh intersection

# S4: 有限控制候选覆盖；joint grid / backup 支持审计，空集仅表示 UNKNOWN
bash scripts/03_run_carla_case.sh support_gap

# 已成功完成的案例聚合、固定配对分析和科学决策门
bash scripts/04_analyze_carla.sh
```

所有案例的逐试验 JSONL、运行异常、零样本/跳过原因与分析 JSON 都保留；**不要把不具备 4 类完整场景的汇总写成全部通过**。

优先检查：

```bash
cat /data0/senzeyu2/dataset/COWP/ride_evidence_20261026/carla/science_gates.json
cat /data0/senzeyu2/dataset/COWP/ride_evidence_20261026/carla/all_scenes_audit.json
```

要提高研究规模而不是先拟合稀有失败：

```bash
export RIDE_ANCHORS=3
export RIDE_REPEATS=3
bash scripts/03_run_carla_case.sh convoy
bash scripts/03_run_carla_case.sh cut_in
bash scripts/03_run_carla_case.sh intersection
bash scripts/03_run_carla_case.sh support_gap
# 额外诊断：在严格嵌套的窄/宽 responder control bank 中寻找正向 witness
bash scripts/03_support_ablation.sh
bash scripts/04_analyze_carla.sh
```

**独立建模注意**：初始物理重叠使用旋转矩形（OBB）而不是外接圆；各场景轨迹仍须检查真实动作是否触发有效交互。`risk_ranking_disagreement.json` 是基于后验物理评测得到的场景内排序分歧，**绝不能当成某个规划器实际选错动作的证据**。

**实验学约束**：`early_slow` 对 `late_slow` 的有符号差异是事先定义的。若从结果中先挑选哪条是坏动作再统计，就会造成 winner's curse。显著性检验以场景 anchor（或 WOMD 原始 segment）为 cluster，不将一个场景中 20 条动作当作 20 个独立交通事件。

## 4. Waymax / WOMD gate（继续使用原 COWP 工程）

Waymax 推荐已有的独立 JAX 环境。不要在 CARLA 环境直接改动 JAX 版本。

```bash
conda activate <your-existing-waymax-env>
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
source scripts/00_env.sh
bash scripts/05_waymax.sh waymax_inspect
bash scripts/05_waymax.sh waymax_smoke
# ONLY if smoke completes, SDC identity/map audit passes:
bash scripts/05_waymax.sh waymax_priority22
```

旧 V4 Waymax 代码是你上传的来源；本工程保留了它的运行入口，并没有在本地服务器验证完 22 个案例。若仍出现 `No module named agents`，必须保留完整 traceback，查看是不是运行错误的命令/环境，因为 `waymax_paired_idm` 本身不是 CARLA BehaviorAgent 入口。

如果原来 `/data0/.../eicdrr_full_validation_20261014` 的 22-ID 开发 cohort **并不存在**，先通过下方第 6.5 节 `scripts/09_native_womd_cohort.sh dev` 从原始 WOMD 构建新盲样本，再使用可直接从该样本冻结 ID 的入口：

```bash
# 运行前确保已进入含 tensorflow、jax、waymax 的 Waymax 环境
source scripts/00_env.sh
bash scripts/05b_waymax_native.sh smoke
# 仅在 smoke 的 complete / map / identity 检查通过后：
bash scripts/05b_waymax_native.sh panel
```

`05b` 会冻结最多 22 个 ID，**不是**历史脚本中的 `waymax_priority22` 原始 22 个研究案例。它是另一个 outcome-blind native WOMD pilot，不能将两者结果混称。

不要仅用 WOMD logged playback 推断真实他车对新的 ego 动作会怎样反应；Waymax IDM 仍然是仿真行为假设。

## 5. 真正验证 RQ3：锁定 baseline 的候选选择，再运行反事实参考

自己将 RACP / HYPE / GameFormer（或作者 checkpoint 的可运行适配版本）的候选选择冻结为 CSV，例如：

```csv
method_id,archetype,anchor,seed,response_family,selected_action,choice_locked_before_rollout
RACP_ADAPTED,convoy,0,17,tm,early_slow,true
HARD_COURTESY,convoy,0,17,tm,late_slow,true
```

**这是 CSV 模式示例，不是任何真实 baseline 已经选出的结果。**`method_id` 必须标明 `ADAPTED` 或作者官方版本。真正的实验需同时保存冻结模型 checkpoint hash、选动作时使用的信息、候选集 ID、推理时延与提交时间，且不能依据 holdout 后验结果调整阈值。

```bash
# 强基线需先冻结 choices CSV，再冻结哈希，并在独立反应 rollout 之前完成
python -m ride.commit --choices /path/to/choices_locked.csv \
  --output /path/to/choices_locked.sha.json
# 只有收集完独立执行结果之后才能运行下面的离线审计
bash scripts/06_compare_locked_planners.sh /path/to/choices_locked.csv /path/to/choices_locked.sha.json
```

如果作者完整基线尚未运行，可以先对 **外部预测分数**实验性测试三种简单、透明的评分器：`safety_only`、`soft_courtesy`、`hard_courtesy`。它们 **不是 RACP、GameFormer 或 HYPE 的官方实现**，亦不能用于宣称强算法系统性失败。首先用自己的 *ex-ante* 预测器填充 `inputs/predictions_SCHEMA_ONLY.csv` 的同名字段（不能用事后 CARLA 观测）；在任何结果 rollout 前锁定：

```bash
bash scripts/06a_lock_proxy_selectors.sh /path/to/predictions.csv hard_courtesy
bash scripts/06a_lock_proxy_selectors.sh /path/to/predictions.csv soft_courtesy
# 每个策略会生成独立 choices.csv 和 choices.sha.json
```

`ride.decision` 计算 **post-hoc evaluation regret**（只用于评估，不是用于在线选择的神谕）。单场景若 ego 本身碰撞、违反规则或无匹配备选，应明确报告，而不是悄悄移出基线分母。

## 6. 数据下载：先后顺序

### P0：WOMD v1.3.1（当前已在用；先检查、只补缺分片）

```bash
source scripts/00_env.sh
bash scripts/01_doctor.sh
# 官方 bucket 自查
# 请按 Waymo Open Dataset 使用条款及当前 GCS 权限操作
gcloud storage ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/
gcloud storage ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/
```

如相应 split 缺失，**在完成权限确认后**复制官方数据：

```bash
mkdir -p "$RAW_WOMD_ROOT/validation" "$RAW_WOMD_ROOT/validation_interactive"
gcloud storage cp 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/*' "$RAW_WOMD_ROOT/validation/"
gcloud storage cp 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/*' "$RAW_WOMD_ROOT/validation_interactive/"
```

上传历史中的 `validation_interactive` 150 个分片有文件名/大小检查，不保证完整 TFExample 解码。使用 `reference_manifests/compact_all_7200_ids.txt` 对 **官方验证划分里新选中的 scene IDs** 排除历史训练和开发缓存；还要按原始 segment/window 去重。

### P1：ReactSim-Bench 的作者候选动作（仅 28 MB；推荐先下载）

```bash
conda activate ride-carla
python -m pip install 'huggingface_hub>=0.27'
source scripts/00_env.sh
bash scripts/07_download_reactsim_data.sh
```

数据实际来源：`zzzy0729/ReactSim-Bench-Data`。它**只包含作者的 off-log ego behaviors，不包含 nuPlan 原始车辆状态、地图和车辆响应**，不能只下载这 28 MB 就声称完成 ReactSim-Bench。

### P1.5：源码基线（只下载源码、独立安装依赖）

```bash
bash scripts/08_clone_external_baselines.sh
```

包括 GameFormer、RA-CBF、Bosch SMART-reactive、ReactSim-Bench、R3R 的官方 GitHub 仓库。请逐个核对作者源代码、checkpoint、原始运行环境与任务接口；这个脚本只下载，**不等于已成功复现强基线**。

### P2：nuPlan 官方数据（需要时再下载）

官方 AWS bucket 可无签名列目录：

```bash
aws s3 ls --no-sign-request s3://motional-nuplan/
```

依据所列目录及你取得许可的 nuPlan 数据版本选取 **maps + mini / relevant DB subsets**。先看实际路径，再执行针对性 `aws s3 sync --no-sign-request s3://motional-nuplan/<official-existing-prefix>/ "$DATASET_ROOT/nuPlan/<matching-subdir>/"`。这里 `<official-existing-prefix>` 是占位符，必须用刚列出的实际目录替换；**不要直接整桶下载，也不要默认原仓库的任何路径都可匿名获取**。完全复用 Bosch SMART-reactive 或 ReactSim 官方基线前，需要单独部署它们的环境、nuPlan devkit、训练权重及按项目说明进行预处理。

## 6.5 原生 WOMD 开发集、完整 150 分片及行为模型观测拟合

为避免 Compact-5k 的 teacher/标签同源风险，保留独立的 WOMD 选择盲采样流程。`native_womd.py` 根据当前/历史的状态和地图交互属性筛选；未来真值仅用于**事后观测评估**，不能用于 ego 反事实动作选择。此操作需要独立 TensorFlow 环境：

```bash
conda create -n ride-womd python=3.10 -y
conda activate ride-womd
python -m pip install 'numpy<2' 'tensorflow-cpu==2.16.2' pytest
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
source scripts/00_env.sh

# 首先仅扫描最多四个分片做链路验证，不代表总体概率
bash scripts/09_native_womd_cohort.sh dev

# 观测响应拟合：IDM 家族与常速度基线的 1s/2s ADE 对照
bash scripts/11_observational_response_calibration.sh

# 只有确认官方 validation 150 个分片全部安装后才运行
# full150 会拒绝不完整分片，并严格使用不封顶的全记录扫描
bash scripts/09_native_womd_cohort.sh full150
```

**注意：** full150 的盲样本包括开发与 ID 哈希分离的 holdout，但仅靠文件划分不能防止研究者反复查看结果造成污染；应在后续算法开发前冻结评测协议，并避免接触 holdout 的未来诊断。

## 7. 单元测试：验证代码，不验证科学结论

```bash
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
python -m pytest -q tests
# 附带 COWP 旧版已知范围内单测
cd COWP
python -m pytest -q tests/test_carla_h2_protocol.py tests/test_waymax_diagnostics.py
```

为了验证审计公式，可用**明确标记合成**的临时数据：

```bash
cd /home/senzeyu2/code/RIDE_RESEARCH_GATE_20261026
python -m ride.synthetic_fixture --output /tmp/ride_SNAPSHOT_SYNTHETIC.jsonl
python -m ride.evidence --input /tmp/ride_SNAPSHOT_SYNTHETIC.jsonl --output /tmp/ride_SNAPSHOT_SYNTHETIC_audit.json
```

**绝对不可将合成 fixture 的结果作为交通失效存在性的论文结果。**

## 8. 提交下一轮科学审查所需的最小文件

- `diagnostics/doctor.log`、`diagnostics/datasets.json`；
- 各场景 `.log` 与 `.jsonl`（含失败 trial）；
- `carla/all_scenes_audit.json`、`carla/risk_ranking_disagreement.json`、`carla/science_gates.json`；
- 已成功执行时 `waymax/waymax_smoke_1.json` 和 `waymax/waymax_priority22.json`；
- 如果运行了真实基线，附上 `choices_locked.csv`、对应 checkpoint hash、`planner_decision_regret.json`。

原始数据无需整包上传；保留 scenario ID、划分、候选和 seed 足够定位。


## 9. 强制科学边界、快速执行顺序与文件打包

实际用户服务器上建议按此顺序执行：

1. `01_doctor.sh` / `02_start_carla_headless.sh`，**先通过实际版本与 TCP 连接**。
2. `03_precommit_scenes.sh` 四类场景，冻结几何选择；单场景 smoke 后执行 `03_run_carla_case.sh`。
3. `03_support_ablation.sh` 和 `04_analyze_carla.sh`，其中 ranking flip 不等于实际 planner 的选动作错误；原始负面结果也必须保留。
4. `09_native_womd_cohort.sh dev`、`11_observational_response_calibration.sh`。不能把模型拟合写成已识别真实 do-effect。
5. **在没有读独立 reaction outcomes 之前**收集强基线的 candidates + scores + chosen action，并锁定 CSV/checkpoint/commit；之后独立 rollout、`06_compare_locked_planners.sh`。
6. RQ2 在至少两种结构独立且观测拟合经过检查的 reactive 环境中复核之前，**不要**将 CARLA 结论当成人类交通系统性失败。

可选打包供复审：

```bash
bash scripts/10_bundle_run_outputs.sh /tmp/RIDE_RESULTS_FOR_REVIEW.tar.gz
```

该脚本不包含原始 WOMD/nuPlan 数据，也不包含所有 NPZ 轨迹；需要定位具体案例时应提供原始场景 ID 与验证性切片数据。

**正式投稿标题尚未冻结。** 是否达到 CCF-A 潜力，不由工程代码测试通过率决定，而由独立可避免第三方风险、强基线的公平失败、方法新颖性与实际规划改善共同决定。与 PDT/iDriveVLA/INTERACT/SIGMA 重叠风险的进一步分析见 `docs/RESEARCH_GAP_CN.md`。
