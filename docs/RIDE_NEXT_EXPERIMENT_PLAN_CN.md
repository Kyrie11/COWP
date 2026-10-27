
# RIDE 当前论证状态与下一阶段实验计划

日期：2026-10-27

## 1. 结论

当前 evidence.zip 不是 RQ1/RQ2 的正向证据，而是一次工程筛查 checkpoint。正确的论文状态是：

- **RQ1：未得到支持。** 风险排序严格翻转为 0；而且这不是一个“证明没有翻转”的充分样本。
- **RQ2：未得到支持。** 任务进展匹配且第三方碰撞可由替代动作避免的配对为 0。
- **RQ3：尚未开始。** 没有锁定并运行强基线，不能声称现有方法失败，也不应开始新的规划算法设计。
- **四类场景：只有 convoy 的一小部分执行过；其余三类在第一帧前就因 CARLA 0.9.16 API 错误退出。**
- **WOMD/Waymax：只能作为数据管线和观测拟合检查，不能作为当前因果结论。**

当前最合适的研究名称仍是：

> **RIDE: Response-Model-Induced Decision Errors in Interactive Driving**

这个标题暂时作为工程验证阶段的工作标题。只有在 RQ1、RQ2 的正例经过独立反应环境复核，并且 RQ3 对强基线形成稳定剩余缺口后，才适合把它作为投稿标题。当前不要把 “RIDE 已经证明了强规划器的系统性错误” 写入论文。

## 2. 当前论文故事和科学问题

与已经上传并审查的 Courteous、FeAR、RACP、GameFormer、HYPE、RA-CBF、Studt/R3R、ReactSim-Bench、SMART-reactive 和交互安全规划工作相比，过宽的命题都不安全：

- “ego 会影响他车”已经是已有工作覆盖的现象；
- “他车可能不按规划器假设响应”与 reactive benchmark 和模型失配研究相邻；
- “预测误差不等于规划质量”已经有决策价值研究；
- “候选集中存在好轨迹但评价器选错”也不能单独作为创新点。

RIDE 应该只验证一个更窄、可证伪的联系：

> 在没有共同响应协议的异质交通中，规划器对其他车辆响应的有限或失准证据，是否会使它在任务进展近似相同的 ego 动作之间选错，从而增加某个可归因的第三方风险或负担？

因此三个研究问题应保持如下定义：

### RQ1：响应模型差异是否真的改变决策？

在相同初始状态、相同 ego 候选动作对、相同观察时域和相同任务匹配规则下，两个独立反应模型是否给出**严格相反的第三方风险排序**？

这里的“排序”必须是候选选择之前就能使用的风险评分或响应证据；不能只在 rollout 之后比较两条轨迹的结果。后验结果排名只能说明模拟器之间有差异，不能直接叫作 planner decision error。

### RQ2：风险是否由 ego 动作诱发且可避免？

对同一个场景和同一组受影响车辆，至少要同时满足：

1. action A 与 action B 的初始状态一致；
2. 两个动作真实执行且 ego 进展匹配；
3. A 在独立反应环境中诱发第三方碰撞、短 TTC 或显著急刹；
4. B 在同一环境、同一评测预算下避免该风险；
5. ego 自身没有通过碰撞或违规换取进展优势；
6. 该差异在多个 anchor、seed 和至少一个结构独立响应环境中复现。

RQ2 不是“某次仿真发生了碰撞”，而是“存在任务匹配的、由 ego 选择诱发的、可由替代动作避免的第三方风险”。

### RQ3：强基线为什么仍会选错？

只有在 RQ1/RQ2 的机制先成立后，才比较充分调参的：

- hard/soft courtesy；
- RACP 类 risk-aware contingency；
- ego-conditioned interaction prediction / HYPE / GameFormer-Planner；
- RA-CBF 等责任感知安全过滤器；
- 必要时将 R3R/Studt 作为假设条件敏感性实验，而不是伪装成相同输入条件的完整 planner。

RQ3 的公平条件是同一初始观测、同一地图和 route goal、同一 ego 候选池、同一反应证据预算、同一 rollout 时域和同一评测器。只要某个方法使用了额外通信、完整未来或其理论保证依赖的共同规则，就必须保留这些假设，不能先删除假设再声称其失败。

## 3. evidence.zip 的逐项结果

### 3.1 CARLA 总体统计

all_scenes_audit.json 给出：

| 项目 | 数值 | 含义 |
|---|---:|---|
| 总 trial | 81 | 四类场景合计 |
| 预注册配对 | 16 | 两个 action pair × 两个 response family × 四类场景 |
| 有效 task-matched 配对 | 2 | 仅 convoy 的 mild pair |
| 严格排序翻转 | 0 | 不能支持 RQ1 |
| cross-family matching collision effect | 0 | 不能支持 RQ2 |
| bootstrap cluster | 1 | 只有一个 anchor，CI=[0,0] 是退化重采样 |
| 强基线 | 未运行 | RQ3 未开始 |

73 个 trial 在第一次 trace 前失败，根因是：

~~~
AttributeError: 'Rotation' object has no attribute 'y'
ride/carla_scenarios.py:264
~~~

CARLA 0.9.16 的属性是 Rotation.yaw。因此：

- cut_in：8/8 失败；
- intersection：17/17 失败；
- support_gap：48/48 失败；
- convoy：8/8 跑通，只是因为 convoy 路径没有调用这个错误分支。

这意味着当前不能从 73 个 trial 推断“没有碰撞”或“没有排序翻转”。它们没有完成实验。

唯一的两个有效配对是 convoy 的 mild_early–mild_late：

- TM：进展差约 2.316 m，两条动作都无第三方碰撞，制动 proxy 和 TTC proxy 差异均为 0；
- BehaviorAgent：进展差约 2.967 m，两条动作都无第三方碰撞，制动 proxy 和 TTC proxy 差异均为 0。

这两组还显示两条动作都触发了 hard_brake=true。轨迹中的最小加速度常接近 -48.15 m/s²，且两动作都一样，疑似控制器瞬态、速度初始化或阈值定义造成的非区分性信号。它不能作为“第三方负担已被观测”的证据；重跑时要同时记录加速度、控制输入、车辆质量、速度更新以及阈值敏感性。

### 3.2 四类场景是否合理

#### Convoy

几何是可执行的直道三车队列，能测试后车制动、TTC 和第三方车辆对风险。但它是人为构造的 stress test，不是 WOMD replay；一个 anchor、一个 seed、一次 repetition 只能验证代码路径，不能支持普遍规律。

当前结论：**可保留为机制筛查场景，但 RQ1/RQ2 尚未通过。**

#### Cut-in

manifest 的两车道配置有交通语义：ego 在相邻车道，follower_a/follower_b 在目标车道，ego 试图进入 gap。但本轮所有 trial 在 waypoint 选择处失败，所以没有验证实际变道。

修复后必须检查：

- ego 的 lane ID 是否真实从相邻车道变为目标车道；
- 变道发生时间和横向位移；
- follower_a 是否真的对 ego 产生响应；
- 轨迹是否经过合法可行的 lane corridor；
- 第三方风险是否来自动作差异，而不是初始重叠或传感器误报。

#### Intersection

道路 ID 和坐标看起来像交叉流，但本轮同样没有真实执行。当前 traffic_light_legality_protocol 仍是未验证的 trigger-volume motion proxy，不能写成红灯合法性或违规因果证据。

修复后必须记录：

- ego 和 follower 是否进入同一冲突区域；
- 进入和离开 junction 的时间；
- traffic-light state 和 stop-line crossing；
- 通过 route arc 或明确道路路径计算 progress；
- 两个 response family 是否在同一初始状态下产生不同的候选风险。

#### Support-gap

这是当前最严重的场景设计错误。旧 manifest 的 support_gap 使用了与 convoy 完全相同的 ego/follower 坐标和 waypoint identity，因此它不是独立的 gap/merge 场景。其 finite-bank ablation 也报告：

- comparison_valid=0；
- all_candidate_executions_recorded=false；
- small/wide no-witness 状态为 UNKNOWN；
- 没有 finite-bank miss；
- 没有 global UNSAT 证明。

我已在工程中把 support_gap 改为真实的相邻车道并线 gap 构造，并把 anchor protocol 升级为 ride_lane_corridor_v3_20261027。旧 support_gap anchors 必须全部作废、重新冻结。有限响应搜索的缺失候选永远只能标为 UNKNOWN。

## 4. WOMD/Waymax 结果如何解释

### 4.1 Observational fit

observational_response_fit.json：

- attempted 320，eligible 310；
- 1 s ADE：constant velocity 0.154 m，cautious 0.294 m，assertive 0.336 m，heterogeneous 0.302 m，TTC-gap 0.163 m；
- constant velocity 在 cautious/assertive/heterogeneous 上分别有 282/310、290/310、288/310 个场景更好；
- 文件自身明确写出：conditioning 是 logged ego future，fit quality 不能建立 human counterfactual validity。

这有一个重要价值：它说明不能把某个 IDM 参数或 TTC rule 仅因为名字更“交互”就当作真实反应模型。它只能用于：

- 检查记录格式和观测拟合管线；
- 估计不同简单模型的 observational residual；
- 预注册响应模型选择和不确定性范围。

它不能支持 do(ego action) 的因果结论。

### 4.2 Native Waymax

native_panel.json：

- attempted 22；
- complete 17；
- missing-file errors 5；
- all-regime matched 0；
- scientific_claim_allowed=false。

现有实现中 IDMRoutePolicy 跟随 logged spatial route，非车辆是 logged expert，progress 使用 initial-heading projection proxy，selected road valid 多数为 false，且没有复现 GameFormer 或 RA-CBF。因此即使某些 cautious/assertive 行出现碰撞差异，也不能当作 RQ1/RQ2 证据。

重跑 Waymax 前先修复 5 个 missing cohort 文件、明确 off-road metric 的 agent index 语义，再把 progress 换成 route arc。否则只把 Waymax 当作接口 smoke test。

## 5. 已实现的工程修改

交付的下一版工程包含以下修改：

1. 将 CARLA 的 Rotation.y 修正为 Rotation.yaw；
2. 添加回归测试，确保 CARLA 0.9.16 waypoint 选择不会再次触发该错误；
3. 将 support_gap 从 convoy alias 改为真实相邻车道 gap/merge 构造；
4. support_gap 的 ego 控制和 lane-validity 检查与 cut-in 一致；
5. 对无 trace 的 trial 保留 simulation_exception 原因，不再把 API 错误压成模糊的 empty_trace；
6. smoke gate 新增 --require-complete，要求每个 trial 至少有一个 trace 且无 simulation exception；finite-bank 的碰撞/越界结果仍可作为已记录结果而不是缺失；
7. 新增 ride.research_status 与 scripts/12_research_status.sh，自动汇总 RQ1/RQ2/RQ3、WOMD、Waymax 和 finite-bank 状态。

由于当前执行环境没有 pytest，已完成：

- python -m compileall -q ride；
- RIDE CLI help smoke；
- yaw 修复的 fake-CARLA waypoint smoke；
- support_gap 构造 smoke；
- 当前 evidence 的 status 汇总。

你在服务器的 conda 环境中应再运行完整 pytest；若环境未安装 pytest，先在工程环境中安装，不要把“未运行测试”写成通过。

## 6. 下一步指令

以下命令都应在新工程目录执行。不要覆盖旧 evidence。

### 6.1 部署并检查环境

~~~
unzip RIDE_next_gate_20261027.zip -d /data0/senzeyu2/dataset/RIDE/
cd /data0/senzeyu2/dataset/RIDE/RIDE_next_gate_20261027

conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export DATASET_ROOT=/data0/senzeyu2/dataset
export RAW_WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
source scripts/00_env.sh

python -c "import carla; from agents.navigation.behavior_agent import BehaviorAgent; print(carla.__file__); print('agents OK')"
python -m pytest -q
bash scripts/01_doctor.sh
~~~

如果 agents 导入失败，先看：

~~~
ls -l "$CARLA_ROOT/PythonAPI/carla/agents/navigation/behavior_agent.py"
python -c "import sys, importlib.util; print(sys.executable); print(importlib.util.find_spec('agents')); print(importlib.util.find_spec('carla'))"
~~~

### 6.2 启动 CARLA 0.9.16 无头服务器

终端 A：

~~~
conda activate mfrp
cd /data0/senzeyu2/dataset/RIDE/RIDE_next_gate_20261027
source scripts/00_env.sh
bash scripts/02_start_carla_headless.sh
~~~

终端 B：

~~~
cd /data0/senzeyu2/dataset/RIDE/RIDE_next_gate_20261027
source scripts/00_env.sh
python -m ride.doctor --server
~~~

必须同时看到 server/client 都是 0.9.16。不要把 0.9.15 的旧 PYTHONPATH 混入当前环境。

### 6.3 先做最小 smoke 和锚点检查

~~~
export CARLA_TOWN=Town04
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_fix_smoke_20261027
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='mild_early mild_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_ALL=1

source scripts/00_env.sh

for case in convoy cut_in intersection support_gap; do
  bash scripts/03_precommit_scenes.sh "$case"
done
~~~

先读每个冻结 manifest：

~~~
for f in "$RIDE_OUTPUT_ROOT"/frozen/*_anchors.jsonl; do
  echo "=== $f"
  head -n 1 "$f"
done
~~~

support_gap 必须显示 ego 与 follower_a 的 lane ID 不同；不能再出现和 convoy 相同的坐标。若某个场景没有合格 anchor，记录为地图模板不适用，换 Town03/Town05，不能手工改坐标。

然后运行最小 outcomes：

~~~
for case in convoy cut_in intersection; do
  export RIDE_SMOKE_REQUIRE_ALL=1
  bash scripts/03_run_carla_case.sh "$case"
done
# support_gap deliberately contains finite-bank candidates that may be
# recorded as collision/off-road outcomes; require complete traces, not all
# candidates to satisfy the normal validity predicate.
export RIDE_SMOKE_REQUIRE_ALL=0
bash scripts/03_run_carla_case.sh support_gap
bash scripts/03_support_ablation.sh
bash scripts/04_analyze_carla.sh
bash scripts/12_research_status.sh
~~~

### 6.4 发现性重跑规模

最小发现性 panel 建议：

- 每类至少 10 个独立 anchor；
- 至少 5 个 seed；
- 每个 anchor/seed 至少 3 个 repetition；
- 保留所有 invalid、missing、off-road 和 red-light proxy 记录；
- 每个 action pair 预先固定，不做事后挑选；
- 首先使用 actor reset 跑通，再在稳定服务器用 reload 做复核。

可以先用：

~~~
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_discovery_20261027
export RIDE_ANCHORS=10
export RIDE_REPEATS=15
export RIDE_SEED=17
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_ALL=0
source scripts/00_env.sh

for case in convoy cut_in intersection support_gap; do
  bash scripts/03_precommit_scenes.sh "$case"
  bash scripts/03_run_carla_case.sh "$case"
done
bash scripts/03_support_ablation.sh
bash scripts/04_analyze_carla.sh
bash scripts/12_research_status.sh
~~~

RIDE_REPEATS=15 会让 seed 取 17,18,...,31，先用于发现性筛查。正式统计时应把 anchor 作为主要 cluster，并把 seed/repetition 作为独立重复；不要用单 anchor bootstrap。

如果 CARLA actor reset 的隐藏状态影响结果，再新建另一目录用：

~~~
export RIDE_WORLD_RESET=reload
~~~

不要把 actor-reset 与 reload 的结果直接合并成同一实验；分别报告 reset protocol。

### 6.5 WOMD 开发队列

当前路径中 validation 和 validation_interactive 各有 150 个 TFRecord shard，training/testing 没有被 doctor 发现。现在不需要下载 training/testing。

先运行：

~~~
conda activate mfrp
cd /data0/senzeyu2/dataset/RIDE/RIDE_next_gate_20261027
source scripts/00_env.sh
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/womd_dev_20261027
bash scripts/09_native_womd_cohort.sh dev
bash scripts/11_observational_response_calibration.sh
~~~

只有在 dev cohort 的 schema、ID、heading、raw shard index 全部通过后，再执行一次：

~~~
bash scripts/09_native_womd_cohort.sh full150
~~~

将 validation_interactive 用于独立的 interaction 场景发现或交互模型测试；不要把反复调参的 validation_interactive 当最终 holdout。WOMD 日志本身仍不能识别 do(ego action) 的人类反应。

### 6.6 ReactSim-Bench 和外部基线

当前脚本下载的只是 ReactSim 的 custom behavior 文件，不是完整闭环数据。先不要为了 RQ3 立刻下载全量 nuPlan。

必要时：

~~~
source scripts/00_env.sh
bash scripts/07_download_reactsim_data.sh
bash scripts/08_clone_external_baselines.sh
~~~

在 RQ1/RQ2 通过前，只做环境和接口 smoke：

1. ReactSim-Bench：获取其 nuPlan mini/dev state + map、scene index 和作者评测脚本，先回放同一批 ego candidate；
2. Bosch SMART-reactive：确认 checkpoint、nuPlan 版本和 rollout API，先作为独立 reactive evaluator；
3. GameFormer-Planner：如果要作为强 planner，使用官方支持的 nuPlan 预处理和 checkpoint，导出 ex-ante candidate choice；
4. RA-CBF：作为责任感知 filter 近邻，报告后处理前后第三方风险与 ego progress；
5. R3R/Studt：只在信息、通信和共同规则假设明确一致时做敏感性实验。

每个外部方法都要记录 git commit、checkpoint hash、预处理版本、可见输入、候选预算和失败原因。不能把“仓库已 clone”写成 baseline 已运行。

## 7. RQ1/RQ2 通过标准和 RQ3 开始条件

### RQ1 通过标准

必须在冻结的 dev/holdout 上同时满足：

- 至少 20 个独立 scene-anchor；
- 每个有效 scene 有两个 action、两个 response family；
- 严格相反风险排序不是 tie；
- 排序定义预注册；
- 进展差和初始状态均在阈值内；
- 至少两个结构独立反应环境出现同方向结果；
- cluster bootstrap 或 paired permutation 的 95% 区间不包含零，或报告明确的效应量和不确定性。

若只出现“模型输出不同”而动作排序不翻转，RQ1 不通过。

### RQ2 通过标准

至少需要：

- 10 个以上独立、任务匹配的正例；
- A/B 的 ego progress 差在预注册阈值内；
- A 的第三方风险显著高于 B；
- ego collision、规则违规和初始重叠排除；
- 至少一个真实记录校准的响应模型和一个独立 reactive environment 复现；
- 不是只靠 hard-brake 阈值一个 proxy；
- 报告第三方碰撞、最小 TTC、减速度、jerk、延误和 route completion 的分项结果。

如果只有一类 CARLA 控制器支持，结论应写成 simulator sensitivity，而不是混合交通机制。

### 进入 RQ3 的门槛

只有以下条件全部满足，才开始强基线和新算法：

1. RQ1 有稳定 candidate-choice reversal；
2. RQ2 有可避免的 third-party risk；
3. support_gap 的 finite bank 有完整记录，所有 no-witness 只标 UNKNOWN；
4. WOMD/ReactSim/SMART 中至少一个独立响应环境支持同类机制；
5. hard courtesy、risk-aware contingency 和至少一个 ego-conditioned planner 已在相同候选池上公平运行；
6. 选择结果不是通过停车或极端保守策略获得。

若 RQ1/RQ2 失败：

- 若风险只在 CARLA 出现，转为 simulator/model sensitivity 论文，不继续包装 RIDE；
- 若响应库有限性是唯一稳定现象，转向 “Unknown-aware behavioral recourse evidence”；
- 若现象在强基线中已被解决，放弃 RIDE 主线，保留场景和审计工程作为后续工作。

## 8. 当前交付物

- RIDE_next_gate_20261027.zip
- evidence_analysis.md
- data_baseline_plan.md
- current_research_status.json

