# RIDE v16：第三方反事实响应差分与规划选择失效——独立取证、论文定位和下一步命令

日期：2026-10-29。**证据作用域**：用户提供的四份 v15 结果和前一轮的 `v14_stress_base_seed171` 原始 CARLA JSONL。这里的数值由原始轨迹重算，不能代替真实道路的反事实证据。新代码通过本地离线测试，没有在用户的 CARLA 0.9.16 服务器上实际执行。

## 1. 科学结论（可以／不可以写进论文）

可写成目前的 CARLA 诊断发现：

1. 延迟 C 的制动控制命令 0.3 s，能够在固定两条 CARLA 物理道路内显著改变其累计制动负担：有效动作中 9/11 条相对同一 cell 的正常 C BehaviorAgent 产生至少 0.25 m/s 的 C 负担绝对差。0 秒延迟的 baseline 只有 0/10 条。`cap=0.85` 并未有效限幅，最大正常 C BehaviorAgent 制动命令仅约 0.30；该 cell 的 0/10 条有效动作负担差超 0.25 m/s。
2. 即使物理路段与 seed 相同，无作用干预（0 延迟且 cap>=正常命令）下，同控制器多次运行也并非完全相同。这是重要噪声对照：无干预 baseline 最大逐动作 C 负担差约 0.114 m/s；cap-only 最大约 0.063 m/s。这些比较不是同一物理独立样本分布的估计。
3. 2026-10-29 的 v15 Seed 171 三组结果各 42 条完整 trial、42 条 valid trial，36 条实际完成并线；B–C 物理碰撞为 0、任何 TTC<1.5 s 为 0、原定义的 C 持续急刹（>=0.2s）为 0，B–C 最小 TTC 在三组中均约 6.15 s 以上。
4. v15 Seed 173 前瞻性选择实验有 18 条审计记录（两套透明选择策略重复）。6 条可比较记录只对应 3 对独立动作，且仅 1 对（Anchor 1 的 mild_early/mild_late）在受控延迟 C 目标条件下显示可避免的 C 负担选择损失（约 0.403 m/s），没有真正 B–C 碰撞和严格模型诱发错误。该实验人为改变 C 响应控制，不能充当与人类行为模型相关的独立响应真值。
5. 主要新增科学观察：单个动作的 C 负担大幅变化 **不必导致** 两个任务匹配候选之间的负担差分发生同等变化。对 Seed 171 完整响应模型的两种 ego 候选配对进行严格比较：延迟-only cell 5 个跨 family 合格动作对，未出现“源模型差分<=0.05、目标差分>=0.25”的虚假平局。延迟+cap 的 6 个合格动作对同样没有虚假平局，其中存在一条事后选择后悔代理（需避免事后挑结果）。因此论文应强调 action contrast，而非 absolute trajectory error 或单动作 burden。

**目前禁止声称**：发生了 B→C 因果中介、严重近失或链式碰撞，C 制动能力被 cap=0.85 削弱，GameFormer/RACP/HYPE/强规划器已失败，问题已被真实驾驶观测反事实支持，或这个小规模 CARLA panel 代表自然交通发生率。

## 2. 我建议的暂定论文题目及科学问题

**RIDE: Decision-Critical Third-Party Response Contrasts in Interactive Driving**（RIDE 缩写沿用工程名，仅作为研究阶段标题）。

上层问题：在没有共享响应协议的异质交通中，**响应证据对 ego 反事实候选所诱发第三方后果差分的错误估计，是否造成可避免的规划选择错误；这种失配在何种第三方响应能力与车流拓扑下会越过安全边界？**

令 `L_C(a;M)` 为模型 `M` 在 ego 动作 `a` 下的 C 后果，定义

`D_M(a,b)=L_C(a;M)-L_C(b;M)`。真实科学目标不是 `L_C(a;M)≈L_C(a;E)`，而是 `D_M(a,b)` 可靠支撑 `a` 与 `b` 的决策；与此同时必须保留绝对危险预算，不允许以相对排名掩盖两个动作都危险的场景。

应分层验证以下因果和决策环节：

- **M0 控制有效性**：实际施加的 C 原始制动请求、延迟命令、实际施加命令、B 和 ego 轨迹与初态匹配；所有未知（UNKNOWN）、重置噪声和观测异常不得删去。
- **RQ1 反事实差分可靠性**：`|D_M|` 很小或顺序错误，而 `|D_E|` 超出同模型无作用对照噪声，影响预注册风险／负担决策；区别稳定可重复的差分压缩和模型共同偏差。
- **RQ2 第三方可避免后果**：来源模型事前选择的 ego 动作在目标策略下造成 C 负担、持续急刹或更高安全风险，存在合法且进展匹配的低损失替代；在宣称 B→C 传播前必须额外进行受控 B 中介干预，不能仅根据时间顺序。
- **RQ3 现有方法剩余缺陷**：至少公平比较 B-only courtesy、B+C hard courtesy、multi-response risk selector，以及适配正确官方实现的交互规划强基线；不可以因为一种自造选择器失败就断言官方方法失败。

## 3. 现有方法的 research gap 与风险

RACP 已考虑多模态策略信念及概率风险，Courteous 已考虑他车代价；HYPE、INTERACT 已使用 ego-conditioned interaction prediction；PDT 已研究从预测评分到规划选择的错误；SIGMA 已研究 decision-critical uncertainty。ReactSim-Bench 已系统评估反应式模型的 agent-agent 碰撞，When Planners Meet Reality 已比较不同 reactive traffic 下 14 种规划方法。因此：

**不新颖**：ego 影响其他车辆；他车可能制动；预测误差使规划变差；使用多响应模型；寻找 UNKNOWN；单纯扩大 CARLA 场景密度。

**值得论证的精确剩余命题**：多模态／社交／风险规划在整体结果合理时，是否仍会将两个任务等效 ego 动作对固定下游车辆链条的不同后果压缩为不可区分的证据，导致 **对特定第三方**的可避免选择错误；并且如果加入同等模型容量或 B+C hard courtesy 仍无法修复，是否需要有条件地验证反事实动作差分的证据机制。这里的研究空白还没有通过强基线验证。

参考文献：
- [Courteous Autonomous Cars](https://arxiv.org/abs/1808.02633)
- [RACP](https://arxiv.org/abs/2402.17387)
- [When Planners Meet Reality](https://arxiv.org/abs/2510.14677)
- [ReactSim-Bench](https://arxiv.org/abs/2606.14058)
- [INTERACT](https://arxiv.org/abs/2609.31137)
- [PDT](https://arxiv.org/abs/2609.02688)
- [SIGMA](https://arxiv.org/abs/2610.09520)

## 4. v16 实现及真实验证边界

在完整上一版源码上新增：

- `ride/v16_cross_stress.py`: 直接读取已完成的 ZIP 或 JSONL，对固定 anchor 和同动作进行控制策略一致性、制动命令激活、源/目标连续负担差、动作对反事实差分、无作用对照及碰撞传感器匹配检查。能拒绝缺 `run_end` 的数据。生成 `docs/v16/V15_UPLOADED_INDEPENDENT_AUDIT.json`。
- `ride/carla/lagged_controller.py` 和 `ride/carla/run.py`: 下一轮 CARLA trial 逐步记录 C 的 raw requested brake、delayed brake、applied brake、cap 是否真正触发，以及延迟是否改变命令；实验结果新增 `v16_C_brake_control_actuation`。**新增记录不追溯改写历史 JSONL。**
- `scripts/47_v16_cap_preflight.sh`: 用完整基线的正常 C command 检查预注册 cap 是否可能生效；0.85 会返回非零退出码 4，除非专门声明用作无效限幅 placebo。
- `scripts/43_v14_safety_stress_carla.sh`: 如指定 `RIDE_V16_CAP_REFERENCE_PANEL` 则预先阻断不生效的限幅；执行后要求实际命令激活验证；不成功不会继续把实验标成已完成的压力证据。
- `ride/v16_transparent_selectors.py`: SOURCE 结束后、TARGET 产生前，锁定 `PROGRESS_ONLY`、`C_ONLY`、`B_ONLY_COURTESY`、`B_C_COURTESY`、`WEIGHTED_B_C` 五种透明参考选择。`scripts/45...` 自动运行锁定和评价；它们绝非作者 GameFormer、RACP 或 HYPE。
- `scripts/46_v16_cross_stress_report.sh`: 四个压力条件与选择结果的只读汇总。
- 增加本地回归测试。未经真实服务器运行，不宣称 v16 的物理实验成功。

**尚未实现**：真实官方 strong planner adapter、物理有效且可检验的 `do(B)` mediator simulator、CARLA 隐状态完整快照克隆，以及通过 nuPlan/ReactSim 学习式响应模型的跨平台确证。不要把这些当作已经完成的新模块。

## 5. 优先执行的服务器命令

### 5.1 安装与验证

```bash
unzip -q RIDE_V16_CAUSAL_RESPONSE_AUDIT_20261029.zip -d /data0/senzeyu2/dataset/RIDE/code/
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V16_CAUSAL_RESPONSE_AUDIT_20261029
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pip install -e '.[test]'
python -m pytest -q
python -c 'import ride.carla.run as r; print(r.__file__)'
python -m ride.doctor --server
```

### 5.2 不运行新 CARLA，先重算历史数据

将变量设为真实文件路径（可以是 `.zip`，或者已解压 `carla/chain_cut_in.jsonl`）：

```bash
export RIDE_V16_BASELINE=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_base_seed171/carla/chain_cut_in.jsonl
export RIDE_V16_CAP=/data0/senzeyu2/dataset/RIDE/evidence/v15_repaired_cap085_seed171/carla/chain_cut_in.jsonl
export RIDE_V16_DELAY=/data0/senzeyu2/dataset/RIDE/evidence/v15_repaired_delay03_seed171/carla/chain_cut_in.jsonl
export RIDE_V16_BOTH=/data0/senzeyu2/dataset/RIDE/evidence/v15_repaired_delay03_cap085_seed171/carla/chain_cut_in.jsonl
export RIDE_V16_LOCKED=/data0/senzeyu2/dataset/RIDE/evidence/v15_locked_c_stress_seed173
export RIDE_V16_REPORT=/data0/senzeyu2/dataset/RIDE/evidence/v16_readonly_stress_audit.json
bash scripts/46_v16_cross_stress_report.sh
```

注意：同一 seed 的两个 anchor 是两个独立物理位置，四组参数 cell 只是重复测量，不能报告成八个独立道路场景。

### 5.3 先检验制动限幅是否可能生效

```bash
export RIDE_V16_CAP_REFERENCE_PANEL="$RIDE_V16_BASELINE"
export RIDE_CHAIN_C_CONTROL_DELAY_S=0
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
bash scripts/47_v16_cap_preflight.sh   # 预期拒绝：cap 未触发

# 新一轮开发实验：0.25 有希望触发，需实际命令日志复核
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.25
bash scripts/47_v16_cap_preflight.sh
```

选择 0.25 是基于开发日志里约 0.30 的最大 C 原始制动命令，**不是人类制动能力的标定值**。该命令只能说明模拟器控制比例可能截断，不能直接把 cap 映射成 2.5 m/s² 之类的物理制动力。不得根据最终 holdout 碰撞是否出现随意更改 cap。

### 5.4 复用原 Seed 171 冻结道路运行一次真正生效的 cap 干预

确保 CARLA 独占服务器已经启动、地图为 Town04：

```bash
export CARLA_TOWN=Town04
bash scripts/29_v9_carla_preflight.sh Town04
export RIDE_V14_STRESS_BASE_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_base_seed171/frozen/chain_cut_in_anchors.jsonl
export RIDE_SEED=171
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export RIDE_WORLD_RESET=actors
export RIDE_CHAIN_C_CONTROL_DELAY_S=0
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.25
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v16_cap025_seed171
bash scripts/43_v14_safety_stress_carla.sh
cat "$RIDE_V14_STRESS_ROOT/carla/v16_C_actuation_gate.json"
```

如果 `command_cap_binding_ticks=0`，则说明预检的可能性没有在真实执行中发生，不能宣称降低了 C 的制动能力；必须记录该 cell 为无效干预。

### 5.5 进一步检验真正制动延迟+限幅条件

```bash
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.25
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v16_delay03_cap025_seed171
bash scripts/43_v14_safety_stress_carla.sh
cat "$RIDE_V14_STRESS_ROOT/carla/v16_C_actuation_gate.json"
```

如果 C 因极端制动失效发生碰撞，需要核查初态、ego/B 执行一致性、cruise 背景风险和传感器实际角色；这是人为 stress test 的证据，不能自动等同真实人类风险。

### 5.6 新的事前选择与透明 courtesy 对照（真正预测后果—选择—执行的门槛）

以下使用新 seed、新输出，不覆盖旧的 `v15_locked...`：

```bash
export RIDE_V16_COURTESY_BUDGET=0.4
export RIDE_SEED=181
export RIDE_ANCHORS=3
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.25
export RIDE_V14_LOCKED_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v16_locked_c_seed181
bash scripts/45_v14_locked_source_to_safety_stress.sh
cat "$RIDE_V14_LOCKED_STRESS_ROOT/v16_transparent_selector_audit.json"
```

该实验锁定五个透明候选选择器的 SOURCE 评分与选择，目标结果在此前不存在。主要观察：B-only courtesy 和 B+C courtesy 是否因为同一个来源响应证据的差分压缩而都选错；以及 B+C courtesy 是否已经足以修复一些 B-only 失误。预算参数 .4 是开发设置，不是默认公平最优参数，需要后续开展 sensitivity sweep；有 `UNKNOWN` 时不可无声地把它视为安全。

### 5.7 明确的停止/转向条件

- 若目标 C 制动延迟可以稳定增加总体负担，但任务匹配候选的 `D(a,b)` 不出现可复现差异，那么仅能声称 **response-latency sensitivity**，不能声称 RIDE 决策错误。
- 若 `D(a,b)` 在同场景多次重复下稳定被来源模型压缩，且预锁选择造成目标中可避免的 C 损失，可进入 hard courtesy 与 risk-aware 强透明基线比较。
- 若增加后车 C 制动压力后仍没有危险 TTC、第三方实际碰撞或其他严重安全事件，取消将“次生碰撞”作为摘要中的确定性动机，保留 avoidable third-party burden 作为主要应用。
- 只有加入受控 B 中介对照，并在至少一种独立的 learned reactive model 与多个物理走廊复现后，才有资格声称链式 B→C 机制。
- RACP/GameFormer/HYPE 仍未运行原版，不能把透明 selector 的失败归咎给它们。
