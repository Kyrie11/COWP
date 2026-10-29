# RIDE v14 用户上传结果复核、v15 协议修复与下一轮实验

**证据日期**：2026-10-29。**来源**：当前上传的 9 个 v14 实验包与 v14 源码，及前轮同一会话的三个 v13 结果包。**状态**：离线复核与本地测试完成；未连接用户的 CARLA 服务器完成新的 v15 物理执行。

## 1. 结论（以完整证据链为准）

1. 不能宣布 RQ1/RQ2 已通过，也不能因应力实验为空宣布“反应延迟无风险”。v14 延迟 / 限幅的三个目录均缺完整 CARLA JSONL；locked stress 也只有工作目录骨架。
2. v13 Seed17 Anchor3 `speed_early/speed_late` 的 TM 目标 C 负担差 0.460085m/s；使用同一锚点、seed 的 v14 TM 重跑测得 0.466517m/s。**目标后果差异具有一次重现线索**。但 v14 的 BehaviorAgent 来源 `speed_early/speed_late` 因 `initial_position_mismatch:follower_b` 而不可用于锁定选择效果，所以原 v13 的完整诊断选择错误**没有复现**。
3. v14 Anchor3 两套 BehaviorAgent→BehaviorAgent 同场景 placebo、Anchor4/5 同模型 placebo 均未报告诊断性模型选择错误。Anchor5 的多个比较却因为 `follower_c` 初态失配成为 UNKNOWN；placebo 的零错误并不能证明隐状态已完全受控。
4. 无延迟、cap1.0 的 v14 基线 42/42 trial `valid`，0 个 B–C 真实碰撞，最小 B–C TTC 约 6.23s，没有原协议的 C 持续 ≥0.2s 急刹。两个名义上相同的 C BehaviorAgent 家族（C delay=0, cap=1.0）只有 12/14 条通过初态+ego+B 轨迹一致性审计。需先评估同模型噪声。
5. 基线 22/42 条被**假设**停止裕度筛查标为负数，**不是实际**碰撞、短 TTC 或真实人类危险概率。静态停车距离公式用于选择测试范围，不能作为 RQ2 已经发生的风险证据。
6. 原先 v13 透明来源模型选错仍是诊断案例，非 GameFormer / RACP / HYPE 已失败；CARLA TM / BehaviorAgent 亦非真实驾驶员的独立反事实真值。

### 重要统计口径

- `v13_target_decision_audit.json` 的 `valid_comparisons` 含 `SOURCE_THIRD_PARTY` 和 `PROGRESS_ONLY` 两套策略，**两行可能来自同一个动作对**。
- Seed107 的 `strict_wrong_model_orders=2` 指两套透明选择器重复记录，不能当作两个独立物理正例。
- 当前两个 v14 正例复现配置在原道路、原 seed 重复，不构成新的独立道路证据；Anchor45 placebo 不能替代 TM 目标端的对应重跑。
- 近失/负担/停止裕度/真实传感器碰撞，必须在论文中分列，不能互相替代。

## 2. 已修复的工程缺陷

### 2.1 `DELAYED_C_PARAMETERS_REQUIRE_EXPLORATORY_CHAIN_FAMILY`

`43_v14_safety_stress_carla.sh` 预先设置了 `tm mixed_b_tm_c_behavior mixed_b_tm_c_delayed_behavior`，但旧版 `03_precommit_scenes.sh` 未将环境中的 families/profiles 传入 `run.py --manifest-only`，于是 Python 回退为 `tm behavior`，在冻结阶段错误报错。现在冻结脚本传入与执行脚本相同的 family/profile，并保留拒绝不受支持 family 的保护性门禁。

### 2.2 `45_v14_locked_source_to_safety_stress.sh` 第二阻断

原来源 family `mixed_b_tm_c_behavior` 必须**不受 C 延迟影响**；但需要记录与目标相同的延迟/限幅作为配对元数据。现在只有显式 `--v14-source-inactive-stress-metadata` 且仅用于该来源 family 才允许这种情况；日志和 manifest 标记参数未生效。目标必须使用 `mixed_b_tm_c_delayed_behavior`，修改确实进入 CARLA 控制器。

### 2.3 v13 锁定器不认识 v14 的新 family

旧 `ride.v13_model_transfer.lock_choices` 只支持 `tm/behavior`，会在 45 脚本来源阶段再次报 `Unsupported response family`。现在只额外放开 **精确的** `mixed_b_tm_c_behavior → mixed_b_tm_c_delayed_behavior` 配对，输出明确的 `V14_CONTROLLED_C_POLICY_STRESS_NOT_MODEL_ERROR`，不把人为制动干预的选择损失计作“响应模型本身的错误”。其他混合 family 仍被拒绝。

### 2.4 一致性与场景复用

- `v15_protocol_check` 不连接 CARLA 便可提前验证整个冻结/来源/目标协议，避免昂贵执行到一半才失败。
- `v15_stress_frozen` 可**复用同一 baseline frozen manifest**，只更新 C 制动延迟和命令上限，保留车辆位置、map、seed、速度和完整路由协议，并记录原始 manifest SHA256。
- `v15_reproducibility_audit` 独立复算有执行记录的实验、验证 source/frozen 的锁定哈希，按真实道路身份汇总，保留 UNKNOWN 和物理模型边界。
- 对 `v14_merge_carla_panels` 新增强制 C-stress 参数配对审计：只要来源/目标参数不同就报错，不能混合比较。
- 共 177 个 pytest 通过、1 个跳过；`python -m compileall` / `bash -n` 应在你的服务器复核。未在这里执行任何 CARLA 新实验。

## 3. 推荐论文主张（分阶段，不要提前冒进）

**研究主线继续保持 RIDE，不要变成单独的 comfort-oriented courtesy 论文。**

> Under independently controlled and heterogeneous traffic response, when do interactive behavior models suppress decision-relevant counterfactual differences in third-party consequences, and when does this evidence-to-choice mismatch cause avoidable losses under task-matched ego alternatives?

中文：在无共享响应协议、独立且异质的交通中，交互响应模型何时会压缩对规划选择关键的特定第三方反事实后果差异？这种证据失配何时使规划器选中存在可避免负担、甚至次生安全危险的 ego 动作？

**先证明“差分稳定可信”才能谈“模型结构差异”。** 以前只看两个模型间 C 负担差分；现在更严谨的检验是把来源/目标的差分误差和同 family 的重复噪声分别估计：

\[D_M(a,b)=L_C^M(a)-L_C^M(b),\qquad \epsilon_{CF}=D_M-D_E.\]

另从相同 scene、相同 family、相同 seed 的重复实验获得 \(N_M=|D_M^{r1}-D_M^{r2}|\)；须证明跨 family 差别显著高于同 family 噪声且会改变实际来源选择，不能仅使用一次 v13 成功案例。

### 三个科学门槛

- **Gate 0，测量与初态可识别**：同一冻结道路上的初始位置、速度、航向、真实 ego 轨迹匹配；同 family 空白对照能够稳定重复；异常物理加速度可审计；UNKNOWN 不剔除。建议工程目标 ≥95% 预注册动作对完成基本有效性检验（工作阈值，非统计定理）。
- **Gate 1，RQ1 和 RQ2 现象**：至少两条空间独立道路上的任务匹配动作对出现稳健的第三方负担差分；模型低估/压缩的影响大于同模型重复噪声；在事前锁定的来源选择上形成可避免损失；同时报告剪切极端减速度和时间滤波下的稳健性。若仅 comfort 有差异，不能写“次生碰撞”。
- **Gate 2，安全放大条件**：基线/限幅/延迟保持同一冻结几何；真实 B–C 碰撞传感器和连续时间近失指标支持行动差异，且 cruise 无相同背景危险；排除 ego、B 轨迹混杂；仅在报告已匹配的响应能力范围内宣称模拟安全效应。更进一步用有控制的 B 轨迹干预检验 B 的中介作用，不能以 B先刹C后刹代替因果证明。
- **Gate 3，其他方法的剩余不足**：先公平调优 B-only courtesy、B+C courtesy、multi-response risk-aware，保留相同候选池/目标/计算预算；如 B+C courtesy 已完全避免问题，则不能写“现有社会规划忽略 C”。之后再接官方 GameFormer-Planner / RACP / HYPE 真实选择，并在 nuPlan/ReactSim 等独立 reactive benchmark 复核。

当前 **Gate 0 尚未稳健通过**，Gate 1 有原始 v13 诊断线索，Gate 2/3 未通过。因此可以做轻量方法设计，不建议完整算法网络训练或锁定 CCF-A 研究贡献。

## 4. 服务器安装：必须验证正在执行的是新代码

注意原错误堆栈指向 `/home/senzeyu2/code/RIDE/ride/carla/run.py`，不是可能解压后的新工程目录。务必检查实际 Python import 路径。推荐保持旧工程只读，新工程另放：

```bash
unzip -q RIDE_V14_REPAIRED_V15_GATE_20261029.zip \
  -d /data0/senzeyu2/dataset/RIDE/code/
cd /data0/senzeyu2/dataset/RIDE/code/RIDE_V14_REPAIRED_V15_GATE_20261029
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pip install -e '.[test]'
python - <<'PY'
import ride.carla.run as r
print('USING_CODE=',r.__file__)
from ride.v15_protocol_check import check
print(check('panel',0.3,1.0)['status'])
print(check('locked',0.3,0.85)['status'])
PY
python -m pytest -q
bash -n scripts/*.sh
python -m ride.doctor --server
```

`USING_CODE` 结果必须是该新工程目录。旧代码仍被导入就先检查当前工作目录、`PYTHONPATH` 和 editable install；不建议在旧工程上原地覆盖改写。

## 5. 首先离线验证三种压力参数：无需重启 CARLA

```bash
python -m ride.v15_protocol_check --mode panel --delay-s 0.3 --brake-cap 1.0
python -m ride.v15_protocol_check --mode panel --delay-s 0.0 --brake-cap 0.85
python -m ride.v15_protocol_check --mode panel --delay-s 0.3 --brake-cap 0.85
python -m ride.v15_protocol_check --mode locked --delay-s 0.3 --brake-cap 0.85
```

必须打印 `OFFLINE_PROTOCOL_VALIDATED_NO_CARLA_EXECUTED`。它只是参数/代码检查，不代表 CARLA 运行成功。

## 6. 修复后复用 Seed171 基线冻结道路，重新运行三种 C stress

前置：CARLA 0.9.16 已无头启动，Town04 preflight 正常；专用服务器无其他车辆。基线冻结文件沿用**你已完成的 v14_stress_base_seed171**：

```bash
export CARLA_TOWN=Town04
export RIDE_V14_STRESS_BASE_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/v14_stress_base_seed171/frozen/chain_cut_in_anchors.jsonl
# 显式记录工作参数，冻结复用将进一步强制原 manifest 值
export RIDE_SEED=171
export RIDE_ANCHORS=2
export RIDE_REPEATS=1
export RIDE_WORLD_RESET=actors
export RIDE_WARMUP_TICKS=20
export RIDE_CRUISE_MPS=7.5
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_CHAIN_BC_GAP_M=20
export RIDE_CHAIN_C_SPEED_DELTA_MPS=-0.5
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS=0.5
```

先只跑 delay=0.3，cap=1.0：

```bash
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=1.0
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v15_repaired_delay03_seed171
bash scripts/43_v14_safety_stress_carla.sh
```

然后做只限幅与组合：

```bash
export RIDE_CHAIN_C_CONTROL_DELAY_S=0
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v15_repaired_cap085_seed171
bash scripts/43_v14_safety_stress_carla.sh

export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
export RIDE_V14_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v15_repaired_delay03_cap085_seed171
bash scripts/43_v14_safety_stress_carla.sh
```

使用这些**全新的输出目录**，不要在原报错目录里续写；新包拒绝覆盖既有结果。每个 stress 目录都应产生 `carla/chain_cut_in.jsonl`、`carla/v14_physical_stress_audit.json`、`carla/v14_hypothetical_stop_margin.json`。检查完整 `run_end` 和 manifest 的非零 delay/cap，而不是只看目录是否创建。

`v15_stress_frozen` 保持 seed 和几何一致，但 actor-only reset 仍不能保证干预初态一致；用 audit 的 `same_controller_C_vs_lagged_C_parity` 和每动作的初态记录筛选有效对照。四种工况应全部报告，含无效和无结果配置。

## 7. 再运行真正的事前选择锁定压力试验

```bash
unset RIDE_V14_STRESS_BASE_FROZEN
export RIDE_SEED=173
export RIDE_ANCHORS=3
export RIDE_CHAIN_C_CONTROL_DELAY_S=0.3
export RIDE_CHAIN_C_BRAKE_COMMAND_CAP=0.85
export RIDE_V14_LOCKED_STRESS_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/v15_locked_c_stress_seed173
bash scripts/45_v14_locked_source_to_safety_stress.sh
```

该实验先执行未延迟 C 的来源、写出哈希锁定选择，再执行受控延迟+限幅 C 的目标，并检验相同初态与 ego 执行轨迹。正常结束将得到 `v13_target_decision_audit.json` 和 `v14_physical_safety_audit.json`。

**重大限制**：这是人为改变 C 控制器的实验，不是两个独立“真实响应模型”的 generalization；新的摘要把相关损失写成 `controlled_C_policy_avoidable_choices`，不会计入 `diagnostic_model_induced_choice_errors`。如果 B 的响应轨迹同时发生大幅变化，不能声称“只有 C 反应时间的影响”，需首先查看 parity。

如果要在完全**同一冻结几何**进行 locked stress，需要为该脚本另行接入指定 frozen manifest 的入口；当前 45 脚本独立重新选取 seed173 场景，不能直接与 seed171 baseline 逐场景配对。因此更严格的四格**控制比较**先以第 6 节同冻结场景的 43 脚本为准。

## 8. 接下来重复 v13 原始正例：测定噪声地板

在 v13 原始目录中读取真实 frozen manifest；不要把 v14 重跑目录的 manifest 再当成一个新物理锚点：

```bash
export RIDE_V14_ORIGINAL_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/v13_dev_behavior_to_tm_seed17/freeze/frozen/chain_cut_in_anchors.jsonl
export RIDE_V14_ANCHORS='3'
export RIDE_V14_SEED=17
export RIDE_V14_SOURCE_FAMILY=behavior
export RIDE_V14_TARGET_FAMILY=tm
unset RIDE_V14_PLACEBO
for k in 2 3 4; do
  export RIDE_V14_ROOT="/data0/senzeyu2/dataset/RIDE/evidence/v15_anchor3_ba_tm_repeat${k}"
  bash scripts/42_v14_replay_frozen_two_stage.sh
done

export RIDE_V14_TARGET_FAMILY=behavior
export RIDE_V14_PLACEBO=1
for k in 2 3 4; do
  export RIDE_V14_ROOT="/data0/senzeyu2/dataset/RIDE/evidence/v15_anchor3_ba_ba_repeat${k}"
  bash scripts/42_v14_replay_frozen_two_stage.sh
done
```

即使同一 seed、同一道路，也要按重复测量计算统计；不要把六次实验写作六个独立交通场景。对最关键的 `speed_early/speed_late`，如果 source 仍无法通过初态配对，优先修复 reset/速度稳定流程，而不是调宽原来的 0.5m 初态容差。

## 9. 再考虑跨平台和 RQ3

建议在 CARLA Gate0/Gate1 达标后使用现有 nuPlan DB 与 ReactSim SMART/MTR、Bosch SMART reactive nuPlan 进行同动作对复核。SMART/MTR 是**反应环境**，不是强规划器。RACP/HYPE 为已经处理多模态/ego-conditioned 交互的竞争论文；必须证明新的方法相比充分调优的 hard courtesy B+C、multi-response risk planning 能额外改善真实候选选择，而不是把模型差异、制动舒适度或三值 UNKNOWN 概念本身当创新。正式强基线的论文主张需真实作者代码、输入等价、候选池/计算预算等价、预先锁定选择、完整记录无效与失败样本。

### 方向停止条件

若多 corridor、重复同模型 placebo 和学习式响应环境下，来源/目标反事实差分误差并未持续大于重复噪声，或不同模型选择未产生可避免损失，应停止把 `response-model-induced decision error` 当成已建立的论文机制。可保留稳健的第三方可避免负担研究，但该方向与 Courteous / RACP 有重合，需另证公平基线无法解决的 gap。若只有人为制动延迟才会撞车，不能宣称一般模型/人类交通的次生碰撞风险已经确立。
