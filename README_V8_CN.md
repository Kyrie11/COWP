# RIDE v8 · 2026-10-28 · 科学论证与可执行协议

> 此版本是 **RQ1/RQ2 可证伪性和 RQ3 环境部署工程**，不是一种已训练/已取得 benchmark 成绩的新规划算法。请先阅读 `docs/V8_FORENSICS_AND_ACTIONS_CN.md`。未在用户的 CARLA、GPU、nuPlan 服务器上运行 v8 物理测试；本地只完成 93 个 Python 测试、编译和 bash 语法验证。

## v8 vs v7 的根本修复

1. `ride/carla/tm_policy.py`：CARLA TM 的 `vehicle_percentage_speed_difference(..., 0)` 是 **以道路限速为参考的 0% 减速**，不是 9m/s。v8 从 `get_speed_limit()` km/h 换算至物理名义速度，并持续按限速段刷新；不可达目标速度不静默截断，直接标为异常。
2. `ride/carla/response_calibration.py` 与 `ride/v8_gate_audit.py`：保留车型/行动的 peak speed、cruise 超 1.5× 名义速度 0.5s 的诊断。**这个指标用于排查对照组校准，不是将超速视为结果中的第三方风险正例**。
3. `ride/protocol_lock.py`：确认阶段先比较冻结开发 manifest 的 `cutin_late_switch_s`、`cutin_lateral_lookahead_m`、ego/rear gaps、rear speed delta、warmup 和 cruise；值不一致即失败，避免 v7 开发 3.5/20 与确认 4.5/15 混用。
4. 冻结确认场景时 `--exclude-anchors-from` 与 `--exclude-radius-m` 排除开发集同 Town 的近距离物理 corridor；新协议号 `ride_lane_corridor_v8_20261028`，旧 anchor 不可混用。不同 seed 不代表不同道路；报告按 corridor 聚类。
5. 碰撞事件补充 other actor 的 `actor_id` 和 `type_id`，方便查清 `external` 是 NPC 还是地图静态资产。仅对今后新实验有效；旧 JSON 不能倒推出类型。
6. `ride/v8_gate_audit.py` 同时审计实验完成率、无效原因、任务匹配分母、严格排序翻转、TM/BehaviorAgent 速度、负对照碰撞、跨确认 corridor 重叠、nuPlan 数据覆盖。结果为 **descriptive / not human causal**，不能声称真实人类反事实因果或强规划器失败。
7. nuPlan 工具：`ride/nuplan_recovery_plan.py` 从 `val_db_coverage.json` 导出真正缺失的 `.db` 文件名和缺失月份；`ride/reactsim_research_subset.py` 可以仅在已安装 nuPlan-devkit 的环境里用官方查询提取 **研究调试小子集**，不会更改 2636 官方 ID 清单。
8. `ride/baseline_readiness.py` 检查 Bosch、GameFormer-Planner、ReactSim 源码/模型是否齐备；`ride/gameformer_headless.py` 非破坏性复制并补丁官方 runner 以抑制 GUI NuBoard；不修改评分器和闭环仿真。

## 环境与 smoke

```bash
# 在服务器将 ZIP 解压到 RIDE/code，cd 到解压得到的 RIDE_V8_RESEARCH_GATE_20261028
conda activate mfrp
python -m pip install -e '.[test]'
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
python -m pytest -q
bash scripts/01_doctor.sh
python -m ride.doctor --server
# CARLA 未启动时，在独立 tmux 终端运行 bash scripts/02_start_carla_headless.sh
export CARLA_TOWN=Town04
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_WARMUP_TICKS=20
export RIDE_SEED=17
export RIDE_ANCHORS=2
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_m0_smoke
bash scripts/23_v8_smoke.sh
```

检查 `carla/cut_in.jsonl` 里的 `tm_speed_configuration`、`max_observed_follower_speed_mps`、`simulation_exception`、`valid`、`ego_collision`、`initial_states`、`target_lane_dwell_max_s`。若没有合格动作 pair，不应扩充 seed。

## 冻结复核（先后顺序）

```bash
export RIDE_DEV_FROZEN=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_m0_smoke/frozen/cut_in_anchors.jsonl
export RIDE_DEV_TRIALS=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_m0_smoke/carla/cut_in.jsonl
export RIDE_CONFIRM_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_confirm
export RIDE_CONFIRM_SEEDS='23 41 73'
export RIDE_ANCHORS=6
# 建议留出 Town05；若无法筛选合格锚点，停止该协议，不修改结果后伪装独立
export CARLA_TOWN=Town05
bash scripts/24_v8_confirm.sh
export RIDE_REPORT_JSON=/data0/senzeyu2/dataset/RIDE/evidence/rq12_v8_confirm/v8_summary.json
bash scripts/25_v8_evidence_report.sh
```

`24_v8_confirm.sh` 会自动进行协议参数验证，并排除开发 manifest 的物理位置；若 Town05 无法形成至少 6 个分离 corridor，则停止并换新实验协议。`RIDE_WORLD_RESET=reload` 是一条独立稳健性实验线，不能混入 `actors` 的统计。

## 现有 nuPlan 的文件恢复方案

```bash
conda activate ride_reactsim_dp
export DATASET_ROOT=/data0/senzeyu2/dataset
export RIDE_REACTSIM_REPO=/data0/senzeyu2/dataset/RIDE/research_baselines/ReactSim_Bench
export RIDE_REACTSIM_WORK=/data0/senzeyu2/dataset/RIDE/reactsim_workspace
export RIDE_NUPLAN_SOURCE_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan
source scripts/00_env.sh
bash scripts/22_reactsim_db_coverage.sh
bash scripts/27_v8_nuplan_recovery.sh
cat "$RIDE_REACTSIM_WORK/diagnostics/v8_nuplan_recovery_plan.json" | head -35
head -20 "$RIDE_REACTSIM_WORK/diagnostics/v8_missing_db_filenames.txt"
```

核对所有存储根目录和官方下载归档的清单后，只恢复有官方来源依据的缺失归档/日志，不重新下载 8,732 个已经存在的 DB；也不能把 nuPlan `test` 混入训练、ReactSim 完整验证。恢复后再运行 `17_reactsim_preprocess_checked.sh inspect-val`，并单独检查 `FILTERED_SCENE_BOUNDARY`、SQL joins/nuPlan devkit 版本问题。存在这两类错误时不能声称官方 2636 完整复现。

可先利用 **标注研究用途、非官方 benchmark** 的小子集测试依赖链：

```bash
bash scripts/28_reactsim_research_subset.sh
# 输出 $RIDE_REACTSIM_WORK/research_only_subset/research_only_val_ids.json
# 及 subset_manifest.json。仅供作者 build_cache.py 的 --scenario_ids_path 做有限 smoke。
```

## Bosch、GameFormer-Planner 及 RQ3 预部署

```bash
export RIDE_BASELINE_ROOT=/data0/senzeyu2/dataset/RIDE/research_baselines
bash scripts/26_v8_baseline_preflight.sh
# RIDE_BASELINE_ROOT/GameFormer-Planner 为实际 nuPlan Planner；原 GameFormer-main 不是
# GameFormer 权重未包含于用户上传源码；需取得对应权重或按作者 train_predictor.py 训练
python -m ride.gameformer_headless \
 --source "$RIDE_BASELINE_ROOT/GameFormer-Planner/run_nuplan_test.py" \
 --dest "$RIDE_BASELINE_ROOT/GameFormer-Planner/run_nuplan_test_ride_headless.py"
```

GameFormer-Planner 对 nuPlan-devkit 的测试版本是 v1.2.2；在 **独立** nuPlan 环境中使用 `RIDE_HEADLESS=1 python run_nuplan_test_ride_headless.py --experiment_name closed_loop_reactive_agents --data_path /data0/.../nuplan/nuplan-v1.1/splits/val --map_path .../nuplan/maps --model_path /真实/权重.pth --scenarios_per_type 1 --total_scenarios 1` 先测试官方原生闭环。没有模型文件时要停止，不能创建随机网络并称为 GameFormer 强基线。

Bosch `interactive-closed-loop` 是 **另一个 learned reactive traffic response environment**，需要单独 nuPlan-devkit 副本与 SMART 副本；请按其官方 README 使用对应 checkpoint，在 `scenario_filter=test_one_scenario` 的可用 DB 上 smoke。它不是 ego planner。若 GameFormer 要在 Bosch SMART 场景中做公平 RQ3 对照，还需要实现二者在相同 nuPlan token、相同输入信息与评测预算下的联通适配，不能仅将两份独立官方 benchmark 总分拼在一张表内。

## 科学禁区

- 不因多次无效 trial 得到“0 次排序翻转”就宣称命题为假；不把无效碰撞算第三方风险正例。
- 开发/确认不同的控制时序、不同 TM 名义速度、重复的 Town04 corridor 不能混合成正式 holdout。
- `cruise` 是机制负对照，不是满足相同任务的真正 RQ2 alternative。
- S4 是有限证据/搜索库消融，不是独立道路拓扑；缺 witness = `UNKNOWN`，不是全局不可行。
- RQ3 需要 **rollout 之前** 有锁定的 strong planner 选择和 checkpoint/hash、状态/候选集/进展配对。已有 `ride.commit` 与 `ride.decision` 提供签名审计，不代表 GameFormer 适配已经完成。
