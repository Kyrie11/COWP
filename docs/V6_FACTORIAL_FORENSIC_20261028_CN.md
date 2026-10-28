# v6 factorial CARLA 结果复核（2026-10-28）

这份复核针对上传的 `rq12_v6_factorial_20261028.zip`。它是一次**发现性筛查**，不是论文最终统计结果。四个参数 cell 都有 `run_end completed=true`，但同一 Town04 corridor 被四个 cell 重复使用，只有 2 个 anchor、1 个 seed、1 次 repetition；不能把 80 条 trial 当作 80 个独立场景。

## 结果

| rear gap | rear speed delta | trials | valid | mechanism executed | task-matched cross-family pairs | strict ranking reversals | third-party A–B collisions | cruise 后车急刹 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 m | 0.0 m/s | 20 | 18 | 16 | 0 | 0 | 0 | 2/4 |
| 35 m | 1.0 m/s | 20 | 18 | 16 | 1 | 0 | 0 | 1/4 |
| 45 m | 0.0 m/s | 20 | 18 | 16 | 3 | 0 | 0 | 4/4 |
| 45 m | 1.0 m/s | 20 | 18 | 16 | 2 | 0 | 0 | 4/4 |

四个 cell 均标记为 `COMPLETE`，但“complete”只表示 JSONL 运行完整，不表示 RQ1/RQ2 通过。

## 失效原因和正负信号

- anchor 1 的 `late_slow` 在 TM 和 BehaviorAgent 中多次因 `offroad_or_lane_departure` 无效；因此晚并线的任务匹配分母被结构性削弱。需要先缩短横向切换或重新冻结一批满足完整走廊的 anchor，不能靠扩大重复次数掩盖。
- 多数正式风险键都是 `(third_party_collision, hard_brake, short_ttc)=(0,0,0)`；四个 cell 没有 A–B 碰撞，也没有严格跨 family 排序翻转。这些结果不能支持 RQ1/RQ2，但样本量也不足以证明现象不存在。
- `rear_gap=45 m` 的 `cruise` 负对照在两个速度设置中都出现后车急刹，说明后车背景行为已经饱和。并线动作造成的额外负担必须报告为相对于同 anchor、同 response family 的 `cruise` 增量，不能把所有急刹归因于 ego 并线。
- 连续制动负担在个别 anchor/参数上对 TM 与 BehaviorAgent 有差异，这是可保留的机制线索；它是事后诊断，不能替换预注册的三元风险键，也不能直接解释为真实驾驶员因果效应。
- `traffic_light_legality_verified=false`、actor-only reset 不恢复所有 Unreal 隐状态、只有 seed 17，这些都把当前结果限定为 simulator pilot。

## 研究判定

| 命题 | 当前状态 | 原因 |
|---|---|---|
| RQ1：响应模型改变任务匹配候选的风险排序 | 未通过，不能下否定结论 | 有效跨 family 配对最多 3 个，严格翻转为 0；还不是在线 planner 的 ex-ante 选择 |
| RQ2：错误动作产生可避免第三方风险 | 未通过，不能下否定结论 | A–B 碰撞为 0，且 cruise 背景制动饱和；没有稳定的 task-matched 可避免效应 |
| RQ3：强基线在同协议下系统性选错 | 未开始 | 没有锁定外部 planner 的 checkpoint、选择记录和独立响应评测 |

因此，当前最准确的论文表述是“场景执行链路已明显改善，但 RIDE 机制仍未被识别”。不能把四个 cell 事后挑一个最有利的 cell 当作正式确认集。

## 下一轮必须修复的实验设计

1. **先修复动作可执行性**：在开发地图上重新冻结至少 6–10 个独立 anchor；对每个 anchor 预先验证 early/late 两条横向走廊的完整驻留与道路有效性；缩短 `late_slow` 的切换延迟或减小 lateral switch 幅度，并把修改后的协议版本升号。
2. **锁定响应前初态**：每个 response family 在干预开始前写出 ego、A、B 的位置、速度、航向摘要和 digest。跨 family 任一角色超出预注册容差就丢弃该 pair，保留为 invalid，不进行事后修正。
3. **保留 `cruise` 机制对照**：主分析使用 early/late 或 mild-early/mild-late 的同任务动作对；`cruise` 只用于报告增量负担和背景风险。
4. **增加独立性**：至少 3 个 seed、每个 seed 6–10 个物理 corridor，并把开发集的参数筛选与 holdout 的正式确认分开。不要把同一个 Town04 corridor 的四个参数 cell 当成独立样本。
5. **再做外部响应复核**：在 ReactSim SMART/MTR 或 Bosch SMART-reactive 环境中复现同一固定初态、同一 A/B 身份和同一 ego A/B 动作。普通 ReactSim benchmark 分数不能替代这个双动作配对协议。
6. **RQ3 延后**：只有当 CARLA 至少出现稳定的 task-matched response disagreement，并且独立 learned reactive evaluator 复现方向，才锁定 GameFormer-Planner、hard courtesy、RA-CBF 等 planner/safety baselines。

## 本版工程改动

- `scripts/14_download_reactsim_weights.sh` 改为 Hugging Face `model` 仓库，固定 revision，支持根目录和 `ckpts/` 两种远程布局，拒绝 Git-LFS pointer，记录 SHA256；不再因为 `Dataset ... not found` 误判为要训练 checkpoint。
- `ride/carla/run.py` 的 CARLA 控制循环原本就是每个记录步一次 `world.tick()`；本版在 manifest 增加 `physics_ticks_per_recorded_step=1` 与 `physics_dt_s`，并加入回归测试，防止未来时间基准漂移。
- `scripts/21_audit_factorial_mechanisms.sh` 可对所有 frozen factorial panel 重新计算 lateral command、实际入道时刻、后车制动开始时刻以及相对 `cruise` 的连续负担；这些输出仍标记为 exploratory。
- `docs/BASELINE_INTEGRATION_MATRIX_CN.md` 明确区分 response evaluator、ego planner、safety filter 和 toy simulator，避免把“源码已下载”写成“强基线已运行”。

## ReactSim DB 覆盖诊断

`inspect.txt` 的 1,501 个 `DB_MISSING` 只能说明这些 log name 不在**当前传给审计器的 trainval 目录**，不能直接区分“源数据没有”与“源数据在 `train_*`/`test` 但没有进入 trainval 软链接”。本版新增 `ride.reactsim_db_coverage` 与 `scripts/22_reactsim_db_coverage.sh`，先在原始 `/data0/senzeyu2/dataset/CapPlan/data/nuplan` 的所有 split 上按 log name 做只读盘点，再决定是否增量获取 DB。

```bash
export RIDE_NUPLAN_SOURCE_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan
bash scripts/22_reactsim_db_coverage.sh
cat "$RIDE_REACTSIM_WORK/diagnostics/val_db_coverage.json"
```

`AVAILABLE_OUTSIDE_STAGED_SPLITS` 表示只需修正软链接视图；`MISSING_IN_ALL_LOCAL_SPLITS` 才表示本地确实没有该 log，需要核对 ReactSim/nuPlan 版本后按清单增量下载或明确记录不可复现。该盘点仍不替代 SQLite token/scene join 审计。

本版的 CARLA 入口增加了两个**预注册前可调**的控制参数：`RIDE_CUTIN_LATE_SWITCH_S` 和 `RIDE_CUTIN_LATERAL_LOOKAHEAD_M`。它们会写入 frozen manifest，执行时若与冻结值不同会直接拒绝。可以在开发地图上用全新输出目录筛查，例如：

```bash
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_ANCHORS=10
export RIDE_REPEATS=1
# 先只运行开发筛查，检查每个 anchor 的 early/late 是否都通过；
# 选定后重新冻结协议，再把参数固定到 holdout。
bash scripts/19_cut_in_factorial_pilot.sh
```

这不是对 v6 失败 trial 的事后修补；它必须在开发集确定后，以新输出根目录、新冻结 manifest 和新的 control-parameter 记录进入确认实验。
