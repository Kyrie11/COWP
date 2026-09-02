# OC-RAP 主方法 Closed-loop 热路径优化说明

## 1. 上一轮优化是否会加速 OC-RAP

会，但只有其中的“通用 closed-loop runner”优化会作用于 `closed_loop.method=ocrap`：

- 同一重规划时刻共享 scene feature，只为每个 candidate 计算 prefix-specific feature；
- `eval_cfg`、root/option 数量与 dataset-quality 配置移出 step loop；
- selected/top-k audit 使用轻量 teacher-label 字典，不再重复序列化 history/map/BEV；
- audit 不再执行未使用的 regime assignment；
- Waymax environment/JIT rollout cache 与分阶段 timing；
- scene journal、原子 partial 和 resume。

以下上一轮内容不直接加速单个 OC-RAP rollout：外部 baseline 的方法级并行、非学习 baseline 跳过训练、GameFormer 输入契约修复。

因此，`label_mode=fast` 的普通 OC-RAP closed-loop 已能受益；`selected/selected_topk/all` 中 teacher/recovery rollout 占比越高，上一轮通用 CPU 优化的整体收益越有限。

## 2. 本轮发现的 OC-RAP 独有热点

### 2.1 每次 replan 将完整 Waymax trajectory 搬到 CPU

旧路径：

1. `raw_scenario_from_waymax_state(... closed_loop_splice ...)`；
2. 对 log/sim trajectory 的 x/y/z/v/heading/size/valid 等字段分别 `device_get`；
3. 复制完整 `A x T` 轨迹并 splice；
4. 构造完整 `RawScenario`；
5. `construct_history` 最终只保留 1 s history 和规划/recovery future window。

这在每个 replan 都重复，并且同步点多。新路径 `construct_history_from_waymax_state`：

- 先在 accelerator 上选择 SDC + `max_agents`；
- 只切 `history + planning/recovery future + gradient 邻点`；
- log/sim 小窗口一次 `jax.device_get`；
- 在小窗口上执行与旧路径相同的 closed-loop splice、`np.gradient` 边界规则、ego transform、route sanitize 和 BEV render；
- 对未知 Waymax layout 自动回退到旧完整路径。

结果 JSON 增加：

```json
"fast_waymax_history": {
  "enabled": true,
  "used": 40,
  "fallbacks": 0,
  "first_error": null
}
```

只有 `fallbacks=0` 时才表示该场景完全走新路径。

### 2.2 BEV 固定网格和极坐标每步重建

BEV radius/resolution 在一次实验中不变。旧代码每步重新生成 `X/Y`，每个 occluder 又重新计算整张网格的 `R/theta`。

修改：

- `grid_coords(radius,resolution)` 使用只读 LRU cache；
- `ego_centered_grid_geometry` 缓存 `X/Y/R/theta/in_range`；
- occlusion shadow 在 ego 原点时复用 cached polar grid。

没有改变任一栅格坐标或 mask 判定。

### 2.3 24 个候选重复 route projection 与邻车筛选

同一 replan 的所有候选具有相同 ego、route 和当前邻车。旧 `_rollout` 对每个 candidate 重复：

- `project_to_route`；
- current valid agents 布尔筛选。

修改为 `generate_candidate_prefixes` 计算一次并传给全部候选。候选顺序、宏动作、控制、utility、hard/harm/feasible 均不变。

合成 CPU micro-benchmark（24 candidates）中，candidate generation 中位耗时从约 22.43 ms 降至 8.51 ms，约 2.64x。该数字只代表候选生成，不是完整 closed-loop 加速比。

### 2.4 feature-only dummy root/option geometry 被每候选重复构造

`label_mode=fast` 下，所有 candidates 的 dummy `root_probs/root_valid/C/m_star/signatures/recovery options` 相同。旧代码仍逐 candidate 分配这些数组，并逐 candidate 计算同一个 BEV unknown ratio。

修改：一次构造共享只读语义的 geometry；unknown ratio 一次计算。训练/teacher-labeled sample 路径不变。

### 2.5 模型推理前后存在多次重复几何修复和 GPU 同步

旧 `predict_samples`：

- 每 candidate 调用 `fix_sample_geometry`；
- 重复创建同样的 option/root-valid tensors；
- `R_dep/R_orc/gap/q/p/C/margins/direct-value` 分 7–9 次 `.cpu().numpy()`。

修改：

- feature-only candidate batch 只修复第一份共享 geometry，通过 tensor `expand` 复用；
- inference dictionaries 共享 history/map/BEV/root/option 数组，只浅复制 candidate-specific 字段；
- 所有模型输出先在 device 端 pack，再一次传回 host。

CPU 小模型 benchmark 基本持平；该优化针对 CUDA stream synchronization，实际收益应在 GPU timing 中确认。

### 2.6 Waymax metrics 与物理几何读取造成多次完整数组传输

旧代码每个 control step：

- 每项 Waymax metric 单独 host transfer；
- x/y/v/yaw/length/width/height/valid 各自把完整 trajectory 拉回 CPU；
- trace 又单独传 x/y；
- timestep/done 多次同步。

修改：

- 在 device 上先切当前 timestep 的 agent vectors；
- Waymax metric values、geometry slice、timestep、done 放入同一 pytree；
- 一次 `jax.device_get`；
- 同一 x/y slice同时用于 clearance/TTC/speed 和 trace。

Waymax metric 的名称、SDC 索引和聚合规则保持不变。

### 2.7 重启后重复 XLA 编译

新增 `waymax.jax_compilation_cache_dir`，并提供 `run_ocrap_closed_loop_optimized.sh` 设置 run-local `JAX_COMPILATION_CACHE_DIR`。首次冷编译仍存在；同硬件、同 JAX/jaxlib 和相同程序形状的后续运行可读取缓存。

## 3. 明确没有修改的评估语义

- WOMD pattern、scenario 顺序和 Waymax reset/step；
- `max_scenarios/max_steps/replan_interval_steps`；
- candidate 数量、宏动作 bank、参数、顺序和随机种子；
- checkpoint、校准阈值、selector 与 intervention budget；
- OC-MERO、root/option 数量和模型输出；
- label mode 及 selected/top-k/all audit label budget；
- Waymax overlap/offroad/log-divergence/kinematic metrics；
- clearance、TTC、FRA、DRS、ODG、NUP、PCD 的定义。

优化脚本默认 `LABEL_MODE=fast`，因为这是原 default config；若你的论文表格要求 `selected_topk` 或 `all`，必须显式保持原 label mode，不能为了速度改成 fast。

## 4. 验证

执行：

```bash
PYTHONPATH=src pytest -q
```

结果：

```text
66 passed
```

新增严格等价测试覆盖：

- fast Waymax window 与 legacy full conversion 的 history/future/acceleration/map/route/BEV/metadata；
- shared-geometry prediction 与旧 prediction 的全部 OC-RAP 输出逐元素相等；
- current-state geometry slice 与 trace XY；
- Waymax metrics + geometry + timestep + done 的合并同步；
- BEV cached Cartesian/polar grids 的一致性和只读性。

本环境没有用户服务器上的 Waymax/TensorFlow、WOMD TFRecord、GPU 和 checkpoint，无法给出可信的端到端 wall-clock 倍数。实际运行后应查看输出 JSON 的 `timing.per_decision_s` 和 `fast_waymax_history`。

## 5. 推荐 A/B 测试

使用相同 checkpoint、calibration、WOMD pattern、scenario count、seed、candidate count、label mode 和 output-independent settings：

旧兼容路径：

```bash
--set closed_loop.fast_waymax_history=false
```

新路径：

```bash
--set closed_loop.fast_waymax_history=true
```

比较：

- `timing.per_decision_s.state_history`；
- `timing.per_decision_s.candidate_features`；
- `timing.per_decision_s.policy_selection`；
- `timing.per_decision_s.waymax_step_metrics`；
- 最终 scene ids、每步 selected candidate/macro、metric trace 和 aggregate metrics。

第一次运行包含 cold XLA compilation；公平 benchmark 应先用相同形状预热一次，或比较第二次运行。
