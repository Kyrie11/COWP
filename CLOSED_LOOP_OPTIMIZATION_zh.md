# COWP 闭环评估加速分析与代码修改说明

## 1. 论文核心目标与算法理解

论文关注一种传统“无碰撞”指标难以识别的交互式规划失效：**false-safe planning（伪安全规划）**。候选轨迹本身可能无碰撞，但它之所以安全，是因为其他交通参与者被迫急刹、突然让行、放弃路权或让出合法间隙，从而替 ego 承担冲突消解负担。

COWP 将这一问题从“礼貌性软代价”提升为**可行性约束**：候选轨迹只有在所有关键交通参与者仍保留足够丰富的、低负担且安全的响应集合时，才满足 non-coercive feasibility（非胁迫可行性）。

整体 pipeline 为：

1. 构建 burden-oriented interaction graph，识别潜在受压迫对象、冲突区间和负担通道。
2. 生成 ego 候选轨迹及宏观意图。
3. 为关键参与者构造 observational、ego-neutral、priority-preserving 等 counterfactual natural alternatives。
4. 在每条 ego 候选条件下构建 safe response set，并计算低负担响应、option preservation、ceding burden 等量。
5. 形成 coercion witness；若高置信 witness 表明候选依赖高负担让行，则将其作为可行性缺陷直接拒绝。
6. 对剩余候选按 conventional safety、NCF 和 ego utility 的 hard-first 逻辑选择；无候选存活时执行保守 fallback。

训练按 response、witness、planner 阶段组织。主要闭环指标包括 CR、EP、FSR、CBS、OPR、HBCR，以及 witness localization / mechanism token 等诊断指标。

## 2. 数据与评估流程理解

数据构建同时使用 WOMD `scenario` proto 和 `tf_example`：

- scenario proto：建立索引并构造 natural/response/witness/planner 等监督标签。
- tf.Example：构建可直接训练与 Waymax 初始化的 tensor cache。
- 选定候选可预先在 Waymax 中 replay，并把 collision/offroad 等 outcome 附加到 cache。
- 真正的 online Waymax closed-loop evaluation 仍直接读取 validation tf.Example 并逐步推进模拟器，不依赖训练候选 replay cache 才能运行。

当前 cache 分析显示 validation 共 5013 个场景，核心阶段字段完整；每场景平均约 50.63 个有效候选，其中固定选取并成功 replay 12 个。附加 outcome 覆盖约为所有有效候选的 23.7%，适合作为部分诊断或辅助监督，而不是完整 online 闭环环境；log-divergence 当前无有限有效值。

## 3. 原实现中的主要性能瓶颈

### 3.1 learned_offline 重复做相同重计算

`run_cowp_v4.sh` 原来为每个方法分别启动一次 `04_eval_closed_loop`：

- 重复加载同一个 checkpoint；
- 重复扫描完整 validation cache；
- 重复执行同一模型前向；
- 方法之间真正不同的部分主要只是候选选择规则和聚合。

对 5 个 learned-offline 方法，这部分近似重复 5 次。

### 3.2 Waymax shard 过滤发生得太晚

原流程先从 TFExample 解析并构造全部 `SimulatorState`，随后才按照：

```python
scenario_index % num_shards == shard_index
```

丢弃不属于当前 shard 的场景。两进程分片时，每个进程仍为完整数据流承担 TFExample decode / state construction，然后丢弃约一半，造成显著重复 CPU 和主机内存开销。

### 3.3 每场景清空 JAX cache

原运行脚本固定传入 `--clear-accelerator-cache`，每个场景后调用 `jax.clear_caches()`。这会使 Waymax/JAX 已编译函数无法跨场景复用，容易触发持续重编译。该选项只适合 OOM 排查，不应作为正常测速或正式评估默认值。

### 3.4 Waymax env/metric 缺少稳定的 JIT 复用

环境 `reset/step` 与标准 metric compute 处于 eager 调用路径，跨场景的稳定 shape 计算没有充分复用已编译程序。

### 3.5 每步重复 device-to-host 同步

- 每个标准 metric 各自读取 SDC index 并 `device_get`；
- metric 结果逐个同步到 host；
- policy diagnostics 中存在多次 `.detach().cpu().item()`，在 GPU policy 下会形成大量细粒度同步点。

这些同步不会改变算子量，但会破坏流水并增加每个 Waymax step 的固定开销。

## 4. 已完成的代码优化

### 4.1 单次前向同时评估多个 learned-offline 方法

涉及：

- `cowp/waymax_eval/rollout.py`
- `cowp/scripts/04_eval_closed_loop.py`
- `run_cowp_v4.sh`

新增 `--methods method_a,method_b,...`，所有方法共享：

- 一次 checkpoint 加载；
- 一次 validation DataLoader 遍历；
- 一次模型前向预测。

每种方法仍调用原来的候选选择与指标聚合逻辑。运行脚本会将共享结果拆回原来的每方法 JSON 文件名，保持下游表格或分析脚本兼容。

同时 learned-offline DataLoader 新增：

- `--num-workers`
- `--prefetch-factor`
- `--pin-memory/--no-pin-memory`
- persistent workers 与 non-blocking device transfer（适用时）

默认仍使用 CPU learned evaluation，以降低数值后端改变导致边界候选翻转的风险；batch 从 1 提升为 32，worker 默认 4。

### 4.2 在原始 TFExample 层提前分片

涉及：

- `cowp/data/parse_tfexample.py`
- `cowp/waymax_eval/dataloader.py`
- `cowp/waymax_eval/rollout.py`

新增 raw-record sharded iterator，在 parse/decode/SimulatorState construction 之前应用原来的全局取模规则。

保留：

- 原全局 `scenario_index`；
- 原 `global_index % num_shards == shard_index` 分配；
- 每 shard 的 `num_scenarios` 含义；
- 合并方式和指标口径。

另外补充支持 Waymax 常见的 `file@N` shard 表达式扩展。

### 4.3 缓存并 JIT Waymax reset/step

新增 `_WaymaxEnvOps`：

- 对固定环境配置缓存 JIT 后的 `reset` 和 `step`；
- 跨场景复用；
- 若当前 Waymax/JAX 版本不支持该 JIT 路径，自动回退到原 eager 调用，不改变行为。

CLI：

```bash
--jit-waymax-env / --no-jit-waymax-env
```

默认开启。

### 4.4 JIT 标准指标并合并 host 同步

`WaymaxStandardMetricAccumulator` 现在：

- 每一步只解析一次 SDC index；
- 可按 metric 缓存 JIT compute；
- metric 失败时逐项自动回退 eager；
- 将多个 metric 结果组织后一次性 `device_get`，减少同步次数；
- 原 episode-level any/mean/max 聚合规则未改。

CLI：

```bash
--jit-waymax-metrics / --no-jit-waymax-metrics
```

默认开启。

### 4.5 批量读取 policy diagnostics

`policy_wrapper.py` 将多个标量 reduction 先在 device 上组成小 tensor，再一次传回 host；候选存在性和 fallback 标志同样批量获取。候选 mask、阈值、hard gate、fallback 与最终 index 选择逻辑未改。

### 4.6 不再默认清空 accelerator cache

`run_cowp_v4.sh`：

```bash
CLEAR_ACCELERATOR_CACHE=0
```

仅在显式设置为 1 时传入 `--clear-accelerator-cache`。同时设置持久化：

```bash
JAX_COMPILATION_CACHE_DIR="$OUT_ROOT/jax_compilation_cache"
```

便于同一配置的后续进程/重复实验复用编译产物。

### 4.7 新增结果等价比较器

新增：

```bash
python -m cowp.scripts.23_compare_eval_outputs \
  --reference old.json \
  --candidate new.json \
  --atol 1e-6 \
  --rtol 1e-6
```

默认只忽略性能实现元数据，递归比较其余 JSON 结构与数值，用于验证优化前后评估输出一致。

## 5. 未改变的内容

本修改没有改变：

- scenario 选择的全局 modulo 规则；
- rollout horizon、动作模式和 Waymax dynamics；
- 方法的候选打分、hard gate、threshold、fallback；
- CR/EP/FSR/CBS/OPR/HBCR 与 Waymax standard metric 的定义；
- episode 内和 episode 间聚合方式；
- 原有单方法 CLI；
- 原有每方法输出文件接口。

新 JIT / prefilter 路径均提供禁用开关和回退路径。

## 6. 推荐运行方式

只使用已有 checkpoint 重新评估：

```bash
cd COWP_closed_loop_optimized
RUN_TRAIN=0 \
CKPT=/path/to/cowp_planner_best.pt \
OUT_ROOT=outputs/cowp_v4_fast \
bash run_cowp_v4.sh
```

按机器调整 learned-offline 数据加载：

```bash
LEARNED_EVAL_BATCH=64 \
LEARNED_EVAL_WORKERS=8 \
LEARNED_EVAL_PREFETCH=2 \
RUN_TRAIN=0 CKPT=/path/to/checkpoint.pt \
bash run_cowp_v4.sh
```

显存不足排查时才启用逐场景清 cache：

```bash
CLEAR_ACCELERATOR_CACHE=1 ... bash run_cowp_v4.sh
```

它会明显拖慢评估。

若某个 Waymax/JAX 版本兼容性异常，可逐项回退：

```bash
PREFILTER_WAYMAX_SHARDS=0 \
JIT_WAYMAX_ENV=0 \
JIT_WAYMAX_METRICS=0 \
... bash run_cowp_v4.sh
```

## 7. 建议的严格 A/B 一致性验证

先在旧代码和新代码中使用相同 checkpoint、配置和场景数运行小规模评估。例如每个方法 20–50 个场景，并固定相同 shard 数与 action mode。

之后比较：

```bash
python -m cowp.scripts.23_compare_eval_outputs \
  --reference /path/to/old_result.json \
  --candidate /path/to/new_result.json \
  --atol 1e-6 --rtol 1e-6
```

建议同时记录：

```bash
/usr/bin/time -v <evaluation command>
```

以及 GPU/JAX profiler，分别报告：

- 启动与首次 JIT 时间；
- steady-state 每场景时间；
- TFExample/state construction 时间；
- policy forward 时间；
- env step 时间；
- metric 时间；
- peak RSS / GPU memory。

## 8. 测试状态与限制

- 静态编译和 shell syntax 检查通过。
- 完整项目测试：**45 passed**。
- 新增测试覆盖：提前分片的全局索引一致性、`@N` 展开、closed-loop 场景选择、标准指标 SDC 读取与 CR 聚合、结果比较器。
- 当前执行环境没有安装完整 Waymax/WOMD 数据与对应 GPU 运行环境，因此无法在这里给出可信的真实 wall-clock 加速倍数。优化依据来自代码路径与重复工作量分析；最终速度应在用户实际 Waymax 环境中按上述 A/B 方法测量。

## 9. 预期收益性质

- learned-offline：模型加载、cache 扫描和神经网络前向从“每方法一次”变成“所有方法共享一次”，该部分的理论重复量由方法数决定；当前 5 方法配置中，重计算部分最多可接近减少到原来的约 1/5，整体速度还受 JSON 聚合和 I/O 占比影响。
- 两 Waymax shards：原来两个进程都构造所有 `SimulatorState`；提前分片后，合计 state construction 约从 2 倍全量降至 1 倍全量。
- 在线 steady-state：避免逐场景清 JAX cache，并复用 JIT env/metric，可消除持续重编译与大量细粒度同步；实际收益取决于 JAX/Waymax 版本、设备、shape 稳定性和 policy 占比。
