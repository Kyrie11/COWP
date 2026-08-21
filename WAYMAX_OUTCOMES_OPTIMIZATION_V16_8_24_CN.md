# COWP v16.8.24 Waymax outcomes 优化说明

## 结论

1. v24 `outcomes` 的算法语义总体正确：对 tensor cache 中选中的 ego candidates 逐条进行 Waymax closed-loop replay，当前 authoritative 配置使用 `absolute_xy_yaw`、`metric_set=safety`，逐 simulator step 计算 Overlap/Offroad，并把 `collision/offroad/rollout_valid` 附加回 candidate 维度。
2. 当前运行使用 `--state-source cache`。v24 tensor cache 已把匹配场景的 WOMD tf.Example feature 以 `womd__*` 存进 NPZ；replay 从这些数组直接恢复 `SimulatorState`。因此 outcomes 阶段不需要再次读取原始 WOMD `uncompressed/scenario` 或 `uncompressed/tf_example`。
3. 原 v24 shell 的“8 小时没有 `_waymax/*.npz`”是可解释的工程行为：每个 split 先把所有 Waymax replay 完整跑完写 JSONL，等待所有 shard 结束后才运行 attach。并且 `replay_split` 被放进 process substitution 再通过 `grep` 取 JSONL 路径，导致 worker 的 tqdm/tee 输出被 grep 消费，终端几乎看不到进度。
4. 本补丁把每个 scene 完成后的 outcomes 立即原子写成一个 `_waymax/*.npz`，JSONL 仍是 source of truth；最后仍运行原 `scripts/12` 做全量 reconciliation 和 `scripts/14` verifier，因此增量写入不会替代最终一致性检查。
5. 不使用 `sampled/adaptive/final` metric 模式做 full authoritative build，也不提高 `done_check_interval`，因为这些选项可能漏掉瞬时 collision/offroad 或改变 early termination；full run 固定 `metric_eval_mode=step`, `metric_eval_interval=1`, `done_check_interval=1`。

## 修改文件

- `cowp/waymax_eval/outcome_attach.py`
  - 新增单 scene outcomes attach。
  - temp NPZ + `os.replace`/`Path.replace` 原子提交，读者不会看到半写文件。
  - 与 `scripts/12_attach_waymax_candidate_outcomes.py` 使用相同 Waymax 字段。
- `cowp/waymax_eval/candidate_replay.py`
  - 每个 scene JSONL 落盘后，把当前完整 scene rows 交给单线程后台 I/O worker 写 NPZ。
  - bounded pending queue，防止写盘落后时无限堆积内存。
  - resume 时先读取已修复 JSONL 中该 scene 的旧 rows，避免续跑时丢掉之前成功 candidate。
  - `profile_detail=scene`：保留每 scene profile，但关闭历史实现每 candidate/step 的细粒度 `perf_counter()`，不改变 rollout/metric 数值。
  - 可选 `jit_env_step/jit_env_reset`，JIT 出错永久 fallback 到 eager。
- `cowp/scripts/13_replay_waymax_candidates.py`
  - 新增 `--profile-detail`, `--attach-output-dir`, `--attach-max-pending`, `--progress-desc`, `--jit-env-reset`。
- `cowp/waymax_eval/rollout.py`
  - `absolute_xy_yaw` 现在强制要求 `StateDynamics`；不再在缺失时错误 fallback 到 action contract 不同的 `DeltaGlobal`。这是 fail-fast correctness fix，不改变正常 v24 环境下的输出。
- `ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh`
  - 去掉 `mapfile < <(replay_split | grep ...)`，tqdm 不再被吞。
  - 双 GPU 各一个 worker/shard；默认拒绝 GPU oversubscription。
  - replay 期间逐 scene 生成 `_waymax/*.npz`。
  - full authoritative semantics 显式锁死为 step metric / done every step。
  - full-run profile 默认 scene level。
  - JIT 默认关闭，必须先过 A/B gate 再打开。
- `CHECK_WAYMAX_JIT_EQUIVALENCE_V16_8_24_CN.sh`
  - 同一 deterministic scene/candidate 集分别 eager/JIT replay。
  - 要求 `(scenario_id,candidate_index)` key 完全一致，`rollout_valid/collision/offroad/steps` 全部一致，才允许 full run 开 JIT。
- `WATCH_WAYMAX_OUTCOMES_V16_8_24_CN.sh`
  - 随时查看每 split JSONL row 数、已生成 NPZ 数和最新日志尾部。

## 推荐运行

### 1. 先检查旧任务是否真的在推进

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
bash WATCH_WAYMAX_OUTCOMES_V16_8_24_CN.sh
```

原实现即使没有 NPZ，`$COWP_ROOT/waymax_replay_v16_8_24/*.jsonl` 也应在每个完成 scene 后增长。如果 JSONL 长时间也是 0 行，重点看相应 `.log`、GPU utilization、首 scene 是否在 JAX compile 或报错，而不是把“没有 NPZ”本身判定为 deadlock。

### 2. 安全基线：不开新 JIT，直接续跑

保留已经产生的 replay JSONL，不要删除。`scripts/13` 的 resume repair 会保留成功 row、重试失败/损坏尾行。

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export WAYMAX_GPUS=0,1
export REPLAY_NUM_SHARDS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MAX_REPLAY_CANDIDATES=24
export REPLAY_HORIZON=80
export WAYMAX_PROFILE_DETAIL=scene
export WAYMAX_JIT_ENV_STEP=0
export WAYMAX_JIT_ENV_RESET=0
bash NEXT_EXECUTION_V16_8_24_CN.sh outcomes
```

### 3. 要进一步加速，先做 JIT 等价性 gate

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export WAYMAX_JIT_CHECK_SCENES=16   # 建议正式前再用 32/64 做一次更强 gate
export WAYMAX_JIT_CHECK_GPU=0
bash CHECK_WAYMAX_JIT_EQUIVALENCE_V16_8_24_CN.sh
```

只有 `pass=true` 后：

```bash
export WAYMAX_JIT_ENV_STEP=1
export WAYMAX_JIT_ENV_RESET=1
bash NEXT_EXECUTION_V16_8_24_CN.sh outcomes
```

## 不建议为了速度修改的参数

- 不降低 `MAX_REPLAY_CANDIDATES=24`：会直接改变 candidate outcome 覆盖率。
- 不降低 `REPLAY_HORIZON=80`：会漏掉后段安全事件。
- 不把 authoritative run 改成 `metric_eval_mode=sampled/adaptive/final`：Waymax OffroadMetric 是当前 timestep 的 metric；采样会漏掉短暂越界，Overlap 也可能同理漏掉瞬时碰撞。
- 不把 `done_check_interval` 从 1 提高：虽然能减少 host sync，但可能多执行本应终止后的 step；除非另做全量/强 A/B 等价性验证。
- 不用 2 个 replay process 争同一张 GPU。两张 A30 时 `REPLAY_NUM_SHARDS=2` + `WAYMAX_GPUS=0,1` 是合理布局。

## 关于删除原 WOMD

当前 outcomes 能继续的前提是三个 core tensor cache 完整存在，而且每个 NPZ 含 `womd__*` Waymax-required feature。你已经在构建 tensor cache 时走了 `--require-waymax-ready`，v24 replay 又固定 `--state-source cache`，所以当前 outcomes 不依赖原始 WOMD TFRecord。

但删除 WOMD 会影响后续：重新 build index/labels/tensor cache；使用 `--state-source tfexample` 的 replay；修复缺失/损坏 core cache；以及任何需要重新扫描 WOMD split/index 的流程。建议把当前 core tensor cache 和 scene-id manifest 当成现在唯一可恢复输入，立即做完整性检查和备份。

## 本地静态/单元测试

本补丁在不安装 Waymax/GPU 的当前环境完成了：

- Python `py_compile`：PASS
- shell `bash -n`：PASS
- 新增 incremental attach/resume/JIT-fallback/shell invariant tests：4/4 PASS
- 原 v24 rebuild-ready + Waymax diagnostics/rollout tests：12/12 PASS

真正的 GPU/JAX 数值等价性无法在当前无你同款 Waymax/CUDA 环境中代替你验证，因此 JIT 默认关闭，并提供了上述 A/B gate。
