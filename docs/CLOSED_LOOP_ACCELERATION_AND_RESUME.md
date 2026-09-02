# OC-RAP closed-loop 加速与断点续跑说明

## 1. 论文与代码逻辑对齐

论文的核心不是单纯预测未来风险，而是把**观测一致性约束下的可恢复性**作为规划原语：

1. 从当前历史状态生成短时域候选前缀；
2. 对每个前缀构造潜在 future/root 与恢复选项；
3. 根据后前缀观测的一致性形成 observation kernel；
4. 计算 root-option recovery margin，并通过 OC-MERO 得到 deployable recoverability、oracle recoverability 和 oracle–deployable gap；
5. 使用校准阈值选择可执行前缀，并在 Waymax 中执行后重新规划；
6. 评价 FRA、DRS、ODG、NUP，以及接触后可部署性、二次碰撞等指标。

`ocrap.cli closed-loop` 是真正的 receding-horizon 流程：Waymax `reset` 一次，每个规划时刻从更新后的 `SimulatorState` 重建历史、生成候选、选择动作、`env.step`，然后继续重规划。优化不能通过减少候选、减少闭环步数、放大重规划间隔或删除审计标签来“提速”，否则会改变闭环轨迹或指标定义。

## 2. 原实现的主要低效点

### 2.1 每个候选重复计算相同场景特征

`_select_prefix -> predict_samples -> sample_to_feature` 对同一 replan 的每个候选重复执行：

- agent history 的邻车筛选与距离排序；
- BEV 均值/方差；
- route、map polyline、dynamic map 的统计与 flatten；
- 大型共享数组的重复遍历。

这些字段在同一重规划时刻对全部候选完全相同，只有 prefix state/control、macro、utility/hard/harm 等候选字段不同。

**修改：**新增 `samples_to_feature_matrix(..., shared_scene=True)`，共享场景部分只计算一次，再与每个候选的 prefix 部分拼接。测试使用 `assert_array_equal` 验证与逐候选旧路径逐元素完全一致。

本机合成 CPU micro-benchmark（16 candidates，较大 history/BEV/map，3200 feature rows）中，仅 feature extraction 部分从 2.90 s 降到 0.34 s，约 8.46×。这不是整体 closed-loop 加速比；整体速度仍取决于 Waymax/JAX、候选生成和 teacher rollout。

### 2.2 每步重复构造不变配置

原 `_rollout_one_scene` 在每个 closed-loop step 都重新复制 `eval_cfg`、重建 `dataset_quality`、解析 `num_roots/num_options`。

**修改：**这些配置在 scene rollout 内不变，全部移到 step loop 外。不会改变任何数值或分支。

### 2.3 审计标签被完整序列化多次

`selected_topk` 审计在标签已经算完后，对选中样本和每个候选多次调用 `DatasetSample.to_npz_dict()`。该函数面向数据集落盘，会复制 history、map、BEV、future metadata、diagnostics 等大量闭环审计根本不使用的数据。

**修改：**新增 `_sample_to_audit_dict`，只取审计实际使用的：

- `root_probs/root_valid`；
- `m_star/option_valid`；
- `r_dep_star/r_orc_star/oracle_gap_star`；
- `candidate_index`。

同一 labeled sample 只构造一次轻量字典。

### 2.4 闭环审计执行了未使用的 regime 标注

`build_labeled_samples_for_candidate_indices` 在生成 teacher labels 后调用 `assign_regimes`，但 closed-loop 的 selected/selected_topk 审计不读取这些 regime labels。

**修改：**函数新增向后兼容参数 `assign_regime_labels=True`；闭环审计传 `False`，训练/数据集构建默认行为不变。

同时复用 feature-only sample 已构造的 recovery options 与 option-valid mask，避免重复创建。

### 2.5 partial 文件每个场景重聚合、重写全部历史

原代码每完成一个 scene：

1. 对全部已完成 scene 再聚合；
2. 把完整 `scene_results` 重新写入 `.json.partial`。

随着场景数增长，累计序列化/I/O 接近二次方增长；而直接写目标文件在进程被杀时还可能留下损坏 JSON。

**修改：**

- 每个完成 scene 追加到 `*.scenes.jsonl`，每条记录独立；
- 完整聚合 `.json.partial` 默认每 4 个 scene 写一次；
- 最后不足 4 个时也强制写最终 partial；
- JSON snapshot 使用同目录临时文件 + `os.replace` 原子替换；
- `closed_loop.resume_fsync=true` 时可要求每条 journal/snapshot 落盘同步，默认关闭以减少同步 I/O。

## 3. 断点续跑设计

### 3.1 断点粒度

支持**完成的 scene 或 bucket target 粒度**续跑：

- 普通 closed-loop：按 `scene_id` 判断完成；
- bucket closed-loop：优先按 `target_key`，否则按 `bucket + scene_id + target_time_index`；
- 已完成的 rollout 不再执行；
- 中断时正在执行的 scene 没有完整记录，因此重启后只重跑该 scene。

不做 step-level `SimulatorState` 恢复，因为 JAX/Waymax simulator state、环境配置、PRNG 和编译缓存需要一起精确持久化；不完整的 step checkpoint 很容易改变轨迹。scene-level 是既安全又不改变评价的断点边界。

### 3.2 生成的文件

以输出 `closed_loop_safe_fast_v39.json` 为例：

- `closed_loop_safe_fast_v39.json`：最终聚合结果；
- `closed_loop_safe_fast_v39.json.partial`：阶段性完整聚合快照；
- `closed_loop_safe_fast_v39.json.scenes.jsonl`：每完成一个 rollout 追加一条，续跑的主增量日志；
- `closed_loop_safe_fast_v39.json.progress.json`：小型状态文件，含 completed/requested/current scene；
- `.log`：脚本已改为 `tee -a`，重启不会覆盖旧日志。

### 3.3 兼容误关前留下的旧 partial

旧代码虽然不会读取 `.json.partial`，但它通常已经包含完整的 `scenes` 列表。新代码默认：

```text
closed_loop.resume=true
closed_loop.resume_allow_legacy_partial=true
```

因此可直接读取旧的、有效 JSON 格式的 `.json.partial` 并跳过已完成 scene。旧文件没有 run fingerprint，只能校验 method 和 bucket dataset，结果中会记录 legacy warning。首次用新代码继续后，新 journal 和新 partial 都带 fingerprint。

如果旧 `.json.partial` 恰好在直接写入时被杀而损坏，新代码会忽略无效 JSON；这种情况下只能依赖新 journal（旧版本没有）或重跑。

### 3.4 防止错误续跑

新文件写入 `run_fingerprint`，覆盖：

- WOMD dataset pattern；
- checkpoint 路径、大小和修改时间；
- method、bucket dataset；
- 所有会影响结果的配置。

只排除 progress/resume/fsync/partial 周期等纯持久化配置。fingerprint 不一致时默认拒绝续跑。只有人工确认等价后才可设置：

```bash
--set closed_loop.resume_force=true
```

不建议常规使用。

## 4. 推荐运行方式

继续你已经中断的同一 `RUN`，保持原来的 `BASE_RUN`、`RUN` 和全部 selector/audit 参数不变，只增加：

```bash
CL_RESUME=true \
CL_PARTIAL_EVERY=4 \
CL_RESUME_FSYNC=false \
BASE_RUN=runs/ocrap_v39_ocrac_balanced \
RUN=runs/ocrap_v39_ocrac_balanced_eval \
RUN_SCALAR_BASELINES=1 \
CONTACT_BRAKE_RESCUE=false \
CONTACT_BRAKE_TAIL=true \
CONTACT_TAIL_BYPASS_PCD_GAIN=true \
CONTACT_TAIL_CHALLENGE_COOLDOWN_BYPASS=true \
CONTACT_TAIL_MAX_CONSECUTIVE=2 \
RUN_NEAR_TAIL=true \
RUN_NEAR_CHALLENGE=true \
NEAR_CHALLENGE_MACROS=brake,yield,merge,stabilize \
NEAR_TAIL_BYPASS_PCD_GAIN=false \
NEAR_TAIL_MAX_CONSECUTIVE=1 \
AUDIT_TARGETS=32 \
AUDIT_LABELS=384 \
AUDIT_MAX_ROLLOUTS=12 \
bash scripts/run_ocrap_v39_ocrac.sh
```

脚本将使用相同输出名，因此自动检测对应 partial/journal。要强制从头跑：

```bash
CL_RESUME=false bash scripts/run_ocrap_v39_ocrac.sh
```

## 5. 如何查看进行到哪里

例如 contact v39 audit：

```bash
python -m json.tool \
  "$RUN/audit_contact_selected_topk_v39_v39.json.progress.json"

tail -n 1 \
  "$RUN/audit_contact_selected_topk_v39_v39.json.scenes.jsonl"

python - <<'PY'
import json, os
run=os.environ['RUN']
p=f"{run}/audit_contact_selected_topk_v39_v39.json.partial"
d=json.load(open(p))
print("completed scenes:", len(d.get("scenes", [])))
print("last scene:", d.get("scenes", [{}])[-1].get("scene_id"))
print("last target:", d.get("scenes", [{}])[-1].get("target_key"))
PY
```

## 6. 对评价结果的影响边界

以下内容**未修改**：

- Waymax reset/step/replan 语义；
- `max_steps`、candidate count、replan interval；
- 模型输入排列与数值；
- selector、gamma、macro 约束；
- teacher/root/recovery rollout 数学；
- FRA、DRS、ODG、NUP、PCD、Waymax metrics 的计算；
- audit target/top-k/label budgets。

因此优化目标是减少重复 CPU 特征处理、无用序列化、无用 regime 后处理和二次方 partial I/O，而不是近似计算。

`selected_topk` 的真正主耗时——counterfactual futures、recovery rollouts、root clustering 和 OC-MERO labels——仍然存在，因为删除或降低它们会改变 audit 指标。safe `label_mode=fast` 通常会更明显地受益于共享特征优化；near/contact selected_topk 的整体加速上限则受 teacher rollout 占比限制。

## 7. 验证结果

在当前代码树执行：

```bash
PYTHONPATH=src pytest -q
```

结果：`52 passed`。

新增测试覆盖：

- shared-scene feature 与旧逐候选 feature 逐元素相等；
- scene journal 中断后续跑，只执行未完成 scene；
- 旧 `.json.partial` 兼容续跑；
- progress 最终状态为 complete。

由于当前环境没有你的 WOMD 文件、Waymax GPU/JAX 编译环境和 checkpoint，未虚构整体 closed-loop wall-clock 加速比。建议在你的机器上比较相同输出配置的 `closed_loop_step`/`scene_done` 时间，并分别统计 safe-fast 与 selected_topk audit。
