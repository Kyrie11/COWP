# OC-RAP 外部 Baseline Closed-Loop 性能诊断与优化

## 1. 论文与实现对齐

论文的核心不是“再增加一个碰撞风险项”，而是把**可恢复性**变成候选动作的准入条件：

1. 对每个候选前缀生成 recovery-sufficient latent roots；
2. 根据执行前缀后的可见信息构建 observation-equivalence / compatibility kernel；
3. 对每个 root–recovery-option 预测或计算 recovery margin；
4. 用 OC-MERO 在不可区分 root 之间寻找共享恢复选项，并做 lower-tail 聚合；
5. 用校准阈值和 CRISP 选择器，在可部署恢复余量足够时保留 nominal，否则拒绝依赖隐藏分支身份的 oracle artifact。

论文主指标是 FRA、ODG、DRS、NUP；contact/post-contact 还包括 secondary collision、stable stop、yaw-rate、route rejoin 和 harm/severity 指标。

当前数据构建使用 WOMD TFExample 与 Waymax closed-loop。near-contact 和 contact 测试均生成 24 个候选前缀，分别保留最多 8/10 个候选做完整 teacher materialization，并使用 12 个 recovery options。

## 2. 代码中实际实现的外部 Baseline

### Near-contact

- `marc_lite`：semantic multipolicy + policy-conditioned roots + dynamic branch point + expected/tail risk contingency scoring。
- `racp_lite`：多模态 root belief、共享/分支 contingency、概率风险加权。
- `expected_risk_filter`：期望风险过滤。
- `cvar_risk_filter`：tail-CVaR 风险过滤。
- `dro_cvar_filter`：在 CVaR 上加入 ambiguity/dispersion 鲁棒项。
- `predictive_safety_filter`：nominal 可通过 backup/barrier 条件时保持 nominal，否则选择最接近且满足约束的候选。
- `oracle_recovery_filter`：只要求每个 root 各自存在一个恢复选项，不要求不可区分 root 共享同一选项。
- `gameformer_lite`：项目内的轻量级 hierarchical interaction Transformer adapter，不是原始 GameFormer 官方代码的逐行复现。

### Contact/post-contact

- `postimpact_mpc_lite`：有限候选格上的 post-impact MPC 近似，综合稳定性、障碍/伤害、附着约束、SBD 和共享恢复能力。
- `post_crash_braking`：碰撞后稳定停车规则，重点控制末端速度、横摆、硬约束和二次碰撞风险。
- `post_collision_restoration`：基于转向/牵引力形状的轨迹恢复启发式。
- `severity_minimization`：不可避免接触下最小化 severity、残余能量、失稳、ODG，并保留 deployability。

这些 `*_lite` 方法是统一候选格与统一 teacher 标签上的 paper-inspired adapters。论文对比时应在表格或附录中明确这一点，避免被理解为作者官方实现或完整复现。

## 3. Closed-loop 慢在哪里

每个场景的每次重规划当前执行链为：

1. 将 Waymax state splice 回 OC-RAP `RawScenario`；
2. 重建 history / map / route / BEV；
3. 生成 24 个 feature-only candidates；
4. 按 baseline 做 sparse preselection；
5. 对 near-contact 最多 8 个、contact 最多 10 个候选生成完整 teacher 标签；
6. 完整标签内部继续生成 counterfactual futures、latent roots、recovery-option rollout、observation compatibility 和 deployable/oracle recovery；
7. baseline 本身只做一个很小的 NumPy/PyTorch 打分与 argmin/argmax；
8. 执行 Waymax step，并计算 closed-loop metrics。

因此主耗时不是 MARC/RACP/CVaR/PSF 的最终评分，而是步骤 5–6 的 teacher materialization 与 Waymax/JAX rollout。

以默认上限、忽略提前终止计算：

- near-contact 7 个非学习方法：`7 × 50 × 40 × 24 = 336,000` 个 feature candidates，`7 × 50 × 40 × 8 = 112,000` 个完整 teacher candidate labels；
- contact 4 个方法：`4 × 50 × 40 × 24 = 192,000` 个 feature candidates，`4 × 50 × 40 × 10 = 80,000` 个完整 teacher candidate labels；
- GameFormer 仍生成 24 个 feature candidates，但只对选中项/审计项做标签，因此与 teacher-required 方法的成本结构不同。

原脚本还把所有方法串行运行，因此每个方法都重新读取相同 WOMD 数据、重新初始化 Waymax/JAX、重新编译共同 kernel，并独立完成整条回放。

不能把所有方法放进同一个共享状态轨迹中直接复用 teacher labels：不同方法在第一次选出不同动作后，Waymax state、后续 observation、候选集与 latent roots 都会分叉。跨方法共享后续标签会改变 closed-loop 定义。安全的加速方式是**方法级独立并行 + 编译缓存 + 避免无效训练扫描**，而不是把不同方法强制共享后续场景状态。

## 4. 已实施修改

### 4.1 结果保持型加速

- near/contact closed-loop 改为 method-per-process 并行调度；GPU 数不足时自动分批。
- `CUDA_DEVICES` 控制可用 GPU，`MAX_PARALLEL` 控制并发进程数。
- 每个进程分配受控的 OMP/MKL/OpenBLAS/NumExpr 线程，避免 CPU 过度订阅。
- 设置共享 `JAX_COMPILATION_CACHE_DIR`，并增加一次小规模 warm-up，避免所有方法冷启动时重复编译共同 Waymax/teacher kernel。
- 设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免每个独立 JAX 进程默认预占大块显存导致并发 OOM。
- 保留原有候选数、sparse label budget、macro diversity、teacher top-k、打分公式、阈值和随机种子；没有用缩短 horizon、减少候选或减少 roots/options 来“换速度”。
- closed-loop 输出新增分阶段计时：`state_history`、`candidate_features`、`teacher_labels`、`policy_selection`、`audit_labels`、`waymax_step_metrics`。

### 4.2 非学习 Baseline 的训练修正

MARC-lite、RACP-lite、风险过滤器、PSF、oracle filter 和四个 contact 方法没有可学习参数。旧脚本逐方法调用 `train-baseline`，实际只是重复扫描 train/val 数据并写注册信息。默认已跳过此阶段；需要数据 sanity check 时可设置 `DO_TRAIN_NONLEARNED=true`，此时也会使用 `external_baselines.training.validate_dataset=false` 做快速注册。

### 4.3 GameFormer-lite 输入契约修正

旧训练路径将 `m_star`、root signatures/features、root probabilities/validity 等 teacher branch tensors 作为模型输入；feature-only online closed-loop 无法获得这些量，只能使用零/均匀占位。这造成：

- offline 输入含 teacher 信息，存在标签泄漏风险；
- train/offline 与 online closed-loop 输入分布不一致；
- 旧 checkpoint 的结果不能作为严格 deployable baseline。

新增 `external_baselines.model.use_teacher_branch_context=false`，训练、offline evaluation 和 closed-loop 统一只使用部署时可获得的 scene/history/prefix/topology features。checkpoint 写入 `input_contract`，运行脚本默认拒绝旧的 legacy checkpoint。

这项修正会改变 GameFormer-lite 的学习结果，属于评估有效性修复，不属于“数值完全不变”的性能优化。七个 near-contact 非学习方法和四个 contact 方法的选择逻辑未改。

## 5. 推荐运行方式

### 5.1 仅重新训练修正后的 GameFormer-lite（一次）

```bash
cd /path/to/OC-RAP-optimized

CUDA_DEVICES=0,1 \
TRAIN_NUM_GPUS=2 \
DO_TRAIN_GAMEFORMER=true \
DO_TRAIN_NONLEARNED=false \
DO_OFFLINE=false \
DO_CLOSED_LOOP=false \
bash scripts/run_near_contact_external_baselines.sh
```

单卡时将 `CUDA_DEVICES=0 TRAIN_NUM_GPUS=1`。

### 5.2 Near-contact：offline + 所有 closed-loop 方法并行

4 张同型号 GPU：

```bash
CUDA_DEVICES=0,1,2,3 \
MAX_PARALLEL=4 \
DO_TRAIN_GAMEFORMER=false \
DO_TRAIN_NONLEARNED=false \
DO_OFFLINE=true \
DO_CLOSED_LOOP=true \
bash scripts/run_near_contact_external_baselines.sh
```

8 张同型号 GPU 可一次启动全部 8 个方法：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PARALLEL=8 \
DO_TRAIN_GAMEFORMER=false \
DO_TRAIN_NONLEARNED=false \
DO_OFFLINE=true \
DO_CLOSED_LOOP=true \
bash scripts/run_near_contact_external_baselines.sh
```

只跑最终 closed-loop：

```bash
CUDA_DEVICES=0,1,2,3 \
MAX_PARALLEL=4 \
DO_TRAIN_GAMEFORMER=false \
DO_TRAIN_NONLEARNED=false \
DO_OFFLINE=false \
DO_CLOSED_LOOP=true \
bash scripts/run_near_contact_external_baselines.sh
```

### 5.3 Contact：所有方法并行，无需训练

```bash
CUDA_DEVICES=0,1,2,3 \
MAX_PARALLEL=4 \
DO_TRAIN_NONLEARNED=false \
DO_OFFLINE=true \
DO_CLOSED_LOOP=true \
bash scripts/run_contact_external_baselines.sh
```

只跑最终 closed-loop：

```bash
CUDA_DEVICES=0,1,2,3 \
MAX_PARALLEL=4 \
DO_TRAIN_NONLEARNED=false \
DO_OFFLINE=false \
DO_CLOSED_LOOP=true \
bash scripts/run_contact_external_baselines.sh
```

### 5.4 单 GPU/双 GPU 注意事项

单 GPU 不建议把 `MAX_PARALLEL` 设为 8。JAX preallocation 虽已关闭，但多个 teacher rollout 进程会争用显存和算力，wall-clock 可能反而更慢。推荐：

- 1 GPU：`CUDA_DEVICES=0 MAX_PARALLEL=1`
- 2 GPU：`CUDA_DEVICES=0,1 MAX_PARALLEL=2`
- 4 GPU：`CUDA_DEVICES=0,1,2,3 MAX_PARALLEL=4`

若显存充足且单方法 GPU 利用率低，可再试每卡 2 进程，例如 4 GPU + `MAX_PARALLEL=8`，但必须先比较 timing 与 OOM 情况。

## 6. 如何确认真正瓶颈和加速比

每个 `closed_loop_<method>.json` 现在含：

```json
{
  "timing": {
    "scene_wall_sum_s": 0.0,
    "totals_s": {
      "state_history": 0.0,
      "candidate_features": 0.0,
      "teacher_labels": 0.0,
      "policy_selection": 0.0,
      "audit_labels": 0.0,
      "waymax_step_metrics": 0.0
    },
    "per_decision_s": {},
    "measured_fraction": 0.0
  }
}
```

脚本还生成 `closed_loop_summary.json`。建议比较：

1. 单方法串行基准：`MAX_PARALLEL=1`；
2. 同样参数下 2/4/8 方法并行的总 wall-clock；
3. `teacher_labels / scene_wall_sum_s`；
4. warm cache 的第二次运行与冷 cache 第一次运行；
5. GPU 利用率、显存和 CPU load。

预期 `policy_selection` 占比应很小；若不是，说明某个 baseline policy 实现出现了意外 Python 循环或张量搬运。若 `teacher_labels` 占主导，则当前诊断成立。

## 7. 验证状态

- 全量单元测试通过：61 passed。
- 新增 GameFormer input-contract 测试。
- 新增 closed-loop timing aggregation 测试。
- 两个运行脚本通过 `bash -n`。
- Python 文件通过 compile check。
- 当前环境没有用户服务器上的 WOMD TFRecord、Waymax GPU runtime 和 baseline checkpoint，因此未声称实际端到端加速倍数；应在目标机器上以新增 timing 字段测量。
