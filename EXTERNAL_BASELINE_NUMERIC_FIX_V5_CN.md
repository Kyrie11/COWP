# 外部 Baseline 数值稳定性与闭环执行修复（V5）

日期：2026-08-27

## 1. 本次修复范围

本次只修改外部 baseline 的训练、WOMD/COWP adapter、loss 数值契约、PDM-Closed-style 规则评分，以及外部 baseline 的 Waymax 执行包装；不修改 COWP 主算法逻辑，不修改用户现有训练/Waymax 命令，也不修改默认输出目录：

`outputs/external_sota5_v16_8_33`

学习型 baseline：GameFormer、DTPP、PLUTO-style、PlanT2-style。
规则 baseline：PDM-Closed-style（同一 rule 模块中的 IDM/Lattice/Frenet 路径也获得相同的数值输入保护）。

## 2. 对论文与 baseline 接口的理解

论文的核心问题是 false-safe planning：ego 轨迹表面无碰撞，但安全依赖其他道路参与者承担高 burden 的让行/急刹/放弃合法 gap。COWP 将这种“safety-by-coercion”从 soft courtesy cost 提升为 non-coercive feasibility 条件，通过 burden-oriented interaction、counterfactual natural alternatives、ego-conditioned safe response sets、root-conditioned option preservation / tail burden 与 coercion witness 做候选过滤和解释。

外部 baseline 训练链路刻意不读取 COWP witness、OPR、burden、false-safe、NCF 标签，而是在同一 WOMD 场景、历史、80-step horizon 和 Waymax evaluator 下比较。代码中的 reference metadata 将 5 个方法标记为跨域/clean-room adaptation，而不是所有方法都声称为作者原生 benchmark 实现。

## 3. 本次 GameFormer 崩溃的直接原因

原始 `cowp/scripts/20_train_external_baseline.py` 在 backward 后直接执行：

```python
torch.nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)
```

PyTorch 的 global L2 norm 通常按梯度 dtype 做归约。即使**每个 FP32 梯度元素都是有限值**，平方和仍可能在 FP32 中溢出为 `Inf`，于是 `clip_grad_norm_` 抛出：

`The total norm ... is non-finite`

项目主训练器 `cowp/scripts/03_train.py` 已经有针对这一情况的 FP64 norm fallback，但外部 baseline trainer 没有复用该保护，因此存在实现不一致。

不过，单凭旧日志不能百分之百证明 batch=260 一定只是 norm-reduction overflow，因为原异常处理存在第二个 bug：

```python
optimizer.zero_grad(set_to_none=True)
bad_grad = _first_nonfinite_gradient(model)
```

它先清空梯度，再查询第一个非有限梯度，所以日志中的：

`first_nonfinite_gradient=None`

本身没有诊断价值。旧日志能确定的是：该 batch 的 forward/loss 检查通过，错误发生在 backward 后的梯度裁剪阶段；不能仅凭 `None` 区分“真实 NaN/Inf 梯度元素”和“所有元素有限但 FP32 global norm 溢出”。

修复后会严格区分两者：

- 若所有梯度元素有限，只是 FP32 norm 归约溢出：用 FP64 重新计算同一个 global L2 norm，并按同样的 max-norm 比例裁剪，然后继续训练。
- 若任一梯度元素真的含 NaN/Inf：保持 fatal-by-default，并在清梯度前记录具体参数路径。

**不要**把 `error_if_nonfinite=False` 当修复。真实 NaN/Inf 时继续按非有限 norm 缩放会污染优化器/参数，只是把首个因果位置推迟到更难排查的地方。

## 4. 训练侧修复

### 4.1 稳定的梯度裁剪

文件：`cowp/scripts/20_train_external_baseline.py`

新增 `_clip_grad_norm_stable`：

1. 先保留 `error_if_nonfinite=True`；
2. 若 PyTorch 报 non-finite norm，先逐参数检查梯度元素；
3. 若存在真实 NaN/Inf，返回参数路径并终止；
4. 若所有元素有限，则用 float64 累加平方和、求 L2 norm、再对原梯度做一次全局缩放；
5. DTPP 仍保持公开实现语义：encoder / decoder 分别 clip，而不是把二者并成一个 norm；两侧 norm 的统计也保持 FP64，避免恢复后又因 `.float()` 再次变 Inf。

新增 epoch 诊断：

- `fp64_grad_norm_fallbacks`
- `max_preclip_grad_norm`

恢复时日志会出现类似：

`recovered finite-gradient fp32 norm overflow with float64 L2 clipping; preclip_norm=...`

若是真实坏梯度，则日志会给：

`nonfinite_gradient_paths=[...]`

### 4.2 修复梯度诊断顺序

先 `_first_nonfinite_gradient` / `_nonfinite_gradient_paths`，再 `optimizer.zero_grad()`，避免旧版永远打印 `None`。

### 4.3 空监督 batch

`valid_samples <= 0` 现在对 GameFormer / DTPP / PLUTO / PlanT2 全部统一跳过，不再只对 DTPP 特判。无有效 future supervision 的 batch 不应更新优化器，也不应把 NaN 指标混进 epoch 汇总。

## 5. Adapter 输入数值契约

文件：`cowp/external_baselines/adapters.py`

核心原则：**先根据 validity + finite geometry 清 validity，再做 zero-fill；不能先 `nan_to_num` 后把坏点继续当 valid。**

修复包括：

- SDC selector 的 `is_sdc` 非有限保护；
- agent type 非有限保护；
- history timestep：`declared valid AND finite state` 才算有效；
- 新增 `sdc_current_valid`；
- future validity 同时要求 finite geometry 和当前 SDC 有效；
- candidate validity 同时要求 finite geometry 和当前 SDC 有效；
- conventional-safe mask 不得重新激活 invalid candidate；
- roadgraph 的 `xyz` 形式和拆分 `x/y` 形式都要求 marked-valid point 的几何有限；
- SDC path marked-valid NaN/Inf 点先失效再参与 nearest/heading；
- `best_candidate_to_logged_ego` 先构造 safe target，再做 residual，防止 masked NaN 中间值。

## 6. 四个学习 baseline 的 loss 修复

### GameFormer

文件：`gameformer_cowp.py`

- GMM/NLL 与 interaction loss 在 loss-side 转 FP32；
- invalid GT 先用 `torch.where` 构造有限 target，再做减法、范数、平方/指数相关运算；
- differentiable zero 从模型输出构造，不再用 raw GT 做 `NaN * 0`；
- all-key-masked Transformer row 强制保留零 SDC anchor，避免 PyTorch attention 在全 mask 行产生 NaN；
- ADE/FDE 只对真实有效 future 统计；
- 返回 `valid_samples`。

### DTPP

文件：`dtpp_cowp.py`

- invalid SDC / all-key-mask 防护；
- neighbor / ego SmoothL1 target 先安全化；
- planner metric target 先安全化；
- 无有效 candidate+future 的 batch 返回 differentiable zero + `valid_samples=0`；
- trainer 保持 encoder/decoder 分别 clip 的 DTPP 语义。

### PLUTO-style

文件：`pluto_cowp.py`

- best-mode 距离在 subtraction 前构造安全 GT；
- regression / auxiliary imitation 都先安全化 target；
- classifier 只在有 supervision 的 scene 上训练；
- 全无监督 batch 不更新；
- metric 只统计 valid future。

### PlanT2-style

文件：`plant2_cowp.py`

- trajectory target 先安全化；
- speed target 只在相邻两个 timestep 都 valid 时构造，避免“前一点缺失、后一点有效”产生虚假的超大速度；
- hazard pair 需要 ego 与 neighbor 同时 future-valid，先 mask 再做距离；
- 全无监督 batch 不更新；
- metric 只统计 valid future。

## 7. 规则 baseline 与 Waymax 闭环修复

### PDM-Closed-style / 其他 rule scorer

文件：`rule_based.py`

- candidate marked-valid 但几何 NaN/Inf 时失效；
- non-finite SDC current numeric state 使该 scene 不可选；
- 非 SDC 坏状态先清 validity 再 zero-fill；
- invalid candidate 在 jerk / curvature / progress 等高阶运算前清零，避免无效 padding 仍触发 overflow/warning。

### 学习 baseline Waymax policy

文件：`waymax_policy.py`

- direct planner 仅在 SDC 当前 observation 有效时执行；
- direct trajectory tensor 有限但 globalize 时溢出/非法，也不会直接炸 rollout，而是落到 causal candidate/emergency path；
- candidate score 必须 finite；
- execution validity 同时继承 adapter validity，不会在 fallback 时把 adapter 已判 invalid 的候选重新激活；
- 当**没有任何有效候选**时，不再执行 padding slot 0；统一调用项目主 Waymax policy 已有的 `_resolve_execution_trajectory`，从当前 ego state 生成 bounded smooth-stop execution fallback。

### Rule Waymax policy

文件：`rule_waymax_policy.py`

同样不再把 padding candidate 0 当真实计划执行；无有效候选时使用 bounded smooth-stop execution fallback，并写入 diagnostic。

## 8. Checkpoint contract 与原命令兼容性

训练数值/validity/loss 语义发生变化，因此 training contract 从：

`v4_explicit_validity_fp32_20260827`

升级为：

`v5_stable_grad_masked_losses_20260827`

这是为了防止 `SKIP_COMPLETED=1` 在**相同输出目录**下静默复用旧 V4 checkpoint。

默认输出目录没有改：

`outputs/external_sota5_v16_8_33`

原训练命令保持不变：

```bash
GPU0=0 GPU1=1 nohup bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all > logs/run.log 2>&1 &
```

原 Waymax 命令也保持不变：

```bash
PARALLEL2=1 \
GPU0=0 GPU1=1 \
WOMD_VALIDATION_TFEXAMPLE_DIR=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation \
bash RUN_5_SOTA_BASELINES_COWP.sh waymax all
```

注意：`20_train_external_baseline.py` 原本就没有 resume semantics。只要某个 baseline 被判定需要重新训练，该入口会清理同目录下该 baseline 的旧 completion/history/checkpoint 后从随机初始化重新训练；本次没有改变这一行为。若需要保留旧 V4 结果用于历史对照，请先自行复制旧输出目录。

Waymax 默认拒绝加载非 V5 checkpoint，以防把旧数值语义混进新结果。`ALLOW_LEGACY_EXTERNAL_CHECKPOINT=1` 只用于明确的历史审计，不建议用于论文最终结果。

## 9. 对上传数据分析包的判断

上传的 compact 分割 manifest 是：train=5000、val=1000、heldout_test=1200。

natural-support diagnostics 中 train/val/heldout 的 `natural_rejection_counts.nonfinite` 都是 0；train 的 `rootless_rate=0`、`lt2_low_burden_rate=0`，且 `parse_errors=[]`。`verify_cache_train.json` 的 `valid_scene_rate_inspected=1.0`，missing/read error 为空。

因此，现有分析包没有证据表明数据集存在“普遍 NaN”。`verify_cache_train.json` 的总 `pass=false` 来自 `irrelevant_blocker_count=58243` 等机制审计项，而不是 non-finite tensor/read failure。

但是该 zip **不包含实际 tensor_cache 的 NPZ 文件**，所以无法在这里重放 shuffle 后的 GameFormer `epoch=2 batch=260`。V5 重新运行到该类 batch 时，新日志会给出最终分类：

- 出现 `recovered finite-gradient fp32 norm overflow...`：说明是 finite-entry / FP32 norm-reduction overflow；
- 出现具体 `nonfinite_gradient_paths=[...]`：说明是模型反向的真实 NaN/Inf，需要按给出的参数路径继续定位。

## 10. 测试结果

通过：

- 新增 V5 数值回归测试：8 passed；
- external baseline + Waymax 定向回归：43 passed，3 warnings；
- `python -m compileall`：通过；
- `bash -n RUN_5_SOTA_BASELINES_COWP.sh`：通过；
- `20_train_external_baseline --help` / `21_eval_external_baseline --help` / `22_eval_rule_baseline --help`：通过。

项目全量测试不能宣称全绿，因为上传的原始代码包本身缺少若干与本次 external baseline 修复无关的文件/接口。例如：

- `test_v16_8_29_recovery_viability.py` 引用当前 `policy_wrapper.py` 不存在的 `_recovery_bridge_viability_mask`；原始未修改副本同样失败；
- `tests/test_v16_8_14_womd_probe_contract.py` 需要包内不存在的 `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`；
- `tests/test_v16_8_15_womd_representation_layout.py` 需要包内不存在的 `NEXT_EXECUTION_V16_8_15_CN.sh`。

在排除前两个已知项后，全测运行到第三个缺失文件前是 `150 passed, 5 skipped`。这些不属于本次 external baseline 训练/闭环修复范围，也没有为了“让测试变绿”而伪造缺失的主算法脚本。

## 11. 额外论文/Bib 审计提示（未修改）

代码 metadata 对外部方法使用：`huang2023gameformer`、`huang2024dtpp`、`cheng2024pluto`、`gerstenecker2025plant20exposingbiases`、`dauner2023parting`。

上传的 Bib 中能找到 `huang2023gameformer`、`huang2024dtpp`、`renz2022plant`、`gulino2023waymax`，但没有找到 `cheng2024pluto`、`gerstenecker2025plant20exposingbiases`、`dauner2023parting`；同时 TeX 中还存在 `\\cite{gameformer}`。这不影响本次代码运行，但建议论文最终编译前统一 citation key 与 baseline 命名/“adaptation”表述。
