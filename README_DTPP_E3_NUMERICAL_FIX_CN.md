# V3：DTPP epoch-3 数值崩溃修复与外部 baseline 训练/评测加固

## 1. 这次报错的实际含义

用户日志显示 DTPP epoch 3 共 313 个 batch，但最终 `samples=3856`。batch size=16，因此只有 241 个 batch 成功进入统计，恰好有 72 个 batch 被旧循环跳过：

- 241 × 16 = 3856；
- 313 - 241 = 72；
- 72 / 313 = 23.003%。

旧 tqdm 在 `continue` 的 skip 分支不刷新 postfix，所以终端末尾仍显示 `skipped=0`，这是显示问题，不代表真的没有 skip。

`max_skip_fraction=2%` 本身不是根因；它正确地阻止了一个已经异常的 epoch 被保存/继续。根因是旧 DTPP 配置与数值防护叠加：

1. V2 默认 `variable_cost=True`，而公开 DTPP `variable_weights` 默认 False；
2. V2 一键脚本给 DTPP 强制 BF16 AMP，而公开训练代码为 FP32；
3. V2 只在 backward 前检查 loss finite，没有检查梯度是否 finite；finite loss 仍可能产生 NaN/Inf gradient，`clip_grad_norm_` 也没有 `error_if_nonfinite=True`，因此一次坏梯度可能污染参数，之后所有 batch 才连续产生 non-finite loss；
4. DTPP ego/map/neighbor/candidate padding mask 在 WOMD ego-frame adaptation 中存在若干“零值等于 padding”的错误假设，会放大数值不稳定。

日志中最后一个成功 batch 已出现 `score_ce=35.9742`，远高于 30-way 分类的正常初始尺度（随机均匀 CE 约 3.4），epoch mean loss 达约 2.59e5；这与 cost/score 尺度失控一致。

## 2. V3 的 DTPP 修改

### 2.1 恢复公开 DTPP 的关键训练默认

标准训练现在默认：

- `variable_cost=False`；
- FP32（`DTPP_AMP=0`）；
- epochs=30；
- batch=16；
- lr=2e-4；
- AdamW weight_decay=0.01（公开源码未显式传该参数，因此使用 PyTorch AdamW 默认 0.01）；
- StepLR(step_size=5, gamma=0.5)；
- encoder 和 decoder **分别** clip grad norm=5.0。

若要做 variable-cost/AMP 消融，可显式设置 `DTPP_VARIABLE_COST=1` / `DTPP_AMP=1`，但不要把它当标准 baseline checkpoint。

### 2.2 训练循环 fail-before-corruption

`20_train_external_baseline.py` 现在：

- forward loss 非 finite：该 batch 不 backward/step；
- backward 后使用 `error_if_nonfinite=True` 检查 gradient norm；
- DTPP encoder/decoder 分别 clip；
- optimizer step 后检查所有 parameter 是否仍 finite；
- 连续 3 次 numerical failure 立即 fail-fast，不再默默跳过剩余几十个 batch；
- 日志报告 `first_nonfinite_gradient`、`score_abs_max`、`weight_max`；
- tqdm 的 skipped 数会在 skip 分支立即刷新；
- arbitrary CUDA/shape/programming RuntimeError 不再被当“malformed batch”吞掉；只有 adapter 数据契约类 `KeyError/ValueError/IndexError` 才允许在 2% gate 内有界跳过；
- DTPP `valid_samples=0` 的完全无监督 batch 不参与优化，也不会用 NaN ADE 污染 epoch metric；
- best-checkpoint 只使用 finite metric。

### 2.3 WOMD ego-frame mask 修复

- stopped ego：用显式 valid channel，而不是 `x+y+yaw+vx+vy==0`；
- stopped neighbor：interaction/collision scorer 使用显式 neighbor valid channel；
- map：padding 定义为“整个 point feature 全零”，不是 `local_x==0`；
- valid stationary stop candidate：使用显式 `candidate_valid`，即使整条候选在 ego frame 中全零也不会被误当 padding；
- MultiheadAttention 3D mask：改为 `repeat_interleave(num_heads)`，使 `[B*num_heads,L,S]` 的 batch/head 排列正确；
- ego-tree attention 恢复公开 DTPP 的 10-step max pooling，而不是 V2 的 strided sampling。

### 2.4 DTPP 仍然是 WOMD shared-tree adaptation

公开 DTPP 原生训练同时使用 30-step first-stage tree loss 和 80-step second-stage tree loss（后者权重 0.2）。本工程的 WOMD/COWP adapter 没有原生 nuPlan 两级 spline tree，而是统一 80-step shared candidate bank，因此仍应在论文中写：

`DTPP (WOMD shared-tree adaptation)`

V3 修复的是源默认、网络/attention/mask/优化稳定性，不把跨 domain adaptation 冒充原生 nuPlan benchmark 的逐字节复现。

## 3. 其他 baseline / evaluation 修复

- PLUTO：contrastive agent dropout 现在同步更新 `agent_valid`；旧版只把 feature 乘零，但 MLP bias 仍会产生“有效 actor token”。
- GameFormer / PLUTO / PlanT2：FDE 不再把 future 全 invalid 的样本按 0 误计入平均。
- offline evaluator：NaN/Inf candidate score 不参与 argmax。
- Waymax evaluator：checkpoint 改为 `strict=True` 加载；不允许 missing/random layer 静默参与 benchmark。
- Waymax direct trajectory：非 finite waypoint/global trajectory 不再静默 nan-to-zero；会进入显式 fallback。
- parallel2：无论其中一个 worker 是否失败都会 reap 两个子进程，并报告各自 exit code，避免 orphan run。
- shell 训练入口现在显式传 `--device`。
- DTPP V2 旧 checkpoint 没有 V3 completion signature，不会被 `SKIP_COMPLETED=1` 误认为可复用。

## 4. 旧 DTPP checkpoint 怎么处理

**不要从 V2 epoch1/2/3 继续训练。** 即使 epoch1/2 看起来 finite，它们已经使用了旧的 variable-cost + BF16 + 错误 mask 配置，不适合作为最终论文 baseline。

只归档 DTPP 目录，保留已完成的 GameFormer：

```bash
mv outputs/external_sota5_v16_8_33/dtpp \
   outputs/external_sota5_v16_8_33/dtpp_failed_v2_$(date +%Y%m%d_%H%M%S)
```

如果 DTPP 目录不存在可以忽略。

## 5. 推荐重新训练命令

### 5.1 继续两 GPU 全部 baseline

```bash
mkdir -p logs
GPU0=0 GPU1=1 \
DTPP_AMP=0 \
DTPP_VARIABLE_COST=0 \
SKIP_COMPLETED=1 \
nohup bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all \
  > logs/run_v3.log 2>&1 &
```

如果 GameFormer 已完整训练，`SKIP_COMPLETED=1` 会复用它。V2 DTPP 因没有 V3 completion signature 会强制重训。由于此前 parallel pair 在 DTPP 失败后终止，PLUTO/PlanT2 通常还未启动，后续会正常继续。

### 5.2 只重训 DTPP

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 \
DTPP_AMP=0 DTPP_VARIABLE_COST=0 \
nohup bash RUN_5_SOTA_BASELINES_COWP.sh train dtpp \
  > logs/dtpp_v3.log 2>&1 &
```

### 5.3 观察数值稳定性

```bash
grep -E "non-finite|bad_grad|first_nonfinite|score_abs_max|weight_max|skipped|epoch summary|updated best" \
  logs/run_v3.log
```

标准稳定训练应看到 `numerical_skipped_batches=0`；若机器/数据上仍触发问题，V3 会在第一次/前三次 numerical failure 附近留下真正原因，而不会等到 epoch 尾部才报 23% skip。

## 6. 训练后闭环命令不变

最终 Waymax 仍使用 raw WOMD validation TFRecord：

```bash
export WOMD_VALIDATION_TFEXAMPLE_DIR=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation

PARALLEL2=1 GPU0=0 GPU1=1 \
bash RUN_5_SOTA_BASELINES_COWP.sh waymax all
```

建议先做短 profiler：

```bash
PROFILE_NUM_SCENARIOS=24 \
bash RUN_5_SOTA_BASELINES_COWP.sh profile all
```

## 7. 验证结果

本次 external-baseline 聚焦回归：

- 18 passed；
- 3 个 warning 均为 PyTorch Transformer nested-tensor performance warning；
- `python -m compileall -q cowp`：PASS；
- `bash -n RUN_5_SOTA_BASELINES_COWP.sh`：PASS。

全仓库 pytest：301 passed / 5 skipped / 8 failed。对原始上传 V2 单独复跑这 8 个 failure 后得到同样结果，因此它们是上传包已有的历史测试债务：6 个测试引用当前 zip 中不存在的旧 launcher shell，2 个测试期待旧 label-semantic fingerprint。V3 没有修改/伪造这些证据协议测试来“制造全绿”。
