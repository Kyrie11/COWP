# v47 OC-TRAC 运行指令

## 0. Near/Contact 数据未完整时先重建

本轮修复了 worker 等待和候选前沿多样性。由于候选 schedule 会改变 NPZ 语义，建议输出到新目录或先备份旧 roots：

```bash
cd /path/to/OC-RAP-v47-OC-TRAC-optimized

OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v47_frontier \
RESUME=0 \
RUN_DIAGNOSTICS=1 \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

请使用一个尚不存在的独立 `OCRAP_ROOT`，因为该脚本会完整构建六个 val/test roots，但不会重建三个 train roots。脚本现在按 Safe、Near、Contact 三对顺序逐对等待，不会覆盖 PID。构建结束后先检查 `${OCRAP_ROOT}/reports`；训练时将 `TRAIN_OCRAP_ROOT` 指向现有训练数据根，将 `EVAL_OCRAP_ROOT` 指向新建的 val/test 根。若不能立即重建，可让两者都指向旧数据根运行 v47 学习修复，但 candidate-frontier 改进不会生效。

## 1. 推荐的开发筛选

```bash
cd /path/to/OC-RAP-v47-OC-TRAC-optimized

OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v47_frontier \
FINAL_RUN=0 \
RETRAIN_CLEAN_BASE=1 \
CLEAN_BASE_RUN=runs/ocrap_v39_ocrac_clean_safe_v47 \
RUN_HELDOUT_TEST=0 \
DATASET_DIAGNOSTICS_DIR=/data0/senzeyu2/dataset/OCRAP_v47_frontier/reports \
RUN_REFERENCE_CLOSED_LOOP_ON_GATE_FAILURE=1 \
bash run_v47_two_gpu_fast_commands.txt
```

该命令会：

1. 全量解冻刷新 clean base，并验证 `freeze_param_prefixes=[]`；
2. 训练 balanced/precision 两个 OC-TRAC 变体；
3. 在 val Near/Contact 上执行 development calibration；
4. 只有证书通过才运行 learned-policy offline 和 Waymax；
5. 即使证书失败，也运行 Safe/Near/Contact nominal reference closed loop。

不要在 development 阶段设置 `RUN_HELDOUT_TEST=1`，避免反复查看 test 后回调模型。

## 2. 只验证三 regime 数据是否支持闭环

```bash
PYTHONPATH=src python tools/check_closed_loop_dataset_support.py \
  --dataset /data0/senzeyu2/dataset/OCRAP/val_safe \
  --womd-pattern /data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord \
  --split val

PYTHONPATH=src python tools/check_closed_loop_dataset_support.py \
  --dataset /data0/senzeyu2/dataset/OCRAP/val_near_contact \
  --womd-pattern /data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord \
  --split val

PYTHONPATH=src python tools/check_closed_loop_dataset_support.py \
  --dataset /data0/senzeyu2/dataset/OCRAP/val_contact \
  --womd-pattern /data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord \
  --split val
```

`schema_supports_closed_loop=true` 只证明 metadata/path contract 成立；实际 runner 仍会在 TFRecord 中核对 scene id。

`WOMD_STRESS` 必须与构建 Near/Contact roots 时使用的 raw source 一致。同步 v47 rebuild 默认使用 standard validation，因此无需设置；旧 roots 若来自 `validation_interactive`，运行时显式覆盖：

```bash
WOMD_STRESS=/path/to/validation_interactive/validation_interactive_tfexample.tfrecord ...
```

## 3. 单独运行三 regime nominal reference closed loop

```bash
RUN=runs/v47_nominal_reference_val \
SAFE_BUCKET=/data0/senzeyu2/dataset/OCRAP/val_safe \
NEAR_BUCKET=/data0/senzeyu2/dataset/OCRAP/val_near_contact \
CONTACT_BUCKET=/data0/senzeyu2/dataset/OCRAP/val_contact \
REFERENCE_BUCKET_SPLIT=val \
REFERENCE_MAX_ROLLOUTS=32 \
REFERENCE_MAX_TARGETS=80 \
REFERENCE_MAX_STEPS=40 \
bash scripts/run_v47_all_regime_reference_closed_loop.sh
```

该输出是 nominal physical reference，不是 learned OC-TRAC 结果。

## 4. 已有有效 v47 checkpoint 时运行 learned-policy 三 regime 闭环

```bash
BASE_RUN=runs/ocrap_v47_trac_balanced \
RUN=runs/ocrap_v47_trac_balanced_val_closed_loop \
SAFE_TEST=/data0/senzeyu2/dataset/OCRAP/val_safe \
NEAR_TEST=/data0/senzeyu2/dataset/OCRAP/val_near_contact \
CONTACT_TEST=/data0/senzeyu2/dataset/OCRAP/val_contact \
SAFE_BUCKET_SPLIT=val \
NEAR_BUCKET_SPLIT=val \
CONTACT_BUCKET_SPLIT=val \
RUN_OFFLINE_EVAL=1 \
RUN_AUDITS=1 \
RUN_SAFE_CLOSED_LOOP=1 \
RUN_SCALAR_BASELINES=1 \
bash scripts/run_ocrap_v47_trac.sh
```

脚本会拒绝读取 `valid_for_active_contract=false` 的 calibration artifact。

## 5. Final contract

先建立与 train/val/test scene-disjoint 的：

```text
cal_near_contact
cal_contact
```

然后执行：

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v47_frontier \
FINAL_RUN=1 \
RETRAIN_CLEAN_BASE=1 \
CLEAN_BASE_RUN=runs/ocrap_v39_ocrac_clean_safe_v47_final \
TRAC_CAL_NEAR_DATA=/data0/senzeyu2/dataset/OCRAP/cal_near_contact \
TRAC_CAL_CONTACT_DATA=/data0/senzeyu2/dataset/OCRAP/cal_contact \
DATASET_DIAGNOSTICS_DIR=/path/to/final/reports \
RUN_HELDOUT_TEST=1 \
bash run_v47_two_gpu_fast_commands.txt
```

Final 模式要求 clean-base marker 和独立 calibration roots。建议分别用 seeds 7、17、27 运行完整闭环，且使用不同 run 目录保存结果。

## 6. Gate 再次失败时的检查顺序

1. `train_summary.json` 中 `init_checkpoint` 是否是本轮 clean base；
2. sampler log 中 `positive_advantage_target=pcd`、`positive_advantage_groups>0`；
3. Near/Contact 的 `candidate_positive_auc` 与 `unconstrained_group_top1_advantage_correlation`；
4. `candidate_harm_auc` 与 verify conditional harmful UCB；
5. oracle frontier 中每个 macro 的正 PCD 样本数；
6. 若 frontier 本身正机会过少，扩展 recovery candidates，而不是降低 gate。
