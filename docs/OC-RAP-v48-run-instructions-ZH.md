# OC-RAP v48 OC-TRAC-SR 执行指令

## 一、解压与环境

```bash
unzip OC-RAP-v48-OC-TRAC-SR.zip
cd OC-RAP-v48-OC-TRAC-SR
export PYTHONPATH=$PWD/src
```

确保可用：PyTorch CUDA、JAX CUDA、TensorFlow、Waymax、WOMD v1.3.1 tf.Example。

## 二、推荐的第一轮：新建训练集 + 使用现有 val 做 scene-disjoint calibration

主脚本默认会把自己放到后台，并把 controller 与子任务日志写入 `OUTPUTDIR/logs`：

```bash
cd /path/to/OC-RAP-v48-OC-TRAC-SR

OUTPUTDIR=runs/ocrap_v48_trac_sr_screening \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_train \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
INIT_CKPT=runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt \
GPU0=0 GPU1=1 \
BUILD_TRAIN=1 \
BUILD_CALIBRATION=0 \
bash run_v48_two_gpu_fast_commands.txt
```

脚本会立即返回 PID：

```bash
cat runs/ocrap_v48_trac_sr_screening/controller.pid
tail -f runs/ocrap_v48_trac_sr_screening/logs/controller.log
```

关键输出：

```text
runs/ocrap_v48_trac_sr_screening/
├── logs/
├── scene_overlap_audit.json
├── teacher_pcd_train_index.jsonl
├── teacher_pcd_train_index_summary.json
├── dataset_splits/
├── candidates/balanced/
├── candidates/precision/
├── chosen_base_run.txt
└── NEXT_COMMANDS.txt
```

原路径：

```text
/data0/senzeyu2/dataset/OCRAP/val_safe
/data0/senzeyu2/dataset/OCRAP/val_near_contact
/data0/senzeyu2/dataset/OCRAP/val_contact
```

不会被改写。calibration/dev 仅在 `OUTPUTDIR/dataset_splits` 中以 scene-disjoint links 创建。

## 三、训练集已构建时跳过重建

```bash
OUTPUTDIR=runs/ocrap_v48_trac_sr_screening_rerun \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_train \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
INIT_CKPT=runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt \
GPU0=0 GPU1=1 \
BUILD_TRAIN=0 \
bash run_v48_two_gpu_fast_commands.txt
```

## 四、单独构建新训练集

```bash
cd /path/to/OC-RAP-v48-OC-TRAC-SR

nohup env \
OUTPUT_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_train \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example \
GPU0=0 GPU1=1 RESUME=1 \
NEAR_RAW_PER_WORKER=5500 \
CONTACT_RAW_PER_WORKER=7000 \
bash scripts/build_v48_train_regimes.sh \
>/data0/senzeyu2/dataset/OCRAP_v48_train/logs/controller.log 2>&1 &
```

质量不足时增加：

```bash
NEAR_RAW_PER_WORKER=8000
CONTACT_RAW_PER_WORKER=10000
```

不要把旧 `train_contact` 直接 append 到新 root。若以后希望作为 auxiliary data 使用，应先实现 per-root importance weight，并限制旧数据占比，而不是无权重混合。

## 五、论文最终版：构建专用 calibration roots

官方 testing/testing_interactive 不提供 future GT，因此这里使用标准 validation：

```bash
cd /path/to/OC-RAP-v48-OC-TRAC-SR

nohup env \
OUTPUT_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example \
GPU0=0 GPU1=1 RESUME=1 \
NEAR_RAW_PER_WORKER=3000 \
CONTACT_RAW_PER_WORKER=4000 \
bash scripts/build_v48_calibration_regimes.sh \
>/data0/senzeyu2/dataset/OCRAP_v48_calibration/logs/controller.log 2>&1 &
```

脚本会自动排除 `EVAL_OCRAP_ROOT` 下已有的 val/test Safe/Near/Contact scene。

随后运行：

```bash
OUTPUTDIR=runs/ocrap_v48_trac_sr_dedicated_cal \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_train \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CALIBRATION_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
INIT_CKPT=runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt \
GPU0=0 GPU1=1 BUILD_TRAIN=0 BUILD_CALIBRATION=0 \
bash run_v48_two_gpu_fast_commands.txt
```

也可让主脚本自动构建：

```bash
BUILD_CALIBRATION=1 \
CALIBRATION_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
... \
bash run_v48_two_gpu_fast_commands.txt
```

## 六、检查 val/test scene 泄漏

在最终 test 前运行：

```bash
python tools/check_scene_overlap_v48.py \
  --train-root /data0/senzeyu2/dataset/OCRAP_v48_train/train_near_contact \
  --train-root /data0/senzeyu2/dataset/OCRAP_v48_train/train_contact \
  --development-root /data0/senzeyu2/dataset/OCRAP/val_near_contact \
  --development-root /data0/senzeyu2/dataset/OCRAP/val_contact \
  --test-root /data0/senzeyu2/dataset/OCRAP/test_near_contact \
  --test-root /data0/senzeyu2/dataset/OCRAP/test_contact \
  --output runs/ocrap_v48_scene_overlap_final.json \
  --fail-on-train-development-overlap \
  --fail-on-development-test-overlap
```

若 development-test overlap 非零，不能把当前 `test_*` 当作真正 held-out；应按 scene 重新切分/重建。

## 七、前台运行（仅调试）

```bash
FOREGROUND=1 \
OUTPUTDIR=runs/ocrap_v48_debug \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_train \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
INIT_CKPT=runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt \
GPU0=0 GPU1=1 BUILD_TRAIN=0 \
bash run_v48_two_gpu_fast_commands.txt
```

## 八、Natural gate 通过后的 development closed-loop

主脚本成功后会生成：

```bash
cat runs/ocrap_v48_trac_sr_screening/NEXT_COMMANDS.txt
```

直接执行其中命令。默认闭环 probe 很小，用于验证 scene matching、运行稳定性和指标字段；确认无误后逐步增加：

```bash
AUDIT_MAX_ROLLOUTS=32
AUDIT_MAX_STEPS=40
AUDIT_TARGETS=64
SAFE_MAX_ROLLOUTS=32
SAFE_MAX_STEPS=40
```

不要在 Natural gate 失败时强制运行 learned policy。可以单独运行 nominal physical reference，但不能把它解释成 v48 效果。

## 九、三随机种子

通过单次 screening 后：

```bash
for SEED in 7 17 37; do
  OUTPUTDIR=runs/ocrap_v48_seed_${SEED} \
  SEED=$SEED \
  TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_train \
  EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
  CALIBRATION_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
  INIT_CKPT=runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt \
  GPU0=0 GPU1=1 BUILD_TRAIN=0 \
  bash run_v48_two_gpu_fast_commands.txt
done
```

主脚本自身会后台化，所以循环会依次创建三个 controller。GPU 资源不足时不要同时启动；可逐个运行。

## 十、常用监控

```bash
# 总控制器
tail -f $OUTPUTDIR/logs/controller.log

# 数据构建
tail -f $TRAIN_OCRAP_ROOT/logs/train_contact_w0.log

# 两个候选训练
tail -f $OUTPUTDIR/candidates/balanced/logs/train_v48_trac_sr.log
tail -f $OUTPUTDIR/candidates/precision/logs/train_v48_trac_sr.log

# Natural gate
cat $OUTPUTDIR/candidates/balanced/calibration/direct_value_risk_near_v48.json
cat $OUTPUTDIR/candidates/balanced/calibration/direct_value_risk_contact_v48.json

# GPU
watch -n 2 nvidia-smi
```

## 十一、可调参数的优先级

先调数据，再调模型：

1. `NEAR_RAW_PER_WORKER` / `CONTACT_RAW_PER_WORKER`；
2. 候选 macro 多样性和 exact positive scene 覆盖；
3. `POSITIVE_GROUP_BOOST`；
4. `ENCODER_LR_SCALE`；
5. `HARM_W`、`FP_W`、`SETWISE_W`；
6. disagreement penalty；
7. 最后才是校准风险阈值。

不得通过放宽 Natural gate 来“制造通过”。
