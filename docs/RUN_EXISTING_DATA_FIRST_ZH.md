# v48.1：优先复用现有数据的执行指令

## 0. 解压与进入代码目录

```bash
unzip OC-RAP-v48.1-REUSE-CAL.zip
cd OC-RAP-v48.1-REUSE-CAL
```

确认现有数据根：

```bash
ls /data0/senzeyu2/dataset/OCRAP/train_near_contact/manifest.csv
ls /data0/senzeyu2/dataset/OCRAP/train_contact/manifest.csv
ls /data0/senzeyu2/dataset/OCRAP/val_safe/manifest.csv
ls /data0/senzeyu2/dataset/OCRAP/val_near_contact/manifest.csv
ls /data0/senzeyu2/dataset/OCRAP/val_contact/manifest.csv
```

## 1. 只做预检查，不启动训练

```bash
PYTHONPATH=src python -m compileall -q src tools
bash -n run_v48_two_gpu_fast_commands.txt
bash -n scripts/train_ocrap_v48_trac_sr.sh
bash -n scripts/build_v48_calibration_regimes.sh
bash -n scripts/recalibrate_v48_on_dedicated_set.sh
```

检查v47 checkpoint路径：

```bash
ls runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt
```

若实际路径不同，只需在正式命令中修改`INIT_CKPT`。

## 2. 推荐的第一轮：现有train + scene拆分现有val

```bash
cd /path/to/OC-RAP-v48.1-REUSE-CAL

OUTPUTDIR=runs/ocrap_v48_1_existing_data_screening \
TRAIN_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
INIT_CKPT=runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt \
CALIBRATION_MODE=proxy_val_split \
CALIBRATION_FRACTION=0.50 \
CALIBRATION_SEED=4801 \
BUILD_TRAIN=0 \
BUILD_CALIBRATION=0 \
STRICT_TRAIN_DATA_GATE=0 \
REUSE_TEACHER_INDEX=1 \
GPU0=0 GPU1=1 \
BATCH_SIZE=96 \
NUM_WORKERS=6 \
PREFETCH_FACTOR=2 \
bash run_v48_two_gpu_fast_commands.txt
```

该脚本默认进入后台。查看：

```bash
tail -f runs/ocrap_v48_1_existing_data_screening/logs/controller.log
```

查看PID：

```bash
cat runs/ocrap_v48_1_existing_data_screening/controller.pid
```

停止controller及其子任务前先查看进程树：

```bash
pstree -ap $(cat runs/ocrap_v48_1_existing_data_screening/controller.pid)
```

## 3. 第一阶段应检查的文件

### 训练数据是否有学习机会

```bash
cat runs/ocrap_v48_1_existing_data_screening/teacher_pcd_train_index_summary.json
```

重点字段：

```text
by_bucket.near.positive_groups
by_bucket.near.positive_scenes
by_bucket.near.max_positive_macro_share
by_bucket.near.top10_positive_scene_share
by_bucket.near.screening_status
by_bucket.near.screening_recommended_action
by_bucket.contact.positive_groups
by_bucket.contact.positive_scenes
by_bucket.contact.max_positive_macro_share
by_bucket.contact.top10_positive_scene_share
by_bucket.contact.screening_status
by_bucket.contact.screening_recommended_action
quality_failures
```

`quality_failures`在本轮只是论文级目标告警，不会自动触发重建。

### 数据角色是否正确

```bash
cat runs/ocrap_v48_1_existing_data_screening/dataset_role_manifest.json
cat runs/ocrap_v48_1_existing_data_screening/development_calibration_overlap_audit.json
```

必须满足：

```text
test_roots_used_during_screening=false
scene overlap=0
```

### 模型是否解决策略级排序

```bash
cat runs/ocrap_v48_1_existing_data_screening/screening_status.json
```

重点比较v47：

```text
candidate_auc
unconstrained_group_top1_correlation
top-1 teacher advantage
harmful rate
verify precision/Wilson LCB
```

Natural gate失败时脚本会停止在策略评估之前，但训练和校准诊断会完整保留，test不会被读取。

## 4. 暂时不要执行的命令

第一轮不要设置：

```bash
BUILD_TRAIN=1
BUILD_CALIBRATION=1
CALIBRATION_MODE=dedicated
```

也不要把任何`test_*`路径传入calibrator或训练early stopping。

## 5. 最终专用calibration构建

标准validation专用构建，默认使用两张A30、按regime顺序运行，每次两个worker并行：

```bash
cd /path/to/OC-RAP-v48.1-REUSE-CAL

mkdir -p /data0/senzeyu2/dataset/OCRAP_v48_calibration/logs

nohup env \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example \
EVAL_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
OUTPUT_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
GPU0=0 GPU1=1 \
CALIBRATION_START_INDEX=11000 \
PARTITION_STRIDE=6 \
SAFE_RAW_PER_WORKER=600 \
NEAR_RAW_PER_WORKER=700 \
CONTACT_RAW_PER_WORKER=700 \
MIN_CAL_SAFE_SCENES=80 \
MIN_CAL_NEAR_SCENES=120 \
MIN_CAL_CONTACT_SCENES=120 \
RESUME=1 \
RUN_DIAGNOSTICS=1 \
bash scripts/build_v48_calibration_regimes.sh \
>/data0/senzeyu2/dataset/OCRAP_v48_calibration/logs/controller.log 2>&1 </dev/null &

echo $! > /data0/senzeyu2/dataset/OCRAP_v48_calibration/controller.pid
```

查看进度：

```bash
tail -f /data0/senzeyu2/dataset/OCRAP_v48_calibration/logs/controller.log
```

每个worker日志在：

```text
/data0/senzeyu2/dataset/OCRAP_v48_calibration/logs/calibration_safe_w*.log
/data0/senzeyu2/dataset/OCRAP_v48_calibration/logs/calibration_near_w*.log
/data0/senzeyu2/dataset/OCRAP_v48_calibration/logs/calibration_contact_w*.log
```

## 6. 专用calibration完成后：不重训，只重新校准

```bash
cd /path/to/OC-RAP-v48.1-REUSE-CAL

OUTPUTDIR=runs/ocrap_v48_1_existing_data_screening \
CALIBRATION_OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP_v48_calibration \
GPU0=0 GPU1=1 \
nohup bash scripts/recalibrate_v48_on_dedicated_set.sh \
>runs/ocrap_v48_1_existing_data_screening/logs/recalibrate_dedicated_controller.log \
2>&1 </dev/null &
```

查看：

```bash
tail -f runs/ocrap_v48_1_existing_data_screening/logs/recalibrate_dedicated_controller.log
cat runs/ocrap_v48_1_existing_data_screening/dedicated_recalibration_status.json
cat runs/ocrap_v48_1_existing_data_screening/chosen_base_run_dedicated.txt
```

## 7. 何时才重建train_contact

先看exact-PCD summary和训练结果。仅在以下组合出现时考虑：

```text
Contact正机会group/scene非常低或高度集中
+ train/development group-top1始终学不成正相关
+ 多个seed结论一致
```

若候选级AUC和训练group-top1明显改善，而proxy/dedicated calibration因scene数或分布偏移失败，应优先增建calibration，不先重建train_contact。
