# COWP 训练加速补丁说明

本补丁做了三类修改：

1. `03_train.py` 增加 CUDA 隐藏告警：如果之前执行过 `export CUDA_VISIBLE_DEVICES=-1`，训练脚本会明确提示当前正在 CPU 上跑。
2. `03_train.py` 增加 `--response-traj-weight 0` / `--no-response-traj`：快速 response 训练时不再加载 `cowp/response/traj`，并跳过 ResponseDecoder 的巨大轨迹输出头。
3. `dataset.py` 将 response/planner 阶段的 key 过滤从宽泛前缀改成显式小 key，避免把不用的大张量读入 CPU/GPU batch。planner 阶段默认不再加载 broad `waymax/`，只有加 `--with-waymax-outcome-labels` 才读取三类 candidate outcome 标量标签。

快速训练建议：

```bash
unset CUDA_VISIBLE_DEVICES
# 或 export CUDA_VISIBLE_DEVICES=0

python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal_v2/tensor_cache_val \
  --stage response \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --no-positive-oversampling \
  --response-traj-weight 0 \
  --resume outputs/checkpoints/representation/cowp_representation_best.pt \
  --output-dir outputs/checkpoints/quick_response
```

注意：`--response-traj-weight 0` 是快速 smoke/初步结果模式。最终论文主结果建议再开轨迹监督，或至少用较小 batch 做短时间 finetune。
