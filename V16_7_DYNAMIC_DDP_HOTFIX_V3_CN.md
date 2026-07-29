# COWP v16.7 动态 DDP Hotfix v3

## 根因

上传包内的说明文件声称 witness/planner 已关闭 `static_graph`，但实际执行文件
`cowp/scripts/03_train.py` 仍在 `freeze_backbone_epochs <= 0` 时把两个阶段设为
`static_graph=True`。原命令固定传入 `FREEZE_BACKBONE_EPOCHS=0`，因此每次都会进入错误分支。

COWP witness/planner 会返回多个 head，最终 loss 又包含按 batch mask/标签决定是否有效的
分支。`response_decoder.mode_head` 等参数可能在某个 batch 参与反传、下个 batch 不参与。
这与 DDP 静态图要求冲突。

## 修复

- witness/planner/response/all：`find_unused_parameters=True`，不设置 `static_graph`；
- 永久冻结的 natural repair：保留 `static_graph=True` 快速路径；
- 保留 `gradient_as_bucket_view=True`；旧 PyTorch 不支持时安全回退；
- 不添加伪零 loss，不改变 `grad=None`、AdamW、loss、前向和评估语义；
- 使用新的 provenance marker `.v16_7_dynamic_ddp_hotfix_v3_applied`，允许已有失败
  `OUT_ROOT` 对此次代码修复做一次可审计兼容续跑。

修复后 witness 日志必须为：

```text
DDP policy: find_unused_parameters=True, static_graph=False, gradient_as_bucket_view=True
```

## 自动跳过与续训

`run_cowp_v16_7_dual_gpu.sh` 的原逻辑保留：

- natural：`checkpoints/natural/cowp_natural_*`；
- transport：`checkpoints/transport/cowp_witness_*`；
- planner：`checkpoints/planner/cowp_planner_*`。

目标 N 个 epoch 在 checkpoint `epoch >= N-1` 时跳过；否则以
`--resume-training` 从 `epoch+1` 继续，并恢复 optimizer、scheduler、early-stop 和 history。
当前错误发生在 witness epoch 0 前几个 batch，未完成一个 epoch 时通常没有可恢复 witness
checkpoint，因此会从已验证 natural checkpoint重新开始 witness；已经完成并原子保存的其他阶段不会重训。
