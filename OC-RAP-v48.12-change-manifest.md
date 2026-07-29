# v48.12 TRIDENT代码变更清单

## 核心算法

- `src/ocrap/models/losses.py`
  - 增加exact-PCD gap-weighted recovery pair tournament损失。
  - 增加按regime分组的policy-top1 benefit/harm跨group pairwise AUC surrogate。
- `src/ocrap/cli/train.py`
  - 将新排序与证据损失参数接入训练配置。
- `configs/default.yaml`
  - 增加TRIDENT损失权重和margin默认项，默认关闭以保持历史配置兼容。

## 训练与checkpoint合同

- `scripts/train_ocrap_v48_12_trident.sh`
  - Stage R显式启用`PREFERENCE_CONDITIONAL_MODE=true`。
  - Stage R默认启用gap-weighted recovery pair监督。
  - Stage E默认启用benefit/harm跨group证据排序，harm权重更高。
  - 保留policy-first/no-fallback和有序三状态概率。
- `scripts/train_ocrap_v48_trac_sr.sh`
  - 增加新损失参数的CLI映射。
- `run_v48_two_gpu_fast_commands.txt`
  - 支持`INIT_CKPT_BALANCED`与`INIT_CKPT_PRECISION`，两个variant可分别继承对应checkpoint。
  - calibration传递macro constraint mode与macro excess budget。

## Natural gate与校准

- `tools/calibrate_policy_risk_v48.py`
  - 报告oracle-positive macro分布。
  - 增加`selected_macro_excess_share`。
  - 支持`absolute`与`opportunity_normalized`两种macro约束。
  - normalized模式下候选排序也按excess concentration而不是raw share。

## 实验编排

- `scripts/run_v48_12_parallel_ablations.sh`
  - 自动运行4组×2 variant共8个任务。
  - 两张A30最多并行2个单GPU任务。
  - 单任务失败不会中断剩余任务。
  - 仅全部任务完成时生成`ABLATIONS_COMPLETE.json`。
- `scripts/recalibrate_v48_12_multiseed.sh`
  - 支持opportunity-normalized macro证书。
- `scripts/recalibrate_v48_12_on_dedicated_set.sh`
  - dedicated calibration使用相同macro证书合同。
- `tools/check_v48_12_learning_gates.py`
  - 分层报告Preference、Evidence和Natural gate。
- `tools/summarize_v48_12_ablations.py`
  - 汇总8个消融任务及policy-top1证据指标。

## 测试

- `tests/test_v48_12_trident.py`
  - gap-weighted recovery pair方向性测试；
  - benefit/harm跨group pairwise证据测试；
  - opportunity-normalized macro concentration测试；
  - Stage-R conditional checkpoint合同测试。

## 文档

- 更新根目录`ALGORITHM_CHANGELOG.md`。
- 新增v48.11结果审计、v48.12运行指令、结构化结果摘要和验证状态。
