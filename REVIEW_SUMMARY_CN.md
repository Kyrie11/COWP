# 第二轮 COWP 代码/数据审查摘要

## 数据是否足够训练

当前 train/val labels 足够进入模型训练。`mean_candidate_endpoint_spread_m≈5.46m` 低于 6.0m 是候选覆盖的软性 warning，不是标签错误，也不会直接破坏 COWP 的核心监督目标。更关键的信号均可用：

- train/val validation error 均为 0；
- train≈20k、val≈3k，规模可用于正式训练起步；
- train/val 分布一致；
- false-safe candidate ratio≈0.365/0.379，能支撑 false-safe 论证；
- positive pair ratio≈0.287/0.294，witness 监督不稀疏；
- ncf_candidate_ratio≈0.188/0.193，可支撑 candidate ranking；
- response_safe_pair_ratio≈0.874/0.869，safe-response set 足够密。

endpoint spread 偏低主要影响 stress candidate 多样性和最终 ablation 说服力。考虑到全量重构成本高，建议先训练，再用 learned-offline 和 Waymax smoke rollout 判断是否需要局部扩充 validation/stress set。

## 闭环不足的原因

不是当前 labels 的性质不足，而是之前代码的 Waymax 闭环路径还没有把 rollout 诊断真正聚合为论文表格可用指标。labels diagnostics 只能说明监督标签质量；闭环实验必须运行 `--mode waymax` 并报告官方 Waymax standard metrics 与 COWP online policy diagnostics。

## 第二轮代码修改

- natural alternatives 改为 weighted set-minADE 监督，避免把无序 counterfactual set 当作固定顺序 tensor。
- Waymax COWP policy wrapper 增加在线诊断：selected candidate、accepted count、fallback、witness probability、OPR、predicted burden、C_i。
- `04_eval_closed_loop.py --mode waymax` 现在输出 `policy_diagnostic_summary`，包含 closed-loop predicted FSR/CBS/OPR/fallback step rate。
- 增加 official Waymax metric 的 scalar summary 聚合。
- 新增两个单测，当前 `15 passed`。

## 建议实验顺序

1. 使用现有 labels 先构建 tensor cache 并训练。
2. 先跑 offline label eval，确认 rule certificate 表格正常。
3. 跑 learned-offline eval，确认模型是否学到 witness 与 candidate ranking。
4. 用 Waymax 小规模 smoke test 跑 50–100 scenarios。
5. 如果模型出现 candidate collapse 或 val/stress ablation 不明显，再考虑只重构 val/stress，而不是直接重构完整 train。

## 第二轮复查结论：AMP、CUDA加载与 critical-agent 监督

1. 本轮报错 `AttributeError: module 'torch.amp' has no attribute 'GradScaler'` 不是 CUDA 驱动问题，而是 PyTorch AMP API 版本兼容问题。训练脚本此前即使没有开启 `--amp` 也会构造 `torch.amp.GradScaler`，在部分 PyTorch 版本中会直接崩溃。现在已改为只有在 `--amp` 且 CUDA 训练时才创建 scaler，并兼容 `torch.cuda.amp.GradScaler`。
2. 训练启动慢的主要原因通常不是没有使用 CUDA，而是默认 `positive_pair_oversampling=true` 会在正式训练前扫描一次全部 `.npz` cache 来计算采样权重。现在这一步有进度条，并会缓存权重，第二次启动会明显变快；若只想快速验证训练链路，可加 `--no-positive-oversampling`。
3. 对 critical agent index 与 model-visible agent tensor 不一致的问题，当前 runtime masking 是安全的：不会把不可见 agent 的监督错误施加给被 clamp 的其他 agent。但它会丢弃不可见 critical slot 的监督信号。是否需要重建 labels/cache 应由 `11_diagnose_tensor_cache_visibility` 的结果决定：如果不可见比例很低，可以继续使用现有 cache；如果比例较高，建议重建 labels 或至少重建 tensor cache。


## v4 额外修复

- 本轮 `scatter_add_` 报错不是原版 cache 的 critical-agent 不适配导致，而是 `--amp` 下 natural source distribution loss 的 dtype 处理错误。
- 已将 source-distribution target accumulation、mixture NLL、set minADE、priority expectation 等 natural 分支关键损失切到 fp32 计算，保留梯度回传并兼容 AMP。
- natural branch minADE 现在按论文中的 OBS / neutral / priority-preserving 相对权重配置计算，不再只是简单三分支平均。
- 旧 cache 仍会 runtime mask 不可见 critical agent；新 cache 仍建议通过 track_id -> WOMD state/id 生成 input_index。
