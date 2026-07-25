# COWP v16 本轮结果诊断与 v16.1 执行方案

## 1. 本轮究竟运行到了哪里

本轮先后完成：

- 90 项旧版回归测试；
- v9 raw/transport cache 复用门禁；
- train/val transport 全量诊断；
- train/val cache alignment；
- natural oracle；
- exact model-facing anchor preflight；
- causal engineering audit。

其中 `model_anchor_preflight_val.json` 已通过：critical mapping、first-step anchor、
typed basis 1 s 和 8 s 均通过。真正错误发生在 natural 训练启动后、epoch `-1`
的第一个 validation batch，训练优化尚未开始。

错误为：

```text
RuntimeError: The size of tensor a (6) must match the size of tensor b (24)
at non-singleton dimension 2
```

根因位于 `_natural_mode_usage_loss`：`mode_source` 原本是 `[B,A,M]`，代码错误地
使用 `psrc[...,0]`，将模式维删除为 `[B,A]`，再与 `[B,A,M,R]` 的 pairwise
logits 广播。真实 batch 中 `A=6、M=24`，因此失败。

## 2. 七项验证结论

| 待验证内容 | 本轮能否验证 | 原因 |
|---|---|---|
| v15/v16 decoder 是否有效 | 否 | 只有零残差解析 basis preflight，无训练后 checkpoint |
| 新 loss 是否有效 | 否 | loss 在首个 validation batch 即报错 |
| OBS residual capacity 是否有效 | 否 | 没有 main/no-capacity 对照结果 |
| natural gate 是否改善 | 否 | 没有 history 和 learned gate |
| planner 是否改善 | 否 | 未进入 transport/planner |
| selector 是否改善 | 否 | 无 learned-offline/BCOT 结果 |
| 在线 Waymax CR/offroad/progress 是否改善 | 否 | 未进入 probe/full Waymax |

因此本轮不存在可归因于模型的实验增益。可确认的仅是数据与工程前置链路通过。

## 3. 现有结果能说明什么

### 3.1 数据链路可靠

- raw/transport 文件数为 train 20,440、val 5,013；
- critical unmapped/invisible 为 0；
- response root 越界为 0；
- selected Waymax rollout success 为 1.0；
- train/val transport 全量诊断通过；
- v9 cache 没有有限 logdiv，继续禁止 logdiv supervision。

### 3.2 v16 初始化存在学习空间，但不是学习结果

同一 2,000 场景 preflight 中：

- typed basis 8 s：2.1775 m；
- OBS 8 s：4.5176 m；
- NEU 8 s：1.1608 m；
- PRIO 8 s：1.1957 m。

OBS 初始值高于 4.0 m gate，说明 learned dynamics residual 必须真实改善 OBS 才能晋级。
这些数值来自零初始化的解析 basis，不能作为 v16 decoder 已有效的证据。

## 4. v16 算法优化是否有益

当前只能评价结构合理性，不能评价经验有效性：

- 动力学积分避免位置、速度、航向互相矛盾，是有效的工程/归纳偏置；
- OBS 专属控制容量为拟合真实曲线提供空间；
- OBS gain、NEU/PRIO preservation 和物理一致性 loss 提供了合理约束；
- learned-vs-basis gate 避免“解析先验好、神经残差无效”仍被宣称成功。

但这些组件是否提升 minADE、planner 或闭环必须由本轮修复后的训练和消融决定。
v16.1 不继续修改算法，避免在没有证据时反复改变变量。

## 5. v15 数据集是否与模型强绑定

不是强绑定。v15 数据协议主要改变：

- natural root 的生成与过滤；
- OBS contamination 权重；
- map compliance；
- 由 natural roots 级联得到的 response/witness/transport 标签。

CNOB loss 直接消费的是 `natural/traj、valid、weight、source`，额外的 contamination/map
字段主要用于审计。换言之，v15 数据集是一个独立的“因果监督协议”假设，不是仅为
某一 decoder 定制的数据格式。

如果 CNOB 在 v9 上无效，不能立即判定 v15 数据无用。建议小规模矩阵：

1. CNOB + v9；
2. CNOB + v15 interaction-heavy pilot；
3. 简化 typed residual + 同一 v15 pilot。

判定规则：

- 两种模型都从 v15 pilot 获益：数据协议有效；
- 简单模型获益而 CNOB 不获益：CNOB 约束或容量有问题；
- CNOB 在 v9 有益、v15 无额外收益：不必全量重建 v15；
- 两种模型均无收益且标签审计正常：v15 标签出发点需重构。

## 6. 论文 method/appendix 的独立风险

这些问题与本轮 crash 无关，但投稿前必须对齐：

1. 论文写的是 graph-conditioned lattice-MPC/OCP，当前 candidate generator 主要是
   constant-acceleration、smooth-stop、lateral-offset 和 conflict-timing primitives。
2. 论文 appendix 写 OBS Transformer-GMM 和 neutral diffusion，当前实现是 typed analytic
   basis + bounded dynamics residual。
3. 论文实验写非 ego 为 logged/learned/rule reactive mixture，当前正式协议仍是 logged replay；
   不能用该协议证明真实 agent reaction 或 burden reduction。
4. 代码有 13 个 macro IDs，其中 `LOGGED_EGO`、`NEUTRAL_EGO` 是辅助 anchor，`PAD` 是填充；
   论文只列了 10 个 deployable intents。应明确区分“可执行意图”和“训练/评测 anchor token”。
5. 固定 macro taxonomy 对交叉口拓扑、掉头、绕障、非机动车/行人交互的覆盖有限。更稳健的
   表述是“拓扑关系 × 顺序关系 × 连续终端条件”的组合式 action descriptor，而非声称固定枚举完备。

在闭环结果出来前建议先记录这些差异，不立即大改论文叙事。若最终采用当前实现，必须将
method/appendix 改为实现一致版本；若坚持论文原架构，则需要补实现而不是保留不真实描述。

## 7. v16.1 工程优化

- 修复 `[B,A,M]` mode mask 维度错误；
- 增加真实维度 forward/loss/backward 测试；
- 整个流程默认 nohup 后台运行；
- 总日志进入 `OUT_ROOT/logs/driver.nohup.log`；
- PID 写入 `OUT_ROOT/logs/driver.pid`；
- 每个阶段继续保留独立 log；
- 新增运行状态脚本；
- natural DataLoader 默认 8 workers/prefetch 2；
- transport/planner 保持 4/1，防止大 tensor 预取造成内存问题；
- 默认 `DIAG_PROFILE=fast`；
- transport 诊断使用线程读取和流式统计；
- cache sufficiency 默认抽样，不再每次扫描 25k 场景；
- 已存在的合法诊断 JSON 自动复用；
- 新增严格 code/config provenance，阻止 stale artifact 混用；
- 提供独立 full data audit，冻结论文版本时再运行。

## 8. 下一轮晋级顺序

1. 运行 v16.1 main natural；
2. 通过 absolute natural gate；
3. 通过 learned-vs-basis effectiveness gate；
4. 运行 no-effectiveness-loss 与 no-OBS-capacity 两项消融；
5. 只有归因 gate 通过，才进入 transport/planner；
6. learned-offline 验证 planner/selector；
7. paired Waymax probe；
8. full Waymax、多 seed、paired CI；
9. 再决定是否构建 v15 pilot/full dataset。
