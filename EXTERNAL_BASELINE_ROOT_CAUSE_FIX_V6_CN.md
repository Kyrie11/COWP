# External Baseline V6：GameFormer/DTPP 根因修复与数值稳定性说明

日期：2026-08-27

## 1. 本次错误与 V5 错误不是同一层

用户新日志表明 V5 的 FP64 梯度范数回退确实起效：GameFormer 在 epoch 2 的 batch 260、263、270 都成功越过了 `clip_grad_norm_` 的 FP32 全局 L2 norm overflow；但这些 batch 的裁剪前梯度范数已经达到约 `3.18e27`、`1.81e27`、`4.31e25`。到 batch 283，问题不再只是“所有梯度元素有限、但 FP32 norm 归约溢出”，而是出现了真正的 NaN/Inf gradient entries。

真正的非有限梯度首先集中在：

- `encoder.lane_encoder.self_line.*`
- `encoder.lane_encoder.left_line.*`
- `encoder.lane_encoder.right_line.*`
- `encoder.lane_encoder.speed_limit.*`
- `encoder.lane_encoder.interpolating.*`
- `encoder.lane_encoder.stop_sign.*`
- `encoder.lane_encoder.pointnet.*`

随后传播到：

- `encoder.fusion_encoder.layers.0.self_attn.in_proj_*`

因此 V5 修复的是“梯度范数计算溢出”这一症状，但新日志证明更上游的 GameFormer map/lane 表示与递归 decoder 仍处在病态数值区间。

> 注意：V5 中 `clip_grad_norm_` / FP64 fallback 会在 `optimizer.step()` 之前缩放梯度。因此 AdamW 并没有直接接收到 `1e27` 的原始梯度幅值。问题是：全局裁剪只能控制向量长度，不能把由错误输入拓扑或近奇异 backward 产生的病态方向变成正确方向。

## 2. 根因 A：V5 把 WOMD roadgraph 的“点流”错误当成了 polyline

### 2.1 WOMD 的语义

WOMD tf.Example 的 roadgraph 是一个采样点集合，关键字段为：

- `roadgraph_samples/xyz`
- `roadgraph_samples/dir`
- `roadgraph_samples/type`
- `roadgraph_samples/id`
- `roadgraph_samples/valid`

其中 `id` 表示该点来自哪个 vector-map feature；`type` 区分 lane center、road line、road edge、crosswalk 等；`dir` 是该 feature 上的方向。

因此不能按数组位置任意把相邻的 50/100 个点当成一条 lane。相邻下标可以属于不同 lane、road line、road edge 或 crosswalk。

### 2.2 V5 GameFormer adapter 的确定性缺陷

V5 `build_gameformer_map()` 的核心行为是：

1. 只取 flat roadgraph 的前 100 个 XY 点；
2. 把这 100 点写入 `map_lanes[..., lane_slot=0, :, :]`；
3. 其余 lane slots 为 0；
4. crosswalk 基本为 0；
5. 这同一条“lane”复制给 ego 和所有需要预测的 neighbor agent。

这意味着：

- feature boundary 被破坏；
- lane center / road line / road edge / crosswalk 可能被拼成同一 polyline；
- 一个点序列内部可能发生不连续空间跳变；
- 所有 actor 获得完全相同的地图上下文，而不是各自附近的 map elements；
- GameFormer `LaneEncoder` 的 `self_line/left_line/right_line/.../PointNet` 首先接收错误结构，和本次实际最先出现 Inf/NaN 的参数路径高度一致。

### 2.3 V5 DTPP adapter 的确定性缺陷

V5 `build_dtpp_map()` 取前 `50*50=2500` 个 roadgraph 点，直接 reshape 为 50 条、每条 50 点的“lane”，并在 flat point stream 上计算相邻点 heading。

因此如果第 k 个点与第 k+1 个点属于不同 feature，heading 仍被当成同一 lane 内的方向；crosswalk 同样没有按真实 map element 单独构造。

这不是 DTPP 原始 map contract。公开 DTPP 数据处理明确按 `LANE / ROUTE_LANES / CROSSWALK` vector-set map elements 提取，而不是把一个 flat point array 等长切块。

新日志只给出了 `dtpp status=1`，没有 DTPP 自身 traceback，所以**不能从这份日志证明 DTPP 的具体报错就是同一个数值异常**。但上述 DTPP map adapter 缺陷是代码层面可确定的问题，本次一起修复。

## 3. 根因 B：V5 GameFormer decoder 与公开源码存在尺度偏差

V5 clean-room implementation 与 MCZhi/GameFormer 公开源码比较后发现两个确定性偏差。

### 3.1 CrossTransformer 多加了一次 query residual

V5：cross-attention 输出后执行近似：

`LayerNorm(attention_output + query)`

公开 GameFormer 源码：

`LayerNorm(attention_output)`

随后才执行 FFN residual。

V6 已去掉该额外 `+ query`。

### 3.2 多模态 future aggregation 使用 sum 而非 mean

V5 interaction decoder：

`(... * softmax(scores)).sum(dim=2)`

公开 GameFormer：

`(... * softmax(scores)).mean(dim=2)`

对于默认 6 modes，这会改变每个 interaction level 的 feature scale，并在递归 decoder 中反复进入 future encoder / self-attention / cross-attention。

V6 已恢复为源码的 `mean(dim=2)`。

### 3.3 FutureEncoder heading 也恢复源码定义

V6 的 future heading 计算同步到公开 GameFormer：

`atan2(dy, dx.clamp(min=1e-3))`

避免 clean-room 版本继续积累不必要的实现差异。

这些源码偏差能够确定会改变模型尺度和递归交互语义；但仅凭当前一个日志文件，不能严谨地宣称它们单独就是 batch 283 的唯一原因。当前最直接的日志因果线索仍是：**lane/map branch 首先失稳，而 V5 map adapter 恰好严重破坏了 WOMD map feature topology**。V6 因此同时修复地图拓扑和 decoder 源码一致性。

## 4. V6 地图重构

### 4.1 GameFormer

V6 使用 `roadgraph_samples/id/type/dir/valid/xyz` 重建 map elements：

- lane center 只接受 WOMD type 1/2/3；
- crosswalk 单独接受 type 18；
- 用 feature id 保持同一真实 vector-map feature 内的点连续；
- 使用 `dir` 提供/辅助真实方向；
- 每个被预测 actor 根据自己的当前 XY 独立选择附近 lane/crosswalk；
- 不再把 road line、road edge 拼进 lane center；
- 不再把一条 fake lane 复制给所有 actor；
- raw valid bit 与 raw finite geometry 必须同时成立。

由于 WOMD tf.Example 没有直接提供 GameFormer Scenario proto 中完整的 lane boundary / speed-limit / traffic-light-to-lane topology，V6 不伪造不存在的语义；缺失属性保持安全默认值。这是 WOMD tf.Example → GameFormer 的 source-aware adaptation，而不是声称逐字段复现 Scenario proto 的所有 map metadata。

### 4.2 DTPP

V6 同样先按 WOMD feature id 分组，再从 ego 周围选择 lane center 和 crosswalk elements；不再按固定点数切 flat stream。

DTPP 原始代码基于 nuPlan map API，本项目使用 WOMD，因此这是“保留 vector-set map element 语义”的跨数据集 adaptation，而不是冒充原 nuPlan map tensor。

## 5. Train / offline eval / Waymax closed-loop 地图语义统一

V5 Waymax 在线 batch 只把 roadgraph 的 `xy + valid` 传给 external adapter，丢失 `ids/types/dir`。如果只修训练 adapter，闭环会再次退回错误的 flat-map 行为，形成 train-test mismatch。

V6：

- `policy_wrapper.py` 保留 Waymax `RoadgraphPoints.ids/types/dir_x/dir_y/valid`；
- 转成与离线 cache 一致的 `roadgraph_samples/id/type/dir/valid/xyz` 风格字段；
- `waymax_policy.py` 使用同一套在线字段；
- 老版本 Waymax 若真的没有 `id/type`，不会伪造一个统一 `id=-1` 把所有点错误合成一个 feature。

因此训练、offline eval 和 Waymax closed-loop 使用同一 map topology contract。

## 6. V6 启动时 map contract 自检

正式 learned baseline 训练/离线评估启动时会抽检数据集首/中/尾样本，要求 roadgraph：

- XY/XYZ 存在；
- `id` 存在；
- `type` 存在；
- `dir` 存在；
- `valid` 存在；
- 五类字段 point count 对齐。

正常日志应出现：

```text
map topology contract baseline=gameformer split=train mode=womd_feature_id_topology ...
map topology contract baseline=gameformer split=val mode=womd_feature_id_topology ...
```

如果正式 cache 缺这些字段，V6 会在训练开始前报数据契约错误，而不是静默运行旧的 V5 flat-map adapter。

项目的 `parse_tfexample.py` 遍历 tf.Example 中所有 feature；`build_cache.py` 再把所有解析字段写成 `womd__...`，所以从正式 WOMD tf.Example 构建的当前 tensor cache 按代码链应保留 `roadgraph_samples/id/type/dir/...`。V6 的启动检查会在用户机器上最终验证这一事实。

`--allow-legacy-flat-map` 仅保留给显式调试，不应在正式实验中开启；原命令不需要加该参数。

## 7. 数值保险层：不再沿病态有限梯度方向更新

V5 FP64 fallback 保留，因为它正确区分：

1. gradient entries 本身有限，但 FP32 global norm reduction overflow；
2. gradient entries 真正含 NaN/Inf。

V6 增加第三类：

3. gradient entries 和 FP64 norm 都有限，但 pre-clip global norm 已是病理级 outlier。

默认：

```text
--max-preclip-grad-norm 1e8
```

如果超过阈值：

- 记录 `top_postclip_gradients`，帮助定位主导 branch；
- `zero_grad()`；
- **不执行 optimizer.step()**；
- 计入 `exploding_gradient_skipped_batches`；
- 连续达到 3 个则中止，防止系统性异常被掩盖。

该机制不是根因修复，而是防止某个孤立病态 batch 在根因修复后仍然把优化器推向不可靠方向。

## 8. 真正 NaN/Inf 的 bounded skip 语义

V5 在第一个 true non-finite gradient 就立即结束整个 baseline；parallel2 因此整个 pair 被判失败。

V6 对 isolated numerical bad batch 做 bounded skip：

- bad batch 永远不执行 optimizer step；
- 默认全 epoch true-numerical skip 上限为 0.5%；
- 连续 3 个 true numerical failures 立即中止；
- 总 skip 上限仍为 2%；
- 日志仍输出具体 `nonfinite_gradient_paths`。

这样一个孤立异常不会浪费整个 20-epoch 训练，但重复/系统性错误仍然硬失败，避免“靠跳数据把实验跑完”。

## 9. Training contract 与原命令兼容

V6 contract：

```text
v6_womd_map_topology_source_fidelity_20260827
```

因为模型 forward/map representation 已改变，旧 V5 checkpoint 不应被静默复用。原训练命令会在**相同输出目录结构**中重新训练 V6 learned baselines。

输出根目录未改变：

```text
outputs/external_sota5_v16_8_33
```

训练命令不变：

```bash
GPU0=0 GPU1=1 nohup bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all > logs/run.log 2>&1 &
```

Waymax 闭环命令不变：

```bash
PARALLEL2=1 \
GPU0=0 GPU1=1 \
WOMD_VALIDATION_TFEXAMPLE_DIR=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation \
bash RUN_5_SOTA_BASELINES_COWP.sh waymax all
```

## 10. 已执行验证

### 10.1 单元/回归测试

V6 新增地图/模型数值测试覆盖：

- 多 WOMD feature id 不被拼成 fake lane；
- lane types 与 crosswalk type 分离；
- 两个 actor 分别取得各自附近 lane；
- topology contract 缺字段可检测；
- GameFormer CrossTransformer 无额外 query residual；
- InteractionDecoder 使用 modal `mean`；
- Waymax `ids/types/dir` 在线传递；
- structured-map GameFormer forward → loss → backward 全参数 gradient finite。

external-baseline + Waymax 专项回归集通过。

### 10.2 实际训练入口 smoke test

使用临时 structured WOMD cache（包含真实形态的 `roadgraph id/type/dir`）调用实际 `20_train_external_baseline.py`：

- GameFormer：完成 epoch、AdamW step、checkpoint；0 numerical skip / 0 exploding skip / 0 FP64 fallback；最大 pre-clip norm 约 1e2；
- DTPP：完成 epoch、AdamW step、checkpoint；0 numerical skip / 0 exploding skip / 0 FP64 fallback；最大 pre-clip norm 约 1e1；
- GameFormer offline eval：通过 topology contract 并完成输出。

这是代码链路验证，不等价于在当前环境复现用户 GPU 上 epoch 2 batch 283。当前环境没有用户的完整 5000-scene tensor cache 与相同 CUDA 训练环境，因此不能声称已经对该精确 batch 做 bitwise reproduction。

## 11. 重跑时优先观察的日志

训练开始后先确认：

```text
contract=v6_womd_map_topology_source_fidelity_20260827
map topology contract ... mode=womd_feature_id_topology
```

在原来 epoch 2 / 60%-70% 附近，健康情况应不再出现 `1e25-1e27` 量级 pre-clip norm。

如果仍出现病态有限梯度，V6 会明确打印：

```text
skipped pathological finite gradient norm preclip_norm=... threshold=1e+08 ...
top_postclip_gradients=[...]
```

如果出现真正 NaN/Inf：

```text
skipped non-finite gradients ...
first_nonfinite_gradient=...
nonfinite_gradient_paths=[...]
```

这些日志可以进一步区分：仍是 map encoder、decoder、loss，还是某个不同 baseline 的独立问题。
