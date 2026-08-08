# COWP v16.8.7 本轮审计与优化结论

## 1. 本次 FileNotFoundError 的根因

`tensor_cache_val_waymax_transport_v16_8` 是 overlay，不是独立物理 cache。`26_augment_transport_labels.py` 在 overlay 模式下把输入 raw NPZ 以符号链接暴露在输出目录，并把 `cowp/transport/*` 写到 hidden sidecar。你删除 `tensor_cache_val_waymax` 后，目录名和链接本身仍可被 `glob("*.npz")` 看见，但链接目标已不存在；因此 `COWPNpzDataset` 建索引正常，直到 `np.load` 才失败。

这不是 `.npz` 内部索引损坏，也不是 10040e572b831a04 这个 scene 特殊。

## 2. 为什么可以改用 tensor_cache_val

旧 `12_attach_waymax_candidate_outcomes.py` 会读取 base tensor cache 的全部数组，原样保留，然后仅追加 `waymax/*` 数组。因此 `tensor_cache_val_waymax` 的 COWP proposal/witness/natural/response 内容与 `tensor_cache_val` 相同。另一方面，proposal ceiling / paired probe 不读取 `waymax/*`，所以直接使用 `tensor_cache_val` 是更干净的 old-bank reference。

同理，transport augmentation 的 REQUIRED_INPUT_KEYS 也不包含 `waymax/*`。如果旧 `.transport_v16_8` sidecar 尚在，可以安全复制 sidecar 并把 base link 指向 `tensor_cache_train/val`，无需重跑 Waymax。

## 3. micro probe 当前真实状态

请求为 64 hard + 128 random，其中场景 `a58c0a42edd65669` 同时出现在两组，因此 union=191。`fresh_probe_profile.jsonl` 恰好有 191 个唯一场景，全部 status=`written`，没有 filter/error。说明 fresh PCHR label generation 已完整完成，当前失败只发生在 compare old cache 阶段。

因此本轮不应重跑 label build。

## 4. 是否现在 full rebuild

当前仍为 HOLD。旧 bank 已知 PBTR proposal ceiling 不可行，但 PCHR 是否实质降低 PBTR floor 尚未得到 paired verdict，因为 compare 在 broken legacy overlay 上失败。先从 `tensor_cache_val` 恢复 191-scene compare；micro 通过后再 1200 strict probe。只有 strict `promote_to_full_rebuild=true` 才进行 full fresh build。

## 5. 如果最终需要 full rebuild，从哪里开始

- index：复用现有 WOMD `index_train.jsonl/index_val.jsonl`；无需重做。
- scene set：用 surviving `tensor_cache_train/val` 的 filename set 作为严格 allowlist，保证新旧 paired。
- labels：必须回到 WOMD Scenario proto fresh 生成。旧 tensor cache 不可能 retrofit PCHR/BCS-RMR proposal tensors、witness 和 NCF。
- tensor cache：从 fresh labels + WOMD tf.Example 合并。
- cached candidate Waymax outcomes：默认不构建；主 mechanism 和真实 online Waymax 不依赖它。
- transport：v16.8.7 直接内嵌 fresh labels/tensor cache，不再做 post-hoc overlay。

## 6. 数据构建速度分析

191-scene profile 中 `label_engine_s` mean≈287.33s。以 label_engine aggregate 为分母：

- safe responses：31416.2 / 54880.0 ≈ 57.2%；
- witness：19043.3 / 54880.0 ≈ 34.7%；
- critical-agent：4071.8 / 54880.0 ≈ 7.4%；
- candidate generation 本身不足 1%。

所以 PCHR/BCS-RMR candidate generation 不是 4 天耗时的原因。真正成本是 response-search + witness/root-conditioned certification。

本轮采取的是**不改变标签质量**的加速：allowlist producer 过滤、scene-set 精确复用、BLAS/TF 单线程 worker、跳过 full-train cached Waymax replay、取消重复 transport augmentation、严格 resume fingerprint。暂不通过缩小 response/root profile budget 获得速度，因为那会改变 q/OPR/NCF ground truth。

## 7. 新 fresh-cache 设计

新 cache 是实体、自包含 NPZ，并直接包含完整：

- proposal provenance；
- natural/response/witness；
- `transport/mode_*`；
- root recovery / low-safe score / target confidence；
- root min safe burden；
- transported OPR；
- canonical root weight。

这样删除任何可选 Waymax outcome cache都不会使 TRAIN_CACHE/VAL_CACHE 失效。

## 8. 工程修复

- dataset 初始化阶段提前检查 broken symlink；
- probe compare 只打开 requested IDs；
- legacy transport overlay 提供可验证 rebase 工具；
- fresh fingerprint 扩大到 safe_response/witness/burden/priority 等所有标签语义源；
- fresh full cache 训练前全量 integrity audit；
- full merge 后重新计算 validation proposal ceiling，若数学上不可行直接阻止 GPU training。

## 9. 当前最短路径

先执行 `RECOVER_V16_8_6_PRIORITY_MICRO_PROBE_FROM_BASE_CN.sh`。如果 micro screen 失败，到此停止并分析 PCHR；如果通过，再跑 400+800 strict probe。这个决策链比现在直接 full rebuild 更节省时间且证据更强。

## 10. 旧“构建数据集指令”的两个可复现性风险

旧指令中 train attach 输出目录是 `tensor_cache_train_waymax`，但紧随其后的 verify 写成 `tensor_cache_train_waymax_bal12_safety`；两者不一致。另一个风险是 train replay 命令最后 `--metric-set safety \` 仍带续行符，如果直接粘贴完整文本，下一行可能被 shell 吞入同一命令。v16.8.7 的 wrapper 不再使用这些手工路径拼接。
