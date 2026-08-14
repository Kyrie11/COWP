# COWP v16.8.15 — WOMD v1.3.1 representation-aware split 修复

## 1. 修复目标

v16.8.14 的 `69_audit_womd_split_layout.py` 把 split 名字与 `scenario` / `tf_example` 做了笛卡尔积，因此会生成并审计本地/官方布局中不存在的组合，例如 `tf_example/training_20s`。即使旧脚本最终只用 primary train/validation 决定 hard PASS，这种报告语义仍然错误，也容易让执行者把“not released”误读成“download missing”。

v16.8.15 改成 representation-aware matrix，不再从 split 名字推断另一个 representation 必然存在。

## 2. 当前 COWP 使用的 WOMD contract

### Primary training

- `uncompressed/scenario/training`: COWP label 权威源（对象轨迹 validity、vector map、lane topology、traffic controls、critical/priority/natural/witness）。
- `uncompressed/tf_example/training`: 与 Scenario `scenario_id` 匹配的 model tensor / Waymax source。

### Primary validation

- `uncompressed/scenario/validation`: smoke、strict、held-out mechanism labels。
- `uncompressed/tf_example/validation`: model tensor / Waymax validation source。

### Secondary interaction stress

- `scenario/validation_interactive`
- `tf_example/validation_interactive`

仅作为单独报告的 interaction-focused stress set。使用前先运行 scenario-ID overlap audit；不能假设它与 standard validation 独立，也不能把重叠样本直接相加形成更大的独立 N。

### Blind testing

- `scenario/testing` + `tf_example/testing`
- `scenario/testing_interactive` + `tf_example/testing_interactive`

测试 future GT 隐藏，因此不能用于生成依赖未来轨迹的 natural / transport / witness / NCF 标签，也不用于本地 logged-future Waymax supervision。需要 blind benchmark 时应走官方 submission/evaluator 口径。

### Scenario-only auxiliary

- `scenario/training_20s`
- `scenario/visualization`

当前 COWP 不使用。尤其 `training_20s` 不能通过把 training glob 替换成 20s glob来加入：当前 label/cache/model contract 明确依赖 10 history + 1 current + 80 future 的 91-step窗口，并且没有 paired `tf_example/training_20s` 供当前 tensor/Waymax 链使用。

## 3. v16.8.15 代码变化

`cowp/scripts/69_audit_womd_split_layout.py` 现在分别定义：

- `SCENARIO_SPLITS`
- `TFEXAMPLE_SPLITS`

对某 representation 不存在的 split，报告：

```json
{
  "applicable": false,
  "released_for_this_representation": false
}
```

不会构造 glob、不会调用 shard resolver、不会计算 missing shard，也不会影响 primary gate。

`split-audit` 的 hard PASS 只依赖：

1. `scenario/training`
2. `tf_example/training`
3. `scenario/validation`
4. `tf_example/validation`

optional interactive / blind test / 20s / visualization 仅盘点，不会因为不存在或未下载而使 primary layout 失败。

## 4. 推荐执行顺序

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE="$SOURCE_DATA_ROOT/tensor_cache_val"

set +e
bash NEXT_EXECUTION_V16_8_15_CN.sh split-audit
SPLIT_AUDIT_RC=$?
set -e

echo "split audit rc=$SPLIT_AUDIT_RC"
cat /data0/senzeyu2/dataset/COWP/formal_v16_8_15_support_smoke/womd_v1_3_1_split_layout.json
```

如果 RC 非 0，只检查 JSON 的 `primary_checks`；`training_20s` / `visualization` 的 tf.Example 侧不会成为失败原因。

然后：

```bash
bash NEXT_EXECUTION_V16_8_15_CN.sh preflight
```

只有 primary 9s train/validation 四个数据源完整且抽样 record contract 正确后，再运行 smoke。

## 5. 不变的算法语义

本版本仅修 WOMD split/repr 使用逻辑。v16.8.13/14 的 mechanism-valid、certificate-valid、PRIO typed source、empirical route evidence、tensor visibility、natural/model-support gates 均保持不变。因此旧 v16.8.14 smoke/strict verdict 不能与 v16.8.15 fingerprint 混用，应从新的 v16.8.15 smoke root 重新 promotion。
