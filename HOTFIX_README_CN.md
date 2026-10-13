# WOMD TFExample 修复补丁

作用于已有的 `/home/senzeyu2/code/COWP` 代码库，覆盖原 `research_probe/native_womd.py` 和研究验证入口；不会修改服务器 `/data0/.../WOMD` 原始数据。

**根因**：原 `build_cohort` 变量 `mode` 被 `for mode,k in ...` 覆盖。第一条 TFExample 解码后模式变成 `interaction`，第二条开始误入未导入 `scenario_pb2` 的 Scenario 分支。

使用：

```bash
cd /home/senzeyu2/code/COWP
cp research_probe/native_womd.py research_probe/native_womd.py.bak
unzip -o /home/senzeyu2/code/COWP_NATIVE_TFEXAMPLE_HOTFIX.zip -d .
export WOMD_RAW=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export COWP_DATA=/data0/senzeyu2/dataset/COWP
bash NEXT_SCIENCE_COMMANDS_CN.sh sanity
bash NEXT_SCIENCE_COMMANDS_CN.sh smoke_raw
bash NEXT_SCIENCE_COMMANDS_CN.sh sample_raw
bash NEXT_SCIENCE_COMMANDS_CN.sh simulate_reactive
bash NEXT_SCIENCE_COMMANDS_CN.sh baseline_readiness
```

详细方法、基线所需 Scenario 下载说明、限制与审计门槛：`SCIENTIFIC_DEBUG_AND_BASELINES_CN.md`。
