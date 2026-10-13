# Native WOMD TFExample 解码修复、H1--H4 证据门槛及强基线接入

## 1. 报错的准确原因和修复

`build_cohort()` 的 `mode` 一开始由 `locate_shards` 返回 `'tfexample'`；在处理第一条记录时，同名变量 `for mode,k in [('blind',...),('interaction',...)]` 被覆盖为 `'interaction'`。后续记录误入 Scenario proto 解码分支；该分支所需的函数局部 `scenario_pb2` 没有导入，因此报 `local variable 'scenario_pb2' referenced before assignment`。报错 1207/1208 并非 1207 条 TFRecord 坏数据。

本补丁做了四项修复：

1. 只用 `record_format` 表示 TFExample/Scenario 格式，`cohort_name` 表示选样桶，二者不共享变量。
2. 使用 eager `tf.data.TFRecordDataset(...).as_numpy_iterator()` 替代弃用的 `tf.compat.v1.io.tf_record_iterator`。
3. `cohort_manifest.json.source_format` 正确记录格式；解码失败错误信息依真实格式生成。
4. 增加连续多条 TFExample 的 mock 回归测试，即使本地没有 TensorFlow 也可测试曾出现的 bug。

**不需要**下载 Scenario proto 来修复此问题。你已有的 `uncompressed/tf_example/validation/validation_tfexample.tfrecord-*` 足够完成原生 WOMD 的观测/盲抽样及简化干预筛查。

## 2. 在你的服务器上安装热修补

请先备份 COWP。将 `COWP_NATIVE_TFEXAMPLE_HOTFIX.zip` 上传到服务器，例如 `/home/senzeyu2/code/`：

```bash
cd /home/senzeyu2/code/COWP
cp research_probe/native_womd.py research_probe/native_womd.py.bak
unzip -o /home/senzeyu2/code/COWP_NATIVE_TFEXAMPLE_HOTFIX.zip -d .

export WOMD_RAW=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export COWP_DATA=/data0/senzeyu2/dataset/COWP

bash NEXT_SCIENCE_COMMANDS_CN.sh sanity
bash NEXT_SCIENCE_COMMANDS_CN.sh discover_raw
bash NEXT_SCIENCE_COMMANDS_CN.sh smoke_raw
```

小规模烟测应在 `$COWP_DATA/research_native_womd/cohort_smoke/cohort_manifest.json` 输出 `counts.bad_records=0` 和 `source_format="tfexample"`。如果没有符合至少两个邻车的 48 条样本，可扩大烟测到 `sample_raw`，但不应产生相同的 `scenario_pb2` 变量错误。

运行真正的盲抽样：

```bash
export WOMD_SPLIT=validation
export WOMD_FORMAT=tfexample
export WOMD_SHARDS=4
export WOMD_PER_SHARD=350
export WOMD_RANDOM=120
export WOMD_INTERACTIVE=120

# 可选：此前 Compact-5k / exact200 的 scenario-ID，一行一个
# export WOMD_EXCLUDE_IDS=/data0/senzeyu2/dataset/COWP/excluded_development_ids.txt

bash NEXT_SCIENCE_COMMANDS_CN.sh sample_raw
bash NEXT_SCIENCE_COMMANDS_CN.sh simulate_reactive
bash NEXT_SCIENCE_COMMANDS_CN.sh baseline_readiness
```

以上新增输出集中于 `/data0/senzeyu2/dataset/COWP/research_native_womd/`，不会在 WOMD 原始目录创建任何新数据。

**提醒**：4 个分片各取前 350 条，只能当作 development / mechanism discovery。正式估计发生率需以预注册方案覆盖足够多原生分片，并在独立场景 IDs 上复核；检验场景候选不能用未来真值来选择。

## 3. 当前 H1--H4 的科学状态（修复后的原生实验尚未运行）

| 假设 | 上轮朝向矩形结论 | 当前能否宣传为科学发现 | 下一道证据门槛 |
|--|--|--|--|
| H1：个体可行、联合不相容 | 2/75，且均在初始时刻有重叠 | 否 | 原生数据排除初始碰撞后，完整动力学 + 高覆盖联合响应搜索下仍有一致失败证据 |
| H2：联合安全响应存在，但独立选择不相容 | 2/75，未证明确由 ego 干预引起 | 否 | 原生独立策略下出现行动相关的失配；反事实 ego 替代动作有可比进展和可测改善 |
| H3：当前可行，执行后无安全恢复方案 | 0/75 | 否 | 正确的后继状态、真实执行动作、备份安全集/可达性搜索、响应库覆盖敏感性 |
| H4：强基线尚未充分处理 H1--H3 | 未做忠实强基线 | 否 | 同场景 IDs、信息预算、动作空间、动力学、时间预算与进展约束的配对闭环比较 |

原生 WOMD 的**历史轨迹是 observational evidence**；不能由其中一条未来轨迹识别 `do(ego_action)` 的反事实结果。当前 `reactive_stress.py` 的恒朝向纵向 IDM-inspired 模型仅是候选产生器，不是经过独立验证的闭环交通仿真。

**现阶段更优先的问题仍是“独立响应的实现可能性是否受 ego 动作影响，且能否被合理替代动作改善？”**；但这一定位本身已经面临 JFP conditional joint distribution 与 GameFormer 多智能体交互推理等已有工作。若无真实干预增益和相对于强基线的可重复 gap，不应提交“首个考虑 joint-realizability”的创新主张。

## 4. GameFormer：什么时候应该接入，如何接入

用户上传的 GameFormer ICCV 2023 仓库 README 明确：

- 提供 WOMD 的交互预测 + 选定场景的 **open-loop planning**；
- **没有 WOMD 的完整 closed-loop planning 源码**；
- `open_loop_planning/data_process.py` 和 `open_loop_test.py` 实际使用 `scenario_pb2`、map features、traffic lights、route/reference line。
- 运行模型必须有与代码匹配的 trained `model_path`；上传源码包中没有该模型权重。

因此只依赖 TFExample 不足以忠实执行 GameFormer 作者开源的 open-loop planning。**当前阶段不需要下载 Scenario 文件来修复报错**；若下一阶段确实要运行 GameFormer，才下载官方原始 Scenario validation，并在 COWP 数据目录保存：

```bash
# 官方下载：需先同意 Waymo license 并具备 Google Cloud 访问权限
mkdir -p /data0/senzeyu2/dataset/COWP/third_party_datasets/womd_scenario
# 请先用 gcloud storage ls 验证你有访问权限、路径确实存在，再复制大数据：
gcloud storage ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/
gcloud storage cp --recursive \
  gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/ \
  /data0/senzeyu2/dataset/COWP/third_party_datasets/womd_scenario/
```

不同版本和官方存储布局可能不完全相同；若 `ls` 报无权限或不存在，须从 Waymo 官方下载页面确认 GCS prefix，不要盲猜复制。

然后在具有 `tensorflow` 和 `waymo_open_dataset.protos.scenario_pb2` 的隔离环境中：

```bash
export WOMD_SCENARIO=/data0/senzeyu2/dataset/COWP/third_party_datasets/womd_scenario/validation
export GAMEFORMER_SRC=/home/senzeyu2/code/COWP/third_party/GameFormer-main
export GAMEFORMER_CKPT=/data0/senzeyu2/dataset/COWP/external_checkpoints/gameformer/open_loop.pt
bash NEXT_SCIENCE_COMMANDS_CN.sh baseline_readiness
bash NEXT_SCIENCE_COMMANDS_CN.sh prepare_gameformer_subset
```

**重要**：`prepare_gameformer_subset` 会逐条读取真正的 Scenario proto，按照 `scenario_id` 查找与原生 TFExample cohort 对应的记录，生成匹配的 `gameformer_cohort.csv`。不能仅从 TFExample 文件名推断 Scenario 文件的分片编号。

如前置数据和**兼容的已训练 checkpoint**齐备，在作者的独立 GameFormer 环境中：

```bash
cd /home/senzeyu2/code/COWP/third_party/GameFormer-main/open_loop_planning
python open_loop_test.py \
  --name cowp_matched_open_loop \
  --test_set "$WOMD_SCENARIO" \
  --model_path "$GAMEFORMER_CKPT" \
  --candidate_csv /data0/senzeyu2/dataset/COWP/research_native_womd/gameformer_cohort.csv \
  --device cuda

cd /home/senzeyu2/code/COWP
export GAMEFORMER_RESULT_CSV=/home/senzeyu2/code/COWP/third_party/GameFormer-main/open_loop_planning/testing_log/cowp_matched_open_loop/testing_log.csv
bash NEXT_SCIENCE_COMMANDS_CN.sh gameformer_report
```

此结果**只能作为原作者 open-loop GameFormer 的结果**，不能用于证明 GameFormer 在没有 V2X 的闭环失效。对于闭环比较，必须再建立 Waymax 动作适配及 reactive agent 执行评测协议，评估 oracle-log replay 与 reactive rollout 的差别。

GameFormer 原始 requirements 使用老版本 torch==1.12.0+cu116 和 waymo-open-dataset-tf-2-11-0，千万不要直接 `pip install -r` 到现有 COWP/Waymax 环境；单独 conda 环境并核实 GPU/CUDA/软件包兼容。

## 5. JFP 和 RA-CBF：不能直接转成闭环基线

**JFP（CoRL 2022）**：你上传的 `luo23a.pdf` 明确研究能量图模型与 pairwise potentials、belief propagation，支持基于 AV 候选轨迹做 conditional joint inference（见第 4--5 页）。它正面占据“联合预测一致性”和“AV 条件下的联合行为预测”一般概念；上传的是论文，**没有源码、预训练权重或可执行控制器**。它可作为强预测比较，需要独立实现或作者可复现模型；不能将它默认为安全控制基线。

**RA-CBF / Learning Responsibility Allocations（ICRA 2023）**：用户提供的仓库 README 说明代码主要用于训练/输出责任分配 `gamma`，闭环仿真依赖未公开源码，数据接口为 `nuScenes/trajdata`，并非 WOMD 或 Waymax。若要比较，应明确设计 ① 责任网络 WOMD 状态特征适配，② 安全过滤器控制器，③ 时域动作约束和道路规则，④ 训练/模型权重来源，⑤ 官方条件与修改版之差别。**没有这些环节不能报告“原版 RA-CBF 在 WOMD 失败”。**

推荐现阶段先使用 `baseline_readiness` 生成能力审计，不运行未知权重的模型。基线加入优先序：

1. 当前共享候选集合上的 pairwise, joint, reactive selector（不是 SOTA）。
2. 原生 WOMD 盲抽样 + Waymax IDMRoutePolicy 等真实 reactive model，与等预算对照，优先验证 H2。
3. GameFormer 用 Scenario + checkpoint 做同 ID open-loop，与 reactive rollout 的干预评测严格区分。
4. RA-CBF 的适配式安全控制闭环（须在论文中说明属于复现/扩展）。
5. 如果 H3 在 high-fidelity 中有正例，再优先忠实复现 Studt 的 shared-rule decentralized contingency MPC；R3R 原方法涉及通信，不能去掉通信还称原版。

## 6. 下一轮回传的文件

- `.../research_native_womd/cohort/cohort_manifest.json`（检查 schema、记录解析、场景抽样）
- `.../research_native_womd/reactive_screen.json`（简化响应筛选及候选介入动作）
- `.../research_native_womd/baseline_readiness.json`（缺少什么基线前置条件）
- 若 GameFormer 真正跑通，还包括 `gameformer_cohort_match.json` 和 `gameformer_open_loop_report.json`。

必须先核查 `counts.bad_records=0`、原生抽样与历史开发 IDs 的交叉是否被排除、`rows/errors`、多策略拟合指标及 ego 动作引起的响应变化，再确定论文新问题。**任何 H1--H4 的 GO/STOP 决策都应以此后的独立实验证据为准。**
