## 2026-10-13 EIC-DRR 科学核验更新（请先阅读）

优先阅读 `SCIENTIFIC_DECISION_EICDRR_CN.md`，按 `bash NEXT_EICDRR_COMMANDS_CN.sh help` 查看新实验。
**新证据仍不足**：原生 WOMD 80 检查/76 有效，三套 IDM 参数共享同一结构；固定 non-ego 碰撞对随 ego 动作变化仅 heterogeneous 一例，安全替代动作进展至少相差 12 m，因此 3 m 双侧进展匹配下为 **0/76**。已添加配对原 JSON 审计、ghost-response 参照、Waymax IDM 试验性闭环适配、将所有 Compact-5k 7,200 个 ID 排除的列表。不要将本包称为已跑通 Waymax 或 GameFormer/RA-CBF/JFP 全部基线。

本地已执行 `python -m pytest -q tests --ignore=tests/test_core.py`，结果 **56 passed, 2 warnings**；所排除的 `test_core.py` 依赖此精简包没有包含的独立 `src/ocrap` 包。无 ignore 的全目录测试还会误收集第三方 GameFormer/RA-CBF demo，缺少它们的单独依赖。请不要把这种 collection error 混同为新增 probe 测试失败，也不要将选择性 56 项冒称为全仓库所有测试通过。Waymax 试验性入口尚未在服务器的版本和数据上运行。

> **2026-10-12 hotfix notice**: `HOTFIX_README_CN.md` and `SCIENTIFIC_DEBUG_AND_BASELINES_CN.md` supersede previous native WOMD sample commands where they differ.

# COWP 精简科学验证包

此为上传 `COWP(1).zip` 的 **新的独立副本**，不会触碰服务器 `/home/senzeyu2/code/COWP`。重要：**不要直接覆盖旧仓库**；旧仓库含大量历史实验与另一个 `src/ocrap` 工作流，本包不包含它们。对旧仓库执行任何删除前，先自行完整备份并比对训练/评测脚本。

## 保留

- COWP 正在使用的 core/data/label/models/planning/geometry/waymax_eval/external_baselines/utils 代码。
- 原 COWP/RCRSO 的主要数据缓存构建、训练、Waymax 评测、sidecar 构建/合并、RCRSO 训练及 Stage-0 测试入口。
- 稳定且无版本号的 `configs/{label,data,eval,train,model}.yaml` 与当前 RCRSO runtime 绑定。
- 部分核心测试、RCRSO 语义测试、新研究探针测试，及用于过滤历史开发集的 `reference_manifests/development_exact200_ids.txt`。
- `run.sh` 为稳定唯一入口，新增科学假设不引入新的 `V...` 脚本名。

## 不包含（**非零风险的工程剪枝**）

- 所有历史 `NEXT_RUN_COMMANDS_V*.sh`、历史版本专用测试/分析、patch、changelog、reference_results 等，不应误认为这些文件永不有用；它们仍保留在你上传的原始 ZIP 供追溯。
- 原仓库 `src/ocrap` **另一套独立算法/实验代码**和专属版本执行脚本；这部分不是当前 COWP/RCRSO 或新探针运行链的依赖。如果你仍运行 OCRA 系列任务，应继续使用旧仓库，而不是本包。
- 服务器 checkpoint、WOMD 和 Compact-5k tensor caches，它们不在上传包里，也不会自动删除或搬迁。

这个清理是**有选择的运行发行包**，不是将 1,232 个历史文件全部重新命名并保证全部工作流可用。该项目的 `outputs/` 是空目录，所有新实验输出都写到此处。启动前检查 `BASE_CKPT` 是否指向可读取 checkpoint，必要时与旧 outputs 做只读链接。

## 首轮最有价值命令

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"
bash run.sh sanity
bash run.sh probe_synthetic
PROBE_SCENES=80 bash run.sh probe_cache
python -m research_probe.probe --cache-dir "$COWP_ROOT/tensor_cache_heldout_test" \
  --exclude-ids reference_manifests/development_exact200_ids.txt \
  --max-scenes 80 --max-candidates 10 --max-neighbors 3 \
  --response-accels=-5,-3,0,2 \
  --output outputs/research_probe/alternative_response_bank.json
bash run.sh analyze_probe
```

也可以 `bash NEXT_RUN_COMMANDS_CN.sh quick` 顺序运行以上全部，但建议首次逐条运行并检查数据路径、错误数。输出回传：

```
outputs/research_probe/development_diagnostics.json
outputs/research_probe/alternative_response_bank.json
outputs/research_probe/sensitivity_analysis.json
```

如果所有诊断为零，并不证明真实场景不存在；但不支持当前简化响应家族下的 gap。如果多个响应家族均有稳定正例，**下一步仍应先接高保真 action execution + independent reactive simulator + strong decentralized MPC baseline**，不要据此直接重写论文 Abstract 成“已证明”。

当前 `logged_conventional / logged_planner_only / logged_cowp` 只作为 Waymax logged-replay 对照。不对同一场景的未来他车轨迹施加 ego 改变的真实因果响应。

## 运行质量状态

本地环境完成 `python -m compileall`、合成几何/载入 schema demo、29 项 RCRSO/critical-input/新探针测试；**没有**服务器 WOMD 数据、完整 Waymax rollout 和 Stage-0 复跑。完整运行前仍需在你的环境独立运行 sanity 并检查依赖版本。

## 当前 75 场景结果审计后的最高优先级（更新）

请先读 `SCIENTIFIC_DECISION_REPORT_CN.md`：当前 14 个 individual-safe / joint-incompatible 场景的 138 条候选 **全部 joint=false**，没有 in-bank 可行反事实；206 条可以检查后继的候选全部未触发 temporal collapse。因此初步故事目前是 **NO SCIENTIFIC GO**，不应立即进行大规模 RCRSO 或 temporal network 迭代。

新增更保守的诊断实验：

```bash
bash NEXT_EXPERIMENT_COMMANDS_CN.sh sanity
bash NEXT_EXPERIMENT_COMMANDS_CN.sh historical_audit
bash NEXT_EXPERIMENT_COMMANDS_CN.sh full_geometry
```

其中 `full_geometry` 在同一组 ID 上跑 circle vs oriented-box 以及 2 个 response banks，自动输出 `primary_geometry_audit.json`、`alternate_geometry_audit.json` 和 `oriented_response_sensitivity.json`。所有输出在本目录 `outputs/research_probe/`，**不会覆盖旧论文数据或 checkpoint**。`oriented-box` 仍非真实 Waymax 动力学/连续碰撞认证。新加脚本不要混淆为 Studt/R3R 的开源实现；R3R 的真实官方 Julia 仓库 https://github.com/MarshallVielmetti/r3r_decentralized_gatekeeper 已公开。
