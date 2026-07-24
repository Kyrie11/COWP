# COWP v16.3 推荐执行顺序

## 1. 先恢复 natural 学习，不运行下游

```bash
BACKGROUND=1 bash NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh
```

## 2. 检查状态

```bash
OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026 \
  bash CHECK_RUN_STATUS_V16_3.sh
```

通过标准：

- `natural_basis_gate: pass=true`；
- `natural_effectiveness_gate: pass=true`；
- `optimizer_steps > 0`；
- `amp_skips = 0`；
- 不存在 `QUALITY_GATES_BYPASSED.txt`。

## 3. 做新 loss / OBS capacity 控制变量消融

```bash
MAIN_OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026 \
  bash RUN_NATURAL_ABLATIONS_V16_3_CN.sh
```

要求 `natural_component_attribution_gate.json` 的 `pass=true`。

## 4. 再运行 transport、planner、offline 和 Waymax full

```bash
OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_3_natural_ablations_v9labels_seed2026/natural_component_attribution_gate.json \
BACKGROUND=1 bash NEXT_RUN_COMMANDS_V16_3_FULL_CN.sh
```

`ALLOW_QUALITY_GATE_FAILURE=1` 只能用于 `NEXT_RUN_COMMANDS_V16_3_ENGINEERING_SMOKE_CN.sh`，其输出不得用于论文结论。
