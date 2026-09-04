# V16.8.44R1 下一步执行指令（仅修复后重跑 frozen lost7）

本版本是 **engineering-only cache-fidelity repair**。科学方法仍为 `cowp_root_conditioned_control_reachable_responder_support`，V16.8.44 的所有 root/burden/certificate/controller/horizon/checkpoint/dataset 与预注册 Gate 均保持不变。

旧 `v16_8_44_root_conditioned_control_reachable_responder_support` 的 lost7 结果不得用于 GO/STOP，也不得拷贝到 R1 输出目录。

```bash
cd COWP_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh sanity
# 只有 index 缺失才执行：
# bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh lost7_parallel2
bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh analyze_lost7
```

R1 会同时启用 `--profile-policy-runtime`，用来定位 `lost7_parallel2` 的实际时间分布，不改变 planner 输出。

## 冻结科学判据

- `lost7_new_rescues < 2/7`：V16.8.44 **STOP**，随后才允许关闭 analytic responder-support completion，并拆分 natural-root validity / responder-environment compatibility / multi-agent joint realizability。
- `lost7_new_rescues >= 2/7`：再运行 retained3；通过总 rescue retention 后才运行 induced9；二者通过后才运行 remaining29。

优先回传完整 R1 output 目录 ZIP；最小回传集合是 merged JSON、failfast gate JSON、两个 shard JSON、wall-seconds 与 runtime profile 字段。
