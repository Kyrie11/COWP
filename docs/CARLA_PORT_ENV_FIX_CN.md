# RIDE / CARLA 0.9.16 无头环境及端口错误修复（2026-10-27）

## 1. 基于用户上传 RIDE.zip 的确定事实

- 原始 `scripts/00_env.sh:6` 使用 `export CARLA_ROOT="${CARLA_ROOT:-/data0/senzeyu2/dataset/CARLA_0.9.16}"`。它在已有环境变量为非空时直接沿用，因此**不会主动生成** `.../CRADrive/CARLA_0.9.15`；出现该目录说明旧环境变量已被继承，或者执行的不是本包中的那个脚本。
- 原始 `scripts/02_start_carla_headless.sh` 将服务端 RPC 端口写死为 `2000`，没有检查 2000/2001 是否被其他 CARLA 或其他用户占用。
- 原始 `scripts/03_run_carla_case.sh` **只启动 Python 客户端** `ride.carla.run`，不会显式开启第二个 CARLA 服务端；它调用的 `run_trial()` 会多次重载 world，每条 trial 都调用 `get_trafficmanager(8000)`，即使 responder family 是 `behavior` 或 `joint_witness`。
- 当前版本 `ride.carla.run` 检查服务端/客户端 0.9.16，并在配对物理实验中使用 `world.apply_settings()` 的 `no_rendering_mode=True`。

CARLA 进程看到 `Exception thrown: bind: Address already in use` 时，最优先调查的是网络监听套接字冲突，而非显存溢出；随后出现的 `Signal 11` 常是 Unreal Engine 对致命错误的退出方式。**实际究竟是 RPC 2000、数据流 2001、TM 8000，还是 map 重载引起的二次初始化，未拿到服务器 `ss` 和客户端时间线前无法实证判定。**

## 2. 修改文件及意图

| 文件 | 修复 |
| --- | --- |
| `scripts/00_env.sh` | 旧版 0.9.15 根目录自动纠正，显式 `RIDE_CARLA_ROOT`，校验 0.9.16 目录名；过滤 0.9.15 的 `PYTHONPATH` 残留；导出统一 RPC/TM 端口 |
| `ride/carla/ports.py` | 标准库端口状态探针，无需 CARLA wheel，不杀进程；启动时遇 2000/2001 占用立即拒绝 |
| `scripts/00_check_ports.sh`、`00_diagnose_carla.sh` | 只读端口、解释器、进程、GPU 信息诊断 |
| `scripts/02_start_carla_headless.sh` | 检查端口冲突；支持 `CARLA_RPC_PORT` 与可选 `CARLA_GPU_ADAPTER` |
| `scripts/03_run_carla_case.sh`、`03_precommit_scenes.sh`、`01b_probe_anchors.sh`、`03_support_ablation.sh` | 明确传递 RPC/Traffic Manager 端口，连接前预检服务端 |
| `ride/doctor.py`、`scripts/01_doctor.sh` | `--server` 连接到选择的端口，并检查双端版本 |
| `ride/carla/run.py` | 避免入口处对已经加载的同名地图重复 `load_world`；每个正式 trial 保留 world reset，但在同图时改用 `reload_world`；只在 TM family 中创建 TM；增加阶段日志和端口元数据 |
| `tests/test_carla_ports_env.py` | 测试继承旧环境变量、端口冲突拒绝、GPU 参数传递、端口不重叠 |

**科学一致性说明：** 仍对每个实验 branch 重置 world 并设置同步固定时间步，以尽量保持初始条件与随机种子一致；没有为了绕过崩溃而偷偷取消重置或复用上一场景的交通状态。因此，当前版本不能保证消除 CARLA 引擎自身所有 `load_world`/`reload_world` 崩溃。新日志便于判断崩溃发生在何处：`RIDE_WORLD_SWITCH`（首次切图）、`RIDE_WORLD_REUSE`（重用）、`RIDE_TRIAL_RESET_BEGIN`（trial 重置之前）、`RIDE_TRIAL_WORLD_RESET`（重置完成）。

## 3. 在服务器上无损定位冲突（不随意杀别人的进程）

```bash
cd /path/to/RIDE_fixed
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
printf 'CARLA_ROOT=%s\n' "$CARLA_ROOT"
realpath scripts/00_env.sh
bash scripts/00_diagnose_carla.sh

# 额外确认 Linux 端口归属；无 ss 时可以使用 lsof / fuser
ss -ltnp | grep -E ':(2000|2001|2002|2003|8000|8100|8102)\b' || true
ps -eo pid,ppid,user,args | grep -E '[C]arlaUE4|[C]arlaUnreal'
```

先检查 GPU 截图上 PID `1976905` 是否为仍在运行的旧 CARLA 进程：

```bash
ps -fp 1976905 || true
readlink -f /proc/1976905/cwd 2>/dev/null || true
tr '\0' ' ' < /proc/1976905/cmdline 2>/dev/null; echo
```

该 PID 仅来自上传截图，在下次启动时可能已不存在、变化或被系统复用。只有**明确确认属于自己、且不再承担其他任务**时，才考虑向该 PID 发送 `kill -TERM <PID>`；绝不在共享服务器执行不加区分的 `pkill -f Carla`。

如果端口 2000 已被一个**正确的** CARLA 0.9.16 服务端使用，不要再在端口 2000 启动第二个服务端：复用现有实例，随后执行 `python -m ride.doctor --server`，确保不影响别人的同步实验。

## 4. 建议流程 A：只使用现有 2000 端口 CARLA（单卡即可）

**终端 A**：仅当端口 2000/2001 空闲时运行：

```bash
cd /path/to/RIDE_fixed
conda activate ride-carla
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2000
export CARLA_TM_PORT=8100
source scripts/00_env.sh
bash scripts/00_check_ports.sh
bash scripts/02_start_carla_headless.sh
```

**终端 B**：服务器就绪后，保持与 A 端相同的环境变量，使用全新结果目录：

```bash
cd /path/to/RIDE_fixed
conda activate ride-carla
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2000
export CARLA_TM_PORT=8100
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_fixed_run_001
export CARLA_TOWN=Town04
export RIDE_ANCHORS=1
export RIDE_REPEATS=1
source scripts/00_env.sh
python -m ride.doctor --server
bash scripts/01b_probe_anchors.sh convoy Town04
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

注意：`03_run_carla_case.sh` 会进行 1 个锚点、2 个 responder families、4 个 ego profiles 的 8 条 rollout；每条都有独立 world reset。`03_precommit_scenes.sh` 只冻结锚点，不做响应物理 rollout。

## 5. 建议流程 B：GPU 1 启动**独立**端口服务端（绕开原有 GPU 0 任务）

如果已有 2000/2001 端口由其他任务占用，或 GPU 0 当前还在服务其他 CARLA 实验：

**终端 A（新 CARLA 服务器）**

```bash
cd /path/to/RIDE_fixed
conda activate ride-carla
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2002
export CARLA_TM_PORT=8102
export CARLA_GPU_ADAPTER=1
source scripts/00_env.sh
bash scripts/00_check_ports.sh
bash scripts/02_start_carla_headless.sh
```

**终端 B（新 RIDE 客户端）**

```bash
cd /path/to/RIDE_fixed
conda activate ride-carla
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2002
export CARLA_TM_PORT=8102
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_gpu1_2002_run_001
source scripts/00_env.sh
python -m ride.doctor --server
bash scripts/01b_probe_anchors.sh convoy Town04
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

**重要：** `-graphicsadapter=1` 是 Unreal Vulkan 图形适配器索引，并不保证恰好映射到 `nvidia-smi` 的 GPU 1。请在启动后用 `nvidia-smi`/UE4 日志验证；若物理映射不同，需在启动命令中试其它合法 adapter 值。不要只设置 `CUDA_VISIBLE_DEVICES=1` 并认为 CARLA Vulkan 一定使用 GPU 1。两个服务端**不能**共用 RPC/streaming 端口；两个独立进程也不表示单场景实现双卡并行计算。

## 6. 如果仍然崩溃，如何快速分类

| 现象 | 优先怀疑 | 下一步 |
| --- | --- | --- |
| 启动时端口探针显示 `rpc LISTENING` 或 `streaming LISTENING` | 已经有监听进程 | 查看 `ss`/`ps`；不重复启动，或更改端口 |
| doctor 连接到 0.9.15 | 旧 CARLA 进程占用了请求端口 | 连接正确的独立端口，核对 server/client 版本 |
| `03_precommit` 之前就崩溃 | server 启动/端口/加载 Town 地图问题 | 看服务端首个 fatal 前的日志与系统监听 |
| `RIDE_WORLD_SWITCH` 之后崩溃 | map `load_world` 崩溃 | 先尝试服务器预载 Town04；核查地图版本/引擎日志 |
| `RIDE_TRIAL_RESET_BEGIN` 后未出现 `RIDE_TRIAL_WORLD_RESET` | trial 的 `reload_world` 期间服务端崩溃 | 需要调试服务端重载稳定性；不能把半截 JSONL 当作成功实验 |
| `RIDE_TRIAL_WORLD_RESET` 之后，TM family 之前/期间失败 | TM/同步模式或 CARLA actor 问题 | 查看 client traceback、TM 8100/8102 监听归属；尝试独立空闲 TM 端口 |
| 有 `VK_ERROR_DEVICE_LOST`、CUDA OOM、OOM-killer 等 | 才考虑显卡/驱动/内存问题 | `nvidia-smi`、`dmesg` 等系统证据；按错误类型排查 |

常用日志过滤命令（`convoy.log` 中含客户端阶段标记）：

```bash
rg 'RIDE_WORLD|RIDE_TRIAL|RIDE_TM|Traceback|Timeout|Exception' \
  "$RIDE_OUTPUT_ROOT/carla/convoy.log" || true
ss -ltnp | grep -E ':(2000|2001|2002|2003|8100|8102)\b' || true
```

## 7. 修改后的验证边界

本次仅在交付环境验证了 Python 语法、Shell 语法、纯单元测试和伪造 TCP/假 CARLA 启动测试。**无法在此机器连接用户服务器的真实 CARLA 0.9.16，也没有真实运行 convoy。** 现阶段仍应优先拿到 CARLA 机制场景的实际 RQ1/RQ2 证据，不要将端口/环境修好误认为论文研究假设已被证实。
