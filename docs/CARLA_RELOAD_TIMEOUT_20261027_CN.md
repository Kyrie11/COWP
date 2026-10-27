# CARLA 0.9.16 `reload_world()` 40s 超时与 Signal 11：工程修复及科学有效性说明

## 1. 诊断结论及依据

用户日志：RPC 2002 与 streaming 2003 已监听；TM 8102 尚未创建；`main()` 显示 `RIDE_WORLD_SWITCH to Town04`，下一步 `run_trial()` 显示 `RIDE_TRIAL_RESET_BEGIN family=tm profile=early_slow target=Town04`，紧接着 `client.reload_world()` 超时 40 秒。

上传的 `ride/carla/run.py` **先在 `main()` 中按需 `load_world(Town04)`，随后第一条 trial 无条件执行 `reload_world()`**。这不是 TCP bind 错误；TM 在重载成功以后才调用，因此不能归咎于 TM 8102。客户端超时不等于直接证明 UE4 发生 Signal 11；必须保留服务端日志确认崩溃而非长时间卡住。

官方说明 `reload_world()` 创建新 world、销毁 actors 但保留 TM 实例；反复切图的实际稳定性取决于 CARLA 引擎/驱动/场景。

## 2. 修复方法与限制

- `ride/carla/run.py` 新增 `--world-reset actors|reload`。**默认 `actors`**：地图只在启动/预冻结阶段加载；每条 trial 在同一 world 中重置同步固定时间步、重置交通信号灯周期、重新创建 RIDE 的三辆车和碰撞传感器，并在 trial 结束销毁所有 RIDE actors、关闭 TM 同步、恢复原 world 设置。
- 允许严格 `reload`（旧试验的地图整体重载）。如果服务端重载失败，抛出清晰的 `CARLA_WORLD_RELOAD_FAILED`，禁止盲目自动重试死亡服务器。
- `actors` 模式检测 world 是否存在其他车辆，有则拒绝继续。**绝不删除其他实验用户或不属于 RIDE 的车辆**。
- `scripts/03_run_carla_case.sh` 新增环境变量 `RIDE_WORLD_RESET`、`RIDE_CARLA_TIMEOUT`、`RIDE_FAMILIES`、`RIDE_PROFILES`，便于只运行一条 smoke trial，再分离 TM 和 BehaviorAgent。
- `ride/carla/health.py` 与 `scripts/02a_probe_server.sh` 提供只读健康检查；可选显式危险测试 `reload`，仅可在空白独占 CARLA 上执行。
- `scripts/02b_collect_carla_diagnostics.sh` 保存端口/进程/GPU/UE 日志快照（只读）。
- `convoy.jsonl` manifest 记录 `world_reset_policy`，trial 记录 `world_reset_protocol`。不能将不同 reset 策略的数据合并为同一实验条件。
- `--timeout` 由 40s 增加至默认 120s，以降低正常地图加载较慢时的误报。**单纯延长超时不会修复已死亡的 UE 进程。**
- 只有重置完成并满足地图初始化、驾驶安全等有效性标准的数据才可计入科学实验。

重要科学限制：`actors` 不会恢复所有隐藏 UE4 世界状态，TM 的状态和全局模拟器时钟可能跨 trial 保留。交通灯周期已重置，但这些措施不足以证明严格复现实验的初始条件完全一致。此模式首先用于诊断和探索；正式配对结论必须检查 `initial_states`、交通灯、seed、world setting 与响应一致性，理想情况下用独立 fresh CARLA server 或稳定时的 strict reload 作复核。

## 3. 迁移步骤：使用独立目录，避免残留不完整 JSONL

新包解压目录假定为 `/home/senzeyu2/code/RIDE_reload_fix`，在相同 Conda 环境执行；原来 `mfrp` 可用且已经 import `carla`/`agents` 时不要求换环境。

```bash
cd /home/senzeyu2/code
unzip RIDE_RELOAD_FIX_20261027.zip
cd RIDE_reload_fix
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2002
export CARLA_TM_PORT=8102
export CARLA_TOWN=Town04
source scripts/00_env.sh
bash scripts/00_diagnose_carla.sh
```

注意**只关闭自己创建且占用 2002/2003 的 CARLA 进程**。例如 `ss -ltnp | grep -E ':(2002|2003)\\b'` 查看 PID，再用 `ps -fp PID` 核对；不要盲目 `pkill -f Carla` 或停止其他用户的服务器。若旧进程已经崩溃，重新启动同端口，启动日志必须由服务端终端保存。

终端 A：

```bash
cd /home/senzeyu2/code/RIDE_reload_fix
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2002 CARLA_TM_PORT=8102 CARLA_GPU_ADAPTER=1
source scripts/00_env.sh
set -o pipefail
bash scripts/02_start_carla_headless.sh 2>&1 | tee ~/carla_2002_server.log
```

上述 `CARLA_GPU_ADAPTER=1` 是 Unreal/Vulkan adapter 索引，未必对应 `nvidia-smi` 的 GPU 1。请在启动后实际检查 GPU PID。

终端 B：

```bash
cd /home/senzeyu2/code/RIDE_reload_fix
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export CARLA_RPC_PORT=2002 CARLA_TM_PORT=8102 CARLA_TOWN=Town04
source scripts/00_env.sh
bash scripts/02a_probe_server.sh
```

如果健康检查失败，不要执行 convoy。如果地图不是 Town04，需要在一个新、独占的 CARLA 实例中通过 `03_precommit_scenes.sh` 首次加载 Town04；正常情况下该脚本只切一次图。若**首次** `load_world(Town04)` 同样崩溃，应先独立排查服务端 Vulkan、地图资源和驱动，不能说已解决。

冻结锚点与第一次 smoke：

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_actors_smoke01
export RIDE_ANCHORS=1 RIDE_REPEATS=1
export RIDE_WORLD_RESET=actors
export RIDE_CARLA_TIMEOUT=120
bash scripts/03_precommit_scenes.sh convoy
# 先不使用 Traffic Manager，单独测试 BehaviorAgent 与车辆物理引擎：
export RIDE_FAMILIES=behavior
export RIDE_PROFILES=early_slow
bash scripts/03_run_carla_case.sh convoy
```

通过标志：日志有 `RIDE_TRIAL_WORLD_REUSE`、`RIDE_WORLD_SETTINGS_APPLIED`、`RIDE_TRAFFIC_LIGHTS_RESET`、最后出现 `run_end completed:true`。没有这些标志、或者 `simulation_exception` / 有效性为 false，都不能算成功；需根据原因具体定位。

TM 独立 smoke：

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_tm_smoke01
export RIDE_FAMILIES=tm
export RIDE_PROFILES=early_slow
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
```

两条试验均正常后才尝试完整 `convoy`（**第三个新输出目录**），移除 `RIDE_FAMILIES` 与 `RIDE_PROFILES` 或设置完整列表：

```bash
export RIDE_OUTPUT_ROOT=/data0/senzeyu2/dataset/RIDE/evidence/convoy_actors_full01
unset RIDE_FAMILIES RIDE_PROFILES
bash scripts/03_precommit_scenes.sh convoy
bash scripts/03_run_carla_case.sh convoy
bash scripts/04_analyze_carla.sh
```

如果 `actors` 模式可运行，不能据此证明 strict `reload` 已经稳定。若要单独测试重载，仅在无其他用户/车辆的专属实例上：

```bash
RIDE_HEALTH_MODE=reload RIDE_ALLOW_DESTRUCTIVE_RELOAD=1 bash scripts/02a_probe_server.sh
```

此测试会清空当前所有车辆。不要在有正在进行的实验或共享 CARLA 服务器上执行。

## 4. 再次崩溃时怎么收集证据

```bash
bash scripts/02b_collect_carla_diagnostics.sh
# 把终端 A 的 ~/carla_2002_server.log 最后 150 行单独保存：
tail -n 150 ~/carla_2002_server.log
# 观察 crash 后端口和 PID：
ss -ltnp | grep -E ':(2002|2003|8102)\\b' || true
```

判断分支：

- `RIDE_TRIAL_WORLD_REUSE` 前就失败：地图连接、服务端连接或地图切换。
- `RIDE_WORLD_SETTINGS_APPLIED` 前失败：同步设置、服务端故障。
- `RIDE_TM_CREATE_BEGIN` 前失败：尚不能指责 Traffic Manager。
- `RIDE_TM_CREATE_BEGIN` 后失败：需确认 TM 端口生命周期与服务器日志。
- BehaviorAgent 能运行但 TM 崩溃：聚焦 Traffic Manager / autopilot 同步及其版本，而不是增加 GPU。
- `RIDE_TRIAL_WORLD_REUSE` 后物理 `world.tick()` 超时：进一步排查 actor/sensor 创建与 tick。
- server 日志有 `Out of memory` / Vulkan Xid：才调查显存、系统 RAM、驱动与地图加载。

## 5. 已执行的本地验证

执行 `PYTHONPATH=. python -m pytest -q`，本地 43 项测试通过；包括 mock world 不重载测试、地图身份检查、拒绝非自有 actor、strict reload 失败诊断，以及保留原有 RIDE 回归和研究门禁测试。**这些测试并未运行真实 CARLA UE4 physics，也没有验证你机器上的 Signal 11 已消失。**

参考：

- CARLA Python API `reload_world`：https://carla.readthedocs.io/en/latest/python_api/
- CARLA 同步设置：https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
- CARLA Traffic Manager：https://carla.readthedocs.io/en/latest/adv_traffic_manager/
- CARLA 无头与无渲染：https://carla.readthedocs.io/en/latest/adv_rendering_options/
