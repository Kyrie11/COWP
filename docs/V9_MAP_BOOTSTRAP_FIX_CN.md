# V9 CARLA Town04 启动与地图初始化修复

## 事件与根因

2026-10-28 用户启动 CARLA 0.9.16 并设置 `CARLA_TOWN=Town04`，但
`RIDE_CARLA_HEALTH_BEFORE` 显示当前地图为 `Carla/Maps/Town10HD_Opt`，
`29_v9_carla_preflight.sh` 因 `RIDE_MAP_MISMATCH_REFUSE_AUTO_LOAD` 退出。
这说明 CARLA RPC 2000 上实际运行的世界没有加载 Town04。仅设置 RIDE 环境变量
不会更改已运行 Unreal 的地图。UE 启动地图的 *位置参数* 在具体发行包中是否生效
必须以 CARLA Python API 读回的 `world.get_map().name` 为准。

这不是版本不匹配：当时服务端和 Python 客户端均报告 0.9.16，且车辆数为 0。
不应移除 preflight 拒绝自动切图的保护，因为 CARLA 加载地图会销毁当前世界，
曾有 Unreal Signal 11/超时风险。`-RenderOffScreen` 也不是
`world.get_settings().no_rendering_mode=True`，后者为 false 并非此次错误。

## 修复范围

- **新增** `ride/carla/map_setup.py`：地图准备与安全检查。要求操作员显式
  `--allow-world-load`；检查服务端/客户端 0.9.16、当前地图、同步模式、动态 actor、
  待加载地图是否可用；最多调用一次 `load_world`，之后重新读回地图确认。
  一旦地图加载失败不重试，避免把崩溃重复触发。
- **新增** `scripts/02c_prepare_carla_town.sh`：明确的人工地图准备入口，
  需要 `RIDE_ALLOW_INITIAL_MAP_LOAD=1`；成功后运行**只读**的 v9 preflight。
- **更新** `ride/carla/health.py`：地图不一致时给出操作步骤；仍然只读。
- **更新** `scripts/02_start_carla_headless.sh`：启动时明确警告地图尚未验证，
  不把 UE 启动参数假定为成功加载。
- **新增** 单元测试，覆盖拒绝隐式加载、actor / 同步模式拦截、地图不存在、
  一次加载、验证失败和异常不重试。所有测试在假 CARLA 对象上执行，
  不代表在实际 GPU/Unreal 上完成了运行验证。

## 推荐运行步骤

在你**独占**的 CARLA 服务器上执行（切换地图会销毁世界，其他实验必须先停止）。

**终端 A**（RIDE 项目目录）：

```bash
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
export RIDE_CARLA_QUALITY=Epic
bash scripts/02_start_carla_headless.sh
```

**终端 B**（同一 RIDE 项目目录）：

```bash
conda activate mfrp
export RIDE_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
source scripts/00_env.sh
export CARLA_TOWN=Town04
RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh Town04
# 如需再次确认（只读）：
bash scripts/29_v9_carla_preflight.sh Town04
```

官方原生替代方式：从 `CARLA_ROOT` 调用
`python PythonAPI/util/config.py --host 127.0.0.1 --port 2000 --map Town04`，
然后依然必须运行 RIDE preflight。该替代方式不包含新脚本的安全检查。

**限制**：静态检查（world 里没有活动 actor 且不同步）不能证明没有远端
客户端即将连接；必须由操作者确认服务器独占、空闲。若仍然是
Town10HD_Opt，检查 RPC 2000/2001 端口实际归哪个 Unreal 进程占用。
若 `load_world` 引发 Signal 11 或超时，停止尝试并审阅 Unreal 日志，
不要将切换失败记作实验结论。
