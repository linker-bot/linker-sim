# 使用指南

本仓库所有可运行命令的实用速查手册。安装步骤见
[installation.md](installation.md)；资产/合成器模型见
[urdf_assets_infra.md](urdf_assets_infra.md)；MJCF 编写规范见
[component_mjcf_authoring.md](component_mjcf_authoring.md)。

下文所有命令均假设已激活 IsaacLab 虚拟环境，且当前位于仓库根目录：

```bash
source ~/opt/IsaacLab/env_isaaclab/bin/activate
cd /path/to/linker-sim
```

如果你的 shell rc 中加载了 ROS 2，请在 Python 命令前加上
`env -u PYTHONPATH -u AMENT_PREFIX_PATH …` 以避免 `lark` / pytest 报错。

---

## 1. 两个入口，一棵配置树

| 入口                       | 配置根                                                          | 用途                                                                       |
|----------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------|
| [scripts/run.py](../scripts/run.py)         | [sim/configs/config.yaml](../sim/configs/config.yaml) | 后端 + 控制器 + 任务 +（可选）录制器的滚动仿真。                          |
| [scripts/replay.py](../scripts/replay.py)   | [sim/configs/replay.yaml](../sim/configs/replay.yaml) | 通过后端回放真实机器人遥测数据，绕过控制器、任务和 `BaseEnv`。            |

二者均为 [Hydra](https://hydra.cc) 入口。命令行中既可用 `group=name`
切换配置组，也可用 `dotted.path=value` 覆盖任意字段。运行输出位于
`outputs/YYYY-MM-DD/HH-MM-SS/`。

配置组位于 [sim/configs/](../sim/configs/)：

- `backend/` — `isaac.yaml`、`mujoco.yaml`
- `robot/` — 包装某个 workstation 的 Hydra 配置
- `controller/` — `joint_pd_bimanual`、`osc_bimanual`（桩）、`ik_pose_bimanual`
- `task/` — `bimanual_reach`、`bimanual_reach_ikpose`
- `recorder/` — `disabled`、`jsonl`、`lerobot`
- `source/` — 回放数据源（如 `data_collection`）

---

## 2. 在 Isaac Sim 中运行

默认配置：`backend=isaac`、`robot=ar5_o6_bench_bimanual`、
`controller=joint_pd_bimanual`、`task=bimanual_reach`、`recorder=disabled`、
`policy=zeros`。

```bash
# 烟雾测试：双臂关节 PD 抵达任务，保持默认姿态。
python scripts/run.py

# 切换 workstation，并用随机游走激发双臂。
python scripts/run.py robot=lkls73_i1_o6_bimanual policy=random_walk

# 无界面 + 限步运行，常用于 CI / 烟雾测试。
python scripts/run.py headless=true max_steps=500

# 多环境（向量化）。
python scripts/run.py num_envs=16 max_steps=200 headless=true
```

热键（窗口模式下）：在视口中按 `R` 重置所有环境；关闭窗口即退出。

仓库内的 workstation（Hydra 配置组 `robot`）：

- O6 手（当前默认）：`ar5_o6_bench_bimanual`（默认）、
  `lkls73_i1_o6_bimanual`、`a7_lite_o6_dc`。
- L6 手（沿用 / 并行）：`ar5_l6_bench_bimanual`、
  `lkls73_i1_bimanual`、`a7_lite_dc`。

常用参数（定义于 [sim/configs/config.yaml](../sim/configs/config.yaml)）：

| 字段                       | 默认值   | 含义                                              |
|----------------------------|----------|---------------------------------------------------|
| `num_envs`                 | 1        | 并行环境数。                                      |
| `env_spacing`              | 2.5      | 各环境间距（米，仅 Isaac）。                      |
| `decimation`               | 4        | 每次 `env.step` 内的物理步数。                    |
| `episode_length_s`         | 8.0      | 自动截断阈值。                                    |
| `reset_joint_noise_scale`  | 0.02     | 重置时每个关节的随机扰动幅度。                    |
| `max_steps`                | 0        | `0` 表示运行到关闭窗口为止。                      |
| `headless`                 | false    | 不开视口。                                        |
| `device`                   | cuda:0   | Torch / Isaac 计算设备。                          |
| `policy`                   | zeros    | `zeros`（保持默认姿态）、`random_walk`（烟雾测试）或 `hold`（不写控制指令 — 用于实时增益调参）。 |

---

## 3. 在 MuJoCo 中运行

MuJoCo 后端仅支持 CPU，且不支持 `rigid_bodies`（即 `task=pick_place`
这类带刚体的场景仅 Isaac 可用）。其他用法一致：

```bash
# 双臂抵达 + 关节 PD + zero 策略，带视口。
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=zeros max_steps=200

# 无界面模式必须指定 max_steps>0（无视口循环可终止）。
python scripts/run.py backend=mujoco headless=true max_steps=400 \
    controller=joint_pd_bimanual task=bimanual_reach

# IK 绝对位姿控制（搭配 task=bimanual_reach_ikpose）。
python scripts/run.py backend=mujoco controller=ik_pose_bimanual \
    task=bimanual_reach_ikpose recorder=jsonl
```

热键（窗口模式下）：在 MuJoCo 视口中按 `R` 重置。

---

## 4. 用 MuJoCo / Isaac 回放真机数据

入口 [scripts/replay.py](../scripts/replay.py) 读取一个 `ReplaySource`
（目前仅 `TelemetryNpzSource`），直接调用 `set_joint_position_target`
驱动 workstation，**不**经过控制器、任务或 `BaseEnv`。手部列通过专用
解码器映射（见 [sim/io/replay/hands.py](../sim/io/replay/hands.py)）。

> **手部解码精度提示。** `linker_l6` / `linker_o6` 把厂商 0–255 字节
> 命令在每个关节 `[lower, upper]` 上做线性插值。真实 Linker Hand 的
> 标定是非线性的，部分关节方向甚至相反，因此**手指姿态只是近似值**
> （手臂跟踪不受影响）。在声称物理保真之前，应用厂商曲线或逐关节
> 查找表替换这两个解码器。文件中以 TODO 标注待办。

```bash
# 真机数据 episode_000025/telemetry.npz，MuJoCo 视口，30 Hz 实时回放。
# 数据本身不纳入版本管理 —— 把它放到默认路径下，或用
# `source.path=...` 覆盖。
python scripts/replay.py robot=a7_lite_dc source=data_collection

# 无界面烟雾：限制 200 帧，关闭实时节流。
python scripts/replay.py robot=a7_lite_dc source=data_collection \
    headless=true realtime=false max_frames=200

# 同一份数据用 Isaac（GPU）回放。
python scripts/replay.py backend=isaac device=cuda:0 \
    robot=a7_lite_dc source=data_collection
```

热键（MuJoCo 窗口模式）：按 `Q` 停止。

### 接入新的录制数据

1. 把 `.npz` 放在可读路径下；默认 key 为 `qpos`，形状 `(T, N)`。
2. 编写 `sim/configs/source/<your_name>.yaml` 描述列布局，模板见
   [sim/configs/source/data_collection.yaml](../sim/configs/source/data_collection.yaml)。
   每个 role 的 `cols: [start, end)` 切片宽度 **必须** 等于该 role 在
   workstation 上的可驱动关节数。
3. 运行：
   ```bash
   python scripts/replay.py robot=<workstation> source=<your_name>
   ```

回放参数（见 [sim/configs/replay.yaml](../sim/configs/replay.yaml)）：
`realtime`（按 `source.hz` 节流）、`max_frames`（截断）、`headless`、
`device`。

---

## 5. 合成 workstation 的 URDF / MJCF

一个 workstation = `recipe.yaml` → `workstation.urdf` +
`workstation.mjcf` + `manifest.yaml`（运行时只读 manifest）。
Recipe 位于 [assets/workstations/](../assets/workstations/)，组件位于
[assets/components/](../assets/components/)。

### 合成

```bash
# 单个 workstation。
python -m linker_sim.tools.composer.compose assets/workstations/a7_lite_dc

# 全量合成。
for ws in assets/workstations/*/; do
    python -m linker_sim.tools.composer.compose "$ws"
done
```

输出 `workstation.urdf`、`workstation.mjcf`（仅在所有组件都提供 MJCF
时生成）以及 `manifest.yaml`。三者都需要提交到版本库。

### 校验

```bash
# 单组件 MJCF 校验（合成前先跑）。
python -m linker_sim.tools.validate_component_mjcf assets/components/arms/a7_lite/variants/left
python -m linker_sim.tools.validate_component_mjcf assets/components/arms/a7_lite/variants/right
python -m linker_sim.tools.validate_component_mjcf assets/components/bases/a7_lite_torso/variants/default

# Workstation 校验：12 项检查（manifest 哈希、URDF 运动学、网格路径、
# drift、URDF↔MJCF 1e-5 m / 1e-5 rad 帧位姿一致性）。
python -m linker_sim.tools.validate_workstation assets/workstations/a7_lite_dc
```

### Drift 守门（CI）

用于捕获 “改了 recipe / 组件但忘了重新合成并提交” 的情况：

```bash
# 全部 workstation。
bash packages/linker-sim/src/linker_sim/tools/ci/check_drift.sh

# 单个 workstation。
bash packages/linker-sim/src/linker_sim/tools/ci/check_drift.sh a7_lite_dc
```

退出码 0 = 干净，1 = 存在 drift。

### 查看运行时看到了什么

```bash
python -m linker_sim.tools.registry_show                 # 列出所有 workstation
python -m linker_sim.tools.registry_show a7_lite_dc      # 打印 roles / joints / frames
```

### 新增一个 workstation

1. 在 `assets/components/{arms,bases,hands}/` 下挑选或新增组件。每个
   组件包含 `meta.yaml` + 各 variant 下的 `<kind>.urdf` + `<kind>.mjcf`
   + `meshes/`，详见 [urdf_assets_infra.md](urdf_assets_infra.md) 与
   [component_mjcf_authoring.md](component_mjcf_authoring.md)。
2. 编写 `assets/workstations/<name>/recipe.yaml`。
3. 合成 → 校验 → 提交。
4. 添加薄薄的 Hydra 包装文件 `sim/configs/robot/<name>.yaml`：
   ```yaml
   # @package _global_
   robot:
     workstation_name: <name>
     role_name: robot
     rigid_bodies: {}
   ```
5. 烟雾测试：
   ```bash
   python scripts/run.py robot=<name> backend=mujoco \
       controller=joint_pd_bimanual task=bimanual_reach max_steps=200
   ```

---

## 6. PD / OSC 增益调参

增益分布在三个层级，按需求选合适的入口修改。

### a) 组件默认值（按 role / 按臂 — 影响所有引用它的 workstation）

每个组件的
[meta.yaml](../assets/components/arms/lkls73_arm/meta.yaml)：

```yaml
default_gains:
  stiffness: 1000
  damping: 4
gain_profiles:
  joint:  { stiffness: 1000, damping: 4 }   # controller=joint_pd_* 时使用
  osc:    { stiffness: 150,  damping: 8 }   # controller=osc_* 时使用
```

修改后必须重新合成 workstation（`python -m linker_sim.tools.composer.compose …`），
并提交新的 `manifest.yaml` / `workstation.urdf`。

### b) 控制器层覆盖（每个控制器配置）

`JointPDControllerCfg` 上的 `stiffness` / `damping` 会在运行时覆盖
manifest 中的默认值：

```yaml
# sim/configs/controller/joint_pd_bimanual.yaml
- _target_: sim.controllers.joint_pd.JointPDController
  cfg:
    role: arm_left
    action_scale: 0.25
    stiffness: 800
    damping: 6
```

也可以在命令行直接覆盖：

```bash
python scripts/run.py controller=joint_pd_bimanual \
    'controller.entries.0.cfg.stiffness=800' \
    'controller.entries.0.cfg.damping=6'
```

OSC 控制器有自己的字段：`actuator_stiffness`、`actuator_damping`、
任务空间的 `stiffness`、`damping_ratio`、`nullspace_stiffness`、
`nullspace_damping_ratio`，详见
[sim/configs/controller/osc_bimanual.yaml](../sim/configs/controller/osc_bimanual.yaml)。

### c) MuJoCo `<position>` 执行器增益（按组件 MJCF）

MuJoCo 把增益固化在模型加载时。刚度放在执行器的 `kp` 上；阻尼放在
**关节**的 `damping` 属性上，因为 MuJoCo 对关节阻尼做隐式积分（无条件
稳定），而执行器的 `kv` 是显式积分，在高数值时会发散。修改各组件 MJCF
（例如 [arm.mjcf](../assets/components/arms/a7_lite/variants/left/arm.mjcf)）：

```xml
<!-- 关节：阻尼在此（隐式积分，稳定） -->
<joint name="L1_Joint" ... damping="20" armature="0.01"/>

<!-- 执行器：kp 在此，kv=0（阻尼在关节上） -->
<position name="L1_Joint_act" joint="L1_Joint" kp="2000" kv="0" ctrlrange="-2.18 3.75"/>
```

改完后：单组件校验 → 重新合成 → 校验 → 提交。这里的增益建议与
manifest 的 `default_gains` 保持一致，以确保 URDF↔MJCF 行为一致。

### d) 实时 PD 增益调参器（通用后端）

使用 `policy=hold` + `+gain_tuner=true` 可在仿真运行时从 JSON 文件热重载
关节 PD 增益。`hold` 策略不输出动作，控制器不会写入目标 — 机器人通过位置
执行器保持当前姿态，你只需调整增益。

```bash
# MuJoCo — 实时调参，默认文件 /tmp/dex_pd_gains.json
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=hold +gain_tuner=true

# 自定义路径
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=hold +gain_tuner=true \
    +gain_tuner_path=/tmp/my_gains.json
```

首次运行时 JSON 文件以 workstation manifest 的 `default_gains` 为种子自动
生成。在另一终端中编辑该文件，改动每 0.5 秒被拾取：

```json
{
  "arm_left":  { "stiffness": 2000, "damping": 20 },
  "arm_right": { "stiffness": 2000, "damping": 20 },
  "hand_left": { "stiffness": 200,  "damping": 4 },
  "hand_right":{ "stiffness": 200,  "damping": 4 }
}
```

调好后将数值固化到组件 MJCF 和 meta.yaml（见上方 a/c 小节）。JSON 文件为
会话级临时文件，不纳入版本管理。

实现：[sim/io/gain_watcher.py](../sim/io/gain_watcher.py)。

### e) OSC 控制器（未实现）

OSC 控制器（`sim/controllers/osc.py`）及其调参器
（`sim/envs/test_osc/gain_tuner_osc.py`）已标记为桩代码——之前的实现
从未经过验证。Hydra 配置 `controller=osc_bimanual` 仍存在，但运行时
会抛出 `NotImplementedError`。

---

## 7. 录制 episode

为 `scripts/run.py` 挂上 recorder 即可：

```bash
# JSONL：每个环境一份 episode 文件，每行对应一次 env.step。
python scripts/run.py recorder=jsonl max_steps=400

# LeRobot 格式数据集，30 fps。
python scripts/run.py recorder=lerobot max_steps=400
```

文件落在 `outputs/YYYY-MM-DD/HH-MM-SS/episodes/`。`recorder=disabled`
（默认）不写任何东西。

---

## 8. 测试

```bash
# 纯 Python 单元测试（无需 GPU）。
env -u PYTHONPATH -u AMENT_PREFIX_PATH pytest tests/ -v
```

完整分层测试流水线（合成器 → registry → 无界面 → 抵达 → 双臂 → 录制器）
见 [TEST_PIPELINE.md](TEST_PIPELINE.md)。

---

## 9. 命令速查表

```bash
# Isaac，全默认
python scripts/run.py

# Isaac，双臂抵达 + JSONL 录制
python scripts/run.py robot=lkls73_i1_bimanual recorder=jsonl max_steps=600

# MuJoCo，关节 PD 烟雾测试
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=zeros max_steps=200

# MuJoCo，IK 绝对位姿
python scripts/run.py backend=mujoco controller=ik_pose_bimanual \
    task=bimanual_reach_ikpose

# 回放真机数据（MuJoCo）
python scripts/replay.py robot=a7_lite_dc source=data_collection

# 回放无界面 / 限帧
python scripts/replay.py robot=a7_lite_dc source=data_collection \
    headless=true realtime=false max_frames=200

# 全量合成 + 校验
for ws in assets/workstations/*/; do python -m linker_sim.tools.composer.compose "$ws"; done
for ws in assets/workstations/*/; do python -m linker_sim.tools.validate_workstation "$ws"; done
bash packages/linker-sim/src/linker_sim/tools/ci/check_drift.sh

# 查看 registry handle
python -m linker_sim.tools.registry_show a7_lite_dc

# 实时 PD 增益调参（MuJoCo，运行时编辑 /tmp/dex_pd_gains.json）
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=hold +gain_tuner=true
```
