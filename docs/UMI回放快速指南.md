# UMI 数据回放快速指南（a7_lite_l6_dc）

从客户端 `.zip` 原始数据到在 `a7_lite_l6_dc` 机器人上回放的完整流程。

## 前置：命令前缀

所有 Python 命令都用这个前缀（MuJoCo 虚拟环境 + UMI-Dex 的 `PYTHONPATH`）：

```bash
PYTHONPATH=~/codes/UMI-Dex/src python
```

## 1. 解压

```bash
unzip -o ep_0007_01045e2b.zip -d data/client_ep_0007
# bag 目录 = data/client_ep_0007/ep_0007_01045e2b/seg_1（含 .mcap + metadata.yaml）
```

## 2. Anchor search（求解手臂锚点，输出 arm_right 轨迹）

```bash
PYTHONPATH=~/codes/UMI-Dex/src python \
    scripts/anchor_search.py <bag 目录> \
    --arm right --hz 30.0 --maxiter 100 \
    --mirror-x --world-rotate-rpy -1.5707 -1.5707 1.5708 \
    --init 0.0667 -0.3599 0.4919 -0.1410 0.0035 -0.0155 \
    --save-npz outputs/umi_replay/ep_arm.npz
```

**手动旋转 / 平移轨迹的方法（都在工作站坐标系下）：**

- `--world-rotate-rpy R P Y`：把**每一帧手腕朝向**整体旋转（弧度，内旋 XYZ），**位置不变**。用来调手掌方向；不参与搜索，设了就固定。上例 `-1.5707 -1.5707 1.5708` = 手掌朝 -z、手指朝 +x。**改了它要重新跑 search**（可达性变了，最优锚点会变）。
- `--world-translate DX DY DZ`：把**整条轨迹**沿 xyz 刚性平移（米），在搜索**之后**施加、不参与优化——固定的手动偏移。**调高度就用它**，例如 `--world-translate 0 0 0.05` 抬高 5cm。施加后会在新位置重新评估并打印跟踪误差（高度变了可达性会变，误差随之变化属正常）。
- `--init` 的后三位 `AR AP AY`：锚点朝向（绕锚点整体旋转轨迹），是搜索的**初值**，会被优化；前三位 `DX DY DZ` 是平移初值，同样会被搜索优化掉（要固定平移用 `--world-translate`）。
- `--mirror-x`：沿 YZ 平面镜像（翻转左右手性，非旋转）。

跟踪误差 RMS 一般在几毫米内；若卡在几十毫米，换一个已收敛的锚点做 `--init` 热启动。

## 3. 拼接手部数据（读 /hand/joint_states，解码成弧度）

```bash
PYTHONPATH=~/codes/UMI-Dex/src python \
    scripts/add_hand_to_npz.py \
    --bag <bag 目录> \
    --arm-npz outputs/umi_replay/ep_arm.npz \
    --out outputs/umi_replay/ep_with_hand.npz \
    --arm right --hz 30.0 --workstation a7_lite_l6_dc
```

- 现役 umi-dex 设备：用默认映射（不加 flag）。
- 旧款 legacy 设备（手指通道顺序不同）：加 `--legacy`。

## 4. 回放 / 可视化

```bash
PYTHONPATH=~/codes/UMI-Dex/src python \
    scripts/replay_ik.py robot=a7_lite_l6_dc source=data_collection \
    ee_poses=outputs/umi_replay/ep_with_hand.npz \
    headless=false realtime=true
```

- `headless=true realtime=false`：不开窗口，打印跟踪误差报告。
- `source=data_collection` 只为满足 Hydra 必填项，在 ee_poses 模式下不生效。
