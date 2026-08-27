# SMP 遥控器部署（sim2sim + sim2real）

把本项目训练的 **SMP Forward/Steering** 策略，用 **FSM + 共享黑板 + 遥控器** 架构部署到
MuJoCo 仿真（PC 手柄）和 G1 真机（宇树无线遥控器）。参考 RoboMimicDeploy_G1 的状态机思路，
功能全部落在本项目 `deploy/` 内，自包含、不依赖训练用的 mjlab。

> 当前支持速度/朝向指令类任务 **Smp-Forward-G1 / Smp-Steering-G1**（观测 101 维，摇杆直接映射
> 为 steering 指令，最适合遥控）。Location/Getup/CMoE 可后续按同一框架扩展。

## 流程总览

```
                       ①导出            ②部署
  训练(uv, mjlab) ──► policy.onnx ──►  deploy/{deploy_mujoco|deploy_real}
   rsl_rl model_*.pt   (含 metadata)    FSM: Passive / FixedPose / Steering
```

策略参数（PD 增益 / 默认关节角 / action_scale / 关节序）**烘焙在 onnx metadata**，部署端
`common/onnx_policy.py` 直接读取——无需手抄，消除 sim2real 漂移。

## ① 导出 ONNX（在训练环境里跑）

```bash
uv run scripts/export_onnx.py --task Smp-Steering-G1 \
    --checkpoint logs/rsl_rl/smp_steering_g1/<run>/model_XXXX.pt
# 生成同目录 policy.onnx（mjlab 内置导出：actor+经验观测归一化折进图，并附 metadata）
```

## ② 安装部署依赖（部署机，可与训练环境分开）

```bash
pip install -r deploy/requirements-deploy.txt      # 真机另装 unitree_sdk2py
```

## ③ 运行

```bash
# sim2sim（MuJoCo + PC 手柄）
python deploy/deploy_mujoco/deploy_mujoco.py --policy path/to/policy.onnx \
    --command-mode steering --max-speed 2.0        # Forward 用 --command-mode forward --max-speed 5

# sim2real（DDS + 宇树无线遥控器）
python deploy/deploy_real/deploy_real.py enp4s0 deploy/deploy_real/configs/g1_smp.yaml \
    --policy path/to/policy.onnx --command-mode steering --max-speed 2.0
```

## 三个状态

| 状态 | 说明 |
| --- | --- |
| **PassiveMode** | 阻尼保护（开机默认，kp=0, kd=8） |
| **FixedPose**   | 位控 2s 平滑到默认站姿并保持 |
| **SteeringMode**| SMP 策略：摇杆→5 维 steering 指令→101 维观测→动作（★核心） |

## 遥控器映射（仿真手柄 / 真机遥控器一致）

| 操作 | PC 手柄(sim) | 宇树遥控器(real) | 效果 |
| --- | --- | --- | --- |
| 进就绪位 | `START` | `start` | → FixedPose |
| 开始行走 | `R1`+`A` | `R1`+`A` | → SteeringMode |
| 阻尼保护 | `L1` | `F1` | → PassiveMode |
| 退出 | `SELECT` | `select` | 下发阻尼后退出 |
| 行进方向+速度 | 左摇杆 | 左摇杆 | 前后=进/退, 左右=侧移；速度=摇杆量 |
| 朝向(steering) | 右摇杆 X | 右摇杆 X | 设置身体朝向角 |

- `forward` 模式：固定 +x 方向/朝向，仅左摇杆前推控速(0~max_speed)，对应 Smp-Forward-G1。
- `steering` 模式：左摇杆定方向+速度、右摇杆定朝向，对应 Smp-Steering-G1。

## 观测布局（101，与 g1_smp_env_cfg 的 actor 观测组同序）

```
[base_lin_vel(3)][base_ang_vel(3)][projected_gravity(3)]
[joint_pos-default(29)][joint_vel(29)][last_action(29)]
[command(5)= tar_dir_x, tar_dir_y, tar_speed, face_dir_x, face_dir_y]  (朝向系)
```

## ⚠️ 真机 base_lin_vel

SMP 观测含机体系线速度 `base_lin_vel`，真机 IMU 测不到、需状态估计。sim2sim 由 MuJoCo 直接提供；
sim2real 里 `deploy_real.py` 的 `estimate_base_lin_vel()` **默认返回零占位**。要真机稳定，二选一：
1. 在 `estimate_base_lin_vel()` 接入你的里程计/状态估计；
2. 训练时用去掉 `base_lin_vel` 的观测配置(Wo-State-Estimation)，重新导出 onnx（部署端会自动按新的
   `num_obs` 组观测——但需相应调整 SteeringMode 里的 96 偏移量，见代码注释）。

## 离线自测（无需 mjlab/mujoco/手柄）

```bash
python deploy/tests/test_steering_deploy.py   # 观测布局 + 关节序重排 + 指令映射 + FSM 切换
```

## 代码结构

```
deploy/
  common/   ctrlcomp(黑板) fsm_utils(枚举+机体系数学) onnx_policy(元数据+双关节序)
            joystick+gamepad(PC 手柄) remote_controller(宇树遥控器) command_helper+rotation_helper
  FSM/      FSMState(基类) FSM(3 状态调度)
  policy/   passive/ fixedpose/ steering/(★SMP 策略+指令)
  deploy_mujoco/deploy_mujoco.py    sim2sim
  deploy_real/deploy_real.py        sim2real（+ configs/g1_smp.yaml）
  unitree_description/              sim 用 G1 mjcf+meshes
  tests/    test_steering_deploy.py
scripts/export_onnx.py             训练环境里导出 onnx
```
