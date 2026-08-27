"""export_onnx.py — 把训练好的 SMP 策略(rsl_rl checkpoint) 导出为部署用 ONNX。

mjlab 内置了导出：MjlabOnPolicyRunner.export_policy_to_onnx() 会把 actor + 经验观测归一化
折进一张图（输入 raw obs → 输出 action），attach_metadata_to_onnx() 再把关节刚度/阻尼/默认角/
逐关节 action_scale/关节序/观测名 等烘焙进 onnx metadata。deploy/ 里的 common/onnx_policy.py 就
从这些 metadata 读取部署参数——因此**部署端无需再手抄任何增益**。

本脚本 = 注册 SMP 任务 + 调 mjlab 导出，参照 mjlab 官方 export 流程编写。
需在装好 mjlab 的环境里运行（本仓库用 uv）：

  uv run scripts/export_onnx.py --task Smp-Steering-G1 \
      --checkpoint logs/rsl_rl/smp_steering_g1/<run>/model_XXXX.pt \
      --output logs/rsl_rl/smp_steering_g1/<run>/policy.onnx

导出得到的 policy.onnx 直接喂给 deploy/deploy_mujoco/deploy_mujoco.py 或 deploy_real/deploy_real.py。
Forward/Steering 的 actor 观测为 101 维；命令模式：Forward 用 --command-mode forward --max-speed 5，
Steering 用 --command-mode steering --max-speed 2（在部署脚本里指定）。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import smp.rl.tasks  # noqa: F401  # 注册 Smp-* 任务到 mjlab registry


def run_export(task: str, checkpoint: str, output: str | None,
               device: str | None, num_envs: int, attach_metadata: bool) -> Path:
    ckpt = Path(checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    out = Path(output).expanduser().resolve() if output else ckpt.parent / "policy.onnx"

    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)
    env_cfg.scene.num_envs = num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=dev, render_mode=None)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=dev)
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=dev)

        out.parent.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(out.parent), out.name)

        if attach_metadata:
            metadata = get_base_metadata(env.unwrapped, ckpt.parent.name)
            attach_metadata_to_onnx(out, metadata)

        print(f"[export_onnx] 导出成功 -> {out}")
        print("[export_onnx] 部署: deploy/deploy_mujoco/deploy_mujoco.py --policy", out)
    finally:
        env.close()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   help="Smp-Forward-G1 / Smp-Steering-G1（先支持这两个速度指令类任务）")
    p.add_argument("--checkpoint", required=True, help="rsl_rl model_*.pt")
    p.add_argument("--output", default=None, help="输出 onnx（默认 checkpoint 同目录 policy.onnx）")
    p.add_argument("--device", default=None)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--no-metadata", action="store_true", help="不烘焙 metadata（不推荐）")
    args = p.parse_args()

    run_export(args.task, args.checkpoint, args.output, args.device,
               args.num_envs, attach_metadata=not args.no_metadata)


if __name__ == "__main__":
    main()
