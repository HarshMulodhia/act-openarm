"""Deploy a trained ACT policy in Isaac Sim with the viewer enabled.

Loads the best checkpoint, runs chunked inference in a live viewer session,
and exits after a configurable number of episodes or on keyboard interrupt.

Usage::

    python openarm_act_project/scripts/deploy_act_openarm.py \\
        --config openarm_act_project/configs/act_openarm_reach.yaml \\
        --checkpoint openarm_act_project/checkpoints/openarm_reach_act/

Required arguments:
    --config        Path to the task YAML config file.
    --checkpoint    Directory containing the trained checkpoint.

Optional arguments:
    --num_episodes  Number of deployment episodes (default: 10).
    --step_hz       Real-time control frequency in Hz (default: 30).
    --device        Torch/sim device (default: cuda:0).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy ACT policy in Isaac Sim.")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--step_hz", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Ensure the viewer is visible during deployment
if not hasattr(args_cli, "headless") or not args_cli.headless:
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports."""

import os
import time

import numpy as np
import torch
import yaml

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_ACT_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "act"))
for _p in (_SRC_DIR, _ACT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openarm_act.envs.openarm_isaac_env import OpenArmIsaacEnv  # noqa: E402
from openarm_act.policies.act_openarm_policy import ACTOpenArmPolicy  # noqa: E402
from openarm_act.utils.io_utils import load_norm_stats  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(args_cli.config)
    obs_cfg = cfg.get("observation", {})
    eval_cfg = cfg.get("evaluation", {})

    max_timesteps: int = eval_cfg.get("max_timesteps", 400)
    camera_names: list[str] = obs_cfg.get("camera_names", ["cam_main"])
    num_joints: int = obs_cfg.get("num_joints", 6)
    step_period = 1.0 / args_cli.step_hz

    checkpoint_dir = os.path.normpath(
        os.path.join(_PROJECT_ROOT, args_cli.checkpoint)
        if not os.path.isabs(args_cli.checkpoint)
        else args_cli.checkpoint
    )

    norm_stats = load_norm_stats(checkpoint_dir)
    policy = ACTOpenArmPolicy.from_checkpoint(checkpoint_dir, cfg, device=args_cli.device)
    policy._model.eval()
    if norm_stats is not None:
        policy.norm_stats = norm_stats

    print(f"\nDeploying ACT policy — task: {cfg['task']}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Running {args_cli.num_episodes} episode(s).  Press Ctrl-C to abort.\n")

    successes = []

    with OpenArmIsaacEnv(
        task=cfg["task"],
        camera_names=camera_names,
        num_joints=num_joints,
        device=args_cli.device,
    ) as env:
        for ep in range(args_cli.num_episodes):
            obs = env.reset()
            policy.reset()
            success = False

            for _t in range(max_timesteps):
                t0 = time.time()

                action = policy.get_action(obs["qpos"], obs["images"])
                obs, success = env.step(action)

                if success:
                    break

                # Real-time pacing
                elapsed = time.time() - t0
                if elapsed < step_period:
                    time.sleep(step_period - elapsed)

            status = "✓ SUCCESS" if success else "✗ failed"
            successes.append(int(success))
            print(f"  Episode {ep + 1}/{args_cli.num_episodes}  {status}")

    rate = float(np.mean(successes)) if successes else 0.0
    print(f"\nDeploy summary — success rate: {rate:.1%}")

    simulation_app.close()


if __name__ == "__main__":
    main()
