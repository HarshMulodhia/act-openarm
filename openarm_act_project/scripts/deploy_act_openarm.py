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
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy ACT policy in Isaac Sim.")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--step_hz", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_project_path(path: str) -> str:
    return os.path.normpath(
        os.path.join(_PROJECT_ROOT, path) if not os.path.isabs(path) else path
    )


def _find_deploy_checkpoint(checkpoint_dir: str) -> str | None:
    best = os.path.join(checkpoint_dir, "policy_best.ckpt")
    if os.path.isfile(best):
        return best
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = sorted(
        f for f in os.listdir(checkpoint_dir) if f.startswith("policy_epoch_")
    )
    return os.path.join(checkpoint_dir, ckpts[-1]) if ckpts else None


CHECKPOINT_DIR = _resolve_project_path(args_cli.checkpoint)
if _find_deploy_checkpoint(CHECKPOINT_DIR) is None:
    print(
        "No trained ACT checkpoint found in "
        f"'{CHECKPOINT_DIR}'.\n"
        "Expected 'policy_best.ckpt' or 'policy_epoch_*.ckpt'.\n"
        "Create one first with:\n"
        f"  /isaac-sim/python.sh scripts/train_act_openarm.py --config {args_cli.config}",
        file=sys.stderr,
    )
    sys.exit(2)

# Use a rendering-capable experience for the live viewer and policy cameras.
args_cli.enable_cameras = True
if args_cli.livestream in (1, 2):
    args_cli.headless = True
else:
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports."""

import time

import numpy as np
import torch
import yaml

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
    camera_names: list[str] = obs_cfg.get("training_camera_names", obs_cfg.get("camera_names", ["cam_main"]))
    camera_setup: dict = obs_cfg.get("camera_setup", {})
    render_camera_names: list[str] = obs_cfg.get("render_camera_names", [])
    num_joints: int = obs_cfg.get("num_joints", 7)
    step_period = 1.0 / args_cli.step_hz

    checkpoint_dir = CHECKPOINT_DIR

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
        camera_setup=camera_setup,
        render_camera_names=render_camera_names,
        success_distance_threshold=eval_cfg.get("success_distance_threshold", 0.03),
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
