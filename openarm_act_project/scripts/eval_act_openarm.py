"""Evaluate a trained ACT policy with deterministic rollouts in Isaac Sim.

Runs standardised evaluation episodes on a hold-out random seed, logs
task success rate, mean episode length, and joint-trajectory plots, then
writes a JSON metrics summary.

Usage::

    python openarm_act_project/scripts/eval_act_openarm.py \\
        --config openarm_act_project/configs/act_openarm_reach.yaml \\
        --checkpoint openarm_act_project/checkpoints/openarm_reach_act/

Required arguments:
    --config        Path to the task YAML config file.
    --checkpoint    Directory containing the trained checkpoint.

Optional arguments:
    --num_rollouts  Number of evaluation episodes (overrides config).
    --seed          Random seed for episode initialisation.
    --headless      Run without viewer window.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACT policy in Isaac Sim.")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_rollouts", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports."""

import json
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
from openarm_act.utils.dataset_tools import write_experiment_summary  # noqa: E402
from openarm_act.utils.io_utils import load_norm_stats  # noqa: E402
from openarm_act.utils.viz_utils import plot_rollout_summary, write_fixed_length_video  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(args_cli.config)
    eval_cfg = cfg.get("evaluation", {})
    obs_cfg = cfg.get("observation", {})

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    num_rollouts: int = args_cli.num_rollouts or eval_cfg.get("num_rollouts", 50)
    max_timesteps: int = eval_cfg.get("max_timesteps", 400)
    camera_names: list[str] = obs_cfg.get("training_camera_names", obs_cfg.get("camera_names", ["cam_main"]))
    render_camera_names: list[str] = obs_cfg.get("render_camera_names", [])
    num_joints: int = obs_cfg.get("num_joints", 7)
    camera_setup: dict = obs_cfg.get("camera_setup", {})
    success_distance_threshold = eval_cfg.get("success_distance_threshold", 0.03)

    # Load policy
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

    # Results directory
    results_dir = os.path.join(checkpoint_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    successes = []
    episode_lengths = []
    video_frames: list[np.ndarray] = []
    video_camera = eval_cfg.get("video_camera", "isometric")
    want_video = bool(eval_cfg.get("render_video", False))

    with OpenArmIsaacEnv(
        task=cfg["task"],
        camera_names=camera_names,
        num_joints=num_joints,
        device=args_cli.device,
        camera_setup=camera_setup,
        render_camera_names=render_camera_names,
        success_distance_threshold=success_distance_threshold,
    ) as env:
        for ep in range(num_rollouts):
            obs = env.reset()
            policy.reset()
            qpos_hist = []
            action_hist = []
            success = False

            for t in range(max_timesteps):
                action = policy.get_action(obs["qpos"], obs["images"])
                qpos_hist.append(obs["qpos"].copy())
                action_hist.append(action.copy())
                obs, success = env.step(action)
                if want_video and ep == 0:
                    frame = obs.get("render_images", {}).get(video_camera)
                    if frame is None:
                        frame = obs["images"].get(camera_names[0])
                    if frame is not None:
                        video_frames.append(frame.copy())
                if success:
                    break

            ep_len = len(qpos_hist)
            successes.append(int(success))
            episode_lengths.append(ep_len)

            # Save rollout plot for every 10th episode
            if (ep + 1) % 10 == 0 or ep == 0:
                plot_rollout_summary(
                    np.array(qpos_hist),
                    np.array(action_hist),
                    output_path=os.path.join(results_dir, f"rollout_{ep:03d}.png"),
                )

            status = "✓" if success else "✗"
            print(f"  Episode {ep + 1:3d}/{num_rollouts}  {status}  len={ep_len}")

    # Aggregate metrics
    success_rate = float(np.mean(successes))
    mean_len = float(np.mean(episode_lengths))
    metrics = {
        "success_rate": round(success_rate, 4),
        "mean_episode_length": round(mean_len, 1),
        "num_rollouts": num_rollouts,
        "checkpoint": checkpoint_dir,
        "task": cfg["task"],
        "seed": args_cli.seed,
    }
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    video_path = None
    if want_video:
        video_path = os.path.join(results_dir, f"inference_{video_camera}_30s_1080p.mp4")
        write_fixed_length_video(
            video_frames,
            output_path=video_path,
            fps=eval_cfg.get("video_fps", 30),
            seconds=eval_cfg.get("video_seconds", 30),
            width=eval_cfg.get("video_width", 1920),
            height=eval_cfg.get("video_height", 1080),
        )

    print(f"\nEvaluation results — success rate: {success_rate:.1%}  "
          f"mean length: {mean_len:.0f}")
    print(f"Metrics saved to: {metrics_path}")
    if video_path:
        print(f"Video saved to: {video_path}")
    write_experiment_summary(
        os.path.join(results_dir, "experiment_summary.json"),
        {
            "stage": "eval",
            "task": cfg["task"],
            "checkpoint": checkpoint_dir,
            "metrics": metrics,
            "video_path": video_path,
            "video_camera": video_camera,
            "video_seconds": eval_cfg.get("video_seconds", 30),
            "video_fps": eval_cfg.get("video_fps", 30),
            "seed": args_cli.seed,
        },
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
