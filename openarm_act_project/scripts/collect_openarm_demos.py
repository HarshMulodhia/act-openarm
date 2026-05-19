"""Collect teleoperated demonstrations from an OpenArm Isaac Lab task.

The script launches Isaac Sim, creates the specified task environment,
streams human teleoperation inputs, and writes successful episodes to HDF5
at the path configured in the YAML config file.

Usage (from workspace root)::

    python openarm_act_project/scripts/collect_openarm_demos.py \\
        --config openarm_act_project/configs/act_openarm_reach.yaml

Required arguments:
    --config    Path to the task YAML config file.

Optional arguments:
    --num_episodes  Override the number of episodes to collect (default: from config).
    --headless      Run without viewer window (default: False).
    --device        Simulation device, e.g. ``cuda:0`` (default: ``cuda:0``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect OpenArm demonstrations for ACT training."
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the task YAML config file.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Number of episodes to collect (overrides config).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after Isaac Sim is running."""

import os
import time

import gymnasium as gym
import omni.log
import torch
import yaml

import openarm.tasks  # noqa: F401 — register OpenArm environments

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Add project src to path
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC_DIR))

from openarm_act.utils.io_utils import EpisodeWriter  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(args_cli.config)
    task: str = cfg["task"]
    coll_cfg: dict = cfg.get("collection", {})
    obs_cfg: dict = cfg.get("observation", {})

    num_episodes: int = args_cli.num_episodes or coll_cfg.get("num_episodes", 50)
    dataset_dir: str = os.path.join(
        os.path.dirname(args_cli.config), "..", coll_cfg.get("dataset_dir", "data/demos_hdf5/demo")
    )
    dataset_dir = os.path.normpath(dataset_dir)
    step_hz: int = coll_cfg.get("step_hz", 30)
    num_success_steps: int = coll_cfg.get("num_success_steps", 10)
    camera_names: list[str] = obs_cfg.get("camera_names", ["cam_main"])
    task_name: str = cfg.get("task_name", task)

    os.makedirs(dataset_dir, exist_ok=True)
    hdf5_path = os.path.join(dataset_dir, "demos.hdf5")

    # -----------------------------------------------------------------------
    # Build environment
    # -----------------------------------------------------------------------
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None

    env = gym.make(task, cfg=env_cfg).unwrapped

    # -----------------------------------------------------------------------
    # Teleoperation device
    # -----------------------------------------------------------------------
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))

    quit_requested = False

    def _quit_cb():
        nonlocal quit_requested
        quit_requested = True

    teleop.add_callback("Q", _quit_cb)
    teleop.reset()

    # -----------------------------------------------------------------------
    # Collection loop
    # -----------------------------------------------------------------------
    episode_idx = 0
    collected = 0
    step_period = 1.0 / step_hz

    print(f"\nCollecting {num_episodes} demonstrations for task '{task}'.")
    print("Controls: WASD/arrows — EE translation | Q — quit\n")

    while collected < num_episodes and not quit_requested:
        obs_raw, _ = env.reset()
        teleop.reset()

        episode_qpos: list = []
        episode_actions: list = []
        episode_images: dict[str, list] = {c: [] for c in camera_names}

        success_steps = 0
        success = False
        step_count = 0

        print(f"Episode {episode_idx + 1}: collecting…", end="", flush=True)

        while True:
            t0 = time.time()
            delta_pose, gripper_cmd = teleop.advance()

            # Build joint-level action from teleop delta (pass-through for simplicity)
            # In a full implementation this would use IK; here we pass the delta
            # directly so the script is runnable as-is.
            num_joints = obs_cfg.get("num_joints", 6)
            action_np = delta_pose[:num_joints].astype("float32")
            action_t = torch.from_numpy(action_np).float().unsqueeze(0).to(args_cli.device)

            obs_raw, _rew, terminated, truncated, info = env.step(action_t)
            step_count += 1

            # Parse qpos
            qpos = None
            if isinstance(obs_raw, dict):
                qpos_t = obs_raw.get("policy", {}).get("joint_pos")
                if qpos_t is not None:
                    qpos = qpos_t.squeeze(0).cpu().numpy()
            if qpos is None:
                qpos = action_np  # fallback

            episode_qpos.append(qpos)
            episode_actions.append(action_np)
            for cam in camera_names:
                raw = (obs_raw.get(f"images/{cam}") or obs_raw.get(cam)) if isinstance(obs_raw, dict) else None
                img = raw.squeeze(0).cpu().numpy().astype("uint8") if raw is not None else \
                    __import__("numpy").zeros((480, 640, 3), dtype="uint8")
                episode_images[cam].append(img)

            # Success check
            if info.get("success", terminated):
                success_steps += 1
            else:
                success_steps = 0

            if success_steps >= num_success_steps:
                success = True
                break

            if truncated or quit_requested:
                break

            # Rate limiting
            elapsed = time.time() - t0
            if elapsed < step_period:
                time.sleep(step_period - elapsed)

        status = "✓ SUCCESS" if success else "✗ failed"
        print(f" {step_count} steps — {status}")

        # Write episode
        import numpy as np
        with EpisodeWriter(hdf5_path, episode_idx, camera_names, task_name) as w:
            for t in range(len(episode_qpos)):
                w.add_timestep(
                    qpos=np.array(episode_qpos[t]),
                    action=np.array(episode_actions[t]),
                    images={c: episode_images[c][t] for c in camera_names},
                )
            w.mark_success(success)

        episode_idx += 1
        if success:
            collected += 1

    env.close()
    simulation_app.close()
    print(f"\nDone — {collected} successful episodes saved to {hdf5_path}")


if __name__ == "__main__":
    main()
