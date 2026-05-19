"""Collect teleoperated demonstrations from an OpenArm Isaac Lab task.

The script launches Isaac Sim, creates the specified task environment,
streams human teleoperation inputs, and writes successful episodes to HDF5
at the path configured in the YAML config file.

Usage (from workspace root)::

    /isaac-sim/python.sh openarm_act_project/scripts/collect_openarm_demos.py \\
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
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after Isaac Sim is running."""

import os
import time

import gymnasium as gym
import omni.log
import numpy as np
import torch
import yaml

# The OpenArm Isaac extension contains imports rooted at ``source.*``.
# Add the extension repository root when running from this project checkout.
_OPENARM_ISAAC_LAB_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "openarm_isaac_lab")
)
if os.path.isdir(_OPENARM_ISAAC_LAB_ROOT) and _OPENARM_ISAAC_LAB_ROOT not in sys.path:
    sys.path.insert(0, _OPENARM_ISAAC_LAB_ROOT)

import openarm.tasks  # noqa: F401 — register OpenArm environments

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Add project src to path
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC_DIR))

from openarm_act.envs.openarm_isaac_env import configure_viewer_env_cfg, warmup_viewer  # noqa: E402
from openarm_act.envs.openarm_isaac_env import configure_camera_env_cfg  # noqa: E402
from openarm_act.policies.act_openarm_policy import ACTOpenArmPolicy  # noqa: E402
from openarm_act.utils.io_utils import load_norm_stats  # noqa: E402
from openarm_act.utils.io_utils import EpisodeWriter  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: str | None) -> str | None:
    if path is None:
        return None
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.normpath(os.path.join(project_root, path) if not os.path.isabs(path) else path)


def _load_expert_policy(coll_cfg: dict, device: str):
    """Load an exported RSL-RL TorchScript policy for expert rollouts."""
    checkpoint = _resolve_project_path(coll_cfg.get("expert_checkpoint"))
    if checkpoint is None:
        if coll_cfg.get("use_pretrained_checkpoint", False):
            raise RuntimeError(
                "collection.use_pretrained_checkpoint is set, but this collector needs an exported "
                "RSL-RL policy.pt path. Run the OpenArm RSL-RL play script once to export policy.pt, "
                "then set collection.expert_checkpoint to that file."
            )
        raise RuntimeError("collection.expert_checkpoint must point to an exported RSL-RL policy.pt file.")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Expert policy file not found: {checkpoint}")
    policy = torch.jit.load(checkpoint, map_location=device)
    policy.eval()
    print(f"Loaded RSL-RL expert policy: {checkpoint}")
    return policy


def _policy_observation(obs_raw):
    if torch.is_tensor(obs_raw):
        return obs_raw
    if isinstance(obs_raw, dict):
        policy_obs = obs_raw.get("policy", obs_raw)
        if torch.is_tensor(policy_obs):
            return policy_obs
        if isinstance(policy_obs, dict):
            terms = [value for value in policy_obs.values() if torch.is_tensor(value)]
            if terms:
                return torch.cat([term.reshape(term.shape[0], -1) for term in terms], dim=-1)
    raise RuntimeError("Could not extract tensor policy observation for expert inference.")


def _get_qpos_from_env(env, num_joints: int):
    try:
        robot = env.scene["robot"]
        joint_ids, _ = robot.find_joints([f"openarm_joint{i}" for i in range(1, num_joints + 1)])
        return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype("float32")
    except Exception as exc:
        omni.log.warn(f"Failed to read joint positions from env; using zeros fallback: {exc}")
        return np.zeros(num_joints, dtype="float32")


def _capture_camera_image(env, camera_name: str, height: int, width: int):
    sensor = None
    sensors = getattr(env.scene, "sensors", {})
    if hasattr(sensors, "get"):
        sensor = sensors.get(camera_name)
    if sensor is None:
        try:
            sensor = env.scene[camera_name]
        except Exception:
            sensor = None
    output = getattr(getattr(sensor, "data", None), "output", None)
    if output is None or "rgb" not in output:
        return np.zeros((height, width, 3), dtype="uint8")
    rgb = output["rgb"]
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu()
        if rgb.ndim == 4:
            rgb = rgb[0]
        img = rgb.numpy()
    else:
        img = np.asarray(rgb)
        if img.ndim == 4:
            img = img[0]
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype("uint8")


def _resolve_action_dim_from_env(env) -> int | None:
    space = getattr(env, "action_space", None)
    shape = getattr(space, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def _ensure_vector_dim(name: str, vec: np.ndarray, expected: int) -> None:
    if vec.ndim != 1 or vec.shape[0] != expected:
        raise RuntimeError(f"{name} has shape {vec.shape}; expected [{expected}].")


def _add_action_noise(action: np.ndarray, noise_cfg: dict, step_idx: int, rng: np.random.Generator) -> np.ndarray:
    if not noise_cfg.get("enabled", False):
        return action.astype(np.float32)
    gaussian_std = float(noise_cfg.get("gaussian_std", 0.0))
    sinusoidal_amp = float(noise_cfg.get("sinusoidal_amplitude", 0.0))
    sinusoidal_period = max(int(noise_cfg.get("sinusoidal_period_steps", 60)), 1)
    noise = np.zeros_like(action, dtype=np.float32)
    if gaussian_std > 0.0:
        noise += rng.normal(0.0, gaussian_std, size=action.shape).astype(np.float32)
    if sinusoidal_amp > 0.0:
        phase = np.linspace(0.0, np.pi, num=action.shape[0], dtype=np.float32)
        noise += sinusoidal_amp * np.sin((2.0 * np.pi * step_idx / sinusoidal_period) + phase)
    return (action + noise).astype(np.float32)


def _obs_for_student_policy(
    env,
    obs_raw,
    camera_names: list[str],
    num_joints: int,
    image_height: int,
    image_width: int,
) -> dict[str, object]:
    qpos = _get_qpos_from_env(env, num_joints)
    images: dict[str, np.ndarray] = {}
    for cam in camera_names:
        raw = (obs_raw.get(f"images/{cam}") or obs_raw.get(cam)) if isinstance(obs_raw, dict) else None
        if raw is not None:
            images[cam] = raw.squeeze(0).cpu().numpy().astype("uint8")
        else:
            images[cam] = _capture_camera_image(env, cam, image_height, image_width)
    return {"qpos": qpos, "images": images}


def _reach_success(env, threshold: float) -> bool:
    try:
        from isaaclab.utils.math import combine_frame_transforms

        robot = env.scene["robot"]
        command = env.command_manager.get_command("ee_pose")
        desired_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        body_ids, _ = robot.find_bodies(["openarm_hand"])
        current_pos_w = robot.data.body_pos_w[:, body_ids[0]]
        distance = torch.norm(current_pos_w - desired_pos_w, dim=1)[0]
        return bool(distance <= threshold)
    except Exception:
        return False


def main() -> None:
    cfg = load_config(args_cli.config)
    task: str = cfg["task"]
    coll_cfg: dict = cfg.get("collection", {})
    obs_cfg: dict = cfg.get("observation", {})

    num_episodes: int = (
        args_cli.num_episodes
        if args_cli.num_episodes is not None
        else coll_cfg.get("num_episodes", 50)
    )
    dataset_dir: str = os.path.join(
        os.path.dirname(args_cli.config), "..", coll_cfg.get("dataset_dir", "data/demos_hdf5/demo")
    )
    dataset_dir = os.path.normpath(dataset_dir)
    step_hz: int = coll_cfg.get("step_hz", 30)
    num_success_steps: int = coll_cfg.get("num_success_steps", 10)
    max_timesteps: int = coll_cfg.get("max_timesteps", 400)
    success_distance_threshold: float = coll_cfg.get("success_distance_threshold", 0.03)
    camera_names: list[str] = obs_cfg.get("training_camera_names", obs_cfg.get("camera_names", ["cam_main"]))
    camera_setup: dict = obs_cfg.get("camera_setup", {})
    image_height: int = obs_cfg.get("image_height", 480)
    image_width: int = obs_cfg.get("image_width", 640)
    task_name: str = cfg.get("task_name", task)
    # Backward compatibility: older configs used `collection.expert_source`.
    collection_mode: str = coll_cfg.get("mode", coll_cfg.get("expert_source", "rsl_rl"))
    if collection_mode not in {"rsl_rl", "keyboard", "dagger"}:
        raise ValueError(f"Unsupported collection.mode: {collection_mode}")
    dagger_cfg: dict = coll_cfg.get("dagger", {})
    noise_cfg: dict = coll_cfg.get("noise", {})
    noise_rng = np.random.default_rng(int(noise_cfg.get("seed", 42)))

    os.makedirs(dataset_dir, exist_ok=True)
    hdf5_path = os.path.join(dataset_dir, "demos.hdf5")

    # -----------------------------------------------------------------------
    # Build environment
    # -----------------------------------------------------------------------
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = expert_source == "rsl_rl"
    configure_viewer_env_cfg(env_cfg)
    configure_camera_env_cfg(env_cfg, camera_setup)
    env_cfg.terminations.time_out = None

    env = gym.make(task, cfg=env_cfg).unwrapped
    env_action_dim = _resolve_action_dim_from_env(env)
    if env_action_dim is not None and env_action_dim != int(obs_cfg.get("num_joints", 7)):
        raise RuntimeError(
            f"Configuration mismatch: observation.num_joints={obs_cfg.get('num_joints', 7)} "
            f"but env action dim is {env_action_dim} for task {task}."
        )
    expert_policy = _load_expert_policy(coll_cfg, args_cli.device) if collection_mode in {"rsl_rl", "dagger"} else None
    student_policy = None
    if collection_mode == "dagger":
        student_checkpoint = _resolve_project_path(dagger_cfg.get("student_checkpoint_dir"))
        if not student_checkpoint:
            raise RuntimeError("collection.dagger.student_checkpoint_dir is required for collection.mode=dagger.")
        student_policy = ACTOpenArmPolicy.from_checkpoint(student_checkpoint, cfg, device=args_cli.device)
        norm_stats = load_norm_stats(student_checkpoint)
        if norm_stats is not None:
            student_policy.norm_stats = norm_stats
        student_policy._model.eval()
        print(f"Loaded ACT student policy: {student_checkpoint}")
    dagger_beta = float(dagger_cfg.get("beta", 0.5))

    # -----------------------------------------------------------------------
    # Teleoperation device
    # -----------------------------------------------------------------------
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5)) if collection_mode == "keyboard" else None

    quit_requested = False

    def _quit_cb():
        nonlocal quit_requested
        quit_requested = True

    if teleop is not None:
        teleop.add_callback("Q", _quit_cb)
        teleop.reset()

    # -----------------------------------------------------------------------
    # Collection loop
    # -----------------------------------------------------------------------
    episode_idx = 0
    collected = 0
    step_period = 1.0 / step_hz

    print(f"\nCollecting {num_episodes} demonstrations for task '{task}' with collection_mode={collection_mode}.")
    if teleop is not None:
        print("Controls: WASD/arrows — EE translation | Q — quit\n")

    while collected < num_episodes and not quit_requested:
        obs_raw, _ = env.reset()
        warmup_viewer(env)
        if expert_policy is not None and hasattr(expert_policy, "reset"):
            try:
                expert_policy.reset()
            except Exception:
                pass
        if student_policy is not None:
            student_policy.reset()
        if teleop is not None:
            teleop.reset()

        episode_qpos: list = []
        episode_actions: list = []
        episode_images: dict[str, list] = {c: [] for c in camera_names}

        success_steps = 0
        success = False
        step_count = 0

        print(f"Episode {episode_idx + 1}: collecting…", end="", flush=True)

        while step_count < max_timesteps:
            t0 = time.time()
            if expert_policy is not None:
                policy_obs = _policy_observation(obs_raw).to(args_cli.device)
                with torch.inference_mode():
                    action_t = expert_policy(policy_obs)
                if isinstance(action_t, tuple):
                    action_t = action_t[0]
                expert_action_np = action_t.squeeze(0).detach().cpu().numpy().astype("float32")
                _ensure_vector_dim("expert_action", expert_action_np, obs_cfg.get("num_joints", 7))
                if student_policy is not None:
                    student_obs = _obs_for_student_policy(
                        env, obs_raw, camera_names, obs_cfg.get("num_joints", 7), image_height, image_width
                    )
                    student_action_np = student_policy.get_action(student_obs["qpos"], student_obs["images"])
                    _ensure_vector_dim("student_action", student_action_np, obs_cfg.get("num_joints", 7))
                    use_expert_for_execution = bool(noise_rng.random() < dagger_beta)
                    exec_action_np = expert_action_np if use_expert_for_execution else student_action_np
                    action_np = expert_action_np
                else:
                    exec_action_np = expert_action_np
                    action_np = expert_action_np
                exec_action_np = _add_action_noise(exec_action_np, noise_cfg, step_count, noise_rng)
                _ensure_vector_dim("exec_action", exec_action_np, obs_cfg.get("num_joints", 7))
                action_t = torch.from_numpy(exec_action_np).float().unsqueeze(0).to(args_cli.device)
            else:
                teleop_cmd = teleop.advance()
                if isinstance(teleop_cmd, tuple):
                    delta_pose = teleop_cmd[0]
                else:
                    delta_pose = teleop_cmd
                if torch.is_tensor(delta_pose):
                    delta_pose = delta_pose.detach().cpu().numpy()
                num_joints = obs_cfg.get("num_joints", 7)
                action_np = delta_pose[:num_joints].astype("float32")
                _ensure_vector_dim("teleop_action", action_np, num_joints)
                exec_action_np = _add_action_noise(action_np, noise_cfg, step_count, noise_rng)
                _ensure_vector_dim("exec_action", exec_action_np, num_joints)
                action_t = torch.from_numpy(exec_action_np).float().unsqueeze(0).to(args_cli.device)

            obs_raw, _rew, terminated, truncated, info = env.step(action_t)
            step_count += 1

            qpos = _get_qpos_from_env(env, obs_cfg.get("num_joints", 7))

            episode_qpos.append(qpos)
            episode_actions.append(action_np)
            for cam in camera_names:
                raw = (obs_raw.get(f"images/{cam}") or obs_raw.get(cam)) if isinstance(obs_raw, dict) else None
                img = raw.squeeze(0).cpu().numpy().astype("uint8") if raw is not None else _capture_camera_image(
                    env, cam, image_height, image_width
                )
                episode_images[cam].append(img)

            # Success check
            if info.get("success", terminated) or _reach_success(env, success_distance_threshold):
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
