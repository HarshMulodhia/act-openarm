"""Thin wrapper around an Isaac Lab gymnasium environment for ACT data collection.

The wrapper provides a stable, typed API to:
  - reset the environment and obtain the initial observation dict,
  - step with a joint-position action and receive the next observation dict,
  - query task success,
  - close cleanly.

This module intentionally avoids importing Isaac Lab at the module level so
that it can be imported in processes where Isaac Lab is not installed (e.g.
for offline dataset inspection).  The actual Isaac Lab imports happen inside
``OpenArmIsaacEnv.__init__``, which must be called after ``AppLauncher`` has
started the simulation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch


DEFAULT_VIEWER_EYE = (2.0, -2.5, 1.6)
DEFAULT_VIEWER_LOOKAT = (0.45, 0.0, 0.25)


def configure_viewer_env_cfg(env_cfg: Any) -> None:
    """Apply a stable viewer camera and responsive render cadence."""
    env_cfg.viewer.eye = DEFAULT_VIEWER_EYE
    env_cfg.viewer.lookat = DEFAULT_VIEWER_LOOKAT
    env_cfg.sim.render_interval = max(1, getattr(env_cfg, "decimation", 1))
    if hasattr(env_cfg, "rerender_on_reset"):
        env_cfg.rerender_on_reset = True


def configure_camera_env_cfg(env_cfg: Any, camera_setup: dict | None) -> None:
    """Attach configured Isaac Lab RGB cameras to the scene config.

    The OpenArm upstream tasks do not currently expose visual observations, so
    this project adds cameras at runtime from the ACT YAML config.  Camera names
    become scene sensor names and can be used as policy or render cameras.
    """
    if not camera_setup:
        return
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    for name, spec in camera_setup.items():
        setattr(
            env_cfg.scene,
            name,
            CameraCfg(
                prim_path=spec["prim_path"],
                update_period=float(spec.get("update_period", 0.0333)),
                height=int(spec.get("height", 480)),
                width=int(spec.get("width", 640)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(spec.get("focal_length", 24.0)),
                    focus_distance=float(spec.get("focus_distance", 400.0)),
                    horizontal_aperture=float(spec.get("horizontal_aperture", 20.955)),
                    clipping_range=tuple(spec.get("clipping_range", (0.1, 1.0e5))),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(spec.get("offset_pos", (0.0, 0.0, 0.0))),
                    rot=tuple(spec.get("offset_rot", (1.0, 0.0, 0.0, 0.0))),
                    convention=spec.get("offset_convention", "ros"),
                ),
            ),
        )


def warmup_viewer(env: Any, num_frames: int = 3) -> None:
    """Render a few frames so the Isaac viewport/livestream shows the stage."""
    sim = getattr(env, "sim", None)
    if sim is None:
        return
    for _ in range(num_frames):
        sim.render()


class OpenArmIsaacEnv:
    """Gymnasium-based wrapper for OpenArm Isaac Lab task environments.

    Args:
        task: Gymnasium task ID registered by the ``openarm`` extension,
            e.g. ``"Isaac-Reach-OpenArm-v0"``.
        camera_names: Camera sensor keys to include in the observation dict.
        num_joints: Degrees of freedom of the arm (used for action validation).
        device: PyTorch device string for tensor operations.
    """

    def __init__(
        self,
        task: str,
        camera_names: list[str],
        num_joints: int,
        device: str = "cuda:0",
        camera_setup: dict | None = None,
        render_camera_names: list[str] | None = None,
        success_distance_threshold: float | None = None,
    ) -> None:
        import gymnasium as gym

        openarm_isaac_lab_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "openarm_isaac_lab")
        )
        if os.path.isdir(openarm_isaac_lab_root) and openarm_isaac_lab_root not in sys.path:
            sys.path.insert(0, openarm_isaac_lab_root)

        import openarm.tasks  # noqa: F401 — registers OpenArm envs

        self.task = task
        self.camera_names = camera_names
        self.render_camera_names = render_camera_names or []
        self.num_joints = num_joints
        self.device = device
        self.success_distance_threshold = success_distance_threshold

        # Disable concatenation so we receive a dict of individual terms.
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(task, device=device, num_envs=1)
        env_cfg.observations.policy.concatenate_terms = False
        configure_viewer_env_cfg(env_cfg)
        configure_camera_env_cfg(env_cfg, camera_setup)
        # Disable time-out so episodes run until success or manual abort.
        env_cfg.terminations.time_out = None

        self._env = gym.make(task, cfg=env_cfg).unwrapped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment and return the first observation dict.

        Returns:
            Observation dict with keys ``qpos`` (float32 array [J]) and
            per-camera image arrays under ``images/<camera_name>``
            (uint8 arrays [H, W, 3]).
        """
        obs_raw, _ = self._env.reset()
        warmup_viewer(self._env)
        return self._parse_obs(obs_raw)

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        """Step the environment with a joint-position action.

        Args:
            action: Target joint positions, float32 array of shape ``[J]``.

        Returns:
            Tuple of ``(observation_dict, success_flag)``.
        """
        action_t = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
        obs_raw, _reward, terminated, truncated, info = self._env.step(action_t)
        success: bool = bool(info.get("success", terminated))
        if not success and self.success_distance_threshold is not None:
            success = self.compute_reach_position_error() <= self.success_distance_threshold
        return self._parse_obs(obs_raw), success

    def capture_camera_image(self, camera_name: str) -> np.ndarray | None:
        """Return the latest RGB image from a scene camera sensor."""
        scene = getattr(self._env, "scene", None)
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", {})
        sensor = sensors.get(camera_name) if hasattr(sensors, "get") else None
        if sensor is None:
            try:
                sensor = scene[camera_name]
            except Exception:
                sensor = None
        if sensor is None:
            return None
        data = getattr(sensor, "data", None)
        output = getattr(data, "output", None)
        if output is None or "rgb" not in output:
            return None
        rgb = output["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu()
            if rgb.ndim == 4:
                rgb = rgb[0]
            rgb_np = rgb.numpy()
        else:
            rgb_np = np.asarray(rgb)
            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        return rgb_np.astype(np.uint8)

    def compute_reach_position_error(self) -> float:
        """Compute OpenArm reach end-effector distance to the active command."""
        try:
            from isaaclab.utils.math import combine_frame_transforms

            robot = self._env.scene["robot"]
            command = self._env.command_manager.get_command("ee_pose")
            desired_pos_b = command[:, :3]
            desired_pos_w, _ = combine_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                desired_pos_b,
            )
            body_ids, _ = robot.find_bodies(["openarm_hand"])
            body_id = body_ids[0]
            current_pos_w = robot.data.body_pos_w[:, body_id]
            return float(torch.norm(current_pos_w - desired_pos_w, dim=1)[0].detach().cpu())
        except Exception:
            return float("inf")

    def close(self) -> None:
        """Close the underlying gymnasium environment."""
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_obs(self, obs_raw: Any) -> dict:
        """Convert raw IsaacLab observation to a plain dict."""
        result: dict = {}

        # joint positions — may be under policy dict or top-level tensor
        if isinstance(obs_raw, dict):
            qpos_t = obs_raw.get("policy", {}).get("joint_pos", obs_raw.get("joint_pos"))
        else:
            qpos_t = obs_raw

        if qpos_t is not None:
            result["qpos"] = qpos_t.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            result["qpos"] = np.zeros(self.num_joints, dtype=np.float32)

        # camera images
        result["images"] = {}
        for cam in self.camera_names:
            img_key = f"images/{cam}"
            raw = None
            if isinstance(obs_raw, dict):
                raw = obs_raw.get(img_key) or obs_raw.get(cam)
            if raw is not None:
                # Expected shape: [1, H, W, 3] or [H, W, 3]
                img_np = raw.squeeze(0).cpu().numpy()
                result["images"][cam] = img_np.astype(np.uint8)
            else:
                scene_img = self.capture_camera_image(cam)
                if scene_img is not None:
                    result["images"][cam] = scene_img
                else:
                    # Placeholder (cameras may not be enabled in headless mode)
                    result["images"][cam] = np.zeros((480, 640, 3), dtype=np.uint8)

        result["render_images"] = {}
        for cam in self.render_camera_names:
            scene_img = self.capture_camera_image(cam)
            if scene_img is not None:
                result["render_images"][cam] = scene_img

        return result

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "OpenArmIsaacEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
