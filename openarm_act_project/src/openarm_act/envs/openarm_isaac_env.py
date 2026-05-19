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

from typing import Any

import numpy as np
import torch


class OpenArmIsaacEnv:
    """Gymnasium-based wrapper for OpenArm Isaac Lab task environments.

    Args:
        task: Gymnasium task ID registered by the ``openarm`` extension,
            e.g. ``"OpenArmUnimanualReach-v0"``.
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
    ) -> None:
        import gymnasium as gym

        import openarm.tasks  # noqa: F401 — registers OpenArm envs

        self.task = task
        self.camera_names = camera_names
        self.num_joints = num_joints
        self.device = device

        # Disable concatenation so we receive a dict of individual terms.
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(task, device=device, num_envs=1)
        env_cfg.observations.policy.concatenate_terms = False
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
        return self._parse_obs(obs_raw), success

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
                # Placeholder (cameras may not be enabled in headless mode)
                result["images"][cam] = np.zeros((480, 640, 3), dtype=np.uint8)

        return result

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "OpenArmIsaacEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
