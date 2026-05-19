"""ACT policy wrapper for online inference in Isaac Sim.

This module wraps the upstream ``ACTPolicy`` from the ``act`` repository
(``/workspace/act/policy.py``) and exposes a simple inference interface that:
  - loads a saved checkpoint,
  - manages the action chunk buffer,
  - optionally performs temporal action aggregation (TAgg).

Usage::

    policy = ACTOpenArmPolicy.from_checkpoint(
        checkpoint_dir="checkpoints/openarm_reach_act/",
        config=cfg,
    )
    policy.reset()
    for obs in rollout:
        action = policy.get_action(obs["qpos"], obs["images"])
        env.step(action)
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch


class ACTOpenArmPolicy:
    """Inference wrapper around the ACT policy for OpenArm tasks.

    Args:
        act_model: An instantiated ``ACTPolicy`` object (from ``act.policy``).
        chunk_size: Number of actions predicted per forward pass.
        camera_names: Ordered list of camera keys matching training config.
        num_joints: Degrees of freedom of the arm.
        temporal_agg: Whether to use temporal action aggregation (TAgg).
            When *True* actions from overlapping chunks are averaged with
            exponentially decaying weights.
        device: Torch device string.
        norm_stats: Optional normalisation statistics dict.  Keys:
            ``qpos_mean``, ``qpos_std``, ``action_mean``, ``action_std``.
    """

    def __init__(
        self,
        act_model: object,
        chunk_size: int,
        camera_names: Sequence[str],
        num_joints: int,
        temporal_agg: bool = False,
        device: str = "cuda:0",
        norm_stats: dict | None = None,
    ) -> None:
        self._model = act_model
        self.chunk_size = chunk_size
        self.camera_names = list(camera_names)
        self.num_joints = num_joints
        self.temporal_agg = temporal_agg
        self.device = device
        self.norm_stats = norm_stats

        # Action buffer for chunk-based control
        self._action_queue: list[np.ndarray] = []
        # TAgg accumulator: shape [chunk_size, J]
        self._all_time_actions: torch.Tensor | None = None
        self._t: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        config: dict,
        device: str = "cuda:0",
    ) -> "ACTOpenArmPolicy":
        """Load an ACT checkpoint and return a ready-to-use policy.

        Args:
            checkpoint_dir: Directory that contains ``policy_best.ckpt`` or
                the highest-numbered ``policy_epoch_*.ckpt`` file.
            config: Flat config dict with keys matching
                ``act_openarm_reach.yaml``.
            device: Torch device string.

        Returns:
            Instantiated and eval-mode ``ACTOpenArmPolicy``.
        """
        import sys

        # Add upstream ACT repo paths so we can import ACTPolicy and its
        # absolute DETR utility imports such as ``util.misc``.
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        act_repo = os.environ.get(
            "ACT_REPO_DIR", os.path.join(os.path.dirname(project_root), "act")
        )
        act_repo = os.path.abspath(act_repo)
        if not os.path.isfile(os.path.join(act_repo, "policy.py")):
            fallback = "/workspace/act"
            if os.path.isfile(os.path.join(fallback, "policy.py")):
                act_repo = fallback
        act_detr = os.path.join(act_repo, "detr")
        for path in (act_detr, act_repo):
            if path not in sys.path:
                sys.path.insert(0, path)

        from policy import ACTPolicy  # type: ignore[import]

        model_cfg = config.get("model", {})
        obs_cfg = config.get("observation", {})
        chunk_size: int = model_cfg.get("chunk_size", 100)
        num_joints: int = obs_cfg.get("num_joints", 6)
        ckpt_path = cls._find_checkpoint(checkpoint_dir)

        # Build policy using ACT factory arguments
        policy_config = {
            "lr": config.get("training", {}).get("lr", 1e-5),
            "num_queries": chunk_size,
            "kl_weight": model_cfg.get("kl_weight", 10),
            "hidden_dim": model_cfg.get("hidden_dim", 512),
            "dim_feedforward": model_cfg.get("dim_feedforward", 3200),
            "lr_backbone": config.get("training", {}).get("lr_backbone", 1e-5),
            "backbone": model_cfg.get("backbone", "resnet18"),
            "enc_layers": model_cfg.get("num_encoder_layers", 4),
            "dec_layers": 7,
            "nheads": model_cfg.get("nheads", 8),
            "camera_names": obs_cfg.get("training_camera_names", obs_cfg.get("camera_names", ["cam_main"])),
        }
        task_cfg = config.get("task", "openarm")
        task_name = (
            task_cfg.get("name", "openarm")
            if isinstance(task_cfg, dict)
            else task_cfg
        )
        argv = sys.argv
        try:
            sys.argv = [
                argv[0],
                "--ckpt_dir",
                checkpoint_dir,
                "--policy_class",
                "ACT",
                "--task_name",
                str(task_name),
                "--seed",
                str(config.get("training", {}).get("seed", 0)),
                "--num_epochs",
                str(config.get("training", {}).get("num_epochs", 1)),
            ]
            act_model = ACTPolicy(policy_config)
        finally:
            sys.argv = argv
        act_model.to(device)

        state = torch.load(ckpt_path, map_location=device)
        loading_status = act_model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}\n{loading_status}")
        act_model.eval()

        return cls(
            act_model=act_model,
            chunk_size=chunk_size,
            camera_names=obs_cfg.get("training_camera_names", obs_cfg.get("camera_names", ["cam_main"])),
            num_joints=num_joints,
            temporal_agg=config.get("evaluation", {}).get("temporal_agg", False),
            device=device,
        )

    @staticmethod
    def _find_checkpoint(checkpoint_dir: str) -> str:
        """Return path to best or latest checkpoint in *checkpoint_dir*."""
        best = os.path.join(checkpoint_dir, "policy_best.ckpt")
        if os.path.isfile(best):
            return best
        ckpts = sorted(
            f for f in os.listdir(checkpoint_dir) if f.startswith("policy_epoch_")
        )
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoint found in '{checkpoint_dir}'. "
                "Run train_act_openarm.py first."
            )
        return os.path.join(checkpoint_dir, ckpts[-1])

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal action buffer (call at the start of every episode)."""
        self._action_queue = []
        self._all_time_actions = None
        self._t = 0

    def get_action(
        self,
        qpos: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute the next joint-position target from the current observation.

        A new ACT forward pass is only triggered when the action buffer is
        empty (i.e. every ``chunk_size`` steps), unless temporal aggregation
        is enabled (in which case it fires every step).

        Args:
            qpos: Current joint positions, float32 ``[J]``.
            images: Dict mapping camera names to uint8 images ``[H, W, 3]``.

        Returns:
            Next joint-position target, float32 ``[J]``.
        """
        if self.temporal_agg:
            return self._get_action_temporal_agg(qpos, images)

        # Chunk-based control: refill buffer when empty
        if not self._action_queue:
            actions = self._forward(qpos, images)  # [chunk, J]
            self._action_queue = list(actions)

        return self._action_queue.pop(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(
        self, qpos: np.ndarray, images: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Run one ACT forward pass and return the raw action chunk.

        Returns:
            float32 numpy array of shape ``[chunk_size, J]``.
        """
        # Normalise qpos if stats are available
        if self.norm_stats is not None:
            qpos_norm = (qpos - self.norm_stats["qpos_mean"]) / (
                self.norm_stats["qpos_std"] + 1e-8
            )
        else:
            qpos_norm = qpos

        qpos_t = (
            torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
        )  # [1, J]

        # Stack camera images into [1, n_cam, C, H, W]
        img_list = []
        for cam in self.camera_names:
            img = images[cam].astype(np.float32) / 255.0  # [H, W, 3]
            img_t = (
                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            )  # [1, C, H, W]
            img_list.append(img_t)
        image_data = torch.stack(img_list, dim=1)  # [1, n_cam, C, H, W]

        with torch.inference_mode():
            a_hat, _, _ = self._model(qpos_t, image_data, None, None)

        actions = a_hat.squeeze(0).cpu().numpy()  # [chunk, J]

        # Denormalise if stats are available
        if self.norm_stats is not None:
            actions = (
                actions * (self.norm_stats["action_std"] + 1e-8)
                + self.norm_stats["action_mean"]
            )
        return actions.astype(np.float32)

    def _get_action_temporal_agg(
        self, qpos: np.ndarray, images: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Temporal action aggregation (TAgg) from the ACT paper.

        Each step fires a new forward pass.  Actions from all previously
        predicted chunks are averaged with exponentially decaying weights,
        where older predictions receive lower weight.
        """
        K = self.chunk_size

        if self._all_time_actions is None:
            self._all_time_actions = torch.zeros(K, K, self.num_joints)

        actions = self._forward(qpos, images)  # [K, J]
        # Shift buffer and insert latest predictions
        self._all_time_actions = self._all_time_actions.roll(-1, dims=0)
        self._all_time_actions[-1] = torch.from_numpy(actions)

        # Exponentially decaying weights: most-recent row has highest weight
        weights = torch.exp(
            torch.arange(K, dtype=torch.float32) / K * torch.log(torch.tensor(K + 1.0))
        )
        weights /= weights.sum()

        # Each row i contains predictions for future steps; extract index self._t % K
        step_idx = self._t % K
        relevant = self._all_time_actions[:, step_idx, :]  # [K, J]
        aggregated = (relevant * weights.unsqueeze(1)).sum(0)  # [J]

        self._t += 1
        return aggregated.numpy().astype(np.float32)
