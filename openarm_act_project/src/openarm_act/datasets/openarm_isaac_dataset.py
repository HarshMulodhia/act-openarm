"""Dataset adapter that maps Isaac Lab HDF5 demonstrations to ACT training inputs.

Expected HDF5 layout produced by collect_openarm_demos.py
(one file per collection session, one group per episode):

    /episode_000/
        observations/
            images/
                cam_main          uint8  [T, H, W, 3]
            qpos                  float32 [T, J]
        actions/
            joint_target          float32 [T, J]
        meta/
            success               bool scalar
            task_name             bytes scalar

Successful episodes are filtered in; partial/corrupt episodes are skipped.
"""

from __future__ import annotations

import os
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class OpenArmIsaacDataset(Dataset):
    """PyTorch Dataset for ACT training from OpenArm Isaac Lab demonstrations.

    Args:
        dataset_dir: Directory containing one or more ``.hdf5`` demo files.
        camera_names: Ordered list of camera keys present in the HDF5 files.
        chunk_size: Number of consecutive timesteps returned as the action
            chunk (matches ACT's ``chunk_size`` / ``num_queries``).
        norm_stats: Optional dict with keys ``qpos_mean``, ``qpos_std``,
            ``action_mean``, ``action_std`` for normalising observations and
            actions.  When *None* no normalisation is applied.
    """

    def __init__(
        self,
        dataset_dir: str | Sequence[str],
        camera_names: Sequence[str],
        chunk_size: int,
        norm_stats: dict | None = None,
    ) -> None:
        if isinstance(dataset_dir, (str, os.PathLike)):
            self.dataset_dirs = [os.fspath(dataset_dir)]
        else:
            self.dataset_dirs = [os.fspath(path) for path in dataset_dir]
        self.dataset_dir = self.dataset_dirs[0] if len(self.dataset_dirs) == 1 else os.pathsep.join(self.dataset_dirs)
        self.camera_names = list(camera_names)
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats

        self._episodes = self._index_episodes()
        if not self._episodes:
            raise RuntimeError(
                f"No successful episodes found in '{self.dataset_dir}'. "
                "Run collect_openarm_demos.py first."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_episodes(self) -> list[dict]:
        """Walk dataset_dir and build an index of successful episodes."""
        episodes: list[dict] = []
        for dataset_dir in self.dataset_dirs:
            if not os.path.isdir(dataset_dir):
                continue
            hdf5_files = sorted(
                f for f in os.listdir(dataset_dir) if f.endswith(".hdf5")
            )
            for fname in hdf5_files:
                path = os.path.join(dataset_dir, fname)
                with h5py.File(path, "r") as f:
                    for key in sorted(f.keys()):
                        if not key.startswith("episode_"):
                            continue
                        grp = f[key]
                        # skip episodes that were not marked successful
                        if not bool(grp["meta/success"][()]):
                            continue
                        T: int = int(grp["observations/qpos"].shape[0])
                        episodes.append({"file": path, "key": key, "length": T})
        return episodes

    @staticmethod
    def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Repeat the last row until *arr* has exactly *target_len* rows."""
        pad = target_len - len(arr)
        if pad <= 0:
            return arr
        return np.concatenate([arr, np.tile(arr[-1:], (pad, 1))], axis=0)

    @staticmethod
    def _pad_images(
        imgs: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Repeat the last frame until tensor has *target_len* frames."""
        pad = target_len - imgs.shape[0]
        if pad <= 0:
            return imgs
        return torch.cat([imgs, imgs[-1:].expand(pad, -1, -1, -1)], dim=0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self._episodes[idx]
        with h5py.File(ep["file"], "r") as f:
            grp = f[ep["key"]]
            T = ep["length"]

            # sample a random window start
            max_start = max(0, T - self.chunk_size)
            start = int(np.random.randint(0, max_start + 1))
            end = min(start + self.chunk_size, T)

            qpos: np.ndarray = grp["observations/qpos"][start:end].astype(np.float32)
            action: np.ndarray = grp["actions/joint_target"][start:end].astype(np.float32)

            images: dict[str, torch.Tensor] = {}
            for cam in self.camera_names:
                raw = grp[f"observations/images/{cam}"][start:end]  # [t, H, W, 3]
                # uint8 -> float [0,1], reorder to [t, C, H, W]
                images[cam] = (
                    torch.from_numpy(raw).float().permute(0, 3, 1, 2) / 255.0
                )

        # Pad short windows (near end of episode) to chunk_size
        qpos = self._pad_to_length(qpos, self.chunk_size)
        action = self._pad_to_length(action, self.chunk_size)
        for cam in self.camera_names:
            images[cam] = self._pad_images(images[cam], self.chunk_size)

        # Optional normalisation
        if self.norm_stats is not None:
            qpos = (qpos - self.norm_stats["qpos_mean"]) / (
                self.norm_stats["qpos_std"] + 1e-8
            )
            action = (action - self.norm_stats["action_mean"]) / (
                self.norm_stats["action_std"] + 1e-8
            )

        return {
            "qpos": torch.from_numpy(qpos),
            "action": torch.from_numpy(action),
            "images": images,
        }

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def compute_norm_stats(self) -> dict:
        """Compute per-joint mean and std across all successful episodes.

        Returns:
            dict with ``qpos_mean``, ``qpos_std``, ``action_mean``,
            ``action_std`` as float32 numpy arrays of shape ``[J]``.
        """
        all_qpos: list[np.ndarray] = []
        all_action: list[np.ndarray] = []

        for ep in self._episodes:
            with h5py.File(ep["file"], "r") as f:
                grp = f[ep["key"]]
                all_qpos.append(grp["observations/qpos"][:].astype(np.float32))
                all_action.append(grp["actions/joint_target"][:].astype(np.float32))

        qpos_cat = np.concatenate(all_qpos, axis=0)
        action_cat = np.concatenate(all_action, axis=0)

        return {
            "qpos_mean": qpos_cat.mean(axis=0),
            "qpos_std": qpos_cat.std(axis=0),
            "action_mean": action_cat.mean(axis=0),
            "action_std": action_cat.std(axis=0),
        }
