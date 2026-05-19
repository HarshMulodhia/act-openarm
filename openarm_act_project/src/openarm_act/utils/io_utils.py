"""HDF5 and file I/O helpers for the openarm_act project.

Conventions:
  - All demo files are stored as ``<dataset_dir>/<session_id>.hdf5``.
  - Each HDF5 file may contain multiple episode groups named
    ``episode_000``, ``episode_001``, ...
  - Normalisation statistics are persisted as ``<checkpoint_dir>/norm_stats.pkl``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 episode writing
# ---------------------------------------------------------------------------


class EpisodeWriter:
    """Context-manager that builds one HDF5 episode group incrementally.

    Usage::

        with EpisodeWriter(hdf5_file, episode_idx, camera_names) as w:
            for obs, action in data:
                w.add_timestep(
                    qpos=obs["qpos"],
                    action=action,
                    images=obs["images"],
                )
            w.mark_success(True)

    The writer flushes to disk on ``__exit__``.
    """

    def __init__(
        self,
        hdf5_path: str,
        episode_idx: int,
        camera_names: list[str],
        task_name: str = "",
    ) -> None:
        self._path = hdf5_path
        self._key = f"episode_{episode_idx:03d}"
        self._camera_names = camera_names
        self._task_name = task_name
        self._qpos_buf: list[np.ndarray] = []
        self._action_buf: list[np.ndarray] = []
        self._image_bufs: dict[str, list[np.ndarray]] = {c: [] for c in camera_names}
        self._success = False

    def add_timestep(
        self,
        qpos: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        """Append one timestep of data."""
        self._qpos_buf.append(qpos.astype(np.float32))
        self._action_buf.append(action.astype(np.float32))
        for cam in self._camera_names:
            self._image_bufs[cam].append(images[cam].astype(np.uint8))

    def mark_success(self, success: bool) -> None:
        """Mark whether the episode succeeded."""
        self._success = success

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._qpos_buf:
            return
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with h5py.File(self._path, "a") as f:
            grp = f.require_group(self._key)
            obs = grp.require_group("observations")
            imgs = obs.require_group("images")

            for cam in self._camera_names:
                imgs.create_dataset(
                    cam,
                    data=np.stack(self._image_bufs[cam]),
                    compression="gzip",
                    compression_opts=4,
                )
            obs.create_dataset("qpos", data=np.stack(self._qpos_buf))

            acts = grp.require_group("actions")
            acts.create_dataset(
                "joint_target", data=np.stack(self._action_buf)
            )

            meta = grp.require_group("meta")
            meta.create_dataset("success", data=np.bool_(self._success))
            meta.create_dataset(
                "task_name", data=np.bytes_(self._task_name)
            )


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------


def count_episodes(dataset_dir: str, success_only: bool = True) -> int:
    """Return the number of episodes in *dataset_dir*.

    Args:
        dataset_dir: Path to the directory containing ``.hdf5`` files.
        success_only: When *True* count only successful episodes.

    Returns:
        Total episode count.
    """
    count = 0
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".hdf5"):
            continue
        with h5py.File(os.path.join(dataset_dir, fname), "r") as f:
            for key in f.keys():
                if not key.startswith("episode_"):
                    continue
                if success_only and not bool(f[key]["meta/success"][()]):
                    continue
                count += 1
    return count


def load_episode(
    hdf5_path: str, episode_key: str, camera_names: list[str]
) -> dict:
    """Load a single episode into memory.

    Returns:
        Dict with keys ``qpos``, ``actions``, and per-camera image arrays.
    """
    with h5py.File(hdf5_path, "r") as f:
        grp = f[episode_key]
        result = {
            "qpos": grp["observations/qpos"][:].astype(np.float32),
            "actions": grp["actions/joint_target"][:].astype(np.float32),
            "success": bool(grp["meta/success"][()]),
            "task_name": grp["meta/task_name"][()].decode(),
            "images": {
                cam: grp[f"observations/images/{cam}"][:] for cam in camera_names
            },
        }
    return result


# ---------------------------------------------------------------------------
# Normalisation statistics persistence
# ---------------------------------------------------------------------------


def save_norm_stats(norm_stats: dict, checkpoint_dir: str) -> None:
    """Persist normalisation statistics to *checkpoint_dir*/norm_stats.pkl."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "norm_stats.pkl")
    with open(path, "wb") as f:
        pickle.dump(norm_stats, f)
    print(f"Saved normalisation statistics → {path}")


def load_norm_stats(checkpoint_dir: str) -> dict | None:
    """Load normalisation statistics from *checkpoint_dir*/norm_stats.pkl.

    Returns *None* if no statistics file is found.
    """
    path = os.path.join(checkpoint_dir, "norm_stats.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        stats = pickle.load(f)
    return stats


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def latest_checkpoint_path(checkpoint_dir: str) -> str | None:
    """Return path to the most recent checkpoint file in *checkpoint_dir*."""
    best = os.path.join(checkpoint_dir, "policy_best.ckpt")
    if os.path.isfile(best):
        return best
    ckpts = sorted(
        Path(checkpoint_dir).glob("policy_epoch_*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    return str(ckpts[-1]) if ckpts else None
