"""Visualisation helpers for the openarm_act project.

Provides lightweight utilities for:
  - saving an episode summary as a grid of frames,
  - plotting training loss curves,
  - rendering a policy rollout summary.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Episode frame grid
# ---------------------------------------------------------------------------


def save_episode_frames(
    images: np.ndarray,
    output_path: str,
    num_frames: int = 8,
    title: str = "",
) -> None:
    """Save a uniform subsample of episode frames as a single image grid.

    Args:
        images: uint8 array of shape ``[T, H, W, 3]``.
        output_path: Destination path (PNG recommended).
        num_frames: Number of frames to sample uniformly from the episode.
        title: Optional title rendered above the grid.
    """
    T = len(images)
    indices = np.linspace(0, T - 1, min(num_frames, T), dtype=int)
    selected = images[indices]

    cols = min(num_frames, 4)
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes_flat = np.array(axes).flatten() if rows * cols > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i < len(selected):
            ax.imshow(selected[i])
            ax.set_title(f"t={indices[i]}", fontsize=7)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_losses: Sequence[float],
    eval_losses: Sequence[float] | None,
    output_path: str,
    eval_interval: int = 1,
) -> None:
    """Plot training (and optional evaluation) loss over epochs.

    Args:
        train_losses: List of per-epoch training losses.
        eval_losses: List of per-eval evaluation losses (may be shorter).
        output_path: Destination path for the PNG figure.
        eval_interval: Number of training epochs between each eval.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train loss", color="steelblue", linewidth=1.2)
    if eval_losses:
        eval_steps = [i * eval_interval for i in range(len(eval_losses))]
        ax.plot(eval_steps, eval_losses, label="eval loss", color="darkorange", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ACT Training Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rollout summary
# ---------------------------------------------------------------------------


def plot_rollout_summary(
    qpos_history: np.ndarray,
    action_history: np.ndarray,
    output_path: str,
    joint_names: Sequence[str] | None = None,
) -> None:
    """Plot joint positions and commanded actions over a single rollout.

    Args:
        qpos_history: float32 array ``[T, J]`` of observed joint positions.
        action_history: float32 array ``[T, J]`` of commanded joint targets.
        output_path: Destination path for the PNG figure.
        joint_names: Optional list of joint label strings (length J).
    """
    T, J = qpos_history.shape
    if joint_names is None:
        joint_names = [f"j{i}" for i in range(J)]

    fig, axes = plt.subplots(J, 1, figsize=(10, J * 1.8), sharex=True)
    if J == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(qpos_history[:, j], label="qpos", linewidth=1.0)
        ax.plot(action_history[:, j], label="target", linewidth=1.0, linestyle="--")
        ax.set_ylabel(joint_names[j], fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Rollout: joint positions vs. commanded targets", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
