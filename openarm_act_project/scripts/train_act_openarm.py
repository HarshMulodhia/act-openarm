"""Train an ACT policy on OpenArm Isaac Lab demonstrations.

Loads the HDF5 dataset produced by ``collect_openarm_demos.py``, builds an
ACTPolicy from the upstream ``act`` repository, and runs the training loop,
saving checkpoints and loss curves to the configured output directory.

Usage::

    python openarm_act_project/scripts/train_act_openarm.py \\
        --config openarm_act_project/configs/act_openarm_reach.yaml

Required arguments:
    --config    Path to the task YAML config file.

Optional arguments:
    --resume    Resume from the latest existing checkpoint.
    --seed      Random seed (overrides config value).
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# ---------------------------------------------------------------------------
# Argument parsing (no Isaac Sim required for training)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train ACT policy for OpenArm.")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Add upstream ACT and project src to sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_ACT_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "act"))

for _p in (_SRC_DIR, _ACT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Remaining imports after path setup
# ---------------------------------------------------------------------------

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from openarm_act.datasets.openarm_isaac_dataset import OpenArmIsaacDataset
from openarm_act.utils.io_utils import load_norm_stats, save_norm_stats
from openarm_act.utils.viz_utils import plot_training_curves

from policy import ACTPolicy  # type: ignore[import]  — upstream act/policy.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_policy(cfg: dict, device: str) -> ACTPolicy:
    model_cfg = cfg.get("model", {})
    obs_cfg = cfg.get("observation", {})
    train_cfg = cfg.get("training", {})
    policy_config = {
        "lr": train_cfg.get("lr", 1e-5),
        "num_queries": model_cfg.get("chunk_size", 100),
        "kl_weight": model_cfg.get("kl_weight", 10),
        "hidden_dim": model_cfg.get("hidden_dim", 512),
        "dim_feedforward": model_cfg.get("dim_feedforward", 3200),
        "lr_backbone": train_cfg.get("lr_backbone", 1e-5),
        "backbone": model_cfg.get("backbone", "resnet18"),
        "enc_layers": model_cfg.get("num_encoder_layers", 4),
        "dec_layers": 7,
        "nheads": model_cfg.get("nheads", 8),
        "camera_names": obs_cfg.get("camera_names", ["cam_main"]),
    }
    policy = ACTPolicy(policy_config)
    policy.to(device)
    return policy


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config(args_cli.config)
    train_cfg = cfg.get("training", {})
    obs_cfg = cfg.get("observation", {})
    model_cfg = cfg.get("model", {})

    seed: int = args_cli.seed if args_cli.seed is not None else train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    dataset_dir = os.path.join(
        _PROJECT_ROOT,
        cfg.get("collection", {}).get("dataset_dir", "data/demos_hdf5/demo"),
    )
    checkpoint_dir = os.path.join(_PROJECT_ROOT, train_cfg.get("checkpoint_dir", "checkpoints/act"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    camera_names: list[str] = obs_cfg.get("camera_names", ["cam_main"])
    chunk_size: int = model_cfg.get("chunk_size", 100)

    dataset = OpenArmIsaacDataset(dataset_dir, camera_names, chunk_size)
    norm_stats = dataset.compute_norm_stats()
    save_norm_stats(norm_stats, checkpoint_dir)
    dataset.norm_stats = norm_stats

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Policy
    policy = build_policy(cfg, device)

    # Optionally resume
    start_epoch = 0
    if args_cli.resume:
        from openarm_act.utils.io_utils import latest_checkpoint_path

        ckpt = latest_checkpoint_path(checkpoint_dir)
        if ckpt:
            state = torch.load(ckpt, map_location=device)
            policy.load_state_dict(state)
            print(f"Resumed from {ckpt}")

    num_epochs: int = train_cfg.get("num_epochs", 2000)
    save_every: int = train_cfg.get("save_every", 100)
    eval_every: int = train_cfg.get("eval_every", 500)

    train_losses: list[float] = []
    best_val_loss = float("inf")

    print(
        f"\nTraining ACT for {num_epochs} epochs | "
        f"dataset size: {len(dataset)} episodes | "
        f"device: {device}\n"
    )

    for epoch in range(start_epoch, num_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            qpos = batch["qpos"].to(device)           # [B, chunk, J]
            actions = batch["action"].to(device)       # [B, chunk, J]
            images = {
                cam: batch["images"][cam].to(device)
                for cam in camera_names
            }

            # ACTPolicy expects image tensor [B, n_cam, C, H, W]
            image_data = torch.stack(
                [images[cam] for cam in camera_names], dim=1
            )

            # Use the first timestep qpos as the current state
            qpos_cur = qpos[:, 0, :]  # [B, J]

            forward_dict = policy(qpos_cur, image_data, actions, False)
            loss = forward_dict["loss"]
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]  loss: {avg_loss:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch + 1:04d}.ckpt")
            torch.save(policy.serialize(), ckpt_path)
            print(f"  → saved checkpoint: {ckpt_path}")

        # Track best checkpoint based on train loss (no sim eval during training)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(policy.serialize(), os.path.join(checkpoint_dir, "policy_best.ckpt"))

    # Final checkpoint
    torch.save(
        policy.serialize(),
        os.path.join(checkpoint_dir, f"policy_epoch_{num_epochs:04d}.ckpt"),
    )

    # Save loss curves
    plot_training_curves(
        train_losses,
        eval_losses=None,
        output_path=os.path.join(checkpoint_dir, "train_loss.png"),
    )
    print(f"\nTraining complete.  Best loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
