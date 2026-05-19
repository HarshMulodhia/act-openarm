"""Dataset validation, augmentation, and experiment summary helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


@dataclass
class DatasetIssue:
    """One validation issue found in an HDF5 dataset."""

    severity: str
    path: str
    message: str


@dataclass
class DatasetValidationReport:
    """Validation result for one or more HDF5 files."""

    dataset_dir: str
    files: int = 0
    episodes: int = 0
    successful_episodes: int = 0
    timesteps: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "files": self.files,
            "episodes": self.episodes,
            "successful_episodes": self.successful_episodes,
            "timesteps": self.timesteps,
            "ok": self.ok,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def _hdf5_files(dataset_dir: str) -> list[Path]:
    path = Path(dataset_dir)
    if not path.is_dir():
        return []
    return sorted(path.glob("*.hdf5"))


def validate_dataset(
    dataset_dir: str,
    camera_names: Sequence[str],
    num_joints: int,
    image_height: int | None = None,
    image_width: int | None = None,
    require_success: bool = True,
) -> DatasetValidationReport:
    """Validate OpenArm ACT HDF5 demonstrations."""

    report = DatasetValidationReport(dataset_dir=dataset_dir)
    files = _hdf5_files(dataset_dir)
    report.files = len(files)
    if not files:
        report.issues.append(DatasetIssue("error", dataset_dir, "No .hdf5 files found."))
        return report

    for hdf5_path in files:
        try:
            h5 = h5py.File(hdf5_path, "r")
        except OSError as exc:
            report.issues.append(DatasetIssue("error", str(hdf5_path), f"Cannot open file: {exc}"))
            continue

        with h5:
            episode_keys = [key for key in sorted(h5.keys()) if key.startswith("episode_")]
            if not episode_keys:
                report.issues.append(DatasetIssue("error", str(hdf5_path), "No episode_* groups found."))
                continue

            for key in episode_keys:
                ep_path = f"{hdf5_path}:{key}"
                report.episodes += 1
                grp = h5[key]
                required = [
                    "observations/qpos",
                    "actions/joint_target",
                    "meta/success",
                    "meta/task_name",
                ]
                required.extend(f"observations/images/{cam}" for cam in camera_names)
                missing = [name for name in required if name not in grp]
                if missing:
                    report.issues.append(
                        DatasetIssue("error", ep_path, f"Missing required datasets: {', '.join(missing)}")
                    )
                    continue

                success = bool(grp["meta/success"][()])
                if success:
                    report.successful_episodes += 1
                elif require_success:
                    report.issues.append(DatasetIssue("warning", ep_path, "Episode is marked unsuccessful."))

                qpos = grp["observations/qpos"]
                actions = grp["actions/joint_target"]
                if qpos.ndim != 2 or qpos.shape[1] != num_joints:
                    report.issues.append(
                        DatasetIssue("error", ep_path, f"qpos shape {qpos.shape} does not match [T,{num_joints}].")
                    )
                if actions.ndim != 2 or actions.shape[1] != num_joints:
                    report.issues.append(
                        DatasetIssue(
                            "error", ep_path, f"joint_target shape {actions.shape} does not match [T,{num_joints}]."
                        )
                    )
                if qpos.shape[0] == 0 or actions.shape[0] == 0:
                    report.issues.append(DatasetIssue("error", ep_path, "Episode has zero timesteps."))
                if qpos.shape[0] != actions.shape[0]:
                    report.issues.append(
                        DatasetIssue("error", ep_path, f"qpos length {qpos.shape[0]} != action length {actions.shape[0]}.")
                    )

                lengths = [qpos.shape[0], actions.shape[0]]
                for cam in camera_names:
                    imgs = grp[f"observations/images/{cam}"]
                    lengths.append(imgs.shape[0])
                    if imgs.ndim != 4 or imgs.shape[-1] != 3:
                        report.issues.append(
                            DatasetIssue("error", ep_path, f"Camera {cam} shape {imgs.shape} is not [T,H,W,3].")
                        )
                    if image_height is not None and imgs.shape[1] != image_height:
                        report.issues.append(
                            DatasetIssue("error", ep_path, f"Camera {cam} height {imgs.shape[1]} != {image_height}.")
                        )
                    if image_width is not None and imgs.shape[2] != image_width:
                        report.issues.append(
                            DatasetIssue("error", ep_path, f"Camera {cam} width {imgs.shape[2]} != {image_width}.")
                        )
                    if imgs.dtype != np.uint8:
                        report.issues.append(DatasetIssue("error", ep_path, f"Camera {cam} dtype is {imgs.dtype}, not uint8."))

                if len(set(lengths)) != 1:
                    report.issues.append(DatasetIssue("error", ep_path, f"Timestep lengths do not match: {lengths}."))
                else:
                    report.timesteps += int(lengths[0])

    if require_success and report.successful_episodes == 0:
        report.issues.append(DatasetIssue("error", dataset_dir, "No successful episodes found."))
    return report


def write_validation_report(report: DatasetValidationReport, output_path: str) -> None:
    """Write a validation report JSON file."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def create_action_noise_dataset(
    source_dataset_dir: str,
    output_dataset_dir: str,
    noise_type: str = "gaussian",
    gaussian_std: float = 0.01,
    sinusoidal_amplitude: float = 0.01,
    sinusoidal_period_steps: int = 60,
    seed: int = 42,
    clip_to_source_range: bool = False,
) -> dict[str, Any]:
    """Copy a dataset and perturb only ``actions/joint_target``."""

    src = Path(source_dataset_dir)
    dst = Path(output_dataset_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dataset_dir}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    files_written = 0
    episodes = 0
    timesteps = 0

    for src_file in sorted(src.glob("*.hdf5")):
        dst_file = dst / src_file.name
        shutil.copy2(src_file, dst_file)
        files_written += 1
        with h5py.File(dst_file, "a") as h5:
            h5.attrs["augmentation/source_dataset_dir"] = str(src)
            h5.attrs["augmentation/noise_type"] = noise_type
            h5.attrs["augmentation/seed"] = seed
            h5.attrs["augmentation/gaussian_std"] = gaussian_std
            h5.attrs["augmentation/sinusoidal_amplitude"] = sinusoidal_amplitude
            h5.attrs["augmentation/sinusoidal_period_steps"] = sinusoidal_period_steps
            h5.attrs["augmentation/actions_only"] = True
            for key in sorted(h5.keys()):
                if not key.startswith("episode_") or "actions/joint_target" not in h5[key]:
                    continue
                ds = h5[key]["actions/joint_target"]
                actions = ds[:].astype(np.float32)
                if noise_type == "gaussian":
                    noise = rng.normal(0.0, gaussian_std, size=actions.shape).astype(np.float32)
                elif noise_type == "sinusoidal":
                    t = np.arange(actions.shape[0], dtype=np.float32)[:, None]
                    phase = rng.uniform(0.0, 2.0 * math.pi, size=(1, actions.shape[1])).astype(np.float32)
                    noise = sinusoidal_amplitude * np.sin((2.0 * math.pi * t / sinusoidal_period_steps) + phase)
                else:
                    raise ValueError(f"Unsupported noise_type: {noise_type}")
                noisy = actions + noise
                if clip_to_source_range:
                    noisy = np.clip(noisy, actions.min(axis=0), actions.max(axis=0))
                ds[...] = noisy.astype(ds.dtype)
                h5[key].attrs["augmentation/noise_type"] = noise_type
                h5[key].attrs["augmentation/actions_only"] = True
                episodes += 1
                timesteps += int(actions.shape[0])

    summary = {
        "source_dataset_dir": str(src),
        "output_dataset_dir": str(dst),
        "noise_type": noise_type,
        "gaussian_std": gaussian_std,
        "sinusoidal_amplitude": sinusoidal_amplitude,
        "sinusoidal_period_steps": sinusoidal_period_steps,
        "seed": seed,
        "clip_to_source_range": clip_to_source_range,
        "files_written": files_written,
        "episodes": episodes,
        "timesteps": timesteps,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(dst / "augmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_experiment_summary(output_path: str, summary: dict[str, Any]) -> None:
    """Write a JSON experiment summary with a UTC timestamp."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), **summary}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
