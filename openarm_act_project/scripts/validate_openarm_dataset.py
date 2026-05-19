"""Validate an OpenArm ACT HDF5 dataset before training."""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openarm_act.utils.dataset_tools import validate_dataset, write_validation_report


def _resolve_project_path(path: str) -> str:
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path) if not os.path.isabs(path) else path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OpenArm ACT HDF5 demonstrations.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    obs_cfg = cfg.get("observation", {})
    dataset_dir = _resolve_project_path(
        args.dataset_dir or cfg.get("collection", {}).get("dataset_dir", "data/demos_hdf5/demo")
    )
    camera_names = obs_cfg.get("training_camera_names", obs_cfg.get("camera_names", ["cam_main"]))
    report = validate_dataset(
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        num_joints=obs_cfg.get("num_joints", 7),
        image_height=obs_cfg.get("image_height"),
        image_width=obs_cfg.get("image_width"),
    )
    output = args.output or os.path.join(dataset_dir, "validation_report.json")
    write_validation_report(report, output)

    print(f"Dataset: {dataset_dir}")
    print(f"Files: {report.files} | episodes: {report.episodes} | successful: {report.successful_episodes}")
    print(f"Timesteps: {report.timesteps} | ok: {report.ok}")
    for issue in report.issues[:20]:
        print(f"[{issue.severity}] {issue.path}: {issue.message}")
    if len(report.issues) > 20:
        print(f"... {len(report.issues) - 20} more issue(s)")
    print(f"Report saved to: {output}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
