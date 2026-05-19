"""Create an action-noise augmented OpenArm ACT dataset."""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openarm_act.utils.dataset_tools import create_action_noise_dataset


def _resolve_project_path(path: str) -> str:
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path) if not os.path.isabs(path) else path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create action-noise augmented OpenArm ACT demos.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--source-dataset-dir", type=str, default=None)
    parser.add_argument("--output-dataset-dir", type=str, default=None)
    parser.add_argument("--noise-type", choices=["gaussian", "sinusoidal"], default=None)
    parser.add_argument("--gaussian-std", type=float, default=None)
    parser.add_argument("--sinusoidal-amplitude", type=float, default=None)
    parser.add_argument("--sinusoidal-period-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    aug_cfg = cfg.get("augmentation", {})

    source = _resolve_project_path(args.source_dataset_dir or aug_cfg.get("source_dataset_dir"))
    output = _resolve_project_path(args.output_dataset_dir or aug_cfg.get("output_dataset_dir"))
    summary = create_action_noise_dataset(
        source_dataset_dir=source,
        output_dataset_dir=output,
        noise_type=args.noise_type or aug_cfg.get("noise_type", "gaussian"),
        gaussian_std=args.gaussian_std if args.gaussian_std is not None else aug_cfg.get("gaussian_std", 0.01),
        sinusoidal_amplitude=(
            args.sinusoidal_amplitude
            if args.sinusoidal_amplitude is not None
            else aug_cfg.get("sinusoidal_amplitude", 0.01)
        ),
        sinusoidal_period_steps=(
            args.sinusoidal_period_steps
            if args.sinusoidal_period_steps is not None
            else aug_cfg.get("sinusoidal_period_steps", 60)
        ),
        seed=args.seed if args.seed is not None else aug_cfg.get("seed", 42),
        clip_to_source_range=aug_cfg.get("clip_to_source_range", False),
    )
    print(f"Noise dataset written to: {output}")
    print(f"Episodes: {summary['episodes']} | timesteps: {summary['timesteps']}")


if __name__ == "__main__":
    main()
