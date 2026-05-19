"""Export an OpenArm RSL-RL checkpoint to a TorchScript expert policy.

This is the bridge between ``openarm_isaac_lab`` RSL-RL training artifacts and
ACT demonstration collection.  It loads an RSL-RL runner checkpoint, exports the
inference actor and normalizer to ``policy.pt``, and prints the path to set as
``collection.expert_checkpoint``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export an OpenArm RSL-RL expert policy.")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint to load.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the Isaac Lab published checkpoint if one exists for the task.",
)
parser.add_argument("--output", type=str, default="checkpoints/openarm_reach_expert/policy.pt")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after Isaac Sim is running."""

import gymnasium as gym
import torch

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_OPENARM_ISAAC_LAB_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "openarm_isaac_lab"))
if os.path.isdir(_OPENARM_ISAAC_LAB_ROOT) and _OPENARM_ISAAC_LAB_ROOT not in sys.path:
    sys.path.insert(0, _OPENARM_ISAAC_LAB_ROOT)

import openarm.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import DistillationRunner, OnPolicyRunner


def _resolve_project_path(path: str) -> str:
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path) if not os.path.isabs(path) else path)


def _resolve_checkpoint(task: str, checkpoint: str | None, use_pretrained: bool) -> str:
    if checkpoint:
        return retrieve_file_path(checkpoint)
    if use_pretrained:
        path = get_published_pretrained_checkpoint("rsl_rl", task)
        if path:
            return path
        raise RuntimeError(f"No published RSL-RL pretrained checkpoint is available for {task}.")
    raise RuntimeError("Provide --checkpoint or --use_pretrained_checkpoint.")


def main() -> None:
    print(f"[INFO] Resolving expert checkpoint for task: {args_cli.task}", flush=True)
    resume_path = _resolve_checkpoint(args_cli.task, args_cli.checkpoint, args_cli.use_pretrained_checkpoint)
    output_path = _resolve_project_path(args_cli.output)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.sim.device = args_cli.device

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading RSL-RL checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_policy_as_jit(policy_nn, normalizer=normalizer, path=output_dir, filename=output_name)
    env.close()
    print(f"Exported expert policy: {output_path}")
    print("Set this in configs/act_openarm_reach.yaml:")
    print(f"  collection.expert_checkpoint: {os.path.relpath(output_path, _PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        simulation_app.close()
