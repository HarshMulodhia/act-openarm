# ACT Policy for OpenArm in Isaac Sim — Complete Pipeline

This document describes the full end-to-end workflow for training and deploying
an [ACT (Action Chunking Transformer)](https://github.com/tonyzhaozh/act) policy
on OpenArm manipulation tasks in Isaac Lab / Isaac Sim.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Workspace Setup](#2-workspace-setup)
3. [Install Dependencies](#3-install-dependencies)
4. [Data Contract](#4-data-contract)
5. [Step A — Collect Demonstrations](#5-step-a--collect-demonstrations)
6. [Step B — Train ACT Policy](#6-step-b--train-act-policy)
7. [Step C — Evaluate in Simulation](#7-step-c--evaluate-in-simulation)
8. [Step D — Deploy in Isaac Sim](#8-step-d--deploy-in-isaac-sim)
9. [Configuration Reference](#9-configuration-reference)
10. [Available Tasks](#10-available-tasks)
11. [Project Structure](#11-project-structure)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| NVIDIA GPU | Ampere or newer recommended |
| Isaac Sim | 4.x (Omniverse) |
| Isaac Lab | matching Isaac Sim version |
| Python | 3.10+ |
| CUDA | 12.x |

The following repositories must exist in `/workspace/`:

```text
/workspace/
  act/                    # git clone https://github.com/tonyzhaozh/act
  isaaclab/               # pre-installed or git clone https://github.com/isaac-sim/IsaacLab
  openarm/                # git clone https://github.com/enactic/openarm
  openarm_isaac_lab/      # git clone https://github.com/enactic/openarm_isaac_lab
  openarm_act_project/    # this project (owned integration layer)
```

---

## 2. Workspace Setup

```bash
cd /workspace

# Clone upstream repos (skip if already present)
git clone https://github.com/enactic/openarm.git
git clone https://github.com/enactic/openarm_isaac_lab.git
git clone https://github.com/tonyzhaozh/act.git

# Install the openarm Isaac Lab extension in editable mode
cd /workspace/isaaclab
python -m pip install -e /workspace/openarm_isaac_lab/source/openarm

# Verify OpenArm tasks are registered
python /workspace/openarm_isaac_lab/scripts/tools/list_envs.py
```

Expected task IDs after installation:

- `OpenArmUnimanualReach-v0`
- `OpenArmUnimanualLift-v0`
- `OpenArmUnimanualCabinet-v0`
- `OpenArmBimanualReach-v0`

---

## 3. Install Dependencies

```bash
# Install the integration package in editable mode
cd /workspace/openarm_act_project
pip install -e .

# Install ACT dependencies
cd /workspace/act
pip install -r conda_env.yaml   # or: conda env create -f conda_env.yaml
```

---

## 4. Data Contract

All demonstration episodes are stored as HDF5 groups within session files under
`openarm_act_project/data/demos_hdf5/<task>/demos.hdf5`.

### HDF5 schema per episode group

```text
/episode_<NNN>/
  observations/
    images/
      cam_main            uint8   [T, H, W, 3]
    qpos                  float32 [T, J]
  actions/
    joint_target          float32 [T, J]
  meta/
    success               bool scalar
    task_name             bytes scalar
```

### Invariants

- Camera names are fixed across all episodes.
- Joint ordering is fixed across collection, training, and deployment.
- Control frequency (``step_hz``) is constant for all recordings.
- Only episodes with ``meta/success = True`` are used for training.

---

## 5. Step A — Collect Demonstrations

### Select a task

Unimanual reach is the recommended starting task.  Edit
`configs/act_openarm_reach.yaml` to adjust the number of episodes and
collection parameters.

### Run the collection script

```bash
cd /workspace/openarm_act_project

python scripts/collect_openarm_demos.py \
    --config configs/act_openarm_reach.yaml
```

Keyboard controls during collection (default device):

| Key | Action |
|---|---|
| `W / S` | EE forward / backward |
| `A / D` | EE left / right |
| `Q / E` | EE up / down |
| `Q` (hold) | Quit collection |

Successful episodes are written to
`data/demos_hdf5/openarm_reach/demos.hdf5`.

### Verify collected data

```python
from openarm_act.utils.io_utils import count_episodes
print(count_episodes("data/demos_hdf5/openarm_reach", success_only=True))
```

---

## 6. Step B — Train ACT Policy

```bash
python scripts/train_act_openarm.py \
    --config configs/act_openarm_reach.yaml
```

The script will:

1. Load all successful episodes from `data/demos_hdf5/openarm_reach/`.
2. Compute per-joint normalisation statistics and save to
   `checkpoints/openarm_reach_act/norm_stats.pkl`.
3. Run the ACT training loop for `num_epochs` epochs.
4. Save periodic checkpoints to `checkpoints/openarm_reach_act/`.
5. Persist `policy_best.ckpt` whenever training loss improves.
6. Output a loss curve plot `checkpoints/openarm_reach_act/train_loss.png`.

### Resume training

```bash
python scripts/train_act_openarm.py \
    --config configs/act_openarm_reach.yaml \
    --resume
```

### Key hyperparameters (in config)

| Parameter | Default | Description |
|---|---|---|
| `model.chunk_size` | 100 | Action chunk length (k) |
| `model.kl_weight` | 10 | KL divergence weight (β) |
| `training.num_epochs` | 2000 | Total training epochs |
| `training.batch_size` | 8 | Batch size |
| `training.lr` | 1e-5 | Learning rate |

---

## 7. Step C — Evaluate in Simulation

Run deterministic evaluation rollouts on a fixed random seed:

```bash
python scripts/eval_act_openarm.py \
    --config configs/act_openarm_reach.yaml \
    --checkpoint checkpoints/openarm_reach_act/
```

The script will:

1. Load `policy_best.ckpt` from the checkpoint directory.
2. Run `evaluation.num_rollouts` episodes from a fixed seed.
3. Log success/failure and episode length per rollout.
4. Save rollout plots to `checkpoints/openarm_reach_act/eval_results/`.
5. Write `eval_results/metrics.json` with aggregate statistics.

### Metrics

```json
{
  "success_rate": 0.84,
  "mean_episode_length": 312.5,
  "num_rollouts": 50,
  "checkpoint": "...",
  "task": "OpenArmUnimanualReach-v0",
  "seed": 42
}
```

### Selection criterion

Use `policy_best.ckpt` (best training loss) as the default.  Promote a
different checkpoint as `policy_best.ckpt` if it achieves higher evaluation
success rate.

---

## 8. Step D — Deploy in Isaac Sim

Run the policy in a live viewer session for qualitative inspection:

```bash
python scripts/deploy_act_openarm.py \
    --config configs/act_openarm_reach.yaml \
    --checkpoint checkpoints/openarm_reach_act/ \
    --num_episodes 10
```

To control the real-time stepping rate:

```bash
python scripts/deploy_act_openarm.py \
    --config configs/act_openarm_reach.yaml \
    --checkpoint checkpoints/openarm_reach_act/ \
    --step_hz 20
```

---

## 9. Configuration Reference

All four lifecycle scripts share a common YAML config.  The structure is:

```yaml
task: <gymnasium task ID>
task_name: <human-readable name>

collection:
  num_episodes: 200       # target number of successful demos
  dataset_dir: data/demos_hdf5/openarm_reach
  step_hz: 30
  teleop_device: keyboard
  num_success_steps: 10

model:
  policy_class: ACT
  chunk_size: 100
  kl_weight: 10
  hidden_dim: 512
  dim_feedforward: 3200
  nheads: 8
  num_encoder_layers: 4
  num_queries: 100        # must equal chunk_size
  backbone: resnet18

observation:
  camera_names: [cam_main]
  image_height: 480
  image_width: 640
  num_joints: 6

training:
  seed: 42
  num_epochs: 2000
  batch_size: 8
  lr: 1.0e-5
  weight_decay: 1.0e-4
  lr_backbone: 1.0e-5
  checkpoint_dir: checkpoints/openarm_reach_act
  save_every: 100
  eval_every: 500
  num_workers: 4

evaluation:
  num_rollouts: 50
  max_timesteps: 400
  temporal_agg: false
  onscreen_render: false
```

---

## 10. Available Tasks

| Task ID | Config | Type | Recommended demos |
|---|---|---|---|
| `OpenArmUnimanualReach-v0` | `act_openarm_reach.yaml` | Unimanual | 200 |
| `OpenArmUnimanualLift-v0` | `act_openarm_lift.yaml` | Unimanual | 300 |
| `OpenArmUnimanualCabinet-v0` | — | Unimanual | 300 |
| `OpenArmBimanualReach-v0` | — | Bimanual | 400 |

RL baseline training scripts for these tasks are in
`openarm_isaac_lab/scripts/reinforcement_learning/` (rl_games, skrl, rsl_rl).

---

## 11. Project Structure

```text
openarm_act_project/
  configs/
    act_openarm_reach.yaml
    act_openarm_lift.yaml
  data/
    demos_hdf5/
      openarm_reach/
        demos.hdf5
  checkpoints/
    openarm_reach_act/
      policy_best.ckpt
      norm_stats.pkl
      train_loss.png
      eval_results/
        metrics.json
  scripts/
    collect_openarm_demos.py   ← Step A
    train_act_openarm.py       ← Step B
    eval_act_openarm.py        ← Step C
    deploy_act_openarm.py      ← Step D
  src/openarm_act/
    __init__.py
    datasets/
      openarm_isaac_dataset.py
    envs/
      openarm_isaac_env.py
    policies/
      act_openarm_policy.py
    utils/
      io_utils.py
      viz_utils.py
  setup.py
  README.md
```

---

## 12. Troubleshooting

### "No successful episodes found"

Ensure `collect_openarm_demos.py` ran to completion and the HDF5 file
contains groups with `meta/success = True`.  Run:

```python
from openarm_act.utils.io_utils import count_episodes
print(count_episodes("data/demos_hdf5/openarm_reach"))
```

### "No checkpoint found"

Run `train_act_openarm.py` first.  Check that `checkpoint_dir` in the config
matches the actual directory path.

### Isaac Sim fails to start

Ensure the Isaac Sim environment is provisioned (e.g. via NVIDIA Brev or a
local GPU workstation) and that `ISAACSIM_PATH` is configured.  Run:

```bash
/workspace/isaaclab/isaaclab.sh -p --version
```

### Import errors for `openarm.tasks`

Reinstall the extension:

```bash
cd /workspace/isaaclab
python -m pip install -e /workspace/openarm_isaac_lab/source/openarm
```

### Import errors for `policy` (ACT)

Confirm that `/workspace/act` is in `PYTHONPATH` or that the scripts are run
from the workspace root so that `sys.path` is resolved correctly.
