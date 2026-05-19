# openarm_act_project

ACT policy integration layer for OpenArm Isaac Lab tasks.

## Overview

This project contains all integration-specific code for training and deploying
an [ACT (Action Chunking Transformer)](https://github.com/tonyzhaozh/act) policy
on OpenArm manipulation tasks in [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
Upstream repositories are kept unmodified.

## Directory Layout

```text
openarm_act_project/
  configs/
    act_openarm_reach.yaml     # Reach task config
    act_openarm_lift.yaml      # Lift task config
  data/
    demos_hdf5/                # Collected demonstrations (HDF5, git-ignored)
  checkpoints/
    openarm_reach_act/         # Trained checkpoints (git-ignored)
    openarm_lift_act/
  scripts/
    export_openarm_rsl_rl_expert.py # RSL-RL checkpoint → policy.pt
    collect_openarm_demos.py   # Step A — demo collection
    validate_openarm_dataset.py # Dataset schema/shape validation
    augment_openarm_dataset.py  # Action-noise dataset generation
    train_act_openarm.py       # Step B — ACT training
    eval_act_openarm.py        # Step C — simulation evaluation
    deploy_act_openarm.py      # Step D — live deployment
  src/openarm_act/
    __init__.py
    datasets/
      openarm_isaac_dataset.py # Isaac HDF5 → ACT dataset adapter
    envs/
      openarm_isaac_env.py     # Isaac Lab environment wrapper
    policies/
      act_openarm_policy.py    # ACT policy inference wrapper
    utils/
      io_utils.py              # HDF5 and file I/O helpers
      viz_utils.py             # Visualisation helpers
  setup.py
```

## Quick Start

**1. Install the package**

```bash
cd /workspace/openarm_act_project
pip install -e .
```

**2. Collect demonstrations**

Export or provide an existing RSL-RL expert policy from `openarm_isaac_lab`,
then set `collection.expert_checkpoint` in the reach config to the exported
`policy.pt`.

```bash
/isaac-sim/python.sh scripts/export_openarm_rsl_rl_expert.py \
  --task Isaac-Reach-OpenArm-v0 \
  --checkpoint /path/to/rsl_rl_checkpoint.pt \
  --output checkpoints/openarm_reach_expert/policy.pt
```

```bash
/isaac-sim/python.sh scripts/collect_openarm_demos.py \
  --config configs/act_openarm_reach.yaml
```

**3. Validate the clean dataset**

```bash
/isaac-sim/python.sh scripts/validate_openarm_dataset.py \
  --config configs/act_openarm_reach.yaml
```

**4. Train the clean baseline**

```bash
/isaac-sim/python.sh scripts/train_act_openarm.py \
  --config configs/act_openarm_reach.yaml
```

**5. Create the action-noise dataset**

```bash
/isaac-sim/python.sh scripts/augment_openarm_dataset.py \
  --config configs/act_openarm_reach.yaml
```

For the clean+noise experiment, set `training.dataset_mode: clean_plus_noise`
and use a separate `training.checkpoint_dir`.

**6. Evaluate and render the 30-second video**

```bash
/isaac-sim/python.sh scripts/eval_act_openarm.py \
  --config configs/act_openarm_reach.yaml \
  --checkpoint checkpoints/openarm_reach_act/
```

**7. Deploy**

```bash
/isaac-sim/python.sh scripts/deploy_act_openarm.py \
  --config configs/act_openarm_reach.yaml \
  --checkpoint checkpoints/openarm_reach_act/
```

## Available Tasks

| Config | Task ID | Type |
|---|---|---|
| `act_openarm_reach.yaml` | `Isaac-Reach-OpenArm-v0` | Unimanual reach |
| `act_openarm_lift.yaml` | `Isaac-Lift-Cube-OpenArm-v0` | Unimanual lift |

See [`../docs/pipeline.md`](../docs/pipeline.md) for the full end-to-end guide.
