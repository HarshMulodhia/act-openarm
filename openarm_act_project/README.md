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
    collect_openarm_demos.py   # Step A — demo collection
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

```bash
python scripts/collect_openarm_demos.py \
  --config configs/act_openarm_reach.yaml
```

**3. Train**

```bash
python scripts/train_act_openarm.py \
  --config configs/act_openarm_reach.yaml
```

**4. Evaluate**

```bash
python scripts/eval_act_openarm.py \
  --config configs/act_openarm_reach.yaml \
  --checkpoint checkpoints/openarm_reach_act/
```

**5. Deploy**

```bash
python scripts/deploy_act_openarm.py \
  --config configs/act_openarm_reach.yaml \
  --checkpoint checkpoints/openarm_reach_act/
```

## Available Tasks

| Config | Task ID | Type |
|---|---|---|
| `act_openarm_reach.yaml` | `OpenArmUnimanualReach-v0` | Unimanual reach |
| `act_openarm_lift.yaml` | `OpenArmUnimanualLift-v0` | Unimanual lift |

See [`../docs/pipeline.md`](../docs/pipeline.md) for the full end-to-end guide.
