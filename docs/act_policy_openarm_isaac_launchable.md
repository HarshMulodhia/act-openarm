# ACT Policy on OpenArm in Isaac Sim

## 1) Goal

Train and deploy an ACT (Action Chunking Transformer) policy for OpenArm tasks in Isaac Sim/Isaac Lab, while keeping upstream repositories clean.

## 2) Upstream Repositories and Responsibilities

- [OpenArm](https://github.com/enactic/openarm)
  - Hardware/software hub and references for OpenArm ecosystem.
- [openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab)
  - Isaac Lab extension with OpenArm task environments (reach, lift, cabinet, bimanual reach) and RL training scripts.
- [ACT](https://github.com/tonyzhaozh/act)
  - Core ACT model, policy wrappers, and training utilities.
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
  - Isaac Lab framework for robot learning in Isaac Sim.

Use these as upstream dependencies; keep custom integration in `openarm_act_project/`.

## 3) Actual Workspace Structure

```text
/workspace/
  act/                               # ACT upstream repo
    conda_env.yaml
    policy.py
    imitate_episodes.py
    utils.py
    constants.py
    detr/                            # DETR backbone used by ACT
    assets/                          # MuJoCo assets (viperx reference)
  isaaclab/                          # Isaac Lab framework (upstream)
    source/
    docker/
    tools/
  openarm/                           # OpenArm hardware/software meta repo (upstream)
    website/                         # Docusaurus documentation site
  openarm_act_project/               # Owned integration project
    configs/
      act_openarm_reach.yaml
    data/
      demos_hdf5/
    checkpoints/
      openarm_reach_act/
    scripts/
      collect_openarm_demos.py
      train_act_openarm.py
      eval_act_openarm.py
      deploy_act_openarm.py
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
  openarm_isaac_lab/                 # OpenArm Isaac Lab extension (upstream)
    source/openarm/
      openarm/tasks/manager_based/openarm_manipulation/
        unimanual/
          reach/
          lift/
          cabinet/
        bimanual/
          reach/
        assets/
        usds/
    scripts/reinforcement_learning/  # RL training (rl_games, skrl, rsl_rl)
    video/                           # Demo GIFs
  tree.log
```

### Why this structure is optimized

1. **Isolation**: no local forks of upstream repos required.
2. **Upgrade safety**: upstream pull/update remains straightforward.
3. **Reproducibility**: configs/data/checkpoints/scripts are colocated in `openarm_act_project/`.
4. **Operational clarity**: one script per lifecycle phase.

## 4) Environment Setup

1. Start an Isaac Sim environment (NVIDIA Brev / local GPU workstation).
2. Open VS Code and verify Isaac Sim and Isaac Lab are accessible.
3. In the workspace:

```bash
cd /workspace
git clone https://github.com/enactic/openarm.git
git clone https://github.com/enactic/openarm_isaac_lab.git
git clone https://github.com/tonyzhaozh/act.git
# isaaclab/ is pre-installed or cloned separately from isaac-sim/IsaacLab
mkdir -p openarm_act_project
```

4. Install the OpenArm extension:

```bash
cd /workspace/isaaclab
python -m pip install -e /workspace/openarm_isaac_lab/source/openarm
```

5. Verify available task environments:

```bash
python /workspace/openarm_isaac_lab/scripts/tools/list_envs.py
```

## 5) Data Contract for ACT Training

ACT training quality depends heavily on clean demonstration schema consistency.

Recommended per-episode HDF5 shape:

```text
/episode_xxx/
  observations/images/cam_main      uint8   [T, H, W, 3]
  observations/qpos                 float32 [T, J]
  actions/joint_target              float32 [T, J]
  meta/success                      bool
  meta/task_name                    string
```

Rules:
- Fixed camera naming across all episodes.
- Fixed joint ordering across data collection/training/deployment.
- Constant control frequency for all recordings.
- Drop corrupt or partial episodes before training.

## 6) End-to-End Workflow

### Step A — Collect demonstrations (Isaac Lab)

- Use an OpenArm task environment (e.g., `OpenArmUnimanualReach-v0`).
- Record images + `qpos` + applied/target action each timestep.
- Save successful episodes into `openarm_act_project/data/demos_hdf5/`.

```bash
python /workspace/openarm_act_project/scripts/collect_openarm_demos.py \
  --config /workspace/openarm_act_project/configs/act_openarm_reach.yaml
```

### Step B — Train ACT

- Use ACT repo as model/training dependency.
- Use `src/openarm_act/datasets/openarm_isaac_dataset.py` to map Isaac HDF5 to ACT inputs.
- Train from config and save checkpoints to `checkpoints/openarm_reach_act/`.

```bash
python /workspace/openarm_act_project/scripts/train_act_openarm.py \
  --config /workspace/openarm_act_project/configs/act_openarm_reach.yaml
```

### Step C — Evaluate in simulation

- Run deterministic eval rollouts on hold-out seeds.
- Log task success rate, trajectory smoothness, and episode length.
- Keep the best checkpoint based on task success, not only train loss.

```bash
python /workspace/openarm_act_project/scripts/eval_act_openarm.py \
  --config /workspace/openarm_act_project/configs/act_openarm_reach.yaml \
  --checkpoint /workspace/openarm_act_project/checkpoints/openarm_reach_act/
```

### Step D — Deploy policy in Isaac Sim

- Load best ACT checkpoint.
- Use chunked inference loop to emit joint targets.
- Run viewer-enabled validation for qualitative behavior checks.

```bash
python /workspace/openarm_act_project/scripts/deploy_act_openarm.py \
  --config /workspace/openarm_act_project/configs/act_openarm_reach.yaml \
  --checkpoint /workspace/openarm_act_project/checkpoints/openarm_reach_act/
```

## 7) Available OpenArm Tasks

| Task ID | Type | Description |
|---|---|---|
| `OpenArmUnimanualReach-v0` | Unimanual | Reach to target position |
| `OpenArmUnimanualLift-v0` | Unimanual | Lift an object |
| `OpenArmUnimanualCabinet-v0` | Unimanual | Open cabinet drawer |
| `OpenArmBimanualReach-v0` | Bimanual | Bimanual coordinated reach |

RL baseline training scripts for these tasks are in `openarm_isaac_lab/scripts/reinforcement_learning/` (supports rl_games, skrl, rsl_rl).

## 8) Optimization Practices

1. **Demonstration quality first**
   - Expert consistency matters more than raw dataset volume.
2. **Temporal consistency**
   - Keep fixed `dt`, action semantics, and camera setup.
3. **Chunk size tuning**
   - Larger chunks improve throughput; smaller chunks improve responsiveness.
4. **Curriculum progression**
   - Train on simpler tasks first (reach) before harder tasks (lift/cabinet).
5. **Checkpoint strategy**
   - Save frequent checkpoints + retain top-k by evaluation success.
6. **Headless vs. viewer**
   - Run collection/evaluation with viewer when debugging; headless for bulk jobs.

## 9) Minimal Script Responsibilities

- `collect_openarm_demos.py`
  - creates env, captures observations/actions, writes HDF5 episodes.
- `train_act_openarm.py`
  - loads config, builds dataset/dataloader, trains ACT, writes checkpoints.
- `eval_act_openarm.py`
  - runs standardized rollouts and writes metrics.
- `deploy_act_openarm.py`
  - loads best checkpoint and runs live policy inference in Isaac Sim.

## 10) Acceptance Criteria

- Demonstrations are schema-consistent and versioned.
- Training is reproducible from config and dataset reference.
- Best checkpoint achieves stable task success in Isaac Sim evaluation.
- Final project structure separates upstream dependencies from `openarm_act_project/`.
