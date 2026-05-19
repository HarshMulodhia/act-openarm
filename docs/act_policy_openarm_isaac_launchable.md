# ACT Policy on OpenArm in Isaac Sim (Isaac Launchable on NVIDIA Brev)

## 1) Goal

Train and deploy an ACT (Action Chunking Transformer) policy for OpenArm tasks in Isaac Sim/Isaac Lab using cloud infrastructure (Isaac Launchable on NVIDIA Brev), while keeping upstream repositories clean and using Hugging Face for dataset/model versioning.

## 2) Upstream Repositories and Responsibilities

- [OpenArm](https://github.com/enactic/openarm)
  - Hardware/software hub and references for OpenArm ecosystem.
- [ACT](https://github.com/tonyzhaozh/act)
  - Core ACT model, policy wrappers, and training utilities.
- [Isaac Launchable](https://github.com/isaac-sim/isaac-launchable)
  - Brev-friendly cloud environment for Isaac Sim + Isaac Lab.

Use these as upstream dependencies; keep custom integration in a separate project layer.

## 3) Final Code Structure (separate integration layer)

```text
~/workspace/
  isaac-launchable/                  # launchable runtime + containers (upstream)
  isaac-lab/                         # Isaac Lab framework (from launchable/upstream)
  openarm/                           # OpenArm upstream repo
  act/                               # ACT upstream repo
  openarm_act_project/               # your owned integration project
    README.md
    pyproject.toml
    configs/
      act_openarm_reach.yaml
      act_openarm_lift.yaml
    data/
      demos_hdf5/
    checkpoints/
      openarm_act/
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
        hdf5_utils.py
        metrics.py
        viz.py
```

### Why this structure is optimized

1. **Isolation**: no local forks of upstream repos required.
2. **Upgrade safety**: upstream pull/update remains straightforward.
3. **Reproducibility**: configs/data/checkpoints/scripts are colocated.
4. **Operational clarity**: one script per lifecycle phase.

## 4) Brev + Isaac Launchable Setup

1. Create/deploy Isaac Launchable in NVIDIA Brev.
2. Open VS Code endpoint and verify Isaac components are present.
3. In the cloud workspace:

```bash
cd ~/workspace
git clone https://github.com/enactic/openarm.git
git clone https://github.com/tonyzhaozh/act.git
# Isaac Launchable/Isaac Lab are already provided by the launchable image.
mkdir -p openarm_act_project
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

- Use OpenArm task environment in Isaac Sim/Isaac Lab.
- Record images + `qpos` + applied/target action each timestep.
- Save successful episodes into `data/demos_hdf5/`.

### Step B — Train ACT

- Use ACT repo as model/training dependency.
- Use custom dataset adapter to map Isaac HDF5 to ACT inputs.
- Train from config (`configs/*.yaml`) and save checkpoints to `checkpoints/openarm_act/`.

### Step C — Evaluate in simulation

- Run deterministic eval rollouts on hold-out seeds.
- Log task success rate, trajectory smoothness, and episode length.
- Keep the best checkpoint based on task success, not only train loss.

### Step D — Deploy policy in Isaac Sim

- Load best ACT checkpoint.
- Use chunked inference loop to emit joint targets.
- Run viewer-enabled validation for qualitative behavior checks.

## 7) Optimization Practices

1. **Demonstration quality first**
   - Expert consistency matters more than raw dataset volume.
2. **Temporal consistency**
   - Keep fixed `dt`, action semantics, and camera setup.
3. **Chunk size tuning**
   - Larger chunks improve throughput; smaller chunks improve responsiveness.
4. **Curriculum progression**
   - Train on simpler tasks first (reach) before harder tasks (lift/manipulation).
5. **Checkpoint strategy**
   - Save frequent checkpoints + retain top-k by evaluation success.
6. **Cloud efficiency**
   - Run collection/evaluation with viewer when debugging; headless for bulk jobs.

## 8) Hugging Face Integration (recommended)

Use Hugging Face for provenance and portability.

### Dataset
- Publish cleaned demonstration snapshots to a HF dataset repo.
- Include schema documentation and collection metadata.

### Model
- Publish selected ACT checkpoints to a HF model repo.
- Add a model card with:
  - task and simulator version
  - dataset reference/version
  - hyperparameters
  - evaluation metrics and limitations

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
- Final project structure separates upstream dependencies from owned integration code.
