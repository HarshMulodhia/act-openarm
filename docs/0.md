# End-to-End ACT Policy Training and Deployment for OpenArm in Isaac Sim via NVIDIA Brev Launchable

## Overview

This report describes a practical, end-to-end pipeline for training and deploying the ACT (Action Chunking Transformer) policy on the OpenArm robot inside NVIDIA Isaac Sim / Isaac Lab, using the official Isaac Launchable template on NVIDIA Brev for cloud-based simulation and training. The guide assumes familiarity with Python and robot learning, but starts from a fresh Brev Launchable and new demonstration data collected in Isaac Lab rather than relying on existing datasets.[^1][^2][^3][^4][^5]

The design goal is a clean project structure that composes three main components: (1) the OpenArm software stack, including the OpenArm Isaac Lab environments, (2) the original ACT implementation from the `tonyzhaozh/act` repository, and (3) the `isaac-sim/isaac-launchable` repository as the base for a Brev Launchable that exposes VS Code and Isaac Sim via browser streaming. The resulting workspace supports recording demonstrations in sim, training an ACT policy on the collected data, and deploying the trained policy back into Isaac Sim for control of OpenArm.[^6][^4][^1]

## Key Repositories and Roles

### OpenArm core repository

The main OpenArm repository provides the meta project, documentation, and links to sub-repositories for hardware, description files, ROS 2 integration, teleoperation, and Isaac Lab environments. It does not itself contain simulation code, but acts as the hub pointing to:[^1]

- `openarm_description`: URDF/xacro description files suitable for simulation and integration.[^1]
- `openarm_ros2`: ROS 2 packages and launch files for hardware and simulated control.[^7]
- `openarm_isaac_lab`: Isaac Lab simulation environments and training tasks for OpenArm.[^8][^1]

### OpenArm Isaac Lab environments

The OpenArm Isaac Lab tasks are distributed in a separate repo (commonly named `openarm_isaaclab` or similar) that is meant to live alongside a cloned copy of Isaac Lab. The typical installation pattern is:[^9][^8]

- Clone Isaac Lab.
- Clone the OpenArm Isaac Lab extension repo into the same parent directory.
- Install the extension in editable mode under `exts/openarm_isaaclab` so that it registers custom OpenArm tasks such as `Isaac-Reach-OpenArm-v0`.[^9]

Once installed and imported, these tasks become available to Isaac Lab training scripts (for example RSL-RL PPO training, or later imitation learning).

### ACT policy repository

The `tonyzhaozh/act` repository implements the ACT policy, including training and evaluation code and example MuJoCo-based environments. Its structure is:[^10][^6]

- `imitate_episodes.py`: main training and evaluation script for ACT.
- `policy.py`: ACT policy adaptor class that wraps the underlying transformer-based model.
- `detr/`: model definitions (transformer decoder layers, attention blocks) adapted from DETR.
- `sim_env.py`, `ee_sim_env.py`: MuJoCo + DM Control environments for joint-space and end-effector control.
- `scripted_policy.py`: scripted policies used to generate data for the example tasks.
- `utils.py`, `constants.py`: utilities and configuration constants for data loading, logging, etc.[^6]

For the OpenArm Isaac Lab project, ACT will be used as a library: the core model and training scripts remain, but the data loading and environment interfaces must be adapted to consume Isaac Lab–generated demonstration files instead of the default ALOHA/gym-aloha datasets.

### Isaac Launchable repository

The `isaac-sim/isaac-launchable` repo provides a Brev-ready template that spins up a VS Code Server, Isaac Lab, Isaac Sim, and a web-based viewer in a single environment using Docker and NVIDIA Omniverse Kit App Streaming. The Launchable includes:[^4][^11]

- A VS Code container for development.
- Isaac Lab pre-installed in a subdirectory (`isaaclab/` or `isaac-lab/`).
- Isaac Sim pre-installed and configured for livestreaming.
- An Omniverse Kit App Streaming client exposed via a `/viewer` URL.[^4]

The README documents both a quickstart flow with a preconfigured Launchable and a “custom Launchable” flow where Brev clones this repo and runs `docker compose up -d` to start the services.[^4]

## High-Level Pipeline

The full end-to-end ACT-on-OpenArm pipeline in Isaac Lab comprises the following stages:

1. **Provision cloud environment**: Use `isaac-sim/isaac-launchable` on Brev to create a Launchable with Isaac Lab and Isaac Sim accessible from the browser.[^4]
2. **Clone and wire repositories**: Inside the VS Code container in the Launchable, clone Isaac Lab (if not already present), the OpenArm Isaac Lab extension repo, the main OpenArm repo, and the ACT repo; install the OpenArm Isaac Lab extension as an Isaac Lab plugin.[^9][^1]
3. **Configure OpenArm Isaac Lab task**: Select or define an OpenArm manipulation task (for example a reaching or pick-and-place benchmark) for which ACT will be trained.[^8][^9]
4. **Collect demonstrations in sim**: Use Isaac Lab’s teleoperation or scripted controllers to record trajectory data of successful task executions, saving them to a dataset format amenable to ACT (for example HDF5 with image sequences, joint states, and actions).[^5][^12]
5. **Adapt ACT data loading**: Extend or wrap `imitate_episodes.py` to read Isaac Lab–generated demonstration files rather than the default ALOHA dataset structure.[^13][^6]
6. **Train ACT in the cloud**: Run ACT training on the Brev GPUs from within the VS Code terminal, monitoring loss and validation success using Isaac Lab simulation for evaluation rollouts.[^6][^4]
7. **Deploy policy back into Isaac Sim**: Export the trained ACT checkpoint; integrate it into an Isaac Lab policy wrapper that runs in an OpenArm task, feeding observations to ACT and sending predicted action chunks to the Isaac Lab articulation controller.[^14][^5]

The remainder of this report proposes a concrete project layout and the detailed steps for each stage.

## Provisioning Isaac Launchable on Brev

### Using the stock Isaac Launchable

The easiest path is to use NVIDIA’s prebuilt Isaac Launchable, which uses the `isaac-sim/isaac-launchable` repo as its template and provisions the following on Brev:

- A Visual Studio Code Server instance accessible via an HTTP port.
- Isaac Sim and Isaac Lab installed in a Dockerized environment.
- A web viewer at `/viewer` that streams the Isaac Sim UI via WebRTC.[^11][^4]

The general deployment steps are:

1. On the Brev website, navigate to the Launchables section and pick the Isaac Launchable that references the `isaac-sim/isaac-launchable` repository.[^11][^4]
2. Click **Deploy Launchable**, then wait until the instance is running and the setup script has completed.[^4]
3. On the instance page, use the TCP/UDP ports section to open the Visual Studio Code Server (typically port 80, HTTP) in a new browser tab.[^4]
4. Log into VS Code Server using the default password (`password` by default, which should be changed) and confirm that the repo files (`isaac-lab`, `docker-compose.yml`, etc.) are present.[^4]

### Creating a custom Launchable (optional)

If the stock Launchable does not exist, a custom Launchable can be created in Brev using the `isaac-sim/isaac-launchable` repo as a bootstrap:

1. Log in to Brev and click **Create Launchable** under the Launchables section.[^11][^4]
2. Select the option indicating there are no initial code files and choose **VM Mode – Basic VM with Python installed**.[^4]
3. Add the following setup script under the “Paste Script” tab:

```bash
#!/bin/bash
git clone https://github.com/isaac-sim/isaac-launchable
cd isaac-launchable/isaac-lab
docker compose up -d
```

4. Expose ports 80, 1024, 47998, and 49100 for VS Code and Kit App Streaming.[^4]
5. Choose an AWS-based GPU instance with RT cores compatible with Isaac Sim.[^4]
6. Finish the Launchable creation; once running, connect to the VS Code Server via port 80 and to the Isaac Sim viewer via `/viewer` on the same hostname.[^4]

With the Launchable running, the next steps are performed inside the VS Code terminal.

## Bootstrapping Isaac Lab + OpenArm + ACT in the Launchable

### Directory layout inside the Launchable

Inside the VS Code environment of the Launchable, the working directory typically contains an `isaac-lab` folder with the Isaac Lab source tree and Docker configuration. The recommended parent structure for integrating OpenArm and ACT is:[^5][^4]

```text
~/workspace/
  isaac-lab/                 # Isaac Lab (from Isaac Launchable)
  openarm/                   # main OpenArm meta-repo (docs, links)
  openarm_isaaclab/          # OpenArm Isaac Lab extension
  act/                       # ACT implementation
  openarm_act_project/       # your glue code / configs
```

This keeps upstream repos separate and places all project-specific scripts and configuration in `openarm_act_project` for clarity.

### Cloning the repositories

From a VS Code terminal inside the Launchable instance:

```bash
cd ~
mkdir -p workspace
cd workspace

# Isaac Lab is already present if using isaac-launchable; if not, clone:
# git clone https://github.com/isaac-sim/IsaacLab isaac-lab

# Main OpenArm meta-repo (for docs and reference)
git clone https://github.com/enactic/openarm.git

# OpenArm Isaac Lab extension (openarm_isaaclab naming may vary)
git clone https://github.com/reazon-research/openarm_isaaclab.git

# ACT implementation
git clone https://github.com/tonyzhaozh/act.git

# Project glue repo (optional; can also be created manually)
mkdir -p openarm_act_project
```

The `openarm_isaaclab` repo should then be installed as an Isaac Lab extension.

### Installing the OpenArm Isaac Lab extension

The OpenArm Isaac Lab extension repo is designed to be installed into Isaac Lab’s `exts` directory as an editable Python module so its tasks register when Isaac Lab runs.[^9]

Assuming Isaac Lab resides in `~/workspace/isaac-lab` and the OpenArm extension in `~/workspace/openarm_isaaclab`:

```bash
cd ~/workspace/openarm_isaaclab
python -m pip install -e exts/openarm_isaaclab
```

After installation, the directory tree should look like:

```text
~/workspace/
  IsaacLab/          # or isaac-lab
  openarm_isaaclab/
    exts/
      openarm_isaaclab/  # installed extension
```

To ensure tasks are registered in standard RL scripts, some Isaac Lab tutorials suggest importing the extension in training and playback scripts, for example by adding:

```python
import openarm_isaaclab.tasks.manipulation
```

at the top of `train.py` and `play.py` in Isaac Lab’s reinforcement learning scripts. For ACT-based imitation learning, the OpenArm tasks will instead be invoked through custom script entry points that use the Isaac Lab Python API.[^12][^9]

## Running Isaac Sim and Isaac Lab in the Launchable

### Isaac Sim viewer

To run Isaac Sim itself in the Launchable using the viewer for interactive debugging or teleoperation visualization, the `isaac-launchable` README recommends using:

```bash
./isaaclab/_isaac_sim/isaac-sim.sh --no-window --enable omni.kit.livestream.webrtc
```

This launches Isaac Sim in headless mode with livestreaming enabled, and the UI can then be accessed at the `/viewer` path on the same host as the VS Code Server.[^4]

### Isaac Lab script execution

For command-line training or data collection tasks that do not require a viewport, Isaac Lab scripts can be run normally. For example, an existing RL example uses:

```bash
python isaaclab/scripts/reinforcement_learning/skrl/train.py \
  --task=Isaac-Ant-v0 --headless
```

to run a headless training job, and a corresponding `play.py` command for evaluation uses `--kit_args="--no-window --enable omni.kit.livestream.webrtc"` to enable the viewer.[^12][^4]

For ACT demonstrations, a similar pattern can be used: headless runs for bulk data collection and optional viewer-enabled runs for debugging.

## Designing the OpenArm ACT Project Structure

### Top-level layout

A practical monorepo structure tying together Isaac Lab, OpenArm, and ACT inside the Launchable looks like this:

```text
~/workspace/
  isaac-lab/                     # Isaac Lab from Launchable
  openarm/                       # upstream OpenArm meta-repo
  openarm_isaaclab/              # OpenArm Isaac Lab tasks
  act/                           # ACT implementation
  openarm_act_project/           # project-specific code
    configs/
      act_openarm_reach.yaml     # task + model hyperparameters
    data/
      demos_hdf5/                # collected demos from Isaac Lab
    scripts/
      collect_openarm_demos.py   # teleop / scripted collection in Isaac Lab
      train_act_openarm.py       # ACT training entrypoint
      eval_act_openarm.py        # evaluation in Isaac Lab
      deploy_act_openarm.py      # deployment policy wrapper for Isaac Lab
    src/
      openarm_act/
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
```

This layout keeps upstream repos untouched while providing explicit locations for:

- Isaac Lab/OpenArm environment wrappers (`envs/`).
- Dataset readers compatible with ACT’s expectations (`datasets/`).
- Policy glue code that embeds ACT inside an Isaac Lab control loop (`policies/`).
- Scripts for data collection, training, evaluation, and deployment (`scripts/`).

## Data Collection in Isaac Lab for ACT

### Required observation and action modalities

ACT, as implemented in the original repository, is designed to work with sequences of visual observations (camera images) plus proprioceptive information (joint positions and possibly velocities), and to predict chunks of future actions that are typically joint position targets. For an OpenArm manipulation task in Isaac Lab, the dataset should contain, at minimum:[^13][^6]

- RGB image frames from one or more Isaac Sim cameras (for example, a wrist camera and a scene camera).
- Joint positions (and optionally velocities) for each OpenArm joint.
- Action sequences corresponding to joint position commands applied by the controller.

For each demonstration episode, data can be stored in an HDF5 file or RLDS-style directory structure similar to existing ACT and LeRobot datasets.[^15][^13]

### Collecting demonstrations with Isaac Lab

An Isaac Lab task for OpenArm (for example `Isaac-Reach-OpenArm-v0`) can be used to collect demonstrations via teleoperation or scripted controllers. A typical collection script inside `openarm_act_project/scripts/collect_openarm_demos.py` would:[^8][^9]

1. Instantiate the OpenArm task environment via Isaac Lab’s Python API, ensuring the extension is loaded.
2. Create cameras attached to the environment (e.g., wrist camera, overhead camera) and configure image capture at a fixed frequency.
3. Accept teleoperation inputs (e.g., joystick/keyboard or direct joint space commands) or use a scripted policy to produce control actions.
4. Log, at each timestep, the images, joint states, and applied actions into an in-memory buffer.
5. On episode completion, write the buffers to an HDF5 file in `data/demos_hdf5/` following a standard ACT-like schema such as:

```text
/episode_000/
  observations/images/cam_main: uint8 [T, H, W, 3]
  observations/qpos: float32 [T, N_joints]
  actions/qpos_target: float32 [T, N_joints]
  meta/success: bool
```

This schema mirrors the structure used in ACT’s ALOHA datasets and LeRobot’s ACT model card examples, which store observations and action arrays at each time step.[^16][^17][^6]

### Using viewer-enabled collection

For debugging and precise teleoperation, Isaac Lab runs can be executed with viewer streaming by passing `--kit_args="--no-window --enable omni.kit.livestream.webrtc"` to the Isaac Lab launch command, then using the `/viewer` tab to see the scene and teleoperate OpenArm. Once the data collection logic is stable, headless runs can batch-collect demonstrations.[^4]

## Integrating Isaac Lab Data with the ACT Codebase

### Adapting ACT’s dataset loader

The ACT repository’s `imitate_episodes.py` script expects datasets in a specific HDF5 layout and uses `utils.py` and `constants.py` to locate and parse them. To avoid forking ACT extensively, a thin adapter can be written in `openarm_act/datasets/openarm_isaac_dataset.py` that:[^6]

- Wraps the Isaac Lab dataset schema in the interface expected by ACT (for example, returning episode trajectories as arrays of images, joint states, and actions).
- Implements any required normalization, scaling, and masking logic (for example cropping images, normalizing joint angles, padding sequences).

One approach is to write a new CLI entry point `train_act_openarm.py` that:

- Imports ACT’s `imitate_episodes` functions or classes.
- Overrides the dataset loading logic to point to `data/demos_hdf5/` with OpenArm-specific configuration.
- Uses an ACT configuration file in `configs/act_openarm_reach.yaml` to specify observation modalities, joint counts, and chunk sizes.

### Example training entry point snippet

Within `openarm_act_project/scripts/train_act_openarm.py`, the main routine could:

```python
from openarm_act.datasets.openarm_isaac_dataset import OpenArmIsaacDataset
from act.imitate_episodes import train_act

if __name__ == "__main__":
    dataset = OpenArmIsaacDataset(
        root_dir="../data/demos_hdf5",
        camera_key="cam_main",
        qpos_key="qpos",
        action_key="qpos_target",
    )

    train_act(
        dataset=dataset,
        ckpt_dir="../checkpoints/openarm_reach_act",
        policy_class="ACT",
        kl_weight=10,
        chunk_size=100,
        hidden_dim=512,
        batch_size=8,
        dim_feedforward=3200,
        num_epochs=2000,
        lr=1e-5,
        seed=0,
    )
```

This pseudocode mirrors the ACT examples but swaps in `OpenArmIsaacDataset` as the data source.[^6]

### Using Hugging Face for datasets and checkpoints

Hugging Face is commonly used to publish ACT-based datasets and checkpoints for simulation and real-world tasks (for example, the `lerobot/act_aloha_sim_insertion_human` model and its corresponding dataset). For the OpenArm project, Hugging Face can be used in two ways:[^17][^16]

- Upload collected OpenArm Isaac Lab demonstrations as a dataset repo, allowing sharing and reuse.
- Publish trained ACT checkpoints as a model repo with a model card describing the task, hyperparameters, and evaluation setup, similar to the LeRobot examples.[^16][^15]

Integrating Hugging Face into the training script is optional but straightforward using the `huggingface_hub` Python library.

## Training ACT in the Brev Launchable

### Environment and dependencies

The ACT repository requires a Python 3.8 environment with PyTorch, MuJoCo, DM Control, and other scientific Python libraries installed, as detailed in the README. Because Isaac Lab and Isaac Sim also have specific Python dependency requirements, the Launchable’s environment should be carefully managed.[^6]

The simplest approach is:

- Use the Python environment provided by Isaac Lab in the Launchable, installing ACT’s additional Python dependencies into that environment if compatible.[^18][^5]
- Alternatively, create a separate conda environment for ACT inside the VS Code container and ensure it can still import Isaac Lab and OpenArm Isaac Lab APIs; this may require path configuration and careful CUDA compatibility checks.

The ACT install from the repo is then performed as:

```bash
cd ~/workspace/act
# Assuming the environment has PyTorch and a compatible Python version
pip install torchvision torch pyquaternion pyyaml rospkg pexpect \
            mujoco==2.3.7 dm_control==1.0.14 opencv-python \
            matplotlib einops packaging h5py ipython
cd detr && pip install -e .
```

as suggested by the ACT README, adjusting where necessary for the existing Launchable environment.[^6]

### Launching ACT training jobs

With the dataset ready and the `train_act_openarm.py` script in place, training can be initiated from within VS Code’s integrated terminal:

```bash
cd ~/workspace/openarm_act_project/scripts
python train_act_openarm.py \
  --config ../configs/act_openarm_reach.yaml \
  --ckpt_dir ../checkpoints/openarm_reach_act
```

The config file can control all hyperparameters and dataset paths, and the training script can log metrics and save checkpoints periodically. If GPU utilization is a concern, Brev’s larger GPU instances can be used by selecting a higher-capacity Launchable configuration.[^11][^4]

ACT-specific tuning tips from the authors—such as continuing training beyond apparent loss plateaus to improve smoothness and success rates—should be followed, especially for visually complex manipulation tasks.[^13][^6]

## Deploying ACT Policies in Isaac Sim with OpenArm

### Policy wrapper in Isaac Lab

To run a trained ACT checkpoint inside Isaac Lab, a wrapper policy can be implemented in `openarm_act/policies/act_openarm_policy.py` that:

- Loads the trained ACT weights from the checkpoint directory.
- Exposes a `compute_action(observation_sequence)` method that takes a recent window of image and joint observations and returns a chunk of future joint target positions.
- Manages internal replay of partial chunks so the environment can query the policy at every Isaac Lab simulation step.

The wrapper can use ACT’s `policy.py` adaptor class internally, but the observation acquisition and action setting are delegated to Isaac Lab’s environment API, accessing the OpenArm articulation handles or control interfaces.

### Integration into an Isaac Lab script

A deployment script `deploy_act_openarm.py` can:

1. Instantiate the OpenArm task environment in Isaac Lab.
2. Initialize the `ACTOpenArmPolicy` with the checkpoint path.
3. At each simulation step:
   - Capture the latest observations (images, joint positions).
   - Feed them into the policy if a new chunk is needed, or step through the current chunk.
   - Apply the resulting joint targets to the OpenArm articulation using Isaac Lab’s control API.
4. Optionally stream the Isaac Sim viewport via the `/viewer` URL to visually monitor policy behavior.[^5][^4]

For convenience, this deployment script can be run with `--kit_args` to enable viewer streaming when needed, and run headless when performing large-scale evaluation.

## Example Code Structure Summary

The final project layout goal is to clearly separate upstream repos from custom glue code while making the ACT-on-OpenArm workflow reproducible from a single Launchable environment:

| Component | Location | Role |
|----------|----------|------|
| Isaac Launchable | `isaac-sim/isaac-launchable` (Brev template) | Provisions VS Code, Isaac Lab, Isaac Sim, and viewer in cloud via Docker.[^4][^11] |
| Isaac Lab | `~/workspace/isaac-lab` | Core robot learning framework and simulation engine.[^5][^19] |
| OpenArm meta repo | `~/workspace/openarm` | Documentation and links for hardware, description, ROS 2, teleop, Isaac Lab environments.[^1] |
| OpenArm Isaac Lab extension | `~/workspace/openarm_isaaclab` | Defines OpenArm tasks and gym environments for Isaac Lab, installed as an extension.[^9][^8] |
| ACT implementation | `~/workspace/act` | Transformer-based Action Chunking policy training and evaluation code.[^6][^10] |
| Project glue code | `~/workspace/openarm_act_project` | Dataset adapters, OpenArm Isaac Lab env wrappers, ACT integration scripts, configs. |

Inside `openarm_act_project`, the primary directories and scripts are:

- `configs/`: YAML config files specifying ACT hyperparameters and dataset paths.
- `data/demos_hdf5/`: demonstration episodes recorded from OpenArm Isaac Lab tasks.
- `src/openarm_act/datasets/`: dataset adapters for reading Isaac Lab demos into ACT’s training pipeline.
- `src/openarm_act/envs/`: environment helpers for interacting with OpenArm tasks in Isaac Lab.
- `src/openarm_act/policies/`: ACT policy wrappers for Isaac Lab deployment.
- `src/openarm_act/utils/`: IO and visualization utilities.
- `scripts/collect_openarm_demos.py`: demonstration collection entry point.
- `scripts/train_act_openarm.py`: ACT training entry point.
- `scripts/eval_act_openarm.py`: evaluation script.
- `scripts/deploy_act_openarm.py`: deployment script for running trained policies in real-time.

This structure enables a clean, modular workflow: collect demonstrations in Isaac Lab using OpenArm environments, train ACT policies on the collected data in the Brev-based Launchable, and deploy them back into Isaac Sim for evaluation and later sim-to-real transfer to physical OpenArm hardware via ROS 2 integration.[^20][^7][^1]

---

## References

1. [GitHub - enactic/openarm: A fully open-source humanoid arm for ...](https://github.com/enactic/openarm) - OpenArm is an open-source 7DOF humanoid arm designed for physical AI research and deployment in cont...

2. [Getting Started with NVIDIA Isaac Sim & Isaac Lab on Brev - YouTube](https://www.youtube.com/watch?v=ta-1RJzcDNs) - ... Isaac Lab, powerful frameworks for robotics simulation, training, and learning on NVIDIA Omniver...

3. [How to set up NVIDIA Robotics Isaac Lab with Isaac Sim ... - YouTube](https://www.youtube.com/watch?v=0FIYWtu6ipw) - ... Isaac Sim or Isaac Lab locally. You launch a cloud environment and start working straight away. ...

4. [GitHub - isaac-sim/isaac-launchable: This repo is used to create a ...](https://github.com/isaac-sim/isaac-launchable) - This repo is used to create a pre-configured Isaac Sim and Isaac Lab environment using the NVIDIA Br...

5. [isaac-sim/IsaacLab: Unified framework for robot learning ... - GitHub](https://github.com/isaac-sim/IsaacLab) - Isaac Lab is a GPU-accelerated, open-source framework designed to unify and simplify robotics resear...

6. [tonyzhaozh/act - GitHub](https://github.com/tonyzhaozh/act) - Contribute to tonyzhaozh/act development by creating an account on GitHub.

7. [enactic/openarm_ros2: ROS2 packages for OpenArm - GitHub](https://github.com/enactic/openarm_ros2) - ROS 2 (Robot Operating System 2) is a modern, open-source framework for building robotic software. h...

8. [发行说明 — Isaac Lab 文档](https://docs.robotsfan.com/isaaclab/source/refs/release_notes.html) - Updates docs on Isaac Sim binary installation path and VSCode integration. Removes remaining depreca...

9. [GitHub - enactic/openarm_isaaclab_experiment](https://github.com/reazon-research/openarm-isaaclab) - These scripts are used for training reinforcement learning models and replaying trained models. Howe...

10. [Action Chunking Transformer (ACT) policies and ALOHA - Zenn](https://zenn.dev/yuto_mo/articles/362c49e99f0395) - The tonyzhaozh/act GitHub repository provides code to train and evaluate ACT on both simulated (gym-...

11. [Docker, Brev, windows advice - Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/docker-brev-windows-advice/346972) - You can also run on Brev. Take a look at GitHub - isaac-sim/isaac-launchable: This repo is used to c...

12. [Train a Humanoid Locomotion Policy with Isaac Lab on DGX Spark](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_isaac_robotics/4_isaac_rfl/) - By the end of this section you'll understand the key stages of the RL training pipeline, including: ...

13. [ACT Policy Explained (2026): Action Chunking with Transformers](https://www.roboticscenter.ai/blog/act-policy-explained) - ACT is an imitation learning algorithm designed for fine-grained manipulation tasks where the robot ...

14. [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/lab) - NVIDIA Isaac Lab is an open-source modular framework for robot learning designed to simplify how rob...

15. [nvidia/ALOHA-Cosmos-Policy · Datasets at Hugging Face](https://huggingface.co/datasets/nvidia/ALOHA-Cosmos-Policy) - ALOHA-Cosmos-Policy is a real-world bimanual manipulation dataset collected on the ALOHA 2 robot pla...

16. [lerobot/act_aloha_sim_insertion_human - Hugging Face](https://huggingface.co/lerobot/act_aloha_sim_insertion_human) - The model was evaluated on the AlohaInsertion task from gym-aloha and compared to a similar model tr...

17. [lerobot/act_aloha_sim_transfer_cube_human - Hugging Face](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) - The model was evaluated on the AlohaTransferCube task from gym-aloha and compared to a similar model...

18. [How to Install Isaac Lab on Ubuntu 22.04 - Robotics Center](https://www.roboticscenter.ai/tutorials/install-isaac-lab/) - This tutorial walks through a clean install on Ubuntu 22.04 with Isaac Sim 4.x, a dedicated conda en...

19. [Welcome to Isaac Lab!](https://isaac-sim.github.io/IsaacLab/) - Isaac Lab is a unified and modular framework for robot learning that aims to simplify common workflo...

20. [The Enactic OpenArm running in Nvidia ISaac Sim 5.1 ... - YouTube](https://www.youtube.com/watch?v=UBeMtl21JOg) - OpenArm Bimanual Teleop in Isaac Sim/Lab + ROS 2 (RViz, Webcam & Keyboard Control) In this video I b...

