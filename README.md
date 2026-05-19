# act-openarm

This repository contains structured documentation for building and integrating an ACT-based policy workflow for OpenArm in Isaac Lab/Isaac Sim.

## Documentation

- `/docs/0.md` — Architecture and project structure
- `/docs/1.md` — Setup and execution workflow
- `/docs/2.md` — Refactor standards and implementation checklist
- `/docs/act_policy_openarm_isaac_launchable.md` — Detailed ACT training/deployment guide and final code structure
- `/docs/pipeline.md` — **Complete end-to-end pipeline guide** (setup → collect → train → eval → deploy)

## Workspace Layout

The actual workspace at `/workspace` contains:

```text
/workspace/
  act/                   # ACT upstream repository
  isaaclab/              # Isaac Lab framework (upstream)
  openarm/               # OpenArm hardware/software meta repository (upstream)
  openarm_act_project/   # ACT integration layer (owned project)
  openarm_isaac_lab/     # OpenArm Isaac Lab extension (upstream)
  tree.log               # Workspace directory listing
```

## Scope

The goal is to keep upstream repositories unchanged and place all integration-specific code in the `openarm_act_project/` layer.

## Upstream References

- https://github.com/enactic/openarm
- https://github.com/enactic/openarm_isaac_lab
- https://github.com/tonyzhaozh/act
- https://github.com/isaac-sim/IsaacLab
