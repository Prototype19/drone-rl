# drone-rl

Reinforcement learning for a Crazyflie 2.1+ quadcopter in [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/), with the intent to deploy sim-to-real onto physical hardware.

## Goals

Train a single policy that can:

1. **Hover** stably at a target position
2. **Follow waypoints** in a known indoor workspace
3. **Avoid static obstacles** using onboard rangefinders

State-and-range observations only — no vision. Off-board inference (laptop or workstation), with commands sent to the drone over a Crazyradio. The Crazyflie's STM32 runs stock Bitcraze firmware.

## Stack

- **Simulation:** Isaac Sim 5.1 + Isaac Lab 2.3.2
- **RL algorithm:** PPO via `rsl-rl-lib` 5.0.1 (Isaac Lab's native trainer)
- **Compute:** PyTorch 2.9 + CUDA 13 on an NVIDIA DGX Spark (aarch64, GB10)
- **Tracking:** TensorBoard + Weights & Biases

See [`SPEC.md`](./SPEC.md) §3 for the full pinned configuration.

## Clone

```bash
git clone https://github.com/Prototype19/drone-rl.git
```

This repo holds project docs, custom Isaac Lab environments, and training configs. It does not include Isaac Sim or Isaac Lab themselves — install those separately per [`SETUP_NOTES.md`](./SETUP_NOTES.md).

## Status

**Phase 1 — Sim foundations.** Validating the stack against stock Isaac Lab tasks (cartpole, then the bundled quadcopter env) before any custom environment work.

The project is structured in phases (§6 of the spec): sim foundations → source deep-dive → custom hover env → domain randomization → waypoint following → obstacle avoidance → sim-to-real decision. Each phase has its own deliverable and exit criteria.

## Repository layout

```
SPEC.md          Source of truth — read first
CLAUDE.md        Orientation for Claude Code sessions
NOTES.md         Conceptual notes as the project progresses
SETUP_NOTES.md   Install commands, version pins, gotchas
EXPERIMENTS.md   One line per training run
envs/            Custom Isaac Lab environments (added in Phase 3+)
scripts/         Thin wrappers around Isaac Lab train/play
configs/         Per-experiment configs
```

`checkpoints/`, `logs/`, `videos/`, and `wandb/` are gitignored.

## Further reading

- [`SPEC.md`](./SPEC.md) — full project specification, phased plan, and conventions
- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/)
- [Bitcraze Crazyflie documentation](https://www.bitcraze.io/documentation/)
