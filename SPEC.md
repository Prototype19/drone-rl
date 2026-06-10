# Drone RL Project Specification

> **For Claude Code:** This is the source of truth for the project. Read this first, every session, before making decisions. If something here conflicts with what's in code, the spec wins until we agree to change the spec. If you think the spec should change, propose the change explicitly before changing code.

---

## 1. Project Summary

Train a reinforcement learning policy in NVIDIA Isaac Lab that controls a Crazyflie 2.1+ quadcopter to:

1. **Hover** stably at a target position
2. **Follow waypoints** sent from the user's laptop
3. **Avoid obstacles** in a known indoor environment

The trained policy will eventually be deployed sim-to-real on a physical Crazyflie 2.1+ with indoor Lighthouse positioning. Hardware purchase is deferred until Phase 1 sim work is complete.

### What this project is NOT
- Not an outdoor drone project (Crazyflie is unsuitable for wind, forests, hilly terrain)
- Not a vision-based perception project (state + range observations only; no cameras)
- Not a multi-drone or swarm project
- Not a from-scratch flight controller project (we use Crazyflie firmware as-is)

---

## 2. Hardware Targets (deferred until Phase 1 complete)

| Item | Choice | Approx. cost |
|------|--------|--------------|
| Drone | Crazyflie 2.1+ Bundle (Bitcraze) | ~$280 |
| Radio | Crazyradio 2.0 | ~$50 |
| Positioning | Lighthouse Positioning Kit + 2 base stations | ~$250 |
| Sensing | Multiranger deck (4-5 directional rangefinders) | ~$50 |
| Spare batteries | 4+ LiPo packs | ~$25 |
| **Total** | | **~$655** |

**Compute split:** Off-board. Policy runs on the user's laptop or Spark; commands are sent to the drone via Crazyradio. The Crazyflie's STM32 runs only the low-level flight controller (Bitcraze firmware, unmodified).

---

## 3. Verified Configuration (as of project start)

This is the actual, verified configuration on the development Spark. Discrepancies between this and what's installed should be investigated.

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Ubuntu 24.04.4 LTS | aarch64 |
| Kernel | 6.17.0-1021-nvidia | NVIDIA's kernel |
| GPU | NVIDIA GB10 | DGX Spark, 128 GiB unified memory |
| NVIDIA driver | 580.159.03 | |
| Python (bundled) | 3.11.13 | Inside Isaac Sim env |
| PyTorch | 2.9.0+cu130 | CUDA 13.0 build |
| CUDA build | 13.0 | Matches PyTorch wheel |
| Isaac Sim | 5.1.0-rc.19 | Built from source @ commit `aa503a9` |
| Isaac Lab repo | 2.3.2 | Cloned @ commit `a859a5f9d` |
| `isaaclab` package | 0.54.3 | Python package version |
| RSL-RL | `rsl-rl-lib` 5.0.1 | Installed in bundled Python env. Note: package doesn't expose `__version__` attribute; check via `pip show rsl-rl-lib` |
| Disk free | 3.4 TB / 3.7 TB | Plenty of room |

**Project paths:**
- Project root: `~/spark-dev-workspace/drone-rl/`
- Isaac Sim source: `~/IsaacSim/`
- Isaac Sim build output: `$ISAACSIM_PATH` = `~/IsaacSim/_build/linux-aarch64/release`
- Isaac Lab: `~/IsaacLab/`
- Bundled Python: `~/IsaacLab/_isaac_sim/python.sh`

**Heads up:** PyTorch 2.9 and CUDA 13 are very recent (released within the past few months). If you hit "this used to work" issues following online tutorials or older docs, version drift is a likely cause. Check version compatibility before assuming a bug.

---

## 4. aarch64 limitations to respect

The Spark is aarch64. The following Isaac Sim features are **NOT available** and must not be used:
- Livestreaming (no `isaac-sim.streaming.sh`, no WebRTC client)
- OBJ file imports (affects URDF importer for OBJ meshes — use STL/USD instead)
- Hub Workstation Cache
- Application Template
- cuRobo / cuMotion
- Isaac Sim App Selector

**All Isaac Lab scripts run with `--headless`. No exceptions until NoMachine remote desktop is configured separately.**

---

## 5. Stack and Conventions

### Required tools
- **Isaac Sim + Isaac Lab** — simulation and env framework
- **RSL-RL** — PPO implementation (Isaac Lab native; matches Arm DGX Spark guide)
- **PyTorch 2.9 with CUDA 13** — bundled in Isaac Sim Python env
- **TensorBoard** — local training visualization
- **Weights & Biases** — experiment tracking (free tier; project name: `drone-rl`)
- **tmux** — REQUIRED for any command expected to run longer than 2 minutes
- **Git + GitHub** — public repo at `Prototype19/drone-rl`; commit at every working state

### Project layout
```
~/spark-dev-workspace/drone-rl/
├── SPEC.md                     # This file. Source of truth.
├── CLAUDE.md                   # Short Claude Code orientation; references SPEC.md
├── NOTES.md                    # Conceptual notes as the user learns
├── SETUP_NOTES.md              # Install commands actually run, gotchas hit
├── EXPERIMENTS.md              # One-line-per-training-run log
├── README.md                   # Public-facing overview
├── envs/                       # Custom Isaac Lab environments
│   └── crazyflie_hover/        # Phase 3 env, etc.
├── scripts/
│   ├── train.py                # Thin wrapper around Isaac Lab train
│   ├── play.py                 # Thin wrapper around Isaac Lab play
│   └── record_video.py         # Helper to scp video to laptop
├── configs/                    # YAML/Python configs for experiments
├── checkpoints/                # gitignored; trained policies
├── logs/                       # gitignored; tensorboard, wandb
└── videos/                     # gitignored; recorded MP4s
```

### Coding conventions
- Type hints on all function signatures
- Docstrings on all public classes and methods (Google style)
- One env per directory; each env has `__init__.py`, `*_env.py`, `*_env_cfg.py`
- Reward terms named clearly: `reward_hover`, `reward_action_smoothness`, etc. Never `r1`, `r2`.
- All randomization ranges go in env config, not hardcoded in env logic
- `print()` for one-off debugging only. Use Python `logging` for anything that stays in.

### Git discipline
- Commit after every working state. "Hover trains to reward > 10" is a commit.
- Tag at end of each phase: `v0.1-hover`, `v0.2-randomization`, etc.
- `.gitignore` excludes: `checkpoints/`, `logs/`, `videos/`, `*.pth`, `wandb/`, `outputs/`, `__pycache__/`

---

## 6. Phased Plan

The project is divided into phases. Do not start phase N+1 until phase N's deliverable is met.

### Phase 1: Sim foundations (~weeks 1-2)
**Goal:** Validate the stack with stock Isaac Lab tasks.

**Tasks:**
- [ ] Train `Isaac-Cartpole-Direct-v0` headlessly to completion. Verify TensorBoard reward curve climbs.
- [ ] Train `Isaac-Quadcopter-Direct-v0` headlessly for 5000+ iterations. Verify hover behavior.
- [ ] Record an MP4 of the trained quadcopter policy. Inspect visually.
- [ ] Set up W&B logging integrated with RSL-RL.
- [ ] Initialize Git repo, push to GitHub.

**Deliverable:** Trained stock quadcopter policy, TensorBoard + W&B both logging, video of hover.

### Phase 2: Source code deep dive (~week 3)
**Goal:** Understand the stock quadcopter env line by line. **No new code in this phase.**

**Tasks:**
- [x] Read `quadcopter_env_cfg.py` end to end. Document each config field in `NOTES.md`. *(Direct env — config is `QuadcopterEnvCfg` inside `quadcopter_env.py`; no separate cfg file.)*
- [x] Read `quadcopter_env.py`. Document each method's purpose in `NOTES.md`.
- [x] Locate and read `CRAZYFLIE_CFG` in `isaaclab_assets`. Note mass, inertia, motor model.
- [x] Read `agents/rsl_rl_ppo_cfg.py`. List which hyperparameters exist (don't tune yet).
- [x] Answer in `NOTES.md`: how is thrust modeled? What's the action space? What termination conditions exist?

**Deliverable:** A 1-2 page summary in `NOTES.md` that explains the stock env from first principles.

### Phase 3: Custom hover env (~week 4)
**Goal:** Hand-built minimal hover environment, training to convergence.

**Tasks:**
- [ ] Copy stock quadcopter env to `envs/crazyflie_hover/`. Rename classes, strip unused code.
- [ ] Reward function: dense, 3 terms max — `reward_position` (Gaussian around origin), `reward_orientation` (penalize tilt), `reward_action` (small action penalty).
- [ ] Observations: position (3), linear velocity (3), orientation as quaternion (4), angular velocity (3). 13 floats.
- [ ] Actions: 4 motor thrust commands, normalized to [-1, 1].
- [ ] Register env via Gym. Confirm it loads with `--headless`.
- [ ] Train to convergence. Define success: drone stays within 0.5m of origin for 95% of an evaluation episode.

**Deliverable:** Trained custom hover policy in `checkpoints/`, video showing stable hover, EXPERIMENTS.md log of the run.

### Phase 4: Domain randomization (~week 5)
**Goal:** Robust hover policy that survives sim-to-real reality gap.

**Add these one at a time. Retrain and verify after each.**

- [ ] Random initial pose and velocity within reasonable bounds
- [ ] Mass randomization: ±20% of nominal Crazyflie mass (~27g ± 5g)
- [ ] Per-motor strength randomization: ±15%
- [ ] External force perturbations during episodes (simulated wind/touch)
- [ ] Observation noise: Gaussian noise on position (σ=2cm) and orientation
- [ ] Action latency: apply action 1 timestep late
- [ ] Center-of-mass random offset: ±5mm in each axis

**Deliverable:** Hover policy that recovers from all perturbations above. Video evidence.

### Phase 5: Waypoint following (~weeks 6-7)
**Goal:** Drone tracks a sequence of 3D goal positions provided by an external command.

**Tasks:**
- [ ] Extend observation space to include relative goal position (3 floats).
- [ ] Modify reward: dense distance-to-goal reward, success bonus on reaching waypoint (within 0.2m for 1s).
- [ ] Implement goal resampling: new random goal in workspace whenever current goal reached, or every N steps.
- [ ] Define workspace: 4m × 4m × 3m volume (matches small indoor flight space).
- [ ] All randomization from Phase 4 stays active.
- [ ] Define success: drone reaches sequence of 10 random waypoints in < 60s of simulated time, 90% success rate.

**Deliverable:** Trained waypoint-following policy. Video of policy flying a known waypoint sequence (e.g., a square or figure-8).

### Phase 6: Obstacle avoidance (~weeks 8-11)
**Goal:** Drone avoids static obstacles while still reaching waypoints.

**Tasks:**
- [ ] Add a Multiranger-like sensor to the sim Crazyflie: 5 ray-cast rangefinders (front, back, left, right, up). Output: 5 distances, clipped to e.g. 4m.
- [ ] Extend observations: + 5 floats (rangefinder distances).
- [ ] Place static obstacles in the env: boxes, walls, pillars. Randomize positions per episode.
- [ ] Reward: keep waypoint reward, add penalty for proximity to obstacles (smooth, not just collision), large penalty for actual collision.
- [ ] Termination on collision.
- [ ] Define success: 80% completion rate on procedurally-generated obstacle courses with 5+ obstacles.

**Deliverable:** Trained obstacle-avoiding waypoint-following policy. Video of policy navigating a procedurally-generated course.

### Phase 7: Decision point
**Not a coding phase.** Re-evaluate whether to proceed to hardware (Phase 8), do more sim work (refine, harder tasks), or stop.

### Phase 8: Sim-to-real (hardware, separate spec)
Out of scope for this document. A separate `SPEC_HARDWARE.md` will be written when Phase 7 decision is "proceed."

---

## 7. Workflow Conventions for Claude Code

### Autonomy mode: Co-pilot
- Claude Code MAY execute individual tasks autonomously: edit files, run training scripts, run tests, search the codebase, read documentation.
- Claude Code MUST get user approval before:
  - Starting a new phase
  - Adding a new dependency
  - Significantly changing the env's observation or action space
  - Modifying the reward function (after Phase 3 — propose changes, don't just make them)
  - Deleting or renaming files outside its own working scratch
  - Spending Spark GPU time on a training run > 30 minutes
- Claude Code SHOULD propose plans (Shift+Tab plan mode or equivalent) before multi-file changes.

### Per-session rituals
1. Start every session by reading this SPEC.md, then NOTES.md and EXPERIMENTS.md if relevant.
2. Confirm the current phase before suggesting work.
3. End each session by updating EXPERIMENTS.md if a training run was attempted, and NOTES.md if a concept was learned.

### Mandatory practices
- **tmux** for any process expected to run > 2 minutes. Never block the session on a long process.
- **--headless** on every Isaac Lab invocation.
- **Verify before declaring done.** If a change should make hover work, run a short training and confirm reward improves.
- **One concept per training run.** Don't change reward AND randomization AND hyperparameters in one run — debugging will be impossible.

### Forbidden practices
- Running Isaac Sim in GUI mode on the Spark (will hang or burn CPU on software rendering)
- Hardcoding paths that vary per machine (use env vars or config)
- Committing trained checkpoints, logs, videos, or W&B caches to Git
- Adding reward terms without recording them in EXPERIMENTS.md
- Hyperparameter tuning before Phase 4 — env correctness first
- Implementing features beyond the current phase

---

## 8. Reward Design Philosophy

**Hover and waypoint:** dense rewards. Composite of multiple small terms. Train fast, fine-tune the policy.

**Obstacle avoidance:** sparser. Heavy penalty for collision, lighter shaping rewards for clearance. Avoid micromanaging trajectory.

**Always:**
- Reward terms expressed as named functions, not anonymous arithmetic
- Weights stored in env config, never literals in env code
- Each reward term logged separately to TensorBoard / W&B for diagnostics

---

## 9. Definitions of Done (per phase)

For each phase, "done" means all three:

1. **Functional:** the deliverable runs and produces the expected behavior
2. **Documented:** EXPERIMENTS.md updated, NOTES.md updated with anything learned
3. **Versioned:** Git commit with descriptive message; tag if it's a phase boundary

---

## 10. Open Questions / Decisions Deferred

Tracked here so they don't get lost. Update as decisions are made.

- **Exact Crazyflie firmware version for eventual sim-to-real:** decide at end of Phase 7
- **Lighthouse vs. alternative positioning:** Lighthouse is the plan, revisit if Bitcraze releases something better
- **W&B project sharing:** private by default; reconsider if collaborating
- **Headless video recording cadence:** every N iterations vs. on-demand only — decide in Phase 1
- **Network architecture:** default to RSL-RL's MLP for now; revisit if performance plateaus

---

## 11. References

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim 5.1 docs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/)
- [Arm DGX Spark + Isaac Lab learning path](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_isaac_robotics/)
- [Bitcraze Crazyflie documentation](https://www.bitcraze.io/documentation/)
- [RSL-RL repository](https://github.com/leggedrobotics/rsl_rl)
- [Davide Scaramuzza's Robotics & Perception Group](https://rpg.ifi.uzh.ch/publications.html) — academic benchmark for drone autonomy

---

*Spec last verified: project start. Update §3 "Verified Configuration" whenever Isaac Sim, Isaac Lab, or core dependencies are upgraded.*
