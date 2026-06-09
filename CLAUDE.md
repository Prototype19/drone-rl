# Drone RL Project — Claude Code Orientation

**Read `SPEC.md` first. It is the source of truth for this project.**

This file is auto-loaded into every Claude Code session. Keep it short. Put detailed plans in SPEC.md.

---

## Project context

- **Goal:** Train a Crazyflie 2.1+ to hover, follow waypoints, and avoid obstacles via RL in Isaac Lab, then deploy sim-to-real.
- **Hardware:** NVIDIA DGX Spark (aarch64, GB10, 128 GiB unified memory).
- **Stack:** Isaac Sim 5.1.0-rc.19 + Isaac Lab 2.3.2 + rsl-rl-lib 5.0.1 + PyTorch 2.9 + CUDA 13 + Python 3.11.13.
- **Project root:** `~/spark-dev-workspace/drone-rl/`
- **Isaac Lab:** `~/IsaacLab/`
- **Isaac Sim:** `~/IsaacSim/`, with `$ISAACSIM_PATH` = `~/IsaacSim/_build/linux-aarch64/release`.
- **Bundled Python:** `~/IsaacLab/_isaac_sim/python.sh`

See SPEC.md §3 for the full verified configuration table including driver and git commits.

---

## Hard rules (never violate)

1. **Always pass `--headless` to Isaac Lab scripts.** Livestream is not supported on aarch64. GUI mode hangs or burns CPU forever.
2. **Always run long commands in tmux.** Anything expected to take more than 2 minutes — builds, training, downloads. SSH drops will kill non-tmux processes and waste work.
3. **Never commit `checkpoints/`, `logs/`, `videos/`, `wandb/`, or `*.pth`.** Use the gitignore.
4. **Never start a new project phase without user approval.** Phases are defined in SPEC.md §6.
5. **Never modify the reward function after Phase 3 without proposing the change first.** Reward changes invalidate prior experiments.
6. **One concept per training run.** Don't change reward + randomization + hyperparameters together — debugging becomes impossible.
7. **No GUI mode for Isaac Sim on the Spark.** Use NoMachine for occasional GUI needs (set up separately, not in this project).

---

## Per-session ritual

1. Read this file (you're doing it).
2. Read `SPEC.md` (or skim, if you've read it recently in this conversation).
3. Check `EXPERIMENTS.md` for the most recent training run.
4. Confirm the current phase before suggesting work.
5. If user asks for code on a multi-file change, propose a plan first (Shift+Tab plan mode).
6. End the session by updating `EXPERIMENTS.md` (if a run happened) and `NOTES.md` (if a concept was learned).

---

## Common commands

Training (always headless, always tmux):
```bash
tmux new -s train
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task <task-name> --headless --num_envs <n>
# Ctrl+b d to detach
```

Playing a trained policy (with video recording):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task <task-name> --num_envs 16 --headless \
    --video --video_length 300
```

TensorBoard:
```bash
tensorboard --logdir logs/ --bind_all --port 6006
# user opens http://<spark-ip>:6006 in laptop browser
```

Killing stuck Isaac runs:
```bash
pkill -f "isaac" ; sleep 3 ; pkill -9 -f "isaac"
```

Quick version check:
```bash
~/IsaacLab/_isaac_sim/python.sh -c "
import torch, isaaclab
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'isaaclab: {isaaclab.__version__}')"
```

---

## Documentation files in this project

- **SPEC.md** — full project description. The source of truth.
- **CLAUDE.md** — this file. Quick orientation.
- **NOTES.md** — conceptual notes the user takes as they learn.
- **SETUP_NOTES.md** — install commands actually run, gotchas hit, version pins.
- **EXPERIMENTS.md** — one line per training run: date, task, settings, final reward, notes.
- **README.md** — public-facing overview (for GitHub).

---

## Git conventions

- **Remote:** `origin` (GitHub, public repo at `Prototype19/drone-rl`)
- **Default branch:** `master`
- **Commit messages:** written from the actual diff, present tense imperative ("Add hover reward term" not "Added" or "Adding"). Multi-line body if the change is non-trivial.
- **Commit cadence:** after every working state. "Hover trains to reward > 10" is a commit.
- **Tag phase boundaries:** `v0.1-cartpole-validated`, `v0.2-quadcopter-stock-trained`, `v0.3-hover-custom-env`, etc.
- **Branch naming for experiments:** `experiment/<short-descriptor>` (e.g., `experiment/dense-energy-penalty`). Merge back to `master` only when the experiment is conclusive.
- **Never force-push to `master`.** If a force-push seems necessary, stop and ask first.
- **Verify `.gitignore` is doing its job before pushing.** `git status` should never show `checkpoints/`, `logs/`, `videos/`, `wandb/`, or `*.pth` files as untracked.
- **Never commit credentials.** API keys, W&B keys, GitHub tokens belong in environment variables or a `.env` file (which is gitignored).

## Things that have already gone wrong (learn from them)

Don't repeat these:
- Running `create_empty.py` without `--headless` → burned 2 hours of CPU on a failed render.
- Building Isaac Sim without tmux → SSH dropped mid-build, lost work, had to restart.
- Looking for `isaac-sim.streaming.sh` → doesn't exist on aarch64 (livestream unsupported).
- Mixing source dir `~/IsaacSim/` with build output `~/IsaacSim/_build/linux-aarch64/release/` — launchers live in the latter, not the former.

---

## Heads-up about recent versions

PyTorch 2.9 and CUDA 13 are very recent. If a tutorial says "this works" and it doesn't on this machine, version drift is a likely cause. Check version compatibility before assuming a real bug.
