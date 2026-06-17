# SPEC: Drone RL — Crazyflie 2.1+ in Isaac Lab

| Field | Value |
|---|---|
| Status | Active |
| Owner | Daron (Prototype19) |
| Created | 2026-06-16 |
| Last updated | 2026-06-16 |
| Repo | `~/spark-dev-workspace/drone-rl/` · GitHub `Prototype19/drone-rl` (public) |
| Target environment | DGX Spark — GB10, aarch64, 128 GiB unified memory, Ubuntu 24.04 |
| Permission overrides | See `.claude/settings.json` in this repo (set via `/permissions-interview`) |

> **For the agent:** This is the source of truth. Read it first, every session. If it conflicts with code, the spec wins until we agree to change the spec — propose spec changes explicitly and log them in §10. The original phase-based spec is archived at `Old_SPEC.md`; this file supersedes it.

---

## 1. Goal & Motivation

**What:** Train a reinforcement-learning policy in NVIDIA Isaac Lab that controls a Crazyflie 2.1+ quadcopter to (1) hover stably, (2) follow waypoints, and (3) avoid static obstacles — using state-and-range observations only (no vision). The policy is intended for eventual sim-to-real deployment onto physical hardware with off-board inference.

**Why:** A hands-on path to learn modern RL-for-robotics end to end on the DGX Spark — from validating a recent, fast-moving stack (Isaac Sim 5.1 / PyTorch 2.9 / CUDA 13 on aarch64) through to a deployable autonomy policy — while building reusable conceptual knowledge captured in NOTES.md and `~/knowledge-base/`.

**Definition of overall success:** A single trained policy flies a Crazyflie in Isaac Sim that holds hover within 0.5 m of a target for 95% of an evaluation episode, follows a 10-waypoint sequence at ≥90% success, and completes obstacle courses (5+ obstacles) at ≥80%, all reproducible headless from a clean clone. Hardware deployment is an explicit go/no-go decision *after* the sim work, not part of this spec's success bar.

---

## 2. Scope

### In scope
- Custom Isaac Lab environments for hover, waypoint following, and obstacle avoidance on the Crazyflie 2.1+ asset.
- PPO training via `rsl-rl-lib` (Isaac Lab native), headless, on the Spark.
- Domain randomization to harden the policy against the sim-to-real reality gap.
- A simulated Multiranger-like ray-cast sensor (5 directional rangefinders) for obstacle sensing.
- TensorBoard experiment tracking; reproducible runs logged in EXPERIMENTS.md.
- Conceptual documentation (NOTES.md) and a public README.

### Out of scope (non-goals)
- **Vision/perception** — no cameras; state + range observations only.
- **Multi-drone / swarm** behavior.
- **Outdoor flight** — Crazyflie is unsuitable for wind/forest/terrain.
- **Custom flight-controller firmware** — Bitcraze firmware is used as-is; we do not modify the STM32 low-level controller.
- **Hardware bring-up and sim-to-real transfer** — deferred to a separate `SPEC_HARDWARE.md` (see Future work).
- **Hyperparameter tuning before the env is correct** — env correctness precedes tuning.

### Future work (explicitly deferred)
- **Sim-to-real (hardware):** Crazyflie 2.1+ bundle, Crazyradio 2.0, Lighthouse positioning, Multiranger deck (~$655 total). Separate spec, written only if M9's decision is "proceed."
- **Weights & Biases tracking:** optional cloud experiment tracking (project `drone-rl`). Requires `WANDB_API_KEY`. Not wired up; TensorBoard covers current needs.
- **Network-architecture exploration:** default to RSL-RL's MLP; revisit only if performance plateaus.

---

## 3. Requirements

### Functional
- **FR-1:** A version-check confirms the pinned stack (PyTorch 2.9.0+cu130, CUDA available, isaaclab 0.54.3) and a stock task trains headlessly to completion. *(M0)*
- **FR-2:** The stock `Isaac-Quadcopter-Direct-v0` task trains to a stable hover and a play-video is recorded. *(M1)*
- **FR-3:** The stock quadcopter env is documented from first principles in NOTES.md (thrust model, action space, terminations, observation vector, robot cfg, PPO hyperparameters). *(M2)*
- **FR-4:** A project-owned custom env `Isaac-Crazyflie-Hover-Direct-v0` (fixed-goal hover) registers, loads headless, and trains to convergence. *(M3)*
- **FR-5:** Domain randomization (init pose/vel, mass ±20%, per-motor ±15%, external force perturbations, observation noise, action latency, CoM offset) is added incrementally and the hover policy stays robust. *(M4–M6)*
- **FR-6:** The env extends to waypoint following: relative-goal observation, distance reward + success bonus, goal resampling, defined workspace. *(M7)*
- **FR-7:** The env extends to obstacle avoidance: 5-ray sensor, randomized obstacles, proximity penalty + collision termination. *(M8)*
- **FR-8:** Every training run is logged in EXPERIMENTS.md (one row) and each phase boundary is git-tagged `v0.x-<descriptor>`. *(all milestones)*

### Non-functional
- **NFR-1:** All Isaac Lab invocations run with `--headless`. No GUI mode on the Spark (livestream unsupported on aarch64).
- **NFR-2:** Any process expected to run > 2 minutes runs inside `tmux`.
- **NFR-3:** Training fits within 128 GiB unified memory at the chosen `num_envs`.
- **NFR-4:** Reward terms are named functions; weights live in env config, never as literals in env logic. Type hints on signatures; Google-style docstrings on public classes/methods.
- **NFR-5:** `checkpoints/`, `logs/`, `videos/`, `wandb/`, `*.pth`, `outputs/`, `__pycache__/` are gitignored and never committed.
- **NFR-6:** One concept per training run — never change reward + randomization + hyperparameters together.

### Constraints
- **Hardware/OS:** DGX Spark, GB10, aarch64, 128 GiB unified memory, Ubuntu 24.04.4, kernel 6.17.0-1021-nvidia, NVIDIA driver 580.159.03. Fixed.
- **aarch64 unsupported features (must not be used):** livestreaming/WebRTC, OBJ imports (use STL/USD), Hub Workstation Cache, Application Template, cuRobo/cuMotion, Isaac Sim App Selector.
- **Pinned stack:** see §4 Stack table. PyTorch 2.9 + CUDA 13 are recent — version drift is the first hypothesis when a tutorial "should work" but doesn't.
- **System updates** go through the DGX Dashboard, not `apt upgrade`. The user is not in the `docker` group by design.
- **Reward function is locked after M3** — changes require a proposal first (invalidates prior experiments).

---

## 4. Technical Approach

### Stack
| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS | aarch64 |
| Kernel | 6.17.0-1021-nvidia | NVIDIA kernel |
| GPU | NVIDIA GB10 | DGX Spark, 128 GiB unified memory |
| NVIDIA driver | 580.159.03 | |
| Python (bundled) | 3.11.13 | inside Isaac Sim env |
| PyTorch | 2.9.0+cu130 | CUDA 13.0 build |
| CUDA build | 13.0 | matches PyTorch wheel |
| Isaac Sim | 5.1.0-rc.19 | built from source @ commit `aa503a9` |
| Isaac Lab repo | 2.3.2 | cloned @ commit `a859a5f9d` |
| `isaaclab` package | 0.54.3 | |
| RSL-RL | `rsl-rl-lib` 5.0.1 | no `__version__` attr; check `pip show rsl-rl-lib` |
| TensorBoard | (bundled) | required local tracker |
| Weights & Biases | — | optional/deferred (see Future work) |

### Architecture overview
```
Policy (PPO, RSL-RL MLP)  ──actions──▶  Isaac Lab Direct env  ──wrench──▶  Crazyflie articulation (Isaac Sim / PhysX)
        ▲                                       │
        └────────────── observations ──────────┘   (4096 parallel envs, headless)

Custom env lives in source/crazyflie_hover/ as an editable external extension,
pip-installed into the Isaac Lab bundled Python. Launched via project wrappers
scripts/train.py and scripts/play.py against ~/IsaacLab.
```
The stock quadcopter env is **wrench-controlled** (1 collective thrust + 3 body moments applied to the base), not per-motor — the four props are cosmetic dummy actuators. See NOTES.md "wrench-controlled point body" and the sim-to-real transfer note for the full rationale; not duplicated here.

### Key design decisions
| Decision | Rationale | Alternatives rejected |
|---|---|---|
| RSL-RL PPO (Isaac Lab native) | Matches the Arm DGX Spark learning path; integrated logging/checkpointing | SB3 / other PPO — extra integration friction |
| Custom env as external extension in `source/crazyflie_hover/` | Project-owned, pip-installed editable, importable for train/play without forking Isaac Lab | In-tree edit of Isaac Lab; `envs/` dir from the original spec (corrected — env is at `source/`) |
| Fixed-goal hover for M3 | Converges far faster/tighter than random-goal (≈0.0016 m vs 0.083 m) — clean baseline | Random-goal hover first |
| Wrench control kept (not per-motor) | Per-motor only adds fidelity on saturation/coupling; real firmware does motor mixing as fixed arithmetic, not learned | Per-motor actions — harder sim-to-real for no learning benefit pre-hardware |
| TensorBoard required, W&B optional | TensorBoard is local/zero-setup and has covered all needs; W&B needs an API key never wired up | W&B as a hard requirement (perpetually deferred → demoted) |
| Descriptive git tags `v0.x-<descriptor>` | Matches existing history (`v0.1-cartpole-validated`, `v0.2-quadcopter-stock-trained`) and is human-readable | `m{n}-verified` template style — breaks tag continuity |
| Two-tier milestone verifiers | RL training is long and stochastic; a fast mechanical gate keeps CI-like checks cheap while the metric bar defines real success | Strict metric-parsing gate (slow/flaky); smoke-only (no quality signal) |

---

## 5. Environment & Setup

**Hardware/OS assumptions:** DGX Spark (GB10, aarch64), Ubuntu 24.04, the pinned stack in §4. Isaac Sim and Isaac Lab are installed outside this repo and are *not* vendored here.

**Setup commands:**
```bash
# from clean clone of drone-rl (assumes Isaac Sim + Isaac Lab already installed per SETUP_NOTES.md):
git clone https://github.com/Prototype19/drone-rl.git ~/spark-dev-workspace/drone-rl
cd ~/spark-dev-workspace/drone-rl

# install the custom env into the Isaac Lab bundled Python (editable):
~/IsaacLab/_isaac_sim/python.sh -m pip install -e source/crazyflie_hover

# sanity: versions match SPEC §4
~/IsaacLab/_isaac_sim/python.sh -c "import torch, isaaclab; print(torch.__version__, torch.cuda.is_available(), isaaclab.__version__)"
```

**Key paths:**
- Project root: `~/spark-dev-workspace/drone-rl/`
- Isaac Lab: `~/IsaacLab/` · bundled Python: `~/IsaacLab/_isaac_sim/python.sh`
- Isaac Sim build: `$ISAACSIM_PATH` = `~/IsaacSim/_build/linux-aarch64/release`
- Custom env: `source/crazyflie_hover/` · launchers: `scripts/train.py`, `scripts/play.py`

**Secrets/credentials required:** `WANDB_API_KEY` (only if optional W&B tracking is enabled — names only, never commit the value).

**Relevant knowledge-base entries:** `~/knowledge-base/` — scan its `README.md` index for Isaac Sim / Isaac Lab / aarch64 entries at session start. Pipeline mechanics (skills, memory, `/retro`) live in global `~/.claude/CLAUDE.md`; this spec does not duplicate them.

---

## 6. Milestones

> Two-tier verifier convention: the **Verifier** is the fast mechanical gate that must pass to close a milestone (env loads headless, smoke run exits 0, artifact exists). The **Success bar** is the documented quality threshold the deliverable must meet — judged at review, not asserted by the gate command (RL runs are long and stochastic). Statuses: ☐ not started / ◐ in progress / ☑ verified / ✗ blocked.

### M0 — Environment validation
- **Status:** ☑ (tag `v0.1-cartpole-validated`)
- **Objective:** Pinned stack is present and a stock task trains headlessly end to end.
- **Deliverables:** version-check output; cartpole training run + TensorBoard curve.
- **Verifier:**
  ```bash
  ~/IsaacLab/_isaac_sim/python.sh -c "import torch, isaaclab; print(torch.__version__, torch.cuda.is_available(), isaaclab.__version__)"
  cd ~/IsaacLab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Cartpole-Direct-v0 --headless --max_iterations 150
  ```
  **Expected result:** prints `2.9.0+cu130 True 0.54.3`; cartpole exits 0 and reward climbs from ~-5 to ~>250.
- **Success bar:** pipeline validated end to end (train + checkpoints + TensorBoard).
- **Review gate:** No (mechanical)
- **Checkpoint:** `v0.1-cartpole-validated`

### M1 — Stock quadcopter trained + hover video
- **Status:** ☑ (tag `v0.2-quadcopter-stock-trained`)
- **Objective:** Stock `Isaac-Quadcopter-Direct-v0` trains to stable hover; video recorded.
- **Deliverables:** trained checkpoint; play MP4; EXPERIMENTS.md row.
- **Depends on:** M0
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** smoke run exits 0; reward logging present.
- **Success bar:** full 5000-iter run → reward >100, `final_distance_to_goal` <0.1 m, `died`=0.0, episodes run full length; hover MP4 recorded.
- **Review gate:** Yes — visual inspection of hover video.
- **Checkpoint:** `v0.2-quadcopter-stock-trained`

### M2 — Stock-env source deep-dive (no new code)
- **Status:** ☑
- **Objective:** Understand and document the stock quadcopter env from first principles.
- **Deliverables:** NOTES.md section covering thrust model, action space, termination conditions, the 12-D observation vector, `CRAZYFLIE_CFG`, and the PPO hyperparameter inventory.
- **Depends on:** M1
- **Verifier:**
  ```bash
  grep -q "from first principles" ~/spark-dev-workspace/drone-rl/NOTES.md && \
  grep -qi "wrench" ~/spark-dev-workspace/drone-rl/NOTES.md && echo OK
  ```
  **Expected result:** prints `OK`; the section addresses all five required topics.
- **Success bar:** a reader unfamiliar with the env can state thrust model, action/obs spaces, and terminations from the doc alone.
- **Review gate:** Yes — human reads the write-up.
- **Checkpoint:** docs commit (no version tag; no code change)

### M3 — Custom fixed-goal hover env
- **Status:** ☑ (reward 146.12, final_distance 0.00452 m, died=0, full episodes; video recorded — tag `v0.3-hover-custom-env`)
- **Objective:** Project-owned `Isaac-Crazyflie-Hover-Direct-v0` registers, loads headless, and trains to convergence on a fixed hover point.
- **Deliverables:** `source/crazyflie_hover/` extension; `scripts/train.py` + `scripts/play.py`; trained checkpoint; hover MP4; EXPERIMENTS.md row.
- **Depends on:** M2
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
      --task Isaac-Crazyflie-Hover-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** task is registered, env loads, smoke run exits 0 with reward logging.
- **Success bar:** full run holds within **0.5 m of the hover point for 95%** of an eval episode (sustained, not just terminal); video confirms stable hover.
- **Review gate:** Yes — eval metrics + video.
- **Checkpoint:** `v0.3-hover-custom-env`

### M4 — Domain randomization #1 (init state, mass, motor strength)
- **Status:** ☐
- **Objective:** Add init pose/velocity randomization, mass ±20% (~27 g ± 5 g), per-motor strength ±15%; hover stays robust. Add one at a time, retrain after each.
- **Deliverables:** randomization ranges in env config; retrained checkpoint(s); EXPERIMENTS.md rows.
- **Depends on:** M3
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
      --task Isaac-Crazyflie-Hover-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** env loads with the new randomization enabled; smoke run exits 0.
- **Success bar:** retrained policy still holds <0.5 m hover with all three randomizations active.
- **Review gate:** Yes
- **Checkpoint:** `v0.4-domain-rand-1`

### M5 — Domain randomization #2 (force perturbations, observation noise)
- **Status:** ☐
- **Objective:** Add external force perturbations (simulated wind/touch) and Gaussian observation noise (position σ≈2 cm, orientation). One at a time, retrain after each.
- **Deliverables:** config ranges; retrained checkpoint(s); EXPERIMENTS.md rows.
- **Depends on:** M4
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
      --task Isaac-Crazyflie-Hover-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** env loads with perturbations + noise enabled; smoke run exits 0.
- **Success bar:** policy recovers from force perturbations and tolerates obs noise; holds <0.5 m.
- **Review gate:** Yes
- **Checkpoint:** `v0.5-domain-rand-2`

### M6 — Domain randomization #3 (action latency, CoM offset)
- **Status:** ☐
- **Objective:** Add 1-timestep action latency and random center-of-mass offset (±5 mm/axis). Hover survives the full perturbation suite.
- **Deliverables:** config ranges; final robust hover checkpoint; EXPERIMENTS.md rows; perturbation-recovery video.
- **Depends on:** M5
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
      --task Isaac-Crazyflie-Hover-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** env loads with latency + CoM offset enabled; smoke run exits 0.
- **Success bar:** policy recovers from **all seven** perturbation types; video evidence.
- **Review gate:** Yes
- **Checkpoint:** `v0.6-domain-rand-full`

### M7 — Waypoint following
- **Status:** ☐
- **Objective:** Drone tracks a sequence of 3D goals: relative-goal observation, distance reward + success bonus, goal resampling, defined 4 m × 4 m × 3 m workspace, all M4–M6 randomization active.
- **Deliverables:** extended env (obs += relative goal); modified reward (proposed first, per locked-reward rule); trained checkpoint; waypoint-flight video; EXPERIMENTS.md row.
- **Depends on:** M6
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
      --task Isaac-Crazyflie-Waypoint-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** extended-obs env registers and loads; smoke run exits 0.
- **Success bar:** reaches a sequence of 10 random waypoints in <60 s simulated time at ≥90% success; video of a known path (square/figure-8).
- **Review gate:** Yes
- **Checkpoint:** `v0.7-waypoint`

### M8 — Obstacle avoidance
- **Status:** ☐
- **Objective:** Drone avoids randomized static obstacles while still reaching waypoints, using a 5-ray Multiranger-like sensor (front/back/left/right/up, clipped ~4 m).
- **Deliverables:** ray-cast sensor (+5 obs floats); randomized obstacles; proximity penalty + collision termination; trained checkpoint; course-navigation video; EXPERIMENTS.md row.
- **Depends on:** M7
- **Verifier:**
  ```bash
  cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
      --task Isaac-Crazyflie-Obstacle-Direct-v0 --headless --num_envs 4096 --max_iterations 50
  ```
  **Expected result:** sensor + obstacles load; smoke run exits 0.
- **Success bar:** ≥80% completion on procedurally generated courses with 5+ obstacles; video.
- **Review gate:** Yes
- **Checkpoint:** `v0.8-obstacle-avoid`

### M9 — Decision point (non-coding gate)
- **Status:** ☐
- **Objective:** Re-evaluate whether to proceed to hardware sim-to-real, do more sim work, or stop. No new code.
- **Deliverables:** a go/no-go decision recorded in §10 Decision Log; if "proceed," a `SPEC_HARDWARE.md` stub is created.
- **Depends on:** M8
- **Verifier:**
  ```bash
  grep -q "M9" ~/spark-dev-workspace/drone-rl/SPEC.md && \
  grep -qiE "proceed|stop|more sim" ~/spark-dev-workspace/drone-rl/SPEC.md && echo "decision recorded"
  ```
  **Expected result:** prints `decision recorded`; Decision Log has an M9 row.
- **Review gate:** Yes — human decision.
- **Checkpoint:** `v0.9-sim-complete`

---

## 7. Verification Strategy

**Test approach:** Verification is per-milestone (§6), two-tier. There is no unit-test suite; the "tests" are the mechanical smoke verifiers plus documented success bars. The single fastest health check across the project:
```bash
# stack + custom env load smoke (no training):
~/IsaacLab/_isaac_sim/python.sh -c "import torch, isaaclab; print(torch.__version__, torch.cuda.is_available(), isaaclab.__version__)"
cd ~/IsaacLab && ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
    --task Isaac-Crazyflie-Hover-Direct-v0 --headless --num_envs 64 --max_iterations 5
```

**Continuous checks:** No CI (single-developer, GPU-bound). Before declaring a milestone done: confirm the mechanical verifier exits 0, then judge the success bar from logs/video. Reward-term and config-weight conventions (NFR-4) checked at review.

**Review process:** Review gate at every milestone boundary except M0 (mechanical). Reward changes after M3 require a written proposal before implementation. `/code-review` may be run on diffs; human review precedes any merge to `master`.

**Reproducibility bar:** Every verified milestone must pass its mechanical verifier from a clean clone (after the editable install), not just the working tree. Training metrics are stochastic — the success bar is judged on a representative run, not bit-reproducibility.

---

## 8. Risks & Open Questions

### Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Version drift (PyTorch 2.9 / CUDA 13 very recent) breaks a tutorial-following step | M | M | Treat version drift as first hypothesis; pin versions in §4; log gotchas via `/retro` to knowledge base |
| aarch64 feature gap blocks a needed Isaac Sim capability | M | M | §3 lists known-unsupported features; check aarch64 support before adopting any new tool/feature |
| Stochastic RL run misses success bar despite correct code | M | L | Two-tier verifier separates "pipeline works" from "metric met"; re-run/seed-vary before assuming a bug |
| Reward change mid-project invalidates prior experiments | L | H | Reward locked after M3; changes need a proposal + Decision Log row; one concept per run |
| SSH drop kills a long run | M | H | tmux mandatory for >2-min processes (NFR-2) |
| Sim-to-real gap (idealized wrench vs real actuator/sensor) | H | M | Out of scope here; addressed by M4–M6 domain randomization and deferred hardware spec |
| Accidental commit of checkpoints/logs/videos | L | M | `.gitignore` (NFR-5); `git status` checked before every push |

### Open questions
| Question | Must resolve before | Owner |
|---|---|---|
| Headless video-recording cadence (every N iters vs on-demand) | M4 | Daron |
| Is W&B ever wired up, or permanently optional? (`WANDB_API_KEY` needed if yes) | M7 | Daron |
| Exact Crazyflie firmware version + setpoint interface for sim-to-real | M9 | Daron |
| Positioning approach for hardware (Lighthouse vs alternative) | M9 | Daron |
| Network architecture — stay on RSL-RL MLP or explore? | revisit only if performance plateaus | Daron |

---

## 9. Agent Operating Notes

- Pipeline mechanics (skills, `~/knowledge-base/`, memory, `/retro`, `/permissions-interview`) are governed by global `~/.claude/CLAUDE.md` — follow it; this spec does not restate it.
- **Per-session ritual:** read CLAUDE.md → read this SPEC → check EXPERIMENTS.md for the last run → confirm the current milestone before suggesting work.
- Always pass `--headless`; always run >2-min commands in `tmux`.
- **Reward function is locked after M3** — propose changes in chat first; never edit silently.
- One concept per training run (reward XOR randomization XOR hyperparameters).
- Do not start a new milestone without owner approval. Do not spend >30 min of GPU time on a run without approval.
- When a verifier fails twice for the same cause, or you hit an open question from §8: stop, summarize, ask. Do not redesign around the obstacle.
- Update EXPERIMENTS.md after any run; update NOTES.md when a concept is learned. Run `/retro` before ending a session that closes a milestone or does significant work.
- Do not modify this spec's milestones, scope, or stack without logging it in §10 and getting owner approval.

---

## 10. Decision Log

| Date | Change | Reason | Approved by |
|---|---|---|---|
| 2026-06-16 | Migrated phase-based spec → milestone/verifier template via `/spec-interview`; archived original as `Old_SPEC.md` | Compatibility with new Claude Code pipeline; add executable verifiers | Daron |
| 2026-06-16 | Mapped 8 phases → M0–M9; split domain randomization into M4–M6 | Each milestone independently verifiable; DR too large for one | Daron |
| 2026-06-16 | Demoted W&B from required tracker to optional/Future work | Never wired up; TensorBoard has covered all needs | Daron |
| 2026-06-16 | Corrected custom-env path `envs/crazyflie_hover/` → `source/crazyflie_hover/`; standardized tags on descriptive `v0.x-<descriptor>` | Match actual repo layout and git history | Daron |
| 2026-06-17 | M3 verified ☑ (reward 146.12, final_distance 0.00452 m, died=0, hover video recorded); M3 open question closed | Wrap-up complete; ready to tag `v0.3-hover-custom-env` | Daron |
