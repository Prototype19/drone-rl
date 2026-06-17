# Notes

Conceptual notes as the project progresses.

> **Terminology note (2026-06-16):** The spec migrated from phases to milestones (M0–M9) — see SPEC.md §6. This journal predates that and still says "Phase N"; the mapping is Phase 1 → M0+M1, Phase 2 → M2, Phase 3 → M3, Phase 4 → M4–M6, Phase 5 → M7, Phase 6 → M8, Phase 7 → M9. Entries below are kept in their original voice.

---

## Session Handoff — 2026-06-08 (Phase 1 COMPLETE ✅)

**For the next Claude session: Phase 1 is done and tagged `v0.2-quadcopter-stock-trained`. Do not start Phase 2 without user approval.**

### Phase 1 deliverables — all met
- ✅ Stack smoke test (PyTorch 2.9.0+cu130, CUDA 13.0, isaaclab 0.54.3, CUDA available).
- ✅ Cartpole smoke test: `Isaac-Cartpole-Direct-v0`, 150 iters, reward -5.4 → 295.35, 32s on GB10. Pipeline validated end-to-end.
- ✅ Quadcopter stock training: `Isaac-Quadcopter-Direct-v0`, 5000 iters, 4096 envs, final reward 123.51, ~26 min on GB10. Stable hover: `final_distance_to_goal` 0.083 m, 0 crashes (`died`=0.0), episodes run full length. Final checkpoint `model_4999.pt`.
- ✅ Hover video recorded: `~/IsaacLab/logs/rsl_rl/quadcopter_direct/2026-06-09_00-21-05/videos/play/rl-video-step-0.mp4` (1280x720, 50 fps, 299 frames, ~6 s).
- ✅ Both runs logged in EXPERIMENTS.md; docs committed; phase tagged `v0.2-quadcopter-stock-trained`.

### Carried forward (not blockers)
- **W&B logging** not yet set up — needs the user's W&B API key as an env var (project `drone-rl`). Deferred to a future session.
- tmux `train` and `tb` sessions torn down at the end of this session.

### Reminders
- Show docs for user approval before staging/committing.
- Don't start Phase 2 without user approval.

---

## Concepts learned

- The `TU: ... renderD128 (VK_ERROR_INCOMPATIBLE_DRIVER)` line in headless Isaac Sim logs on aarch64 is a **benign Vulkan probe failure**, not a real error — PhysX runs on CUDA, so training and inference are unaffected.
- Isaac Lab bakes the **local-timezone wall-clock into log-dir names** at creation (e.g. `quadcopter_direct/2026-06-09_00-21-05/`); these are immutable run IDs and won't reflect a later timezone change. On-disk file mtimes are absolute and always display correctly in the current zone.
- **Why the quadcopter reward terms are negative except `distance_to_goal`** (`source/isaaclab_tasks/.../direct/quadcopter/quadcopter_env.py`). The reward is one positive *objective* plus two negative *regularizers*:
  - `distance_to_goal` (scale **+15**): the objective. Mapped as `1 - tanh(dist/0.8)` ∈ [0,1] — equals 1 sitting on the goal, decays toward 0 as the drone drifts away. Dominates the total reward.
  - `lin_vel` (scale **-0.05**) and `ang_vel` (scale **-0.01**): penalties on `Σ(velocity²)`. They don't describe the task; they shape *how* it's done — penalizing speed and spin so the drone is smooth and still, not zooming/tumbling.
  - **Key insight:** "reach the goal" alone is satisfied by flying *through* the goal at speed. The velocity penalties are what turn "reach it" into "reach it **and hold still**" — the optimum is goal-reward maxed *and* penalties ≈ 0 simultaneously, i.e. a stable hover.
  - Diagnostic value: penalties near zero at convergence (our run: `lin_vel` -0.017, `ang_vel` -0.164, `distance_to_goal` 12.54) is the *signature* of a clean hover. Early in training they're large negatives (flailing drone). This positive-objective + negative-regularizers pattern is standard across locomotion/control RL (velocity, energy, action-rate, joint-limit penalties).

---

## Phase 2 — Stock quadcopter env, from first principles

**Deliverable for Phase 2.** Explains `Isaac-Quadcopter-Direct-v0` end to end. This is a **Direct** workflow env, so there is no manager/`*_env_cfg.py` split — the config dataclass (`QuadcopterEnvCfg`) and the env logic (`QuadcopterEnv`) live in one file. Four files cover the whole env:

- `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py` — config + env
- `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/agents/rsl_rl_ppo_cfg.py` — PPO hyperparameters
- `source/isaaclab_assets/isaaclab_assets/robots/quadcopter.py` — `CRAZYFLIE_CFG` robot asset
- the `cf2x.usd` asset on Isaac Nucleus — where mass/inertia actually live

### The one big idea: it's a wrench-controlled point body, not a real quadrotor

The single most important thing to internalize: **the policy does not control four motors.** It outputs one collective thrust and three body torques, which are applied directly to the drone's base as a force + moment ("wrench"). The four propeller joints in the USD spin (initial velocity ±200 rad/s, alternating sign) but are driven by **dummy actuators with zero stiffness and zero damping** — they're cosmetic, for the video. There is no per-rotor thrust, no motor mixing, no rotor dynamics. This is a deliberate abstraction that makes the control problem clean; sim-to-real later will have to bridge from this idealized wrench to real per-motor PWM.

### Config fields (`QuadcopterEnvCfg`)

| Field | Value | Meaning |
|---|---|---|
| `episode_length_s` | 10.0 | Episode is 10 s of sim time. |
| `decimation` | 2 | Policy acts every 2 physics steps. |
| `sim.dt` | 1/100 | Physics at 100 Hz. |
| → derived control rate | 50 Hz | `dt × decimation = 0.02 s` per policy step → **500 control steps/episode** (matches our logged `ep_len` 499, 0-indexed). |
| `action_space` | 4 | collective thrust + 3 body moments. |
| `observation_space` | 12 | see obs breakdown below. |
| `state_space` | 0 | Critic uses the same obs as the actor (no privileged state). |
| `debug_vis` | True | Draws the goal as a small cuboid marker. |
| `scene.num_envs` | 4096 | Parallel envs (overridable on CLI). |
| `scene.env_spacing` | 2.5 | Metres between env origins. |
| `terrain` | flat plane | Ground, friction 1.0/1.0, restitution 0. |
| `robot` | `CRAZYFLIE_CFG` | Spawned at `/World/envs/env_.*/Robot`. |
| `thrust_to_weight` | 1.9 | Max collective thrust = 1.9 × weight. |
| `moment_scale` | 0.01 | Scales the 3 body-moment actions (N·m). |
| `lin_vel_reward_scale` | -0.05 | Linear-velocity penalty weight. |
| `ang_vel_reward_scale` | -0.01 | Angular-velocity penalty weight. |
| `distance_to_goal_reward_scale` | 15.0 | Goal objective weight. |

### Methods (`QuadcopterEnv`)

- `__init__` — allocates action/thrust/moment/goal buffers and the per-term episode-sum log dict; finds the `"body"` index; reads robot **mass at runtime** from PhysX (`root_physx_view.get_masses().sum()`) and computes `weight = mass × |gravity|`. (So thrust is always scaled to the *actual* asset mass, not a hard-coded number.)
- `_setup_scene` — builds the articulation + terrain, clones the 4096 envs, adds a dome light.
- `_pre_physics_step(actions)` — clamps actions to [-1, 1], then maps them (see thrust model below). Runs once per policy step.
- `_apply_action` — writes the computed force + torque onto the body via the wrench composer. Runs every physics step.
- `_get_observations` — builds the 12-D obs (below); returns `{"policy": obs}`.
- `_get_rewards` — the three-term reward (documented in "Concepts learned" above); accumulates episode sums.
- `_get_dones` — returns `(died, time_out)` (termination conditions below).
- `_reset_idx` — logs episodic reward averages + metrics (`final_distance_to_goal`, termination counts), resets the robot to default state, **samples a new goal**, and on full resets randomizes `episode_length_buf` so envs don't all reset in lockstep (avoids throughput spikes). Goal sampling: `x,y ∈ U(-2, 2)` around the env origin, `z ∈ U(0.5, 1.5)`.
- `_set_debug_vis_impl` / `_debug_vis_callback` — create/update the goal cuboid marker.

### The three required answers

**1. How is thrust modeled?** One collective thrust along **body +z**:
```
thrust_z = thrust_to_weight × weight × (a0 + 1) / 2     # a0 ∈ [-1, 1]
```
So `a0 = -1` → 0 N (free-fall), `a0 = 0` → 0.95 × weight (slowly sinking), `a0 = +1` → 1.9 × weight (max climb). Hover sits near `a0 ≈ 0.053` (thrust = weight). It is a single net force on the base — **not** four rotor thrusts. Body moments come from `a1:3 × moment_scale` (0.01 N·m full-scale) about the three body axes.

**2. What is the action space?** 4-D continuous, clamped to [-1, 1]:
- `a0` → collective thrust, mapped to [0, 1.9·W] as above.
- `a1, a2, a3` → roll/pitch/yaw body moments, each scaled by `moment_scale = 0.01`.

**3. What termination conditions exist?** From `_get_dones`:
- `died` — altitude out of bounds: `z < 0.1 m` (crashed/too low) **or** `z > 2.0 m` (escaped upward). A real terminal failure.
- `time_out` — reached the 10 s / 500-step horizon. A bootstrap truncation, not a failure.

(In our Phase 1 run `died = 0.0` and episodes ran full length — every env survived to time-out, the signature of a stable hover.)

### Observation vector (12-D, all body-frame)

| Slice | Source | Dims |
|---|---|---|
| linear velocity | `root_lin_vel_b` | 3 |
| angular velocity | `root_ang_vel_b` | 3 |
| projected gravity | `projected_gravity_b` | 3 |
| goal position (in body frame) | `subtract_frame_transforms(...)` → `desired_pos_b` | 3 |

Note there is **no absolute position or orientation** in the obs — the policy only ever sees the goal *relative to itself* and which way is down (projected gravity). That's what makes the learned hover translation-invariant and directly portable to any start position.

### `CRAZYFLIE_CFG` (the robot asset)

- USD: `{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd`. Gravity enabled, gyroscopic forces on, self-collisions off, 4 position / 0 velocity solver iterations.
- Init: spawn at `z = 0.5 m`; 4 motor joints `m1..m4` given initial spin ±200 rad/s (alternating sign, like a real X-config quad).
- Actuators: a single `"dummy"` `ImplicitActuatorCfg` over all joints with **stiffness 0, damping 0** → confirms the props are unpowered/cosmetic (see "one big idea" above).
- **Mass & inertia are not in this Python file** — they're baked into `cf2x.usd` and read at runtime. Nominal Crazyflie 2.x mass ≈ **0.027 kg (~27 g)**; the env never hard-codes it, it queries PhysX so thrust auto-scales to whatever the USD says.

### PPO hyperparameters (`rsl_rl_ppo_cfg.py`) — inventory only, no tuning

- **Runner:** `num_steps_per_env = 24`, `max_iterations = 200` (⚠️ our Phase 1 run did 5000 — that was a **CLI override**, the file default is 200), `save_interval = 50`, `experiment_name = "quadcopter_direct"`.
- **Policy (actor-critic):** `init_noise_std = 1.0`, actor/critic hidden dims `[64, 64]`, `elu` activation, obs normalization **off** for both.
- **Algorithm:** `clip_param = 0.2`, `entropy_coef = 0.0`, `value_loss_coef = 1.0` (clipped), `num_learning_epochs = 5`, `num_mini_batches = 4`, `learning_rate = 5e-4` with `adaptive` schedule, `gamma = 0.99`, `lam = 0.95`, `desired_kl = 0.01`, `max_grad_norm = 1.0`.

These are the knobs available for Phase 4+ tuning; per the SPEC we do **not** touch them yet.

---

## Concepts learned (cont.)

- **Does the stock wrench-control policy transfer to a real Crazyflie 2.1+?** The honest answer separates two axes that are easy to conflate:
  - **Action granularity** (wrench = 1 thrust + 3 moments, *vs.* 4 individual motor commands) — mostly a *representation* choice, not a transfer blocker. A real quad has a **fixed motor-mixing matrix** that converts `(thrust, τ_roll, τ_pitch, τ_yaw)` → 4 motor PWMs; the Crazyflie firmware already does this in its "power distribution" stage. So a thrust+moments policy maps onto a standard interface point in the real stack — the mixing is arithmetic, **not something to learn**. Training a net to rediscover a known constant matrix just buys a harder sim-to-real problem.
  - **Actuator fidelity** (idealized instant-perfect wrench *vs.* a realistic motor model) — **this is what actually breaks transfer.** The stock env applies the commanded wrench instantly, perfectly, from ground-truth state. The real drone has motor spin-up lag (~tens of ms), nonlinear PWM→thrust, battery-voltage sag, **saturation coupling** (all 4 motors share headroom — can't max thrust *and* max yaw at once; the wrench model lets the policy request infeasible combinations), control/sensor latency, and EKF state estimation noise (IMU + Lighthouse) instead of perfect state.
  - **Key takeaway:** you do **not** need per-motor actions to fly the real drone properly. Per-motor control only buys fidelity on saturation/coupling — and without a realistic motor model behind it, 4 ideal thrusters are just as idealized as one ideal wrench. The fix for transfer is a better actuator/sensor model + **domain randomization**, which is exactly what **Phase 4** is for. Action-space granularity is a **Phase 7/8** (hardware, separate spec) decision; the likely deployment is "policy outputs thrust + body-rates, firmware does the mixing," with per-motor as a fallback only if coupling turns out to matter. (Verify the 2.1+ firmware setpoint interfaces against current Bitcraze docs before locking a hardware approach.)

---

## Session Handoff — 2026-06-17 (Phase 3 / Milestone M3 COMPLETE ✅)

**For the next session: M3 (custom fixed-goal hover env) is done and ready to tag `v0.3-hover-custom-env`. Do NOT start M4 (domain randomization) without user approval.**

### M3 deliverables — all met
- ✅ Custom task `Isaac-Crazyflie-Hover-Direct-v0` — project-owned external extension in `source/crazyflie_hover/` (+ launchers `scripts/train.py`, `scripts/play.py`), installed editable into the Isaac Lab Python. Faithful copy of the stock quadcopter env; **fixed-goal hover** (env-origin xy, z=1.0) the only deviation. Env code committed `d2384e3`.
- ✅ Trained to convergence: 4096 envs, 3000 iters, headless. Converged by ~iter 580, flat after. **Final: mean reward 146.12, `final_distance_to_goal` 0.00452 m, `died`=0.0, `ep_len` 499 (full episodes).** Final checkpoint `model_2999.pt`. Log dir `~/IsaacLab/logs/rsl_rl/crazyflie_hover/2026-06-10_00-08-59/`.
- ✅ Success bar (within 0.5 m of the hover point for 95% of an eval episode) **passed with huge margin, and confirmed sustained** — the episode-averaged `distance_to_goal` reward is **14.69 of a 15 max**, meaning the drone is essentially on the goal across the *whole* episode, not just at the final step (final_distance is only the terminal sample).
- ✅ Hover video recorded 2026-06-17: `logs/rsl_rl/crazyflie_hover/2026-06-10_00-08-59/videos/play/rl-video-step-0.mp4` (1280x720, 50 fps, 299 frames, ~6 s).
- ✅ EXPERIMENTS.md row added.

### Gotcha hit during wrap-up
- Stock rsl_rl `play.py` treats `--checkpoint` as a **direct file path**, not a basename within `--load_run`. Passing `--checkpoint model_2999.pt` → `FileNotFoundError`; pass the **full path** (`.../2026-06-10_00-08-59/model_2999.pt`) instead. The `renderD128 (VK_ERROR_INCOMPATIBLE_DRIVER)` line above it in the log is the known-benign aarch64 Vulkan probe, not the cause.

### Remaining for next session
- Commit these docs (show user first), then **tag `v0.3-hover-custom-env`** per CLAUDE.md, and push (use the `gh` HTTPS fallback if ssh-agent isn't loaded — see memory).
- **Ask user before starting M4 (domain randomization).**

### Concept learned — fixed-goal vs random-goal hover
- The custom env's only change from stock is a **fixed** hover target instead of a per-reset random goal. Effect: convergence is **faster and tighter** — converged by ~iter 580 to `final_distance_to_goal` ~4.5 mm, vs the stock random-goal env's ~83 mm in Phase 1. Intuition: a fixed goal removes the goal-generalization burden, so the policy can overfit a single setpoint to near-zero steady-state error. The trade-off (deferred): this policy only knows one hover point; M7 (waypoint following) reintroduces goal generalization on purpose.

### Reminders
- ⚠️ Per CLAUDE.md rule #5, the reward function is now **LOCKED** after M3 — any change needs a proposal first.
- W&B logging still optional/deferred (needs user's API key as env var) — now demoted to optional in SPEC §2 Future work.
