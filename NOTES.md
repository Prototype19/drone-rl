# Notes

Conceptual notes as the project progresses.

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
