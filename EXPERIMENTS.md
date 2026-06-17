# Experiments

One line per training run. Newest at top.

| Date | Task | Settings | Iters | Final reward | Outcome | Notes |
|------|------|----------|-------|--------------|---------|-------|
| 2026-06-10 | Isaac-Crazyflie-Hover-Direct-v0 | rsl_rl PPO, custom env, fixed-goal hover (env-origin xy, z=1.0), 4096 envs, headless | 3000 | 146.12 | ✅ converged | M3 (Phase 3) custom hover env. Converged by ~iter 580, then flat. final_distance_to_goal 0.00452m, died=0.0, ep_len 499 (full episodes). Episode-avg distance_to_goal reward 14.69/15 → sustained on-goal hover, not just terminal — passes the "within 0.5m for 95% of episode" bar with huge margin. Hover video 1280x720, 50fps, 299 frames (recorded 2026-06-17). Log dir: logs/rsl_rl/crazyflie_hover/2026-06-10_00-08-59/ (final checkpoint model_2999.pt). |
| 2026-06-08 | Isaac-Quadcopter-Direct-v0 | rsl_rl PPO, stock cfg, 4096 envs, headless | 5000 | 123.51 | ✅ converged | Phase 1 stock-env training. 1579s (~26min) on GB10. Stable hover: final_distance_to_goal 0.083m, 0 crashes (died=0.0), episodes run full length (ep_len 499). Final checkpoint model_4999.pt. Hover video recorded (1280x720, 50fps, 299 frames). Log dir: logs/rsl_rl/quadcopter_direct/2026-06-09_00-21-05/ |
| 2026-06-08 | Isaac-Cartpole-Direct-v0 | rsl_rl PPO, stock cfg, default num_envs, headless | 150 | 295.35 (from -5.4 @ iter 0) | ✅ converged | Phase 1 smoke test. 32s wall on GB10. Pipeline validated end-to-end: train + checkpoints + TensorBoard. Log dir: logs/rsl_rl/cartpole_direct/2026-06-09_00-13-23/ |
