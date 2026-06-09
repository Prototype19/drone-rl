# Experiments

One line per training run. Newest at top.

| Date | Task | Settings | Iters | Final reward | Outcome | Notes |
|------|------|----------|-------|--------------|---------|-------|
| 2026-06-08 | Isaac-Quadcopter-Direct-v0 | rsl_rl PPO, stock cfg, 4096 envs, headless | 5000 | 123.51 | ✅ converged | Phase 1 stock-env training. 1579s (~26min) on GB10. Stable hover: final_distance_to_goal 0.083m, 0 crashes (died=0.0), episodes run full length (ep_len 499). Final checkpoint model_4999.pt. Hover video recorded (1280x720, 50fps, 299 frames). Log dir: logs/rsl_rl/quadcopter_direct/2026-06-09_00-21-05/ |
| 2026-06-08 | Isaac-Cartpole-Direct-v0 | rsl_rl PPO, stock cfg, default num_envs, headless | 150 | 295.35 (from -5.4 @ iter 0) | ✅ converged | Phase 1 smoke test. 32s wall on GB10. Pipeline validated end-to-end: train + checkpoints + TensorBoard. Log dir: logs/rsl_rl/cartpole_direct/2026-06-09_00-13-23/ |
