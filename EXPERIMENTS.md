# Experiments

One line per training run. Newest at top.

| Date | Task | Settings | Iters | Final reward | Outcome | Notes |
|------|------|----------|-------|--------------|---------|-------|
| 2026-06-09 | Isaac-Cartpole-Direct-v0 | rsl_rl PPO, stock cfg, default num_envs, headless | 150 | 295.35 (from -5.4 @ iter 0) | ✅ converged | Phase 1 smoke test. 32s wall on GB10. Pipeline validated end-to-end: train + checkpoints + TensorBoard. Log dir: logs/rsl_rl/cartpole_direct/2026-06-09_00-13-23/ |
