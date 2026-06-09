# Notes

Conceptual notes as the project progresses.

---

## Session Handoff — 2026-06-08 (Phase 1 in progress)

**For the next Claude session: read this, then check tmux + the run dir before doing anything.**

### Done this session
- ✅ Stack smoke test passed (PyTorch 2.9.0+cu130, CUDA 13.0, isaaclab 0.54.3, CUDA available).
- ✅ Cartpole smoke test: `Isaac-Cartpole-Direct-v0`, 150 iters, reward -5.4 → 295.35, 32s on GB10. Logged in EXPERIMENTS.md, committed (`492245b`). Pipeline validated end-to-end.

### In flight RIGHT NOW (started ~22:21, 2026-06-08)
- **Quadcopter 5000-iter run** running detached in **tmux session `train`** on the Spark.
  - Command: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Quadcopter-Direct-v0 --headless --max_iterations 5000`
  - 4096 envs, ~0.28s/iter, total ETA ~23 min. Reward was ~124-132 and climbing around iter ~1200.
  - Console log: `logs/quadcopter_run1.log`
  - Run dir / checkpoints: `~/IsaacLab/logs/rsl_rl/quadcopter_direct/2026-06-09_00-21-05/` (saves `model_*.pt` every 50 iters; final ~`model_4999.pt`)
- **TensorBoard** running in **tmux session `tb`**, port 6006 → http://192.168.1.219:6006 (logdir = all of `logs/rsl_rl`).
- Note: the `TU: ... renderD128 (VK_ERROR_INCOMPATIBLE_DRIVER)` line in the log is a benign headless-aarch64 Vulkan probe failure, NOT a real error. PhysX runs on CUDA.

### Pending follow-ups when the run finishes (to close out Phase 1)
1. Confirm convergence / hover behavior (check final reward in TensorBoard or log; `tmux attach -t train`).
2. **Record hover video**: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Quadcopter-Direct-v0 --num_envs 16 --headless --video --video_length 300` (loads latest checkpoint automatically). Inspect the MP4.
3. **Log the run in EXPERIMENTS.md** (newest at top) — same table format as the Cartpole row.
4. Still open in Phase 1: **W&B logging setup** (needs user's W&B API key as env var; project `drone-rl`).
5. Commit docs; tag phase boundary when Phase 1 deliverable is fully met (`v0.2-quadcopter-stock-trained` per CLAUDE.md tag convention).
6. Tear down `train` and `tb` tmux sessions once everything is logged.

### Reminders
- Show docs for user approval before staging/committing.
- Don't start Phase 2 without user approval.

---

## Concepts learned

_(none yet — this session was pipeline validation, no new concepts)_
