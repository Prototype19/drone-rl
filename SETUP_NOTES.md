## Initial setup verified — [today's date]

### Environment
- OS: Ubuntu 24.04.4 LTS (aarch64)
- Kernel: 6.17.0-1021-nvidia
- GPU: NVIDIA GB10
- Driver: 580.159.03
- Disk: 126 GB used / 3.7 TB total
- RAM: 121 GiB total, 117 GiB free

### Software stack
- Isaac Sim: 5.1.0-rc.19 (commit aa503a9), built from source
- Isaac Lab: 2.3.2 (commit a859a5f9d), package isaaclab 0.54.3
- Python (bundled): 3.11.13
- PyTorch: 2.9.0+cu130
- CUDA: 13.0
- rsl-rl-lib: 5.0.1

### Paths
- $ISAACSIM_PATH = /home/daron/IsaacSim/_build/linux-aarch64/release
- Bundled Python: ~/IsaacLab/_isaac_sim/python.sh
- Project root: ~/spark-dev-workspace/drone-rl/

### Verified working
- nvidia-smi detects GPU
- Isaac Sim launches (in non-headless mode → windowing errors as expected on aarch64, but loads to "App is loaded" state)
- Isaac Lab tutorials are runnable with --headless

### Known limitations on this aarch64 platform
See SPEC.md §4 — livestream, OBJ import, Hub Cache, App Template, cuRobo, App Selector all unavailable.
