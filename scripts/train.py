#!/usr/bin/env python
# Copyright (c) 2026, drone-rl project. BSD-3-Clause.

"""Project train launcher.

Registers the drone-rl external tasks (importing ``crazyflie_hover`` runs its
``gym.register`` -- import-light, no SimulationApp needed) and then delegates to
Isaac Lab's stock rsl_rl ``train.py``. Use exactly like the stock script:

    ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/train.py \
        --task Isaac-Crazyflie-Hover-Direct-v0 --headless --num_envs 4096

All CLI args are forwarded verbatim to the stock script.
"""

import os
import runpy
import sys

ISAACLAB_TRAIN = os.path.expanduser(
    "~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
)

# The stock script does a directory-local ``import cli_args``; put its directory
# on sys.path so that import resolves when we run it via runpy.
sys.path.insert(0, os.path.dirname(ISAACLAB_TRAIN))

# Register our external task(s) before handing off. Safe to do pre-app-launch:
# gym.register stores entry points as strings and does not import isaaclab here.
import crazyflie_hover  # noqa: E402, F401

runpy.run_path(ISAACLAB_TRAIN, run_name="__main__")
