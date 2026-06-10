#!/usr/bin/env python
# Copyright (c) 2026, drone-rl project. BSD-3-Clause.

"""Project play launcher.

Registers the drone-rl external tasks and delegates to Isaac Lab's stock rsl_rl
``play.py``. Use exactly like the stock script:

    ./isaaclab.sh -p ~/spark-dev-workspace/drone-rl/scripts/play.py \
        --task Isaac-Crazyflie-Hover-Direct-v0 --num_envs 16 --headless \
        --video --video_length 300

All CLI args are forwarded verbatim to the stock script.
"""

import os
import runpy
import sys

ISAACLAB_PLAY = os.path.expanduser(
    "~/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"
)

# The stock script does a directory-local ``import cli_args``; put its directory
# on sys.path so that import resolves when we run it via runpy.
sys.path.insert(0, os.path.dirname(ISAACLAB_PLAY))

# Register our external task(s) before handing off (import-light, pre-app-launch).
import crazyflie_hover  # noqa: E402, F401

runpy.run_path(ISAACLAB_PLAY, run_name="__main__")
