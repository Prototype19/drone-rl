# Copyright (c) 2026, drone-rl project. BSD-3-Clause.

"""Agent (RL algorithm) configurations for the Crazyflie hover task.

Kept import-light: the PPO config module is referenced by string entry point in
the task registration and imported lazily at run time, so this package's import
does not require isaaclab_rl or a running SimulationApp.
"""
