# Copyright (c) 2026, drone-rl project. BSD-3-Clause.

"""Crazyflie fixed-point hover task (drone-rl Phase 3).

A faithful copy of the stock ``Isaac-Quadcopter-Direct-v0`` env, specialized to
hover at a fixed point. Same 12-D ego-centric observation, same wrench action
(1 collective thrust + 3 body moments), and same 3-term reward as the stock env
-- the only deviation is that the goal is a constant hover point instead of a
per-episode random target (see ``crazyflie_hover_env.py``).

Registration is intentionally import-light: ``gym.register`` stores the entry
points as strings, so importing this package does NOT pull in isaaclab and does
NOT require a running SimulationApp. That lets a thin launcher import this
package before delegating to Isaac Lab's stock train/play scripts.
"""

import gymnasium as gym

##
# Register Gym environment.
##

gym.register(
    id="Isaac-Crazyflie-Hover-Direct-v0",
    entry_point="crazyflie_hover.crazyflie_hover_env:CrazyflieHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "crazyflie_hover.crazyflie_hover_env:CrazyflieHoverEnvCfg",
        "rsl_rl_cfg_entry_point": "crazyflie_hover.agents.rsl_rl_ppo_cfg:CrazyflieHoverPPORunnerCfg",
    },
)

# M7: waypoint-following variant. Subclasses the hover env; tracks a sequence of
# resampling 3D goals in a 4x4x3 m workspace with an added success bonus. See
# crazyflie_waypoint_env.py.
gym.register(
    id="Isaac-Crazyflie-Waypoint-Direct-v0",
    entry_point="crazyflie_hover.crazyflie_waypoint_env:CrazyflieWaypointEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "crazyflie_hover.crazyflie_waypoint_env:CrazyflieWaypointEnvCfg",
        "rsl_rl_cfg_entry_point": "crazyflie_hover.agents.rsl_rl_ppo_cfg:CrazyflieWaypointPPORunnerCfg",
    },
)
