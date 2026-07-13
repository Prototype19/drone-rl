# Copyright (c) 2026, drone-rl project. BSD-3-Clause.
#
# M7 waypoint-following env. Subclasses the M3-M6 hover env and changes ONLY the
# task: instead of a single fixed hover point, the goal is a waypoint sampled in a
# 4 m x 4 m x 3 m workspace that RESAMPLES the moment the drone reaches it, so the
# policy learns to fly a continuous sequence of 3D goals.
#
# Reward (locked-reward exception, SPEC Decision Log 2026-07-12, rule #5): the
# three M3 terms (distance 1-tanh(d/0.8) x15, lin_vel x-0.05, ang_vel x-0.01) are
# UNCHANGED -- inherited verbatim via super()._get_rewards(), now measured to the
# current waypoint -- plus one added DISCRETE success bonus (+success_bonus_scale)
# on the step the drone first enters success_radius of the current waypoint. The
# hover reward/task (Isaac-Crazyflie-Hover-Direct-v0) is untouched.
#
# Everything else (12-D ego-relative obs, wrench action, all M4-M6 domain
# randomization) is inherited unchanged. Termination raises the altitude ceiling
# (2.0 -> 3.5 m) so the taller workspace isn't clipped by the hover death bound.

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from crazyflie_hover.crazyflie_hover_env import CrazyflieHoverEnv, CrazyflieHoverEnvCfg


@configclass
class CrazyflieWaypointEnvCfg(CrazyflieHoverEnvCfg):
    # Longer episodes so the policy sees several waypoint transitions per episode
    # (hover was 10 s; a waypoint chase needs room to reach multiple goals).
    episode_length_s = 20.0

    # Waypoint workspace: a 4 m x 4 m x 3 m box about each env origin. xy is a
    # +/- offset about the origin; z is an absolute height band.
    waypoint_xy_range = 2.0          # m, +/- about env origin -> 4 m span
    waypoint_z_range = (0.5, 3.0)    # m, absolute height band (~3 m tall)

    # Success: a waypoint counts as reached within success_radius (pure proximity,
    # no speed gate -- fly-through following). Reaching grants a one-time discrete
    # bonus and immediately resamples the next waypoint.
    success_radius = 0.15            # m
    success_bonus_scale = 10.0       # one-time reward per waypoint reached

    # Termination bounds. Ceiling raised from the hover env's 2.0 m so the drone
    # can climb to the top of the taller workspace without triggering `died`.
    death_floor = 0.1               # m
    death_ceiling = 3.5             # m

    # Optional scripted path for evaluation/video only: a sequence of (x, y, z)
    # offsets about the env origin. When set, waypoints cycle through this list in
    # order (advancing on reach) instead of random sampling, so the drone traces a
    # known shape (e.g. a square) for the deliverable video. None -> random
    # waypoints (training and random eval). Does not affect the trained policy.
    eval_path: tuple | None = None


@configclass
class CrazyflieWaypointSquareEnvCfg(CrazyflieWaypointEnvCfg):
    # Video variant: fly a fixed 3 m square at 1.5 m altitude (corners as offsets
    # about the env origin). Loads the same trained waypoint policy (the task is
    # goal-relative, so a scripted goal sequence needs no retraining).
    eval_path = (
        (1.5, 1.5, 1.5),
        (1.5, -1.5, 1.5),
        (-1.5, -1.5, 1.5),
        (-1.5, 1.5, 1.5),
    )


class CrazyflieWaypointEnv(CrazyflieHoverEnv):
    cfg: CrazyflieWaypointEnvCfg

    def __init__(self, cfg: CrazyflieWaypointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Add the success-bonus term to the episode-sum log (the inherited
        # _reset_idx logs every key in _episode_sums, so this is picked up for free
        # as Episode_Reward/success_bonus).
        self._episode_sums["success_bonus"] = torch.zeros(self.num_envs, device=self.device)
        # Per-env count of waypoints reached this episode (logged at reset).
        self._waypoints_reached = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        # Scripted-path (video) state: per-env index into cfg.eval_path. Unused when
        # eval_path is None (random waypoints).
        self._path_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.eval_path is not None:
            self._eval_path = torch.tensor(self.cfg.eval_path, dtype=torch.float, device=self.device)

    def _sample_waypoint(self, env_ids: torch.Tensor) -> None:
        """Sample a fresh waypoint in the workspace for ``env_ids`` (M7).

        xy is a uniform offset about the env origin; z is an absolute band. Used
        both at reset and mid-episode when a waypoint is reached.
        """
        origins = self._terrain.env_origins[env_ids]
        # Scripted path (video): cycle to the next fixed offset for each env.
        if self.cfg.eval_path is not None:
            idx = self._path_idx[env_ids] % self._eval_path.shape[0]
            self._desired_pos_w[env_ids] = origins + self._eval_path[idx]
            self._path_idx[env_ids] += 1
            return
        # Random waypoint in the workspace (training / random eval).
        n = len(env_ids)
        r = self.cfg.waypoint_xy_range
        self._desired_pos_w[env_ids, 0] = origins[:, 0] + self._rand(n, -r, r)
        self._desired_pos_w[env_ids, 1] = origins[:, 1] + self._rand(n, -r, r)
        self._desired_pos_w[env_ids, 2] = self._rand(n, self.cfg.waypoint_z_range[0], self.cfg.waypoint_z_range[1])

    def _get_rewards(self) -> torch.Tensor:
        # Inherited three-term hover reward (measured to the current waypoint), plus
        # the M7 discrete success bonus. super() also accumulates the three terms'
        # episode sums, so they stay logged exactly as before.
        reward = super()._get_rewards()
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached = distance_to_goal < self.cfg.success_radius
        bonus = reached.float() * self.cfg.success_bonus_scale
        reward = reward + bonus
        # Logging / bookkeeping
        self._episode_sums["success_bonus"] += bonus
        self._waypoints_reached += reached.int()
        # Resample the next waypoint for every env that just reached one. This runs
        # before _get_observations, so the next obs already points at the new goal.
        reached_ids = reached.nonzero(as_tuple=False).squeeze(-1)
        if reached_ids.numel() > 0:
            self._sample_waypoint(reached_ids)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_z = self._robot.data.root_pos_w[:, 2]
        died = torch.logical_or(root_z < self.cfg.death_floor, root_z > self.cfg.death_ceiling)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Inherited reset: logging, robot state, all M4-M6 randomization, and the
        # (fixed-point) goal assignment -- which we override with a sampled waypoint
        # just below. final_distance_to_goal logged by super() is to the episode's
        # last active waypoint.
        super()._reset_idx(env_ids)
        # M7 metric: waypoints reached this episode (before zeroing the counter).
        self.extras["log"]["Metrics/waypoints_reached"] = (
            self._waypoints_reached[env_ids].float().mean().item()
        )
        # Replace the inherited fixed goal with a fresh workspace waypoint and clear
        # the per-episode counters. Reset the scripted-path index first so a video
        # episode starts at the first path point.
        self._path_idx[env_ids] = 0
        self._sample_waypoint(env_ids)
        self._waypoints_reached[env_ids] = 0
